#!/usr/bin/env python3
"""API-only Luna evaluation on frozen test inputs with the condition-check prompt.

No training, CUDA imports, local model loads, or signals to other processes.
Prepare once; run a small prefix, then resume the same run without regenerating it.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from eval_gui_reflection import parse_verdict, save_json, summarize
from eval_qreflector_prompt import render_user
from qreflector_luna_correction import read_api_key
from qreflector_sft_bf16 import (
    DATA,
    ROOT,
    digest,
    load_images,
    read_rows,
    strict_structure,
    validate_dataset,
)

ENDPOINT = "https://api.openai.com/v1/responses"
COMPARATOR = ROOT / "artifacts/evaluations/qreflector_epoch9_condition_check_v1_20261006_gpu0"


def now():
    return datetime.now(UTC).isoformat()


def append_json(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_request(user, config):
    """Accept only a user message, so references never enter the API request."""
    if user["role"] != "user" or [c["type"] for c in user["content"]] != ["text", "image", "image"]:
        raise ValueError("Expected user text followed by BEFORE and AFTER")
    content = [{"type": "input_text", "text": user["content"][0]["text"]}]
    image_hashes = []
    for image in load_images({"messages": [user]}, DATA, config["image_max_side"]):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        pixels = buffer.getvalue()
        image_hashes.append(hashlib.sha256(pixels).hexdigest())
        content.append({"type": "input_image", "detail": "high",
                        "image_url": "data:image/png;base64," + base64.b64encode(pixels).decode()})
    payload = {"model": config["model"], "store": False,
               "reasoning": {"effort": config["reasoning_effort"]},
               "max_output_tokens": config["max_output_tokens"],
               "input": [{"role": "user", "content": content}]}
    # No additional system instructions, schema constraint, reference or tools.
    return payload, image_hashes


def response_record(response):
    text = "".join(c.get("text", "") for item in response.get("output", [])
                   if item.get("type") == "message" for c in item.get("content", [])
                   if c.get("type") == "output_text")
    refusals = [c.get("refusal", "") for item in response.get("output", [])
                if item.get("type") == "message" for c in item.get("content", [])
                if c.get("type") == "refusal"]
    return {"response_id": response.get("id"), "response_model": response.get("model"),
            "response_status": response.get("status"), "full_response": text,
            "refusals": refusals, "incomplete_details": response.get("incomplete_details"),
            "usage": response.get("usage") or {}}


def score_response(record, expected):
    # A truncated/refused answer is invalid even if a partial verdict is visible.
    predicted = (parse_verdict(record["full_response"])
                 if record["response_status"] == "completed" and not record.get("refusals") else None)
    return {**record, "expected": expected, "predicted": predicted,
            "correct": predicted == expected,
            "valid_four_field_json": strict_structure(record["full_response"])}


def prepare(output):
    validate_dataset(DATA, images=True)
    source = ROOT / "docs/prompts/qreflector_condition_check_v1.txt"
    original = (DATA / "prompt_template.txt").read_text()
    replacement = source.read_text()
    comparator_plan = json.loads((COMPARATOR / "plan.json").read_text())
    assert comparator_plan["prompt_sha256"] == digest(source)
    assert comparator_plan["data_sha256"] == digest(DATA / "test.jsonl")
    assert comparator_plan["image_max_side"] == 512
    comparator_inputs = read_rows(COMPARATOR / "inputs_manifest.jsonl")
    inputs = []
    rows = read_rows(DATA / "test.jsonl")
    for i, row in enumerate(rows, 1):
        user, metadata = render_user(row, original, replacement)
        assert {"sample_id": i, **metadata} == comparator_inputs[i - 1]
        inputs.append({"sample_id": i, "user": user, **metadata})
    assert len(inputs) == 1082
    output = output.resolve()
    if ROOT / "artifacts/evaluations" not in output.parents:
        raise ValueError("Use a new artifacts/evaluations subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "responses").mkdir()
    (output / "prompt_template.txt").write_text(replacement)
    with (output / "inputs_manifest.jsonl").open("x") as stream:
        for record in inputs:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    files = [Path(__file__), source, DATA / "test.jsonl", DATA / "prompt_template.txt",
             DATA / "FROZEN.json", DATA / "image_sha256.json", COMPARATOR / "plan.json",
             COMPARATOR / "inputs_manifest.jsonl", output / "inputs_manifest.jsonl",
             output / "prompt_template.txt"]
    files += [ROOT / "scripts" / name for name in (
        "eval_gui_reflection.py", "eval_qreflector_prompt.py", "qreflector_sft_bf16.py",
        "run_qreflector_bf16.py", "qreflector_luna_correction.py", "qreflector_opsd.py")]
    plan = {"created_utc": now(), "model": "gpt-6-luna", "endpoint": ENDPOINT,
            "reasoning_effort": "medium", "max_output_tokens": 4096,
            "image_max_side": 512, "image_detail": "high", "rows": 1082,
            "concurrency": 4, "max_api_calls": 1200, "timeout_seconds": 90,
            "retry_policy": "At most 3 attempts on transport/429/5xx only; never retry incorrect or malformed model answers",
            "output_constraints": "Prompt-only JSON; no API schema or extra instructions",
            "scoring": "Final Yes/No only; invalid, refused or incomplete outputs count as wrong",
            "reference_in_api_input": False, "gpu_used": False,
            "comparison_run": str(COMPARATOR), "prompt_sha256": digest(source),
            "data_sha256": digest(DATA / "test.jsonl"),
            "comparison_note": "Identical user text, ordered screenshots and 512px resize to epoch 9. Model preprocessing and decoding differ: Luna medium reasoning, 4096 reasoning+output token cap; Qwen greedy, 1024 generated tokens. Not compute-budget matched.",
            "pinned_files": {str(p.resolve()): digest(p) for p in files}}
    save_json(output / "plan.json", plan)
    save_json(output / "status.json", {"status": "prepared", "completed": 0, "total": 1082})
    print(json.dumps({"prepared": str(output), "rows": 1082, "model": plan["model"]}), flush=True)


class APIClient:
    def __init__(self, run, config):
        self.run, self.config = run, config
        self.key = read_api_key(ROOT / ".env")
        self.ledger = run / "api_requests.jsonl"
        self.lock = threading.Lock()
        self.calls = sum(r.get("event") == "request_started" for r in (
            read_rows(self.ledger) if self.ledger.exists() else []))

    def event(self, item):
        with self.lock:
            append_json(self.ledger, {"utc": now(), **item})

    def generate(self, sample):
        payload, image_hashes = build_request(sample["user"], self.config)
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        destination = self.run / "responses" / f'{sample["sample_id"]:04d}.json'
        if destination.exists():
            cached = json.loads(destination.read_text())
            if cached["request_sha256"] != fingerprint:
                raise ValueError("Cached request does not match current inputs")
            return cached
        started = time.monotonic()
        for attempt in range(1, 4):
            with self.lock:
                if self.calls >= self.config["max_api_calls"]:
                    raise RuntimeError("API request cap reached")
                self.calls += 1
                append_json(self.ledger, {"utc": now(), "event": "request_started",
                    "sample_id": sample["sample_id"], "attempt": attempt,
                    "call_number": self.calls, "request_sha256": fingerprint})
            request = Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
                "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=self.config["timeout_seconds"]) as reply:
                    response = json.load(reply)
            except HTTPError as error:
                self.event({"event": "http_error", "sample_id": sample["sample_id"],
                            "status": error.code, "attempt": attempt})
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f"API HTTP {error.code}; evaluation can be resumed") from None
                try:
                    delay = max(2 ** attempt, float(error.headers.get("Retry-After", "0")))
                except (ValueError, TypeError):
                    delay = 2 ** attempt
                if delay > 60:
                    raise RuntimeError("API requested retry delay greater than 60 seconds") from None
                time.sleep(delay)
                continue
            except (URLError, TimeoutError, ConnectionError):
                self.event({"event": "network_error", "sample_id": sample["sample_id"], "attempt": attempt})
                if attempt == 3:
                    raise RuntimeError("API network failure after three attempts") from None
                time.sleep(2 ** attempt)
                continue
            record = {"sample_id": sample["sample_id"], **response_record(response),
                      "request_sha256": fingerprint, "prompt_sha256": sample["prompt_sha256"],
                      "context_sha256": sample["context_sha256"], "image_paths": sample["image_paths"],
                      "submitted_image_png_sha256": image_hashes,
                      "latency_seconds": time.monotonic() - started, "completed_utc": now()}
            save_json(destination, record)
            self.event({"event": "response_received", "sample_id": sample["sample_id"],
                        "response_status": record["response_status"], "usage": record["usage"]})
            return record
        raise RuntimeError("Unreachable retry state")


def report(run, results, status, started, error=None):
    summary = summarize(results)
    summary["valid_four_field_json_rate"] = (sum(r["valid_four_field_json"] for r in results) / len(results)
                                             if results else None)
    summary["api_incomplete_outputs"] = sum(r["response_status"] != "completed" for r in results)
    summary["usage"] = {key: sum(r["usage"].get(key, 0) for r in results)
                        for key in ("input_tokens", "output_tokens", "total_tokens")}
    summary["reasoning_tokens"] = sum(r["usage"].get("output_tokens_details", {}).get("reasoning_tokens", 0)
                                      for r in results)
    save_json(run / "summary.json", summary)
    save_json(run / "status.json", {"status": status, "pid": os.getpid(), "updated_utc": now(),
        "completed": len(results), "total": 1082, "elapsed_this_invocation_seconds": time.monotonic() - started,
        "error": error})


def paired_report(run, results):
    path = COMPARATOR / "predictions.jsonl"
    if not path.exists():
        return
    peers = {r["sample_id"]: r for r in read_rows(path)}
    pairs = [(r, peers[r["sample_id"]]) for r in results if r["sample_id"] in peers]
    for a, b in pairs:
        assert a["prompt_sha256"] == b["prompt_sha256"] and a["expected"] == b["expected"]
        assert a["image_paths"] == b["image_paths"]
    save_json(run / "paired_epoch9.json", {"matched_rows": len(pairs), "total": 1082,
        "complete_comparison": len(pairs) == 1082,
        "both_correct": sum(a["correct"] and b["correct"] for a, b in pairs),
        "luna_only_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
        "qreflector_only_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
        "both_wrong": sum(not a["correct"] and not b["correct"] for a, b in pairs),
        "luna_summary": summarize([a for a, _ in pairs]),
        "qreflector_summary": summarize([b for _, b in pairs])})


def evaluate(run, limit):
    run = run.resolve()
    with (run / ".evaluation.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((run / "plan.json").read_text())
        for name, expected in config["pinned_files"].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"Pinned file changed: {name}")
        samples = read_rows(run / "inputs_manifest.jsonl")
        rows = read_rows(DATA / "test.jsonl")
        labels = {i: json.loads(r["messages"][1]["content"][0]["text"])["answer"].lower()
                  for i, r in enumerate(rows, 1)}
        path = run / "predictions.jsonl"
        results = read_rows(path) if path.exists() else []
        ids = {r["sample_id"] for r in results}
        assert len(ids) == len(results)
        for r in results:
            assert r["expected"] == labels[r["sample_id"]]
            assert r["prompt_sha256"] == samples[r["sample_id"] - 1]["prompt_sha256"]
            assert score_response(r, labels[r["sample_id"]]) == r
        client = APIClient(run, config)
        started = time.monotonic()
        pending = iter(s for s in samples[:limit] if s["sample_id"] not in ids)
        error = None
        report(run, results, "running", started)
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            futures = {}

            def refill():
                while not error and len(futures) < config["concurrency"]:
                    sample = next(pending, None)
                    if sample is None:
                        break
                    futures[pool.submit(client.generate, sample)] = sample["sample_id"]

            refill()
            while futures:
                ready, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in ready:
                    sample_id = futures.pop(future)
                    try:
                        record = score_response(future.result(), labels[sample_id])
                        append_json(path, record)
                        results.append(record)
                    except Exception as exc:
                        # Never print raw API bodies, headers or credentials.
                        error = f"sample {sample_id}: {type(exc).__name__}"
                        print(error, flush=True)
                report(run, results, "draining_after_error" if error else "running", started, error)
                if ready:
                    print(json.dumps({"completed": len(results), "total": 1082,
                                      "correct": sum(r["correct"] for r in results)}), flush=True)
                refill()
        report(run, results, "failed" if error else ("completed" if len(results) == 1082 else "paused_after_prefix"),
               started, error)
        paired_report(run, results)
        if error:
            raise RuntimeError("Evaluation stopped after an API or local error; inspect the sanitized ledger")
        if len(results) == 1082:
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                         for r in sorted(results, key=lambda r: r["sample_id"])))
            temporary.replace(path)
        print(json.dumps({"status": "completed" if len(results) == 1082 else "paused_after_prefix",
                          "summary": summarize(results)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1082)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1082:
        parser.error("limit must be between 1 and 1082")
    if args.mode == "prepare":
        prepare(args.output)
    else:
        evaluate(args.output, args.limit)


if __name__ == "__main__":
    main()
