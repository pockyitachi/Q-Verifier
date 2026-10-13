#!/usr/bin/env python3
"""Independent Haiku API evaluation using the exact prepared Luna test inputs."""
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
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from eval_gui_reflection import save_json, summarize
from eval_qreflector_luna import append_json, now, score_response
from qreflector_sft_bf16 import DATA, ROOT, digest, load_images, read_rows, validate_dataset

MODEL = "claude-haiku-4-5-20251001"
ENDPOINT = "https://api.anthropic.com/v1/messages"
SOURCE = ROOT / "artifacts/evaluations/qreflector_luna_condition_check_v1_20261006_api"
QREF = ROOT / "artifacts/evaluations/qreflector_epoch9_condition_check_v1_20261006_gpu0"


def read_key(env_file):
    value = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not value and env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if "=" not in line or line.startswith("#"):
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "ANTHROPIC_API_KEY":
                value = candidate.strip().strip("\"'")
                break
    if not value or value.lower() in ("your-api-key", "your_anthropic_api_key", "changeme"):
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    return value


def build_request(user, config):
    if user["role"] != "user" or [c["type"] for c in user["content"]] != ["text", "image", "image"]:
        raise ValueError("Expected exactly one user text and two ordered screenshots")
    content = [{"type": "text", "text": user["content"][0]["text"]}]
    hashes = []
    for image in load_images({"messages": [user]}, DATA, config["image_max_side"]):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        hashes.append(hashlib.sha256(data).hexdigest())
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.b64encode(data).decode()}})
    return {"model": config["model"], "max_tokens": config["max_output_tokens"],
            "thinking": {"type": "enabled", "budget_tokens": config["thinking_budget_tokens"]},
            "messages": [{"role": "user", "content": content}]}, hashes


def response_record(response):
    blocks = response.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    raw_usage = response.get("usage") or {}
    input_tokens = sum(raw_usage.get(k, 0) or 0 for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    output_tokens = raw_usage.get("output_tokens", 0) or 0
    stop = response.get("stop_reason")
    thinking_count = (raw_usage.get("output_tokens_details") or {}).get("thinking_tokens")
    return {"response_id": response.get("id"), "response_model": response.get("model"),
            "response_status": "completed" if stop == "end_turn" else "incomplete",
            "stop_reason": stop, "full_response": text,
            "refusals": ["provider_refusal"] if stop == "refusal" else [],
            "thinking_block_present": any(b.get("type") in ("thinking", "redacted_thinking") for b in blocks),
            "thinking_tokens": thinking_count, "provider_usage": raw_usage,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                      "total_tokens": input_tokens + output_tokens}}


def prepare(run):
    run = run.resolve()
    if ROOT / "artifacts/evaluations" not in run.parents:
        raise ValueError("Use a new artifacts/evaluations directory")
    validate_dataset(DATA, images=True)
    source_plan = json.loads((SOURCE / "plan.json").read_text())
    for path, expected in source_plan["pinned_files"].items():
        assert digest(Path(path)) == expected, path
    inputs = read_rows(SOURCE / "inputs_manifest.jsonl")
    assert len(inputs) == 1082
    run.mkdir(parents=True, exist_ok=False)
    (run / "responses").mkdir()
    for name in ("prompt_template.txt", "inputs_manifest.jsonl"):
        (run / name).write_bytes((SOURCE / name).read_bytes())
    files = {**source_plan["pinned_files"], str(SOURCE / "plan.json"): digest(SOURCE / "plan.json"),
             str(Path(__file__).resolve()): digest(Path(__file__)),
             str(run / "prompt_template.txt"): digest(run / "prompt_template.txt"),
             str(run / "inputs_manifest.jsonl"): digest(run / "inputs_manifest.jsonl")}
    config = {"created_utc": now(), "model": MODEL, "endpoint": ENDPOINT,
              "api_version": "2023-06-01", "rows": 1082, "image_max_side": 512,
              "thinking_budget_tokens": 2048, "max_output_tokens": 4096,
              "concurrency": 4, "max_api_calls": 1200, "timeout_seconds": 90,
              "prompt_sha256": source_plan["prompt_sha256"], "data_sha256": source_plan["data_sha256"],
              "pinned_files": files, "gpu_used": False, "reference_in_api_input": False,
              "scoring": "Final Yes/No only. Incomplete/refused/unparseable answers count as wrong.",
              "retry_policy": "Three attempts maximum for transport/429/5xx only; never retry based on correctness or output format",
              "output_constraints": "Same prompt-only JSON requirement as Luna; no API schema, extra system prompt or tools",
              "comparison_note": "User text, image order and resized PNG pixels match Luna exactly. Haiku thinking budget 2048 is not equivalent to Luna medium reasoning; model tokenization and decoding differ.",
              "source_run": str(SOURCE)}
    save_json(run / "plan.json", config)
    save_json(run / "status.json", {"status": "prepared_awaiting_api_key", "completed": 0, "total": 1082})
    print(json.dumps({"prepared": str(run), "rows": 1082, "model": MODEL}), flush=True)


class Client:
    def __init__(self, run, config, env_file):
        self.run, self.config = run, config
        self.key = read_key(env_file)
        self.lock = threading.Lock()
        self.ledger = run / "api_requests.jsonl"
        self.calls = sum(r.get("event") == "request_started" for r in (
            read_rows(self.ledger) if self.ledger.exists() else []))

    def event(self, record):
        with self.lock:
            append_json(self.ledger, {"utc": now(), **record})

    def generate(self, sample):
        payload, hashes = build_request(sample["user"], self.config)
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        destination = self.run / "responses" / f'{sample["sample_id"]:04d}.json'
        if destination.exists():
            cached = json.loads(destination.read_text())
            assert cached["request_sha256"] == fingerprint
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
                "x-api-key": self.key, "anthropic-version": self.config["api_version"],
                "Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=self.config["timeout_seconds"]) as reply:
                    response = json.load(reply)
            except HTTPError as exc:
                self.event({"event": "http_error", "sample_id": sample["sample_id"],
                            "http_status": exc.code, "attempt": attempt})
                if exc.code not in (429, 500, 502, 503, 504, 529) or attempt == 3:
                    raise RuntimeError(f"Anthropic HTTP {exc.code}") from None
                try:
                    delay = max(2 ** attempt, float(exc.headers.get("Retry-After", "0")))
                except (ValueError, TypeError):
                    delay = 2 ** attempt
                if delay > 60:
                    raise RuntimeError("Retry delay exceeds 60 seconds; resume later") from None
                time.sleep(delay)
                continue
            except (URLError, TimeoutError, ConnectionError):
                self.event({"event": "network_error", "sample_id": sample["sample_id"], "attempt": attempt})
                if attempt == 3:
                    raise RuntimeError("Anthropic transport failure") from None
                time.sleep(2 ** attempt)
                continue
            if response.get("type") != "message" or response.get("model") != self.config["model"]:
                raise RuntimeError("Unexpected Anthropic response type or model")
            record = {"sample_id": sample["sample_id"], **response_record(response),
                "request_sha256": fingerprint, "prompt_sha256": sample["prompt_sha256"],
                "context_sha256": sample["context_sha256"], "image_paths": sample["image_paths"],
                "submitted_image_png_sha256": hashes, "latency_seconds": time.monotonic() - started,
                "completed_utc": now()}
            save_json(destination, record)
            self.event({"event": "response_received", "sample_id": sample["sample_id"],
                        "stop_reason": record["stop_reason"], "usage": record["provider_usage"]})
            return record
        raise RuntimeError("Unreachable retry state")


def report(run, results, status, started, error=None):
    summary = summarize(results)
    summary["valid_four_field_json_rate"] = (sum(r["valid_four_field_json"] for r in results) / len(results)
                                             if results else None)
    summary["api_incomplete_outputs"] = sum(r["response_status"] != "completed" for r in results)
    summary["usage"] = {k: sum(r["usage"].get(k, 0) for r in results)
                        for k in ("input_tokens", "output_tokens", "total_tokens")}
    counts = [r["thinking_tokens"] for r in results if r.get("thinking_tokens") is not None]
    summary["thinking_tokens"] = sum(counts) if counts else None
    summary["thinking_token_reporting_rows"] = len(counts)
    save_json(run / "summary.json", summary)
    save_json(run / "status.json", {"status": status, "pid": os.getpid(), "updated_utc": now(),
        "completed": len(results), "total": 1082,
        "elapsed_this_invocation_seconds": time.monotonic() - started, "error": error})


def comparisons(run, results):
    for name, directory in (("luna", SOURCE), ("qreflector_epoch9", QREF)):
        path = directory / "predictions.jsonl"
        if not path.exists():
            continue
        # Other evaluations may still be appending; ignore an unfinished final line.
        raw = path.read_bytes()
        complete = raw[:raw.rfind(b"\n") + 1]
        peers = {r["sample_id"]: r for r in (json.loads(line) for line in complete.splitlines())}
        pairs = [(r, peers[r["sample_id"]]) for r in results if r["sample_id"] in peers]
        for a, b in pairs:
            assert a["expected"] == b["expected"] and a["prompt_sha256"] == b["prompt_sha256"]
            assert a["image_paths"] == b["image_paths"]
            if "submitted_image_png_sha256" in b:
                assert a["submitted_image_png_sha256"] == b["submitted_image_png_sha256"]
        save_json(run / f"paired_{name}.json", {"matched_rows": len(pairs), "total": 1082,
            "complete_comparison": len(pairs) == 1082,
            "both_correct": sum(a["correct"] and b["correct"] for a, b in pairs),
            "haiku_only_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
            "comparator_only_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
            "both_wrong": sum(not a["correct"] and not b["correct"] for a, b in pairs),
            "haiku_summary": summarize([a for a, _ in pairs]),
            "comparator_summary": summarize([b for _, b in pairs])})


def evaluate(run, limit, env_file):
    run = run.resolve()
    with (run / ".evaluation.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((run / "plan.json").read_text())
        for path, expected in config["pinned_files"].items():
            assert digest(Path(path)) == expected, f"Pinned file changed: {path}"
        samples = read_rows(run / "inputs_manifest.jsonl")
        labels = {i: json.loads(r["messages"][1]["content"][0]["text"])["answer"].lower()
                  for i, r in enumerate(read_rows(DATA / "test.jsonl"), 1)}
        output = run / "predictions.jsonl"
        results = read_rows(output) if output.exists() else []
        ids = {r["sample_id"] for r in results}
        assert len(ids) == len(results)
        for record in results:
            assert record["prompt_sha256"] == samples[record["sample_id"] - 1]["prompt_sha256"]
            assert score_response(record, labels[record["sample_id"]]) == record
        started = time.monotonic()
        try:
            client = Client(run, config, env_file)
        except RuntimeError:
            report(run, results, "awaiting_api_key", started, "ANTHROPIC_API_KEY is not configured")
            raise
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
                        append_json(output, record)
                        results.append(record)
                    except Exception as exc:
                        error = f"sample {sample_id}: {type(exc).__name__}"
                        print(error, flush=True)
                report(run, results, "draining_after_error" if error else "running", started, error)
                if ready:
                    print(json.dumps({"completed": len(results), "total": 1082,
                                      "correct": sum(r["correct"] for r in results)}), flush=True)
                refill()
        comparisons(run, results)
        status = "failed" if error else ("completed" if len(results) == 1082 else "paused_after_prefix")
        report(run, results, status, started, error)
        if error:
            raise RuntimeError("Haiku evaluation stopped; inspect sanitized API ledger and resume")
        if len(results) == 1082:
            temporary = output.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                         for r in sorted(results, key=lambda r: r["sample_id"])))
            temporary.replace(output)
        print(json.dumps({"status": status, "summary": summarize(results)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "check-key"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--limit", type=int, default=1082)
    args = parser.parse_args()
    if args.mode == "check-key":
        print(json.dumps({"anthropic_key_configured": bool(read_key(args.env_file))}))
        return
    if not args.output or not 1 <= args.limit <= 1082:
        parser.error("provide --output and a limit between 1 and 1082")
    if args.mode == "prepare":
        prepare(args.output)
    else:
        evaluate(args.output, args.limit, args.env_file)


if __name__ == "__main__":
    main()
