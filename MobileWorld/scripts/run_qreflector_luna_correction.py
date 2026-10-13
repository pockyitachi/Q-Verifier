#!/usr/bin/env python3
"""Run an explicitly authorized Luna correction pilot and its final offline test."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from eval_gui_reflection import summarize
from qreflector_luna_correction import ROOT, load_plan
from qreflector_opsd import digest, read_json, write_json
from run_qreflector_bf16 import verify_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run, smoke = args.run_dir.resolve(), args.smoke_run.resolve()
    plan, _ = load_plan(run)
    smoke_plan, _ = load_plan(smoke)
    config = plan["config"]
    if not all(config[k] == smoke_plan["config"][k] for k in config if k != "max_steps"):
        raise ValueError("Smoke and production configurations differ")
    if smoke_plan["optimizer_batches"] != 1 or plan["optimizer_batches"] != 111:
        raise ValueError("Expected one smoke batch and 111 production batches")
    python = str(ROOT / "artifacts/envs/qreflector-native/bin/python")
    evaluation = run / "test-final"
    commands = [
        ("train", [python, str(ROOT / "scripts/qreflector_luna_correction.py"), "train",
                   "--run-dir", str(run), "--gpu-uuid", args.gpu_uuid,
                   "--memory-cap-gib", "40", "--reserve-free-gib", "8"]),
        ("test-final", [python, str(ROOT / "scripts/qreflector_sft_bf16.py"), "evaluate",
                        "--data", config["data"], "--adapter", str(run / "final"),
                        "--output", str(evaluation), "--gpu-uuid", args.gpu_uuid,
                        "--memory-cap-gib", "40", "--image-max-side", "512",
                        "--limit", "1082", "--max-new-tokens", "1024", "--resume-results"]),
    ]
    sources = [Path(__file__), ROOT / "scripts/qreflector_sft_bf16.py",
               ROOT / "scripts/eval_gui_reflection.py", ROOT / "scripts/run_qreflector_bf16.py"]
    baseline = args.baseline_predictions.resolve()
    runner_plan = {"method": plan["method"], "not_token_kl_opd": True,
        "training_plan_sha256": digest(run / "plan.json"), "smoke_run": str(smoke),
        "gpu_uuid": args.gpu_uuid, "memory_cap_gib": 40, "reserve_free_gib": 8,
        "stages": [{"name": name, "argv": argv} for name, argv in commands],
        "script_sha256": {str(path): digest(path) for path in sources},
        "baseline_predictions": str(baseline), "baseline_sha256": digest(baseline),
        "test_sha256": digest(Path(config["data"]) / "test.jsonl"),
        "evaluation_policy": "Predetermined final adapter; unchanged v6 prompt, final Yes/No only",
        "initialization": "Original epoch-6 SFT adapter; fresh optimizer; smoke weights discarded",
        "process_policy": "Spawn and wait for own children only; never signal existing processes"}
    with (run / ".runner.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "runner_plan.json").exists():
            if read_json(run / "runner_plan.json") != runner_plan:
                raise ValueError("Runner plan changed")
        else:
            write_json(run / "runner_plan.json", runner_plan)
        if args.prepare_only:
            print(json.dumps({"prepared": str(run), "stages": [c[0] for c in commands]}))
            return
        status = read_json(smoke / "status.json")
        metrics = read_json(smoke / "metrics.jsonl")
        if not (status["status"] == "complete" and status["step"] == 1
                and status["source_unchanged"] and status["gpu_uuid"] == args.gpu_uuid
                and metrics["batch_examples"] == 8 and metrics["batch_accepted"] >= 6
                and metrics["gradient_norm"] > 0):
            raise ValueError("Smoke run must complete one finite update with at least 6/8 usable targets")
        if digest(smoke / "final/adapter_model.safetensors") == plan["pinned_files"][
                str(Path(config["adapter"]) / "adapter_model.safetensors")]:
            raise ValueError("Smoke student did not change")
        if read_json(smoke / "review.json").get("status") != "passed":
            raise ValueError("Inspect smoke responses and save a passing review before launch")
        logs = run / "logs"
        logs.mkdir(exist_ok=True)
        completed = []
        stage = "startup"
        try:
            for stage, argv in commands:
                for path, expected in runner_plan["script_sha256"].items():
                    if digest(path) != expected:
                        raise ValueError(f"Pinned evaluator changed: {path}")
                stage_status = run / "status.json" if stage == "train" else evaluation / "status.json"
                if stage_status.exists() and read_json(stage_status).get("status") == "complete":
                    completed.append(stage)
                    continue
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                with (logs / f"{stage}.log").open("a") as log:
                    child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    write_json(run / "pipeline_status.json", {"status": "running", "stage": stage,
                        "runner_pid": os.getpid(), "child_pid": child.pid, "completed_stages": completed,
                        "started_at_utc": datetime.now(UTC).isoformat()})
                    code = child.wait()
                if code or read_json(stage_status).get("status") != "complete":
                    raise RuntimeError(f"{stage} did not complete; see {logs / (stage + '.log')}")
                if stage == "train" and read_json(stage_status)["examples_seen"] != 882:
                    raise ValueError("Not all training rows were visited")
                completed.append(stage)
            stage = "report"
            test = [json.loads(s) for s in (Path(config["data"]) / "test.jsonl").read_text().splitlines()]
            before = verify_predictions(baseline, test)
            after = verify_predictions(evaluation / "predictions.jsonl", test)
            if not all(a["input_tokens"] == b["input_tokens"] for a, b in zip(before, after)):
                raise ValueError("Baseline input tokenization changed")
            initial, final = summarize(before), summarize(after)
            write_json(run / "comparison.json", {"method": plan["method"], "not_token_kl_opd": True,
                "baseline_epoch6": initial, "luna_correction_final": final,
                "accuracy_delta_percentage_points": 100 * (final["accuracy"] - initial["accuracy"]),
                "matched_prompts_labels_input_token_counts": True,
                "score": "Final Yes/No only; invalid outputs count as incorrect",
                "test_sha256": runner_plan["test_sha256"],
                "corrected_errors": sum(not a["correct"] and b["correct"] for a, b in zip(before, after)),
                "new_errors": sum(a["correct"] and not b["correct"] for a, b in zip(before, after))})
            load_plan(run)
            if digest(baseline) != runner_plan["baseline_sha256"]:
                raise ValueError("Baseline results changed")
            write_json(run / "pipeline_status.json", {"status": "complete", "completed_stages": completed,
                "completed_at_utc": datetime.now(UTC).isoformat()})
        except BaseException as error:
            write_json(run / "pipeline_status.json", {"status": "failed", "stage": stage,
                "error": f"{type(error).__name__}: {error}", "completed_stages": completed})
            raise


if __name__ == "__main__":
    main()
