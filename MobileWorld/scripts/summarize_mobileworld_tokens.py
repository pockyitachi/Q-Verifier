"""Refresh token summaries without contacting models or touching an evaluation.

PYTHONPATH=src python scripts/summarize_mobileworld_tokens.py LOG_ROOT
Pass --no-reflection for an actor-only run. Unknown usage stays null/blank.
"""

import argparse
import json
from pathlib import Path

from mobile_world.runtime.utils.token_usage import write_token_usage_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--tasks", nargs="+", help="All assigned task names; defaults to eval report or logs")
    parser.add_argument("--output-dir", help="Write summaries here instead of in the run directory")
    args = parser.parse_args()
    tasks = args.tasks
    reflection_enabled = not args.no_reflection
    if tasks is None:
        reports = sorted(args.log_root.glob("eval_report_*.json"))
        if reports:
            report = json.loads(reports[-1].read_text())
            tasks = [row["task_name"] for row in report["tasks_with_results"]]
            tasks += report.get("tasks_with_no_results", [])
            if report.get("metadata", {}).get("reflection_mode") == "none":
                reflection_enabled = False
        else:
            tasks = [p.parent.name for p in args.log_root.glob("*/traj.json")
                     if "_backup_" not in p.parent.name]
            ledger = args.log_root / "token_usage.jsonl"
            if ledger.exists():
                tasks += [json.loads(line)["task_name"] for line in ledger.read_text().splitlines()]
            print("No eval report: denominator includes discovered tasks only; use --tasks for all assigned tasks.")
    summary = write_token_usage_report(str(args.log_root), tasks, reflection_enabled, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
