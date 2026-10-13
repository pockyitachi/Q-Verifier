"""Small append-only token log and paper-table summary; no model dependencies."""

import csv
import json
from pathlib import Path
from threading import Lock


_WRITE_LOCK = Lock()
_FIELDS = ("prompt_tokens", "completion_tokens")


class TaskTokenRecorder:
    """Keep calls outside trajectory backups so environment retries are included."""

    def __init__(self, log_root: str, task_name: str):
        self.path = Path(log_root) / "token_usage.jsonl"
        self.task_name = task_name
        self._append({
            "event": "task_start",
            "prior_trajectory": (Path(log_root) / task_name / "traj.json").exists(),
        })

    def _append(self, event: dict) -> None:
        with _WRITE_LOCK:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"task_name": self.task_name, **event}) + "\n")

    def record(self, role: str, usage: dict | None) -> None:
        self._append({"event": "call", "role": role, "usage": usage})


def summarize_token_usage(log_root: str, tasks: list[str], reflection_enabled: bool) -> dict:
    """Average over ALL assigned tasks, including failures; null means incomplete.

    Cached input and hidden reasoning are subsets of input/output, respectively.
    They must not be added again. SDK-internal retries without returned usage
    cannot be measured; this is model usage accounting, not a billing estimate.
    """
    tasks = list(dict.fromkeys(tasks))
    buckets = {
        task: {
            role: {"reported_calls": 0, "missing_usage_calls": 0,
                   "known_prompt_tokens": 0, "known_completion_tokens": 0}
            for role in ("actor", "reflector")
        }
        for task in tasks
    }
    started = set()
    unmetered_history = set()
    path = Path(log_root) / "token_usage.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                task = event["task_name"]
                if task not in buckets:
                    continue
                if event["event"] == "task_start":
                    if task not in started and event.get("prior_trajectory"):
                        unmetered_history.add(task)
                    started.add(task)
                    continue
                counts = buckets[task][event["role"]]
                usage = event.get("usage")
                if not usage or not all(
                    type(usage.get(field)) is int and usage[field] >= 0 for field in _FIELDS
                ):
                    counts["missing_usage_calls"] += 1
                else:
                    counts["reported_calls"] += 1
                    for field in _FIELDS:
                        counts[f"known_{field}"] += usage[field]

    rows = []
    for task in tasks:
        row = {"task_name": task}
        for role, counts in buckets[task].items():
            complete = (task in started and task not in unmetered_history
                        and counts["missing_usage_calls"] == 0)
            if role == "reflector" and not reflection_enabled:
                complete = True
            row[role] = {**counts, "complete": complete}
            for field in _FIELDS:
                row[role][field] = counts[f"known_{field}"] if complete else None
        row["combined"] = {
            field: sum(row[role][field] for role in ("actor", "reflector"))
            if all(row[role][field] is not None for role in ("actor", "reflector")) else None
            for field in _FIELDS
        }
        rows.append(row)

    summary = {
        "total_tasks": len(tasks),
        "scope": "All recorded calls, including action retries and environment/task retries. "
                 "Reuse a log directory only when resuming the same experiment.",
        "counting": "API usage or model token IDs, including visual input. Cached input and "
                    "reasoning output are already included. Unknown usage is null, not zero. "
                    "SDK-internal retries without returned usage are not observable.",
    }
    for role in ("actor", "reflector", "combined"):
        complete_tasks = sum(all(row[role][field] is not None for field in _FIELDS) for row in rows)
        stats = {"complete_tasks": complete_tasks}
        if role != "combined":
            for key in ("reported_calls", "missing_usage_calls", "known_prompt_tokens",
                        "known_completion_tokens"):
                stats[key] = sum(row[role][key] for row in rows)
        for field in _FIELDS:
            total = sum(row[role][field] for row in rows) if complete_tasks == len(tasks) else None
            stats[field] = total
            stats[f"mean_{field}_per_task"] = total / len(tasks) if total is not None and tasks else None
        summary[role] = stats
    return {"summary": summary, "tasks": rows}


def write_token_usage_report(log_root: str, tasks: list[str], reflection_enabled: bool,
                             output_dir: str | None = None) -> dict:
    report = summarize_token_usage(log_root, tasks, reflection_enabled)
    destination = Path(output_dir or log_root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "token_usage_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    columns = ["task_name"] + [
        f"{role}_{field}" for role in ("actor", "reflector", "combined") for field in _FIELDS
    ]
    with (destination / "token_usage_by_task.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["tasks"]:
            writer.writerow({"task_name": row["task_name"], **{
                f"{role}_{field}": row[role][field]
                for role in ("actor", "reflector", "combined") for field in _FIELDS
            }})
    return report["summary"]
