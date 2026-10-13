"""Create a task-grounded dataset variant without a supplied subgoal field.

Keep all actions, histories, images, labels, reasons and postconditions intact.
Remove three explicit goal statements embedded in v3 preconditions; keep the
original values in audit sidecars. Never overwrite an existing dataset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROMPT = """You are a mobile GUI reflector performing post-action verification.
Judge whether this single action made valid progress toward the user's task.
A task may require several actions. Useful intermediate progress, such as exposing
an app icon by opening its folder, can be Yes without completing the entire task.
A screen change alone is not sufficient: the change must be useful and consistent
with the task. Action descriptions and history may contain mistaken assumptions;
use the task and visible evidence rather than treating those claims as facts.

Formulate the action's precondition and expected postcondition using the task goal,
current action, available action history, and BEFORE screenshot. The expected
postcondition should describe a useful outcome of this individual step, including
a valid intermediate outcome; do not rewrite it merely to match a failed outcome.
Compare the BEFORE and AFTER screenshots and explain the observed evidence.
Both screenshots are provided together in this single post-action call.

=== Context ===
{context}

=== Image Reference ===
<image_0>: Image BEFORE the action
<image_1>: Image AFTER the action

=== Required Output ===
Return only one JSON object with these string fields in this order:
"precondition": the prerequisites for this step;
"postcondition": the expected useful outcome of this step;
"reason": a brief explanation grounded in the observed transition;
"answer": exactly "Yes" or "No".
"""

GOAL_STATEMENTS = {
    ("test", 449): "The current subgoal is to sort the expenses by name or amount to group duplicates together. ",
    ("test", 643): "The current subgoal is to use the teal color to continue the drawing. ",
    ("test", 679): "The current subgoal is to enable bulk selection mode. ",
}
MENTION = re.compile(r"\bsub[ -]?(?:goal|task)s?\b", re.I)
SPLITS = {"train": 882, "val": 71, "test": 1082}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def context_of(row):
    text = row["messages"][0]["content"][0]["text"]
    return json.JSONDecoder().raw_decode(text.split("=== Context ===\n", 1)[1])[0]


def target_of(row):
    return json.loads(row["messages"][1]["content"][0]["text"])


def migrate(source: Path, destination: Path):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("The new dataset must be outside the source dataset")
    protected = {str(p): sha(p) for p in source.iterdir() if p.is_file()}
    original_meta = json.loads((source / "metadata.json").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".no-subgoal-", dir=destination.parent) as temp:
        output = Path(temp) / "dataset"
        output.mkdir()
        # Calculate symlinks for the final destination, not the staging directory.
        (output / "images").mkdir()
        for namespace in ("train_val", "test"):
            image_root = (source / "images" / namespace).resolve(strict=True)
            (output / "images" / namespace).symlink_to(
                os.path.relpath(image_root, destination / "images"), target_is_directory=True
            )

        stats, converted, changes, mentions = {}, {}, [], []
        image_count, image_paths = 0, set()
        for split, count in SPLITS.items():
            old_rows = read_rows(source / f"{split}.jsonl")
            refs = read_rows(source / f"{split}.references.jsonl")
            assert len(old_rows) == len(refs) == count
            new_rows, inputs, new_refs, field_changes = [], [], [], Counter()
            for line, (row, ref) in enumerate(zip(old_rows, refs), 1):
                old_context = context_of(row)
                assert set(old_context) == {"goal", "subgoal", "history", "action"}
                context = {k: v for k, v in old_context.items() if k != "subgoal"}
                assert set(context) == {"goal", "history", "action"}
                new = copy.deepcopy(row)
                new["messages"][0]["content"][0]["text"] = PROMPT.format(
                    context=json.dumps(context, ensure_ascii=False, indent=2)
                )
                old_target = target_of(row)
                target = dict(old_target)
                statement = GOAL_STATEMENTS.get((split, line))
                if statement:
                    assert target["precondition"].count(statement) == 1
                    target["precondition"] = target["precondition"].replace(statement, "")
                    assert target["precondition"].strip() and not MENTION.search(target["precondition"])
                    new["messages"][1]["content"][0]["text"] = json.dumps(target, ensure_ascii=False)
                    field_changes["precondition"] += 1

                images = new["messages"][0]["content"][1:]
                assert len(images) == 2 and all(p["type"] == "image" for p in images)
                assert images == row["messages"][0]["content"][1:]
                for image in images:
                    path = Path(image["image"])
                    assert not path.is_absolute() and ".." not in path.parts
                    assert path.parts[:2] in (("images", "train_val"), ("images", "test"))
                    image_path = (source / path).resolve(strict=True)
                    assert image_path.is_file()
                    image_paths.add(str(image_path))
                    image_count += 1
                assert context == {k: v for k, v in old_context.items() if k != "subgoal"}
                assert all(target[k] == old_target[k] for k in ("postcondition", "reason", "answer"))
                if not statement:
                    assert new["messages"][1] == row["messages"][1]

                delta = {"split": split, "line": line, "removed_subgoal": old_context["subgoal"],
                         "source_sha256": protected[str(source / f"{split}.jsonl")]}
                if statement:
                    delta["precondition_edit"] = {
                        "old": old_target["precondition"], "new": target["precondition"],
                        "removed_sentence": statement.strip(),
                        "reason": "A claimed intermediate objective is not a before-state prerequisite. Remove it without changing the action, observed UI statements or verdict.",
                    }
                changes.append(delta)
                updated_ref = copy.deepcopy(ref)
                updated_ref["no_subgoal_revision"] = {
                    "source_directory": str(source), "source_line": line,
                    "removed_subgoal": old_context["subgoal"],
                    "precondition_goal_statement_removed": statement is not None,
                }
                new_refs.append(updated_ref)
                new_rows.append(new)
                inputs.append({"messages": [new["messages"][0]]})
                converted[(split, line)] = (context, target)

                # Lexical audit only: an action/reason may naturally describe an
                # intermediate objective. Do not erase original action history,
                # change labels, or treat every mention as a semantic defect.
                for field, value in {"action": context["action"], "history": context["history"],
                                     **target}.items():
                    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    if MENTION.search(text):
                        mentions.append({"split": split, "line": line, "field": field, "text": text,
                                         "status": "preserved_source_wording_not_a_required_input_field"})

            write_rows(output / f"{split}.jsonl", new_rows)
            write_rows(output / f"{split}.inputs.jsonl", inputs)
            write_rows(output / f"{split}.references.jsonl", new_refs)
            stats[split] = {"rows": count, "removed_input_subgoal_fields": count,
                            "changed_preconditions": field_changes["precondition"],
                            "labels": dict(Counter(target_of(r)["answer"] for r in new_rows))}

        pending = read_rows(source / "review_required.jsonl")
        for item in pending:
            context, target = converted[(item["split"], item["line"])]
            item["context"], item["current_target"] = context, target
            item["review_note"] = "Inherited unresolved review issue. Context and target refreshed to this version; references to a supplied subgoal in the original issue are historical. Removing that input does not automatically adjudicate the label."
        write_rows(output / "review_required.jsonl", pending)
        write_rows(output / "input_changes.jsonl", changes)
        write_rows(output / "remaining_wording_audit.jsonl", mentions)
        (output / "prompt_template.txt").write_text(PROMPT)

        checks = {
            "passed": True, "splits": stats,
            "all_input_context_keys": ["goal", "history", "action"],
            "standalone_subgoal_fields_remaining": 0,
            "prompt_requests_supplied_subgoal": False,
            "goals_actions_histories_unchanged": True,
            "postconditions_reasons_labels_unchanged": True,
            "precondition_goal_statements_removed": len(GOAL_STATEMENTS),
            "image_references": image_count, "unique_original_image_files": len(image_paths),
            "copied_image_files": 0,
            "inherited_review_records": len(pending),
            "inherited_review_cases": len({(r['split'], r['line']) for r in pending}),
            "remaining_wording_mentions": len(mentions),
            "all_protected_sources_unchanged": all(sha(Path(p)) == v for p, v in protected.items()),
            "scope": "Input-contract migration, three explicit precondition sentence removals, and lexical audit. Not an exhaustive semantic or visual relabeling.",
        }
        assert checks["all_protected_sources_unchanged"]
        (output / "validation_report.json").write_text(json.dumps(checks, indent=2) + "\n")
        metadata = {
            "schema_version": "qreflector_condition_generation_v1",
            "dataset_version": "condition_generation_v4_no_subgoal",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_version_directory": str(source),
            "conversion_script": str(Path(__file__).resolve()),
            "conversion_script_sha256": sha(Path(__file__)),
            "prompt_sha256": sha(output / "prompt_template.txt"),
            "input_fields": ["goal", "history[].step", "history[].action", "action", "before_image", "after_image"],
            "output_fields": ["precondition", "postcondition", "reason", "answer"],
            "splits": stats, "protected_source_sha256": protected,
            "target_policy": "All Yes/No labels, reasons and postconditions preserved. Three explicit subgoal statements removed from preconditions; full deltas in input_changes.jsonl.",
            "history_policy": original_meta["history_policy"],
            "image_policy": original_meta["image_policy"],
            "test_policy": original_meta["test_policy"],
            "condition_targets_ready_for_unreviewed_training": False,
            "review_policy": "Existing review decisions are not automatically applied or resolved. The v3 review website and its answer files remain unchanged.",
            "limitations": [
                "Inherited unresolved annotation issues remain; see review_required.jsonl and the previous version's metadata.",
                "Some original action descriptions, histories and reasons use the word subgoal/subtask. These are retained source statements, not a required context field; see remaining_wording_audit.jsonl. Lexical inspection cannot establish complete semantic independence from old annotations.",
                "This migration changes the prompt and three preconditions as well as dropping a field; it is not a field-only ablation.",
                "Both screenshots are supplied in one call. The prompt does not establish independent pre-execution condition prediction.",
                "The MobileWorld runtime prompt has not been changed; using this input contract online requires a separate integration change.",
            ],
        }
        metadata["output_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        output.rename(destination)
    # Symlinks resolve at their final location.
    for split in SPLITS:
        for row in read_rows(destination / f"{split}.inputs.jsonl"):
            for image in row["messages"][0]["content"][1:]:
                assert (destination / image["image"]).resolve(strict=True) == (source / image["image"]).resolve(strict=True)
    assert all(sha(Path(p)) == v for p, v in protected.items())
    print(json.dumps({"output": str(destination), **checks}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.source, args.output)


if __name__ == "__main__":
    main()
