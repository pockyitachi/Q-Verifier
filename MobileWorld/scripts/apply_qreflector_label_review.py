"""Apply the user's two label-review decisions to a new dataset version."""

import argparse
import copy
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from repair_qreflector_condition_placeholders import get_user_text, sections, sha256


SEMANTICS = (
    "Judge whether one action makes valid progress toward the current subgoal "
    "in the global-task context. A subgoal may require multiple actions. "
    "Useful intermediate progress, such as exposing an app icon by opening its "
    "folder, can be Yes without completing the subgoal. Conditions describe "
    "the individual step, and reasons must state the actual observed effect."
)
AMAZON_PRE = (
    "The closed SPA4 folder containing Amazon Shopping must be visible and "
    "accessible on the launcher so the tap can expose the app icon."
)
AMAZON_POST = (
    "The SPA4 folder is expanded and the Amazon Shopping icon is available "
    "for the next tap. This is valid intermediate progress toward opening "
    "Amazon Shopping; the app itself need not be launched by this action."
)
REASONS = {
    60: "Google Chrome opened successfully and displays a restored Google Scholar "
    "tab. The current subgoal is to open Chrome, and the address bar is available "
    "for the next recipe-search action. A blank page is not required.",
    191: "The tap expanded the SPA4 folder and exposed the Amazon Shopping icon. "
    "This advances the current subgoal by making the app available for the next "
    "tap. Amazon Shopping has not launched yet; completing this subgoal can "
    "require more than one action.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    assert not output.exists() and source != output
    source_hashes = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
    annotations = json.loads((source / "condition_annotations.json").read_text())
    changes = []
    decisions = {}
    lines = (source / "train.jsonl").read_text().splitlines(keepends=True)
    for number, task, before_answer in [
        (60, "2_recipes_chinese", "No"),
        (191, "Web_Shopping_1_ENG", "Yes"),
    ]:
        sample = json.loads(lines[number - 1])
        expected_image = f"images/{task}_1_before.jpg"
        assert any(c.get("image") == expected_image for c in sample["messages"][0]["content"])
        text_item = get_user_text(sample)
        transition, history, spans = sections(text_item["text"])
        assert not history
        assert transition["action"] == annotations[task]["action"]

        def replace(mapping, key, value, field):
            if mapping[key] != value:
                changes.append({"split": "train", "line": number, "task": task,
                                "field": field, "old": mapping[key], "new": value,
                                "annotation_source": "user_review_20261001"})
                mapping[key] = value

        if number == 191:
            replace(transition["from_state"], "precondition", AMAZON_PRE,
                    "current.from_state.precondition")
            replace(transition["from_state"], "postcondition", AMAZON_POST,
                    "current.from_state.postcondition")
            replace(transition["to_state"], "postcondition", AMAZON_POST,
                    "current.to_state.postcondition")
            start, end = spans[0]
            text_item["text"] = (text_item["text"][:start] + json.dumps(transition, indent=2)
                                 + text_item["text"][end:])
            annotations[task]["replacement"] = copy.deepcopy(transition["from_state"])
        answer_item = sample["messages"][1]["content"][0]
        answer = json.loads(answer_item["text"])
        assert answer["answer"] == before_answer
        replace(answer, "answer", "Yes", "assistant.answer")
        replace(answer, "reason", REASONS[number], "assistant.reason")
        answer_item["text"] = json.dumps(answer, ensure_ascii=False)
        lines[number - 1] = json.dumps(sample, ensure_ascii=False) + "\n"
        decisions[task] = {"status": "resolved_by_user", "split": "train", "line": number,
                           "previous_answer": before_answer, "answer": "Yes",
                           "reason": REASONS[number], "label_semantics": SEMANTICS}
        annotations[task]["review_note"] = decisions[task]

    assert len(changes) == 6
    counts = Counter(json.loads(json.loads(line)["messages"][1]["content"][0]["text"])["answer"]
                     for line in lines)
    assert counts == {"Yes": 518, "No": 364}
    metadata = json.loads((source / "metadata.json").read_text())
    repair = metadata["annotation_repair"]
    repair.update({
        "version": "initial_condition_placeholders_v2_user_reviewed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Placeholder repair plus the two explicitly user-reviewed label/reason/condition corrections.",
        "condition_semantics": SEMANTICS,
        "labels_actions_splits_and_image_paths_preserved": False,
        "actions_splits_and_image_paths_preserved": True,
        "label_changes": 1, "reason_changes": 2,
        "pending_review": {}, "resolved_review": decisions,
        "previous_version_directory": str(source), "previous_version_sha256": source_hashes,
        "limitations": [
            "Only confirmed placeholders and the two reviewed cases were repaired; other annotations were not comprehensively audited against the clarified semantics.",
            "The original input/output schema is retained; condition-generation training targets are not created.",
            "This annotation version does not change the provenance of previously trained models."
        ],
    })
    repair["splits"]["train"]["label_counts"] = dict(counts)
    cumulative = [json.loads(line) for line in (source / "repair_changes.jsonl").read_text().splitlines()]
    by_field = {(c["split"], c["line"], c["field"]): c for c in cumulative}
    for change in changes:
        key = (change["split"], change["line"], change["field"])
        if key in by_field:
            assert by_field[key]["new"] == change["old"]
            by_field[key]["new"] = change["new"]
            by_field[key]["annotation_source"] = change["annotation_source"]
        else:
            cumulative.append(change)
    repair["changed_fields"] = len(cumulative)
    assert len(cumulative) == 196
    assert source_hashes == {name: sha256(source / name) for name in source_hashes}
    output.mkdir(parents=True)
    (output / "train.jsonl").write_text("".join(lines))
    (output / "val.jsonl").write_bytes((source / "val.jsonl").read_bytes())
    (output / "images").symlink_to(os.path.relpath((source / "images").resolve(), output), target_is_directory=True)
    for name, value in [("metadata.json", metadata), ("condition_annotations.json", annotations),
                        ("review_required.json", {}), ("review_resolved.json", decisions)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    for name, entries in [("repair_changes.jsonl", cumulative), ("review_changes.jsonl", changes)]:
        (output / name).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in entries))
    print(json.dumps({"output": str(output), "train_labels": dict(counts),
                      "rows_changed_since_v1": 2, "field_changes_since_v1": len(changes),
                      "reviewed_cases_resolved": len(decisions)}, indent=2))


if __name__ == "__main__":
    main()
