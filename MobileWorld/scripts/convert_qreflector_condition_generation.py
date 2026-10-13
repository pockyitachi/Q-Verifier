"""Convert train/validation/test together without modifying source data or images.

This is a traceable format migration, not semantic reannotation. Unreviewed
legacy conditions remain provisional supervision and are identified in sidecars.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROMPT = """You are a mobile GUI reflector performing post-action verification.
Judge whether this single action made valid progress toward the current subgoal
in the context of the global task. One subgoal may require several actions.
Useful intermediate progress, such as exposing an app icon by opening its folder,
can be Yes without completing the entire subgoal. A screen change alone is not
sufficient: the change must be useful and consistent with the task.

Formulate the action's precondition and expected postcondition using the goal,
subgoal, action, action history, and BEFORE screenshot. The expected postcondition
should describe a useful outcome of this individual step, including a valid
intermediate outcome; do not rewrite it merely to match a failed outcome.
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
OUTPUT_FIELDS = ["precondition", "postcondition", "reason", "answer"]
KNOWN_PLACEHOLDERS = [
    "The device is ready for interaction.",
    "The user can perform actions on the current screen.",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_source(sample):
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant"]
    content = sample["messages"][0]["content"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    images = [c["image"] for c in content if c["type"] == "image"]
    assert len(texts) == 1 and len(images) == 2
    text = texts[0]
    goal = re.search(r"=== Global Task ===\s*(.*?)\s*=== Current Subgoal ===", text, re.S)
    subgoal = re.search(r"=== Current Subgoal ===\s*(.*?)\s*=== Current Transition ===", text, re.S)
    assert goal and subgoal
    transition_start = re.search(r"=== Current Transition ===\s*", text).end()
    history_start = re.search(r"=== Previous Steps \(latest \d+\) ===\s*", text).end()
    decoder = json.JSONDecoder()
    transition = decoder.raw_decode(text[transition_start:])[0]
    history = decoder.raw_decode(text[history_start:])[0]
    answer = json.loads(sample["messages"][1]["content"][0]["text"])
    assert answer["answer"] in ("Yes", "No") and isinstance(answer["reason"], str)
    return goal.group(1).strip(), subgoal.group(1).strip(), transition, history, answer, images


def normalize_action(action):
    """Unify nested training actions and flat AndroidWorld actions without loss."""
    kind = action.get("type", action.get("name"))
    assert isinstance(kind, str) and kind
    description = action.get("description", "")
    assert isinstance(description, str)
    if "arguments" in action:
        assert set(action) <= {"type", "name", "description", "arguments"}
        arguments = copy.deepcopy(action["arguments"])
    else:
        arguments = {k: copy.deepcopy(v) for k, v in action.items()
                     if k not in {"type", "name", "description"}} or None
    return {"type": kind, "arguments": arguments, "description": description}


def convert_sample(sample, split, line, reviewed):
    goal, subgoal, transition, history, answer, images = parse_source(sample)
    # Action-only history avoids exposing old condition/validity annotations.
    context = {
        "goal": goal, "subgoal": subgoal,
        "history": [{"step": h["step"], "action": normalize_action(h["action"])} for h in history],
        "action": normalize_action(transition["action"]),
    }
    annotation = reviewed.get((split, line))
    if annotation is not None:
        assert transition["action"] == annotation["action"]
        assert transition["from_state"] == annotation["replacement"]
        pre, post = (transition["from_state"][key] for key in ["precondition", "postcondition"])
        condition_source = {"precondition": "current.from_state.precondition",
                            "postcondition": "current.from_state.postcondition"}
        status = "previously_reviewed_initial_transition"
    else:
        pre = transition["from_state"]["precondition"]
        post = transition["to_state"]["postcondition"]
        condition_source = {"precondition": "current.from_state.precondition",
                            "postcondition": "current.to_state.postcondition"}
        status = "legacy_unreviewed_for_condition_generation"
    assert all(isinstance(v, str) and v.strip() and v not in KNOWN_PLACEHOLDERS for v in (pre, post))
    target = {"precondition": pre, "postcondition": post,
              "reason": answer["reason"], "answer": answer["answer"]}
    namespace = "test" if split == "test" else "train_val"
    new_images = []
    for value in images:
        path = Path(value)
        assert not path.is_absolute() and path.parts[0] == "images" and ".." not in path.parts
        new_images.append(str(Path("images") / namespace / Path(*path.parts[1:])))
    converted = {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(context=json.dumps(context, ensure_ascii=False, indent=2))},
            *[{"type": "image", "image": p} for p in new_images],
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False)}]},
    ]}
    reference = {
        "sample_id": line, "split": split, "source_line": line,
        "source_images": images, "converted_images": new_images,
        "condition_source_fields": condition_source, "condition_review_status": status,
        "original_sample": sample,
    }
    return converted, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training, testing, output = args.training_source.resolve(), args.test_source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    assert output not in (training, testing)
    annotations = json.loads((training / "condition_annotations.json").read_text())
    reviewed = {(a["source_split"], a["source_line"]): a for a in annotations.values()}
    protected = {str(p): digest(p) for directory in (training, testing)
                 for p in directory.iterdir() if p.is_file()}
    sources = {"train": training / "train.jsonl", "val": training / "val.jsonl", "test": testing / "test.jsonl"}
    expected_counts = {"train": 882, "val": 71, "test": 1082}
    products, stats = {}, {}
    for split, source in sources.items():
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        assert len(rows) == expected_counts[split]
        converted, references, labels, review_counts = [], [], Counter(), Counter()
        for line, sample in enumerate(rows, 1):
            row, reference = convert_sample(sample, split, line, reviewed)
            for image_path in reference["source_images"]:
                assert (source.parent / image_path).resolve(strict=True).is_file()
            labels[json.loads(row["messages"][1]["content"][0]["text"])["answer"]] += 1
            review_counts[reference["condition_review_status"]] += 1
            converted.append(row)
            references.append(reference)
        products[f"{split}.jsonl"] = converted
        products[f"{split}.references.jsonl"] = references
        # Reference-free files can be handed directly to an inference driver.
        products[f"{split}.inputs.jsonl"] = [{"messages": [r["messages"][0]]} for r in converted]
        stats[split] = {"rows": len(rows), "labels": dict(labels),
                        "condition_review_status": dict(review_counts),
                        "source": str(source), "source_sha256": digest(source)}
    assert protected == {p: digest(Path(p)) for p in protected}
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for namespace, directory in [("train_val", training), ("test", testing)]:
        (output / "images" / namespace).symlink_to(
            os.path.relpath((directory / "images").resolve(strict=True), output / "images"),
            target_is_directory=True,
        )
    for name, rows in products.items():
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "prompt_template.txt").write_text(PROMPT)
    metadata = {
        "schema_version": "qreflector_condition_generation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_script": str(Path(__file__).resolve()), "conversion_script_sha256": digest(Path(__file__)),
        "prompt_sha256": digest(output / "prompt_template.txt"), "splits": stats,
        "output_fields": OUTPUT_FIELDS,
        "input_fields": ["goal", "subgoal", "history[].step", "history[].action", "action", "before_image", "after_image"],
        "history_policy": "Preserve original step order and action details; remove state descriptions, conditions and reference validity labels.",
        "condition_mapping": "Use the reviewed initial-state pair for 41 reviewed current transitions. Otherwise migrate from_state.precondition and to_state.postcondition verbatim as provisional targets. Preserve every original field in per-split references.",
        "label_policy": "All labels and reasons are unchanged from the input source versions. Yes accepts valid intermediate task progress; a subgoal can span multiple actions.",
        "test_policy": "Only user messages are model input. Assistant conditions are provisional references, not validated condition-generation gold; evaluate final answer against the original Yes/No labels.",
        "image_policy": "Relative paths through two directory symlinks; no image bytes are copied or changed. Original image directories must remain available.",
        "protected_source_sha256": protected,
        "semantic_reannotation_performed": False,
        "condition_targets_ready_for_unreviewed_training": False,
        "limitations": [
            "Legacy conditions sometimes describe observed results, later actions, or completion of the whole subgoal instead of this action's prerequisites/expected progress. Format conversion does not repair these semantics.",
            "Existing labels/reasons outside the two user-reviewed training cases have not been reaudited under the clarified intermediate-progress criterion.",
            "Single-call generation with both screenshots does not establish independent pre-execution condition prediction."
        ],
        "review_examples": [
            {"split": "train", "line": 1, "issue": "The migrated postcondition describes the wrong recipient remaining in the To field, an observed failure rather than the intended email address."},
            {"split": "train", "line": 2, "issue": "The subgoal/conditions describe returning home, while the action/Yes reason describe performing a search."},
            {"split": "train", "line": 25, "issue": "The action scrolls the app drawer, while the legacy postcondition requires Google Play Store to be open."}
        ],
    }
    metadata["output_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    assert protected == {p: digest(Path(p)) for p in protected}
    print(json.dumps({"output": str(output), "splits": stats, "images_copied": 0}, indent=2))


if __name__ == "__main__":
    main()
