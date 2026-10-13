"""Build a separate audited dataset. Original datasets and screenshots are read-only.

Usage: python scripts/qreflector_synthetic_audit/build.py
Review judgments are explicit in review.py; this script does not infer labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_qreflector_condition_generation import parse_source, normalize_action
from repair_qreflector_condition_semantics import context_of
from review import REVIEWS

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '../datasets/condition_generation_v4_no_subgoal'
OUTPUT = ROOT / '../datasets/condition_generation_v5_synthetic_audited'
RAW = ROOT / '../training/SFT/training_data_corruption_1'
OLD_REVIEW = ROOT / '../review/condition_generation_v4_no_subgoal'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def jsonl(p, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

def target(row):
    return json.loads(row["messages"][1]["content"][0]["text"])

def image_paths(row, base):
    return [(base / c["image"]).resolve(strict=True)
            for c in row["messages"][0]["content"] if c["type"] == "image"]

def set_target(row, value):
    assert set(value) == {"precondition", "postcondition", "reason", "answer"}
    assert value["answer"] in {"Yes", "No"}
    assert all(isinstance(v, str) and v.strip() for v in value.values())
    assert not re.search(r"[\u3400-\u9fff]", json.dumps(value, ensure_ascii=False))
    row["messages"][1]["content"][0]["text"] = json.dumps(value, ensure_ascii=False)

def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    protected_paths = [p for p in SOURCE.iterdir() if p.is_file()]
    for directory in [RAW, ROOT / '../training/SFT/training_data_64',
                      ROOT / '../training/SFT/training_data_filtered',
                      ROOT / '../testing_data_androidworld']:
        protected_paths.extend(p for p in directory.glob("*.json*") if p.is_file())
    protected = {str(p): sha(p) for p in protected_paths}
    answer_path = OLD_REVIEW / "answers.json"
    # Snapshot only. Do not mutate the live review sidecar or apply drafts.
    answer_bytes = answer_path.read_bytes()
    answers = json.loads(answer_bytes)
    assert answers["dataset_id"] == "qreflector-v4-5d4241363e400966"
    rows = {s: read(SOURCE / f"{s}.jsonl") for s in ("train", "val", "test")}
    refs = read(SOURCE / "train.references.jsonl")
    assert [len(rows[s]) for s in ("train", "val", "test")] == [882, 71, 1082]
    assert len(refs) == 882
    image_hashes = {}
    def ih(p):
        key = str(p)
        if key not in image_hashes:
            image_hashes[key] = sha(p)
        return image_hashes[key]
    index = defaultdict(list)
    for split, split_rows in rows.items():
        for line, row in enumerate(split_rows, 1):
            assert set(context_of(row)) == {"goal", "action", "history"}
            for phase, p in zip(("before", "after"), image_paths(row, SOURCE)):
                index[ih(p)].append({"split": split, "source_line": line, "phase": phase})

    raw_index = defaultdict(list)
    raw_rows = read(RAW / "train.jsonl")
    assert len(raw_rows) == 235
    for line, row in enumerate(raw_rows, 1):
        hashes = tuple(ih(p) for p in image_paths(row, RAW))
        raw_index[hashes].append(line)
    # Check actual goal+action overlap with the generic image-swap generator.
    generic_keys = set()
    generic_files = []
    for p in (ROOT / '../training/Data_Corruption/corrupted_data').glob("*.jsonl"):
        generic_files.append({"path": str(p), "sha256": sha(p)})
        for row in read(p):
            goal, _, transition, _, _, _ = parse_source(row)
            generic_keys.add((goal, json.dumps(normalize_action(transition["action"]), sort_keys=True)))

    audited, quarantine, new_train, new_refs, line_map, confirmed = [], [], [], [], [], []
    duplicate_pairs = defaultdict(list)
    overlaps, synthetic_paths = [], set()
    for line, (old_row, old_ref) in enumerate(zip(rows["train"], refs), 1):
        row, ref = copy.deepcopy(old_row), copy.deepcopy(old_ref)
        context = context_of(row)
        old = target(row)
        review = REVIEWS.get(line)
        if review:
            assert all("corrupt_" in p.name for p in image_paths(row, SOURCE))
            hashes = tuple(ih(p) for p in image_paths(row, SOURCE))
            synthetic_paths.update(str(p) for p in image_paths(row, SOURCE))
            duplicate_pairs[hashes].append(line)
            source_matches = raw_index[hashes]
            assert len(source_matches) == 1, (line, source_matches)
            raw_line = source_matches[0]
            raw_goal, _, raw_transition, _, _, _ = parse_source(raw_rows[raw_line - 1])
            assert raw_goal == context["goal"]
            assert normalize_action(raw_transition["action"]) == context["action"]
            overlap = [hit for h in hashes for hit in index[h]
                       if hit["split"] != "train" or hit["source_line"] < 648]
            if overlap:
                overlaps.append({"line": line, "hits": overlap})
            replacement = None
            if review["status"] != "quarantine":
                replacement = {
                    "precondition": review["precondition"] or old["precondition"],
                    "postcondition": review["postcondition"] or old["postcondition"],
                    "reason": review["evidence"],
                    "answer": "Yes" if review["status"] == "positive" else "No",
                }
                set_target(row, replacement)
            evidence = {"sample_id": f"train-{line}", "source_line": line,
                "status": review["status"], "reviewer": "assistant_visual_audit",
                "review_method": "Both source screenshots inspected in contact sheets, with enlarged inspection for ambiguous text edits. Task and recorded action checked; no model endpoint was called.",
                "before_observation": review["before"], "after_observation": review["after"],
                "decision_evidence": review["evidence"], "issues": review["issues"],
                "original_target": old, "replacement_target": replacement,
                "changes": {k: {"old": old[k], "new": replacement[k]} for k in old
                            if replacement is not None and old[k] != replacement[k]},
                "image_sha256": dict(zip(("before", "after"), hashes)),
                "byte_identical_pair": hashes[0] == hashes[1],
                "direct_source": {"path": str(RAW / "train.jsonl"), "line": raw_line,
                    "matched_by": "Both image SHA-256 hashes plus exact task and normalized current action",
                    "row_sha256": hashlib.sha256(json.dumps(raw_rows[raw_line - 1], sort_keys=True).encode()).hexdigest()},
                "matches_generic_generator_goal_and_action": (context["goal"], json.dumps(context["action"], sort_keys=True)) in generic_keys,
                "corruption_operation": None,
                "corruption_operation_status": "No per-example mutation log found; do not infer image replacement from the corrupt_ filename.",
                "output_train_line": None,
            }
            audited.append(evidence)
            if review["status"] == "quarantine":
                quarantine.append({"sample_id": f"train-{line}", "source_line": line,
                    "training_eligible": False, "current_target": None,
                    "input": {"messages": [row["messages"][0]]},
                    "audit": evidence, "source_reference": ref})
                line_map.append({"sample_id": f"train-{line}", "source_line": line,
                                 "output_line": None, "status": "quarantine"})
                continue
            ref["synthetic_audit"] = copy.deepcopy(evidence)
            ref["condition_review_status"] = "assistant_visually_audited"
        elif f"train-{line}" in {"train-138", "train-155"}:
            saved = answers["reviews"].get(f"train-{line}")
            if saved and saved["status"] == "done":
                fields = saved["fields"]
                # A changed task or action needs a separate explicit migration.
                assert fields["goal"] == context["goal"]
                assert fields["action_type"] == context["action"]["type"]
                assert fields["action_description"] == context["action"]["description"]
                assert json.loads(fields["action_arguments"]) == context["action"]["arguments"]
                replacement = {k: fields[k] for k in old}
                set_target(row, replacement)
                record = {"sample_id": f"train-{line}", "source_line": line,
                    "source_review_file": str(answer_path), "review": saved,
                    "changes": {k: {"old": old[k], "new": replacement[k]} for k in old if old[k] != replacement[k]}}
                confirmed.append(record)
                ref["user_confirmed_review"] = record
                ref["condition_review_status"] = "user_confirmed"
        assert row["messages"][0] == old_row["messages"][0], "Inputs must be preserved exactly"
        output_line = len(new_train) + 1
        ref["v5_sample_id"] = f"train-{line}"
        ref["output_line"] = output_line
        ref["v4_source_line"] = line
        if review:
            audited[-1]["output_train_line"] = output_line
            ref["synthetic_audit"]["output_train_line"] = output_line
        new_train.append(row)
        new_refs.append(ref)
        line_map.append({"sample_id": f"train-{line}", "source_line": line,
                         "output_line": output_line, "status": review["status"] if review else "retained_original"})

    assert len(audited) == 235 and len(quarantine) == 13 and len(new_train) == 869
    assert not overlaps, "Cross-split/original image overlap needs explicit handling"
    assert not any(len(v) > 1 for v in duplicate_pairs.values())
    assert not any(a["matches_generic_generator_goal_and_action"] for a in audited)
    assert sorted(a["direct_source"]["line"] for a in audited) == list(range(1, 236))
    assert len(confirmed) == 2
    training_labels = dict(Counter(target(r)["answer"] for r in new_train))
    assert training_labels == {"No": 347, "Yes": 522}
    unchanged_nonsynthetic = sum(new_train[i] == rows["train"][i] for i in range(647))
    assert unchanged_nonsynthetic == 646

    with tempfile.TemporaryDirectory(prefix=".synthetic-audit-", dir=OUTPUT.parent) as temp:
        staged = Path(temp) / "dataset"
        staged.mkdir()
        (staged / "images").mkdir()
        for namespace in ("train_val", "test"):
            (staged / "images" / namespace).symlink_to(os.path.relpath(
                (SOURCE / "images" / namespace).resolve(), OUTPUT / "images"), target_is_directory=True)
        jsonl(staged / "train.jsonl", new_train)
        jsonl(staged / "train.inputs.jsonl", [{"messages": [r["messages"][0]]} for r in new_train])
        jsonl(staged / "train.references.jsonl", new_refs)
        jsonl(staged / "train.line_map.jsonl", line_map)
        jsonl(staged / "synthetic_audit.jsonl", audited)
        jsonl(staged / "quarantine.jsonl", quarantine)
        jsonl(staged / "applied_user_reviews.jsonl", confirmed)
        (staged / "previous_review_answers.snapshot.json").write_bytes(answer_bytes)
        for split in ("val", "test"):
            for suffix in ("jsonl", "inputs.jsonl", "references.jsonl"):
                name = f"{split}.{suffix}"
                shutil.copyfile(SOURCE / name, staged / name)
                assert sha(staged / name) == sha(SOURCE / name)
        shutil.copyfile(SOURCE / "prompt_template.txt", staged / "prompt_template.txt")
        pending = []
        for item in read(SOURCE / "review_required.jsonl"):
            if item["split"] == "train" and item["line"] in {2, 138, 155, 743}:
                continue
            pending.append(item)
        assert len({(p['split'], p['line']) for p in pending}) == 26
        for q in quarantine:
            pending.append({"split": "train", "line": q["source_line"],
                "line_semantics": "v4_source_line_not_v5_output_line", "sample_id": q["sample_id"],
                "output_train_line": None, "training_eligible": False,
                "issue": q["audit"]["decision_evidence"], "current_target": None,
                "context": context_of(q["input"])})
        jsonl(staged / "review_required.jsonl", pending)
        source_ok = all(sha(Path(p)) == h for p, h in protected.items())
        images_ok = all(sha(Path(p)) == image_hashes[p] for p in synthetic_paths)
        assert source_ok and images_ok
        checks = {"passed": True, "all_235_reviewed": True, "train_rows": len(new_train),
            "train_labels": training_labels, "val_rows": 71, "test_rows": 1082,
            "quarantined": 13, "retained_synthetic_negative": 217, "synthetic_relabelled_yes": 5,
            "non_synthetic_rows_unchanged": unchanged_nonsynthetic,
            "non_synthetic_user_confirmed_changed_ids": [c["sample_id"] for c in confirmed if c["changes"]],
            "all_retained_input_messages_unchanged": True, "prompt_unchanged": True,
            "validation_and_test_files_byte_identical": True,
            "all_image_paths_resolve": True, "copied_image_files": 0,
            "all_protected_sources_unchanged": source_ok, "synthetic_image_bytes_unchanged": images_ok,
            "byte_identical_synthetic_pairs": [a["source_line"] for a in audited if a["byte_identical_pair"]],
            "duplicate_synthetic_image_pairs": [], "exact_synthetic_image_overlap_with_other_splits_or_original_train": overlaps,
            "raw_source_235_of_235_matched": True,
            "semantic_scope": "Assistant visual audit, not independent expert certification or causal reconstruction of synthetic generation.",
        }
        dump(staged / "validation_report.json", checks)
        metadata = {"schema_version": "qreflector_condition_generation_v1",
            "dataset_version": OUTPUT.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_version_directory": str(SOURCE), "protected_source_sha256": protected,
            "scripts": {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('review.py'))},
            "splits": {"train": {"rows": 869, "labels": training_labels},
                       "val": {"rows": 71, "labels": dict(Counter(target(r)['answer'] for r in rows['val']))},
                       "test": {"rows": 1082, "labels": dict(Counter(target(r)['answer'] for r in rows['test']))}},
            "input_fields": ["goal", "history[].step", "history[].action", "action", "before_image", "after_image"],
            "output_fields": ["precondition", "postcondition", "reason", "answer"],
            "sample_identity": "train.line_map.jsonl maps stable v4 IDs to new physical train lines; quarantined IDs have null output_line.",
            "image_policy": "Original image directories are linked, not copied or changed.",
            "quarantine_policy": "13 uncertain examples are excluded from train.jsonl; quarantine.jsonl has no executable assistant target. Its old targets are historical evidence only.",
            "label_policy": "Accept useful single-action task progress, including opening search, recent apps, or removing a blocking sheet. Do not require complete task success, exact actor prose, or a fresh browser tab.",
            "repair_policy": "Recorded goals/actions/histories/images and the input prompt are unchanged. Rewrite retained synthetic target fields from screenshot evidence; never alter an erroneous executed action to make it succeed.",
            "test_policy": "Validation and test JSONL, user-only inputs, and references are byte-identical to v4; no test labels or images changed.",
            "source_trace": {"direct_source": str(RAW), "matched_rows": 235,
                "generic_generator_files_compared": generic_files, "generic_goal_action_matches": 0,
                "per_sample_corruption_operation_available": False},
            "distribution_notes": {"synthetic_tasks": 27,
                "actions": dict(Counter(context_of(r)['action']['type'] for r in rows['train'][647:])),
                "records_with_argument_value_2399": sum(2399 in (context_of(r)['action']['arguments'] or {}).values() for r in rows['train'][647:]),
                "records_with_argument_value_1079": sum(1079 in (context_of(r)['action']['arguments'] or {}).values() for r in rows['train'][647:]),
                "interpretation": "Repeated app-opening and short/swapped gestures limit diversity. Repeated coordinate values are a diagnostic observation, not proof of a specific generator or coordinate bug."},
            "condition_targets_ready_for_unreviewed_training": False,
            "limitations": ["The remaining 647 original training examples were not exhaustively visually reaudited in this task; only two existing user-confirmed reviews were imported.",
                "26 inherited test-condition/action ambiguities remain; test files were deliberately preserved.",
                "Exact hash checks do not rule out near-duplicate images or related trajectories across splits.",
                "The 235 cases come from 27 tasks and include repetitive, easy failures; validity repairs do not establish high diversity or hard-negative quality.",
                "No retained sample is certified as a particular image-stitch operation without a mutation log.",
                "Both screenshots were visible during annotation and remain in a single-call prompt; this does not demonstrate independent before-only condition prediction.",
                "No training, offline evaluation, MobileWorld run, model service change, or GPU operation was performed."],
        }
        metadata["output_sha256"] = {p.name: sha(p) for p in staged.iterdir() if p.is_file()}
        dump(staged / "metadata.json", metadata)
        os.replace(staged, OUTPUT)
    for row in new_train:
        assert len(image_paths(row, OUTPUT)) == 2
    print(json.dumps({"output": str(OUTPUT), **checks}, indent=2))

if __name__ == "__main__":
    main()
