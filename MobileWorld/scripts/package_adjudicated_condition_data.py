#!/usr/bin/env python3
"""Build a separate v8 from explicit visual adjudications; no API or GPU calls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / '../datasets'
SOURCE = DATASETS / "condition_generation_v7_condition_check"
FROZEN = DATASETS / "condition_generation_v6_user_reviewed"
REVIEW = ROOT / "artifacts/data_preparation/qreflector_v8_adjudication_20261006"
FIELDS = ("precondition", "postcondition", "reason", "answer")
HEADINGS = ("Precondition check:", "Postcondition check:", "Task progress:")
README = """Q-Reflector v8: condition-check supervision with disputed cases adjudicated

This is a NEW version. v6 and v7 are preserved. No training was started.
Canonical training files: train.jsonl (880 accepted), val.jsonl (71 accepted).
The source training set still has 882 records; two are explicitly quarantined.
Test: 1,082 rows copied byte-for-byte from v7, including its updated prompt.
Test labels and assistant references remain unchanged from frozen v6.
Screenshots remain path references via the images symlink; no images are copied.

Scope and provenance
All 87 v7 disputes (76 train, 11 validation) were visually adjudicated using
task, actual action, available history, and both screenshots. 53 labels changed,
32 labels were retained, and 2 records were quarantined. Explanations and
conditions were corrected where needed. The other v7 records inherit their
API-assisted annotations, including five previous visual corrections. This is
not a claim that all 953 source records received a new independent manual audit.

Decision standard
Yes requires a correctly realized useful local effect AND valid task progress.
The whole task or an old subgoal need not be completed by one transition.
Subgoal is absent from the model input. Conditions describe the task-relevant
expectation from goal, current action, history and BEFORE, not the observed
failure rewritten as a desired outcome. Reason checks the evidence and progress.

Useful progress includes opening a relevant folder or app switcher, entering a
correct query, moving forward through relevant content, necessary permissions,
and entering a search's loading view. For a compound shortcut, a completed useful
component may earn Yes; the reason must explicitly say which component remains
unverified. This is task-relevant transition acceptance, not an assertion that
every low-level subcommand completed successfully.

Actual coordinates and visible effects take precedence over mistaken action
narratives. A different but useful route can be accepted when its relevance is
concrete. Inspecting a plausible candidate and newly learning it is ineligible
can be useful information gain; continuing to wait on an already-known wrong
candidate does not provide that same progress.

Note entry tolerates nonessential punctuation, quotation and automatic list
formatting differences. Substantive target/field errors remain failures. Known
transcription imperfections are explicitly described rather than hidden.
The screenshots verify entry of supplied notes, not the truth of every original
source claim or persistence after leaving the editor.

Routine focus preparation can be accepted when the correct editable field and
keyboard are ready. Explicitly requested stopping at a relevant screen may be
valid even without a visual change. Arbitrary desktop waits, repeated boundary
scrolls, unrelated setup, and premature navigation away are not useful progress.
Claims about global best/nearest choices or hidden persistence are not inferred
from a limited visible comparison.

Quarantine
train-243 and train-589 have conflicting Triller/Troolls app identity evidence.
One pair shows Play Store's Triller Open control launching a Troolls-branded
screen. The other reopens Troolls from the launcher. The screenshots verify the
launch but cannot settle whether this is the intended app. Both are excluded
from training; final_label is null. Historical source/candidate labels in audit
records are provenance, not approved ground truth. No user action is required
to use the accepted subset.

Files
train/val.jsonl: accepted SFT examples; all targets are English and ordered as
precondition, postcondition, reason, answer. Reason has three explicit checks.
train/val.line_map.jsonl: output line to ORIGINAL source line and stable ID.
train/val.inputs.jsonl, references.jsonl, labels.jsonl: aligned sidecars.
all_source_audit.jsonl: all 953 source records, including quarantine dispositions.
adjudication_decisions.jsonl: every disputed case with old/new targets and reason.
quarantine.jsonl: excluded records, contexts and evidence, not an SFT file.
prior_v7_review_decisions.jsonl: previous five visual corrections, retained.
adjudication_report.json: counts, checks and practical limitations.
DATASET_MANIFEST.json: hashes and source provenance.

Experiment interpretation
This revision changes supervision, 53 labels, and training membership; it is not
just a prompt-format change. Validation has six revised labels (three in each
direction), so comparisons must use a common validation version. Test was not
relabelled. The old loader pins v6 and assumes 882 training rows; configure and
validate a new run explicitly before training on v8. Do not silently redirect an
existing experiment. This version is prepared, not automatically frozen or run.
"""


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, values):
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def target(row):
    return json.loads(row["messages"][1]["content"][0]["text"])


def check_target(value, sample_id):
    if list(value) != list(FIELDS) or value["answer"] not in ("Yes", "No"):
        raise ValueError(f"{sample_id}: invalid target schema")
    if any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError(f"{sample_id}: empty target")
    if re.search(r"[\u3400-\u9fff]", " ".join(value.values())):
        raise ValueError(f"{sample_id}: non-English target")
    reason = value["reason"]
    if (not reason.startswith(HEADINGS[0])
            or any(reason.count(h) != 1 for h in HEADINGS)
            or reason.index(HEADINGS[1]) > reason.index(HEADINGS[2])):
        raise ValueError(f"{sample_id}: invalid check order")


def verify_sources():
    for base, manifest_name in ((SOURCE, "DATASET_MANIFEST.json"), (FROZEN, "FROZEN.json")):
        manifest = json.loads((base / manifest_name).read_text())
        for name, expected in manifest["artifact_sha256"].items():
            if digest(base / name) != expected:
                raise ValueError(f"Source changed: {base / name}")
    if digest(REVIEW / "review_source.snapshot.jsonl") != digest(SOURCE / "review_required.jsonl"):
        raise ValueError("Review snapshot no longer matches source")
    image_manifest = json.loads((FROZEN / "image_sha256.json").read_text())
    for name, metadata in image_manifest.items():
        if digest(FROZEN / name) != metadata["sha256"]:
            raise ValueError(f"Frozen image changed: {name}")
    return len(image_manifest)


def build(output):
    if output.exists():
        raise ValueError("Output already exists; do not overwrite a data version")
    verified_images = verify_sources()
    disputed = {r["sample_id"]: r for r in rows(REVIEW / "review_source.snapshot.jsonl")}
    decisions_list = rows(REVIEW / "decisions.jsonl")
    decisions = {r["sample_id"]: r for r in decisions_list}
    if len(decisions) != len(decisions_list) or set(decisions) != set(disputed):
        raise ValueError("Adjudications must cover every disputed ID exactly once")
    for sid, decision in decisions.items():
        source = disputed[sid]
        for key in ("source_line", "context_sha256", "original_target", "candidate_target"):
            if decision[key] != source[key]:
                raise ValueError(f"Stale decision: {sid}: {key}")
        if decision.get("disposition", "accept") not in ("accept", "quarantine"):
            raise ValueError(f"Invalid disposition: {sid}")
        check_target(decision["final_target"], sid)
        if not Path(decision["inspected_pair"]).is_file():
            raise ValueError(f"Missing inspected evidence pair: {sid}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(
        ROOT / "artifacts/models/qreflector_base_bf16/Qwen3-VL-8B-Instruct/tokenizer.json"))
    prompt = (SOURCE / "prompt_template.txt").read_text()
    prefix, suffix = prompt.split("{context}")
    prepared, summaries, all_audits, quarantine, lengths = {}, {}, [], [], []
    for split, expected_rows in (("train", 882), ("val", 71)):
        candidates = rows(SOURCE / f"{split}.candidates.jsonl")
        refs = rows(SOURCE / f"{split}.references.jsonl")
        inputs = rows(SOURCE / f"{split}.inputs.jsonl")
        original = rows(FROZEN / f"{split}.jsonl")
        if {len(candidates), len(refs), len(inputs), len(original)} != {expected_rows}:
            raise ValueError(f"Unexpected source size: {split}")
        accepted, selected_refs, mappings, selected_inputs, labels = [], [], [], [], []
        changes, kept, quarantined = Counter(), 0, 0
        for line, (row, ref, inp, old) in enumerate(zip(candidates, refs, inputs, original, strict=True), 1):
            sid = f"{split}-{line}"
            if ref["sample_id"] != sid or ref["source_line"] != line:
                raise ValueError(f"Source order changed: {sid}")
            user = row["messages"][0]
            raw = user["content"][0]["text"]
            if not raw.startswith(prefix) or not suffix or not raw.endswith(suffix):
                raise ValueError(f"Prompt mismatch: {sid}")
            context = raw[len(prefix):-len(suffix)]
            if (hashlib.sha256(context.encode()).hexdigest() != ref["context_sha256"]
                    or json.loads(context) != ref["context"]
                    or set(json.loads(context)) != {"goal", "action", "history"}
                    or inp != {"messages": [user]}
                    or user["content"][1:] != old["messages"][0]["content"][1:]
                    or target(row) != ref["candidate_target"]
                    or target(old) != ref["original_target"]):
                raise ValueError(f"Context, images or reference mismatch: {sid}")
            decision = decisions.get(sid)
            if bool(ref["audit"]["flags"]) != bool(decision):
                raise ValueError(f"Unaccounted source audit flag: {sid}")
            audit = copy.deepcopy(ref)
            audit["source_audit"] = audit.pop("audit")
            audit["status"] = "accepted_after_visual_adjudication" if decision else "accepted_from_v7"
            if decision and decision.get("disposition") == "quarantine":
                audit.update(status="quarantined", final_target=None, final_label=None,
                             reason=decision["note"], category=decision["category"])
                audit["provisional_checks"] = {k: decision["final_target"][k] for k in FIELDS[:-1]}
                quarantine.append({**audit, "input": inp, "use_for_training": False})
                all_audits.append(audit)
                quarantined += 1
                continue
            value = copy.deepcopy(decision["final_target"] if decision else target(row))
            check_target(value, sid)
            serialized = json.dumps(value, ensure_ascii=False)
            n_tokens = len(tokenizer.encode(serialized, add_special_tokens=False).ids)
            if n_tokens >= 1024:
                raise ValueError(f"{sid}: target exceeds generation budget")
            lengths.append(n_tokens)
            final = copy.deepcopy(row)
            final["messages"][1]["content"][0]["text"] = serialized
            changed_fields = [k for k in FIELDS if value[k] != ref["candidate_target"][k]]
            if changed_fields and not decision:
                raise ValueError(f"Unreviewed target change: {sid}")
            if value["answer"] != ref["original_target"]["answer"]:
                changes[f'{ref["original_target"]["answer"]}->{value["answer"]}'] += 1
            elif decision:
                kept += 1
            audit.update(final_target=value, final_label=value["answer"], target_tokens=n_tokens,
                         changed_fields_from_v7=changed_fields)
            if decision:
                audit.update(adjudication_note=decision["note"], category=decision["category"])
            accepted.append(final)
            selected_inputs.append(inp)
            selected_refs.append(audit)
            all_audits.append(audit)
            mappings.append({"output_line": len(accepted), "source_line": line, "sample_id": sid})
            labels.append({"sample_id": sid, "answer": value["answer"]})
        prepared[split] = {"jsonl": accepted, "inputs.jsonl": selected_inputs,
                           "references.jsonl": selected_refs, "line_map.jsonl": mappings,
                           "labels.jsonl": labels}
        summaries[split] = {"source_rows": expected_rows, "accepted_rows": len(accepted),
                            "quarantined_rows": quarantined,
                            "labels": dict(Counter(target(r)["answer"] for r in accepted)),
                            "label_changes": dict(changes), "reviewed_label_kept": kept}

    original_test, revised_test = rows(FROZEN / "test.jsonl"), rows(SOURCE / "test.jsonl")
    if len(original_test) != 1082 or len(revised_test) != 1082:
        raise ValueError("Unexpected test size")
    for old, new in zip(original_test, revised_test, strict=True):
        if (old["messages"][1] != new["messages"][1]
                or old["messages"][0]["content"][1:] != new["messages"][0]["content"][1:]):
            raise ValueError("Test reference or image changed")

    output.mkdir(parents=True, exist_ok=False)
    (output / "images").symlink_to(FROZEN / "images", target_is_directory=True)
    for split, files in prepared.items():
        for suffix_name, records in files.items():
            write_rows(output / f"{split}.{suffix_name}", records)
    write_rows(output / "all_source_audit.jsonl", all_audits)
    write_rows(output / "quarantine.jsonl", quarantine)
    public_decisions = copy.deepcopy(decisions_list)
    for decision in public_decisions:
        if decision.get("disposition") == "quarantine":
            provisional = decision.pop("final_target")
            decision["provisional_checks"] = {k: provisional[k] for k in FIELDS[:-1]}
            decision["final_label"] = None
            decision["use_for_training"] = False
        else:
            decision["disposition"] = "accept"
            decision["final_label"] = decision["final_target"]["answer"]
    write_rows(output / "adjudication_decisions.jsonl", public_decisions)
    shutil.copyfile(REVIEW / "review_source.snapshot.jsonl", output / "review_source.snapshot.jsonl")
    shutil.copyfile(SOURCE / "review_decisions.jsonl", output / "prior_v7_review_decisions.jsonl")
    for name in ("prompt_template.txt", "test.jsonl", "test.inputs.jsonl", "test.labels.jsonl",
                 "test.input_manifest.jsonl"):
        shutil.copyfile(SOURCE / name, output / name)
        if digest(output / name) != digest(SOURCE / name):
            raise ValueError(f"Copy mismatch: {name}")
    (output / "test.source_references.jsonl").symlink_to(FROZEN / "test.references.jsonl")
    (output / "image_sha256.json").symlink_to(FROZEN / "image_sha256.json")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready_after_adjudication_not_frozen", "source": str(SOURCE),
        "dataset_version": output.name, "splits": summaries,
        "reviewed_disputes": len(decisions), "prior_v7_visual_reviews": 5,
        "labels_changed": sum(sum(s["label_changes"].values()) for s in summaries.values()),
        "reviewed_labels_kept": sum(s["reviewed_label_kept"] for s in summaries.values()),
        "quarantined_ids": [q["sample_id"] for q in quarantine],
        "unadjudicated_disputes": 0, "accepted_train_val_rows": len(lengths),
        "test_rows": 1082, "test_changed_from_v7": False,
        "test_labels_and_assistant_references_unchanged_from_v6": True,
        "prompt_changed_from_v7": False, "images_copied": 0,
        "frozen_images_hash_verified": verified_images,
        "maximum_target_text_tokens": max(lengths), "mean_target_text_tokens": sum(lengths) / len(lengths),
        "source_artifacts_hash_verified": True, "training_started": False, "gpu_used": False,
        "api_calls_this_adjudication": 0,
        "limitations": [
            "87 disputed records were visually adjudicated this round; the other records were not all manually re-reviewed.",
            "Remaining accepted records inherit v7 API-assisted annotations and its five prior visual corrections.",
            "Two app-identity conflicts are excluded, not assigned negative ground truth.",
            "The accepted training set has 880 rows, not 882. Preserve source IDs via line maps.",
            "Validation contains revised labels; compare models on the same validation version.",
            "Test retains legacy references and labels; no test relabeling was performed.",
            "This changes supervision and training membership as well as the older v6 prompt, so it is not a pure prompt ablation.",
            "This is assistant adjudication, not independent human ground truth or a measured reliability guarantee.",
            "The existing SFT loader pins v6 and expects 882 rows; explicit loader/preflight work is required before using v8.",
            "Text-token checks are not a multimodal training collator preflight.",
        ],
    }
    write_json(output / "adjudication_report.json", report)
    write_json(output / "status.json", {"status": report["status"], "unadjudicated_disputes": 0,
                                       "quarantined_rows": len(quarantine), "training_started": False})
    (output / "README.txt").write_text(README)
    write_json(output / "DATASET_MANIFEST.json", {
        **report,
        "artifact_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "source_manifest_sha256": digest(SOURCE / "DATASET_MANIFEST.json"),
        "source_frozen_sha256": digest(FROZEN / "FROZEN.json"),
        "adjudication_source_sha256": digest(REVIEW / "decisions.jsonl"),
        "packager_sha256": digest(Path(__file__)),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATASETS / "condition_generation_v8_adjudicated")
    build(parser.parse_args().output.resolve())
