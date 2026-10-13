"""Import the completed 2026-10-02 synthetic review into a new, immutable version.

Usage: python scripts/qreflector_synthetic_audit/import_user_reviews.py EXPORT.json
Labels come from the user's export. Explicit English edits below align the
conditions and explanations; these edits are not attributed to the user.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair_qreflector_condition_semantics import context_of

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '../datasets/condition_generation_v5_synthetic_audited'
OUTPUT = ROOT / '../datasets/condition_generation_v6_user_reviewed'
CASES = ROOT / '../review/condition_generation_v5_synthetic_audit/cases.json'
EXPECTED_EXPORT_SHA = "2764e7d1d0b1904b2d0731ce5c06016dbb6ccb069831285e40c474b697fa5db8"
TARGET_KEYS = ("precondition", "postcondition", "reason", "answer")
RESTORED = {679, 698, 707, 713, 748, 797, 834, 844, 853, 860, 876, 877, 881}

# Only text is edited here. No label override is permitted.
TEXT_EDITS = {
    679: {
        "postcondition": "Tripadvisor returns from the Cantor Arts Center detail page to the museum search results, making it possible to locate The Foster Museum next.",
        "reason": "The screen remains on the Cantor Arts Center detail page. The gallery image changes, but the museum search results are not restored, so the intended return-navigation step is not completed.",
    },
    698: {
        "postcondition": "Tripadvisor returns from the Cantor Arts Center detail page to the museum search results so that The Foster Museum can be located next.",
        "reason": "The tap opens the Cantor Arts Center photo gallery rather than returning to the museum search results. It does not complete the return-navigation step needed to continue to the next museum.",
    },
    707: {
        "postcondition": "The Explore interface opens with search available for locating The Foster Museum. Searching for that museum is a later action.",
        "reason": "The after screen shows traveler insights, ratings, and shuttle information for Cantor Arts Center. It remains within this museum's details instead of opening the Explore search interface needed for the next museum lookup.",
    },
    713: {
        "postcondition": "The existing 6:00 AM alarm is switched on. The separate 6:00 PM alarm is not the requested alarm.",
        "reason": "The tap opens the 6:00 PM alarm editor while the 6:00 AM alarm remains disabled. Opening the PM entry does not enable the existing 6:00 AM alarm requested for this step.",
    },
    748: {
        "precondition": "The app drawer is open and the Chrome icon is already visible beside Chevron. Opening Chrome requires activating that icon; the recorded action is a swipe despite the description calling it a tap.",
        "postcondition": "Chrome opens with its browser interface available for entering the Winter Olympics query. Entering the query is a later action.",
        "reason": "The recorded swipe shifts the app drawer slightly, while Chrome remains visible and unopened. It neither launches the browser nor exposes a previously inaccessible Chrome icon.",
    },
    797: {
        "postcondition": "The incorrect query test@example.com is cleared from the search editor so the 55-inch 4K OLED TV query can be entered next.",
        "reason": "The tap selects My store & nearby stores in the open Select stores sheet. The sheet remains open and the query test@example.com is not cleared, so this step does not prepare an empty query field for the TV search.",
    },
    834: {
        "precondition": "Booking.com's Stays form shows Champaign, a date range, and the row 1 room, 2 adults, 0 children above Search. The recorded point (815, 1083) lies in that room-and-guest row, which can open the occupancy settings.",
        "postcondition": "The room-and-guest selector opens so the hotel's search parameters can be inspected or adjusted before submitting the search. Hotel results need not appear in this step.",
        "reason": "The tap opens Select rooms and guests. This provides access to the occupancy parameters for the Champaign hotel search and is a useful preparatory step. It does not yet submit the search or display hotel results.",
    },
    844: {
        "postcondition": "Google Maps completes a search for the current restaurant query within the displayed map area. A completed refresh may legitimately return the same restaurants.",
        "reason": "The after screenshot shows the same map and restaurant listing, with no visible confirmation of a completed refresh. The intended search-area update is not demonstrated in the provided transition.",
    },
    853: {
        "precondition": "Booking.com's Stays form shows Champaign, a date range, and the row 1 room, 2 adults, 0 children above Search. The recorded point (663, 1089) lies in the room-and-guest row rather than the Search button.",
        "postcondition": "The room-and-guest selector opens, allowing the occupancy parameters to be checked or adjusted before the Champaign hotel search is submitted.",
        "reason": "The tap opens the room-and-guest selector. Inspecting these hotel-search parameters is a useful preparatory step even though the action description incorrectly calls this a search submission. The hotel results are not displayed yet.",
    },
    860: {
        "postcondition": "The Call action for Desserts ETC. opens an interface displaying that bakery's phone number, making it available to copy into Notes later.",
        "reason": "The tap opens the bakery's Menu tab with cake images rather than its Call interface. The phone number needed for the note is not displayed, so this phone-number retrieval step is not completed.",
    },
    876: {
        "postcondition": "Google Maps completes a refresh of the current restaurant query for the displayed map area. Successful completion does not require additional restaurants or a visibly different result list.",
        "reason": "The map and restaurant result remain unchanged, and the screenshots provide no visible evidence of the requested search-area refresh. Refresh completion is not demonstrated by this transition.",
    },
    877: {
        "postcondition": "The Google search widget is activated for text entry, with the keyboard available to enter top country songs of 2023. Opening text input is a useful intermediate outcome; completed typing and submission require separate evidence.",
        "reason": "The keyboard opens after the search-widget interaction. This is accepted as intermediate progress toward entering the song query. The query text and search results are not visible, so this judgment does not claim that typing or submission succeeded.",
    },
    881: {
        "reason": "The tap enables the in-store filter and opens Select stores rather than the query editor. The keyboard is not opened and the incorrect query remains, so the search field is not ready for the intended text correction.",
    },
}

REFRESH_CAVEAT = (
    "The user confirmed No, but unchanged screenshots cannot distinguish a failed "
    "refresh from a completed refresh with identical results. No execution or network "
    "log was supplied. This is a user-adjudicated label, not independently verified "
    "refresh failure. The training reason states lack of observed completion, not "
    "a proven ignored tap."
)
CAVEATS = {
    844: REFRESH_CAVEAT,
    876: REFRESH_CAVEAT,
    877: (
        "The user accepts keyboard opening as useful partial progress for the "
        "compound Tap_Type_and_Enter action. The screenshots do not establish the "
        "focused field, successful typing/submission, or the cause of the obscured "
        "search field. No UI-defect cause is asserted in the training target."
    ),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def target(row):
    return json.loads(row["messages"][1]["content"][0]["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    export_bytes = args.export.read_bytes()
    assert hashlib.sha256(export_bytes).hexdigest() == EXPECTED_EXPORT_SHA
    export = json.loads(export_bytes)
    page = json.loads(CASES.read_text())
    assert export["dataset_id"] == page["dataset_id"] == "qreflector-v5-audit-8a4141c613e6ddba"
    assert export["dataset_version"] == SOURCE.name
    assert export["schema_version"] == 1
    cases = {c["id"]: c for c in page["cases"]}
    reviews = export["reviews"]
    assert set(reviews) == set(cases) and len(reviews) == 18
    assert all(r["status"] == "done" for r in reviews.values())
    source_meta = json.loads((SOURCE / "metadata.json").read_text())
    protected = {str(p): sha(p) for p in SOURCE.iterdir() if p.is_file()}
    protected.update(source_meta["protected_source_sha256"])
    protected[str(CASES)] = sha(CASES)
    protected[str(args.export.resolve())] = EXPECTED_EXPORT_SHA
    assert all(sha(Path(p)) == h for p, h in protected.items())
    assert all(sha(SOURCE / name) == h for name, h in source_meta["output_sha256"].items())

    source_rows = read(SOURCE / "train.jsonl")
    source_refs = read(SOURCE / "train.references.jsonl")
    maps = read(SOURCE / "train.line_map.jsonl")
    audit_by_id = {a["sample_id"]: a for a in read(SOURCE / "synthetic_audit.jsonl")}
    quarantine = {q["sample_id"]: q for q in read(SOURCE / "quarantine.jsonl")}
    assert {int(s.split('-')[1]) for s in quarantine} == RESTORED == set(TEXT_EDITS)
    assert len(source_rows) == len(source_refs) == 869 and len(maps) == 882
    rows, refs, new_maps, audits, applied = [], [], [], [], []
    earlier_applied = read(SOURCE / "applied_user_reviews.jsonl")
    caveat_records = []
    image_hashes = {}
    v4 = Path(source_meta["previous_version_directory"])
    v4_rows = read(v4 / "train.jsonl")

    for m in maps:
        sid, line = m["sample_id"], m["source_line"]
        old_output = m["output_line"]
        if old_output is None:
            q = quarantine[sid]
            row, ref = copy.deepcopy(q["input"]), copy.deepcopy(q["source_reference"])
            before_target = None
        else:
            row, ref = copy.deepcopy(source_rows[old_output - 1]), copy.deepcopy(source_refs[old_output - 1])
            before_target = target(row)
        ctx = context_of(row)
        assert set(ctx) == {"goal", "action", "history"}
        record = None
        if sid in reviews:
            r = reviews[sid]
            f = r["fields"]
            assert f["goal"] == ctx["goal"]
            assert f["action_type"] == ctx["action"]["type"]
            assert f["action_description"] == ctx["action"]["description"]
            assert json.loads(f["action_arguments"]) == ctx["action"]["arguments"]
            raw_target = {k: f[k] for k in TARGET_KEYS}
            assert raw_target["answer"] in {"Yes", "No"}
            adjusted = {**raw_target, **TEXT_EDITS.get(line, {})}
            assert adjusted["answer"] == raw_target["answer"]
            assert all(isinstance(v, str) and v.strip() for v in adjusted.values())
            assert not re.search(r"[\u3400-\u9fff]", json.dumps(adjusted, ensure_ascii=False))
            msg = {"role": "assistant", "content": [{"type": "text", "text": json.dumps(adjusted, ensure_ascii=False)}]}
            row["messages"] = [row["messages"][0], msg]
            record = {
                "sample_id": sid, "source_line": line,
                "source_review_file": str(args.export.resolve()), "source_review_sha256": EXPECTED_EXPORT_SHA,
                "review": copy.deepcopy(r), "previous_active_target": before_target,
                "final_target": adjusted, "restored_from_quarantine": old_output is None,
                "label_source": "user_confirmed_export",
                "text_source": "assistant_english_alignment" if line in TEXT_EDITS else "unchanged_user_confirmed_target",
                "assistant_text_edits": {k: {"old": raw_target[k], "new": adjusted[k]} for k in TARGET_KEYS if raw_target[k] != adjusted[k]},
                "evidence_caveat": CAVEATS.get(line),
                "legacy_notes_policy": "Export notes contain previous assistant caveats and are preserved as provenance, not used as target text or attributed as newly authored user evidence.",
            }
            applied.append(record)
            ref["user_confirmed_review"] = copy.deepcopy(record)
            ref["condition_review_status"] = "user_label_confirmed_with_assistant_text_alignment" if line in TEXT_EDITS else "user_confirmed"
            if line in CAVEATS:
                caveat_records.append({"sample_id": sid, "source_line": line,
                    "answer": adjusted["answer"], "training_eligible": True,
                    "label_source": "user_confirmed_export", "evidence_caveat": CAVEATS[line]})
        assert row["messages"][0] == v4_rows[line - 1]["messages"][0]
        assert target(row)["answer"] in {"Yes", "No"}
        new_line = len(rows) + 1
        assert new_line == line
        if sid in audit_by_id:
            previous = audit_by_id[sid]
            audit = copy.deepcopy(previous)
            if record:
                audit["previous_v5_audit"] = copy.deepcopy(previous)
                audit["status"] = "positive" if target(row)["answer"] == "Yes" else "negative"
                audit["reviewer"] = "user_label_review_with_assistant_text_alignment"
                audit["review_method"] = "User marked the attached review done; assistant aligned English conditions/reasons. The exact original export is retained."
                audit["decision_evidence"] = target(row)["reason"]
                audit["replacement_target"] = target(row)
                audit["changes"] = {k: {"old": audit['original_target'][k], "new": target(row)[k]} for k in TARGET_KEYS if audit['original_target'][k] != target(row)[k]}
                audit["review_resolution"] = copy.deepcopy(record)
                audit["previous_issues"] = audit["issues"]
                audit["issues"] = ["user_label_retained_with_evidence_caveat"] if line in CAVEATS else []
            audit["output_train_line"] = new_line
            audit["output_dataset_version"] = OUTPUT.name
            ref["synthetic_audit"] = copy.deepcopy(audit)
            audits.append(audit)
            for phase, item in zip(("before", "after"), [c for c in row['messages'][0]['content'] if c['type'] == 'image']):
                p = (SOURCE / item['image']).resolve(strict=True)
                image_hashes.setdefault(str(p), sha(p))
                assert image_hashes[str(p)] == previous['image_sha256'][phase]
        ref.update({"v6_sample_id": sid, "v4_source_line": line, "output_line": new_line})
        rows.append(row)
        refs.append(ref)
        new_maps.append({**m, "previous_v5_output_line": old_output, "output_line": new_line,
                         "status": "user_confirmed" if record else m['status']})

    labels = dict(Counter(target(r)['answer'] for r in rows))
    assert len(rows) == 882 and labels == {"No": 357, "Yes": 525}
    assert len(applied) == 18 and sum(r['restored_from_quarantine'] for r in applied) == 13
    assert len(audits) == 235
    assert Counter(a['status'] for a in audits) == {"negative": 227, "positive": 8}
    assert all(rows[m['output_line'] - 1] == source_rows[m['output_line'] - 1] for m in maps if m['output_line'] is not None and m['source_line'] < 648)
    # Compare using stable IDs, since restoring quarantined rows shifts physical lines.
    assert all(rows[m['source_line'] - 1] == source_rows[m['output_line'] - 1] for m in maps if m['output_line'] is not None)
    pending = [r for r in read(SOURCE / "review_required.jsonl") if r['split'] != 'train']
    assert len(pending) == 26
    assert Counter(r['review']['fields']['answer'] for r in applied if r['restored_from_quarantine']) == {"Yes": 3, "No": 10}

    with tempfile.TemporaryDirectory(prefix=".user-review-import-", dir=OUTPUT.parent) as tmp:
        staged = Path(tmp) / "dataset"
        staged.mkdir()
        (staged / "images").mkdir()
        for namespace in ("train_val", "test"):
            (staged / "images" / namespace).symlink_to((SOURCE / 'images' / namespace).resolve(), target_is_directory=True)
        jsonl(staged / 'train.jsonl', rows)
        jsonl(staged / 'train.inputs.jsonl', [{"messages": [r['messages'][0]]} for r in rows])
        jsonl(staged / 'train.references.jsonl', refs)
        jsonl(staged / 'train.line_map.jsonl', new_maps)
        jsonl(staged / 'synthetic_audit.jsonl', audits)
        jsonl(staged / 'applied_user_reviews.jsonl', earlier_applied + applied)
        jsonl(staged / 'review_import_changes.jsonl', applied)
        jsonl(staged / 'evidence_caveats.jsonl', caveat_records)
        jsonl(staged / 'review_required.jsonl', pending)
        jsonl(staged / 'quarantine.jsonl', [])
        (staged / 'user_review_export.snapshot.json').write_bytes(export_bytes)
        shutil.copyfile(SOURCE / 'quarantine.jsonl', staged / 'previous_v5_quarantine.snapshot.jsonl')
        shutil.copyfile(SOURCE / 'prompt_template.txt', staged / 'prompt_template.txt')
        for split in ('val', 'test'):
            for suffix in ('jsonl', 'inputs.jsonl', 'references.jsonl'):
                name = f'{split}.{suffix}'
                shutil.copyfile(SOURCE / name, staged / name)
                assert sha(staged / name) == sha(SOURCE / name)
        # Read the written training artifacts back and validate alignment.
        assert read(staged / 'train.jsonl') == rows
        assert len(read(staged / 'train.inputs.jsonl')) == len(read(staged / 'train.references.jsonl')) == 882
        for split in ('train', 'val', 'test'):
            for row in read(staged / f'{split}.jsonl'):
                paths = [c['image'] for c in row['messages'][0]['content'] if c['type'] == 'image']
                assert len(paths) == 2
                assert all((staged / p).resolve(strict=True).is_file() for p in paths)
        assert all(sha(Path(p)) == h for p, h in protected.items())
        assert all(sha(Path(p)) == h for p, h in image_hashes.items())
        checks = {
            "passed": True, "train_rows": 882, "train_labels": labels,
            "val_rows": 71, "test_rows": 1082, "applied_reviews": 18,
            "restored_from_quarantine": 13, "restored_labels": {"Yes": 3, "No": 10},
            "existing_869_training_rows_unchanged": True, "quarantined": 0,
            "all_882_inputs_identical_to_v4": True, "user_labels_preserved": True,
            "validation_and_test_files_byte_identical": True, "prompt_unchanged": True,
            "raw_export_preserved_byte_identical": True, "all_image_paths_resolve": True,
            "copied_image_files": 0, "source_files_unchanged": True, "synthetic_images_unchanged": True,
            "retained_evidence_caveats": sorted(CAVEATS), "remaining_test_review_cases": 26,
            "scope": "Integrity checks and explicit review import; not an independent certification of every label or condition.",
        }
        dump(staged / 'validation_report.json', checks)
        report = """Q-Reflector v6 — completed user review import

Input: the user's 2026-10-02 review export for the v5 synthetic audit page.
All 18 cases were marked done. The exact export is saved unchanged.

Changes
  Restore 13 quarantined examples: 3 Yes (834, 853, 877), 10 No.
  Confirm the five previously repaired Yes labels: 678, 743, 781, 813, 845.
  Training: 882 rows; 525 Yes and 357 No.
  All 869 previously retained training rows are unchanged.
  Synthetic subset: 235 rows; 8 Yes and 227 No.
  Existing user-confirmed reviews 138 and 155 are retained.

English annotation alignment
  The user's labels are unchanged. Assistant edits to precondition,
  postcondition and reason are separately recorded in review_import_changes.jsonl.
  834/853: opening room/guest settings is the accepted useful hotel-search step.
  877: keyboard opening is accepted partial progress; typing/submission is not claimed.
  No examples claim an unchanged screen when a gallery, editor or filter opened.
  Prior assistant notes are provenance only and are not copied into training targets.

Evidence limits retained
  844/876 retain the user's No labels. Same-result refreshes remain observationally
  ambiguous; no execution or request log was provided to establish failure.
  877 does not visually establish the focused field or why the query field is hidden.
  These caveats are in evidence_caveats.jsonl; no extra human confirmation is claimed.
  The other training examples were not all independently re-reviewed in this import.
  26 existing test-condition/action ambiguities remain in review_required.jsonl.
  Labels concern useful task progress; they do not certify full execution of every
  primitive within a compound action. Condition text was not annotated in a blinded
  before-only experiment.

Preservation
  Original datasets and v5 are unchanged. Validation 71 and test 1,082, including
  their inputs/references, are byte-identical to v5. All 882 training inputs are
  unchanged from v4. Screenshot paths use directory links; no images were copied.
  Live review-page answers were not edited. No GPU/model/experiment was started.

Provenance
  train.line_map.jsonl maps stable train-N IDs to the current physical row.
  applied_user_reviews.jsonl includes earlier and new user confirmations.
  synthetic_audit.jsonl retains previous audit history and current targets.
  previous_v5_quarantine.snapshot.jsonl is historical, not an active exclusion list.
  The user confirmed labels and supplied reasons; final English wording adjustments
  are assistant-authored, not separately user-approved.
"""
        (staged / 'IMPORT_REPORT.txt').write_text(report)
        meta = {
            "schema_version": source_meta['schema_version'], "dataset_version": OUTPUT.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_version_directory": str(SOURCE), "protected_source_sha256": protected,
            "script_sha256": {str(Path(__file__).resolve()): sha(Path(__file__))},
            "review_import": {"dataset_id": export['dataset_id'], "export_sha256": EXPECTED_EXPORT_SHA,
                "exported_at": export['exported_at'], "completed_reviews": 18, "restored_rows": 13,
                "label_source": "user_confirmed_export", "assistant_text_alignment_rows": sorted(TEXT_EDITS)},
            "splits": {**source_meta['splits'], "train": {"rows": 882, "labels": labels}},
            "input_fields": source_meta['input_fields'], "output_fields": list(TARGET_KEYS),
            "sample_identity": "Stable train-N IDs and source_line refer to v4. In v6 all 882 IDs are restored in source order; output_line is the physical v6 row.",
            "image_policy": source_meta['image_policy'], "label_policy": source_meta['label_policy'],
            "quarantine_policy": "All 13 v5 quarantined cases restored after explicit user labels; active quarantine is empty. Historical quarantine is retained as a snapshot.",
            "repair_policy": "Preserve all user labels and original inputs. Explicit assistant English alignment is separately logged; raw user text and earlier caveats remain in provenance.",
            "test_policy": source_meta['test_policy'], "source_trace": source_meta['source_trace'],
            "distribution_notes": source_meta['distribution_notes'],
            "evidence_caveat_ids": [f'train-{n}' for n in sorted(CAVEATS)],
            "condition_targets_ready_for_unreviewed_training": False,
            "limitations": source_meta['limitations'] + [
                "The user adjudicated all 18 reviewed labels; assistant English alignments were not separately user-approved.",
                "844/876 No labels remain observationally ambiguous without refresh-completion logs.",
                "877 Yes accepts keyboard opening as partial progress and does not establish successful compound-action completion.",
            ],
        }
        meta['output_sha256'] = {p.name: sha(p) for p in staged.iterdir() if p.is_file()}
        dump(staged / 'metadata.json', meta)
        os.replace(staged, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), **checks}, indent=2))


if __name__ == '__main__':
    main()
