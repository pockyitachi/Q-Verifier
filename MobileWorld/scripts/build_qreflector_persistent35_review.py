#!/usr/bin/env python3
"""Package a manual visual audit; never modify a dataset or recompute scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "artifacts/analysis/qreflector_error_overlap_20261009"
OUT = ROOT / "artifacts/analysis/qreflector_persistent35_review_20261009"
DATA = ROOT / '../datasets/condition_generation_v9_after_evidence'
DEFAULT_BROWSER_SOURCE = Path(
    "artifacts/vendor/android_world_plus/android_world/task_evals/single/browser.py"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser-source", type=Path, default=DEFAULT_BROWSER_SOURCE,
        help="BrowserDraw task source; relative paths are based on MobileWorld/.",
    )
    return parser.parse_args(argv)


def browser_source(path):
    return (ROOT / path).resolve()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    source = browser_source(parse_args(argv).browser_source)
    if not source.is_file():
        raise FileNotFoundError(
            f"BrowserDraw source not found: {source}; set --browser-source."
        )
    summary = json.loads((PRIOR / "summary.json").read_text())
    ids = summary["groups"]["all_24"]["always_wrong_ids"]
    notes = json.loads((OUT / "notes.json").read_text())
    assert len(notes) == len(ids) == 35
    assert sorted(r["id"] for r in notes) == sorted(ids)
    assert len({r["id"] for r in notes}) == 35
    sources = {r["sample_id"]: r for r in read_jsonl(PRIOR / "persistent_cases.jsonl")}
    refs = {r["sample_id"]: r for r in read_jsonl(DATA / "test.source_references.jsonl")}
    dataset_lines = (DATA / "test.jsonl").read_text().splitlines()
    protected = json.loads((OUT / "protected_hashes_before.json").read_text())
    for item in summary["dataset_checks"]:
        protected[item["path"]] = item["sha256"]
    for item in summary["runs"].values():
        protected[item["path"]] = item["sha256"]
    for item in summary["baselines"].values():
        protected[item["path"]] = item["sha256"]
    checks = {p: sha(Path(p)) == expected for p, expected in protected.items()}
    assert all(checks.values()), "A protected source changed; inspect before packaging."

    source_lines = source.read_text().splitlines()
    excerpt = "\n".join(f"{i + 1}: {source_lines[i]}" for i in range(569, 596))
    (OUT / "browserdraw_source_evidence.txt").write_text(
        f"Source: {source}\nSHA256: {sha(source)}\n"
        "The local implementation matches the visible task layout. The exact HTML "
        "loaded during the historical rollout was not recovered; this is supporting "
        "source evidence, not a verified historical runtime dump.\n\n" + excerpt + "\n"
    )
    cases = []
    overlay = []
    trained_names = list(summary["runs"])
    for note in sorted(notes, key=lambda r: r["id"]):
        sid = note["id"]
        src = sources[sid]
        assert src["wrong_of_24"] == 24
        assert all(not src["predictions"][n]["correct"] for n in trained_names)
        original = [m for m in refs[sid]["original_sample"]["messages"] if m["role"] == "assistant"]
        images = [(DATA / image).resolve() for image in src["images"]]
        assert len(images) == 2 and all(p.is_file() for p in images)
        proposal = note["proposed_label"]
        assert proposal in (None, "yes", "no")
        if note["category"] == "model_error":
            assert proposal == src["label"]
        changed_proposal = proposal is not None and proposal != src["label"]
        status = (
            "strong_label_change_proposal" if changed_proposal and note["confidence"] == "high"
            else "conditional_label_change_proposal" if changed_proposal
            else "criterion_boundary_hold" if note["category"] == "label_question"
            else "insufficient_evidence_hold" if note["category"] == "insufficient_evidence"
            else "retain_reference_model_error"
        )
        record = {
            "sample_id": sid,
            "source_sample_id": f"test-{sid}",
            "original_label": src["label"],
            "proposed_label": proposal,
            "review_status": status,
            "category": note["category"],
            "subtype": note["subtype"],
            "confidence": note["confidence"],
            "visual_evidence": note["evidence"],
            "repair_recommendation": note["repair"],
            "trajectory": src["trajectory"],
            "context": src["context"],
            "before_image": str(images[0]),
            "after_image": str(images[1]),
            "image_sha256": [sha(p) for p in images],
            "original_row_sha256": hashlib.sha256(dataset_lines[sid - 1].encode()).hexdigest(),
            "original_reference_response": original,
            "trained_predictions": {n: src["predictions"][n]["predicted"] for n in trained_names},
            "representative_model_response": {
                "run": "Sol_round1_geometry",
                "response": src["predictions"]["Sol_round1_geometry"]["full_response"],
            },
            "source_evidence": str(OUT / "browserdraw_source_evidence.txt") if note["subtype"] == "noninteractive_reference_swatch" else None,
            "applied_to_dataset": False,
            "eligible_for_training": False,
        }
        cases.append(record)
        overlay.append({k: record[k] for k in [
            "sample_id", "source_sample_id", "original_label", "proposed_label",
            "review_status", "category", "subtype", "confidence", "visual_evidence",
            "repair_recommendation", "original_row_sha256", "image_sha256",
            "applied_to_dataset", "eligible_for_training",
        ]})

    counts = dict(Counter(c["category"] for c in cases))
    status_ids = {status: [c["sample_id"] for c in cases if c["review_status"] == status]
                  for status in sorted({c["review_status"] for c in cases})}
    metadata = {
        "scope": "All 35 reference disagreements common to the 24 aligned trained settings.",
        "reviewer": "Assistant visual review; labels and predictions were visible. Not a blinded or independent human adjudication.",
        "visual_review": "All 35 original-resolution image pairs were inspected through the image-view tool. Ten were first inspected in the preceding audit; 25 were newly inspected here. The viewer may scale display size.",
        "category_counts": counts,
        "status_ids": status_ids,
        "source_test_sha256": sha(DATA / "test.jsonl"),
        "selection_limits": [
            "These cases were selected for persistent model/reference disagreement, not sampled randomly.",
            "Category proportions do not estimate label error prevalence across 1,082 cases.",
            "The 24 runs include correlated checkpoints, not 24 independent replications.",
            "Only 35 of the 615 cases ever misclassified across the runs are covered here.",
            "A label question is not an independently confirmed mislabel.",
            "Explanation findings refer to inspected representative outputs; all runs share the verdict disagreement, not necessarily the same reasoning.",
            "This audit does not establish image compression or label noise as the cause of the overall accuracy plateau.",
        ],
        "original_dataset_changed": False,
        "reported_scores_changed": False,
        "new_inference_or_training_started": False,
        "test_examples_must_not_be_used_as_training_targets": True,
    }
    dump(OUT / "review.json", {"metadata": metadata, "cases": cases})
    dump(OUT / "summary.json", metadata)
    (OUT / "adjudication_overlay.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in overlay))
    fields = ["source_sample_id", "original_label", "proposed_label", "review_status", "category",
              "subtype", "confidence", "visual_evidence", "repair_recommendation", "before_image", "after_image"]
    with (OUT / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)

    report = [
        "Q-Reflector persistent-35 visual audit (2026-10-09)",
        "", "Completed: 35/35 before/after pairs; original labels and reported scores unchanged.",
        f"Categories: {json.dumps(counts)}", "",
        "Strong label-change proposals (14): " + str(status_ids["strong_label_change_proposal"]),
        "  Nine rely primarily on visible app/target contradictions; five BrowserDraw cases also use local source evidence.",
        "Conditional proposals (4): " + str(status_ids["conditional_label_change_proposal"]),
        "Criterion-boundary holds (2): " + str(status_ids["criterion_boundary_hold"]),
        "Insufficient-evidence holds (5): " + str(status_ids["insufficient_evidence_hold"]),
        "Retain reference, model error (10): " + str(status_ids["retain_reference_model_error"]),
        "", "Interpretation and repair order:",
        "1. Preserve this as a versioned adjudication overlay. It contains proposals, not automatic label replacements.",
        "2. Resolve the local-action/progress criterion consistently: a one-character correction or useful recovery may succeed before the whole task is complete. An arbitrary successful UI interaction is not automatically useful progress.",
        "3. For 545/850, distinguish strict multi-select intent from useful single-item deletion access. Do not silently switch policies between examples.",
        "4. For 23/100/114/120/382, obtain focus, clipboard, selected-item or timing evidence where available. Unchanged screenshots alone do not establish failure.",
        "5. Model repairs should target tiny text edits, control identity and unsupported progress claims using new TRAINING trajectories. Do not feed these test cases or their corrected answers into SFT/OPD.",
        "6. A resolution/crop experiment is a future hypothesis test. No inference, GPU job, prompt deployment or training was launched by this audit.",
        "7. Before publishing revised benchmark scores, use an independent, consistently applied adjudication protocol including controls beyond model-disagreement cases, then rescore every compared model on the same version.",
        "", "Limits:",
        *["- " + limit for limit in metadata["selection_limits"]],
        "- BrowserDraw source matches the layout but is not the recovered historical runtime HTML.",
        "", "Files:",
        "review.csv: compact per-case decisions, evidence, repair recommendations and image paths.",
        "review.json: full context, original references, model responses and provenance.",
        "adjudication_overlay.jsonl: separate machine-readable proposals; applied_to_dataset=false.",
        "notes.json: manually authored review findings.",
        "browserdraw_source_evidence.txt: supporting code excerpt and hash.",
        "verification.json: source/prediction integrity checks.",
    ]
    (OUT / "REPORT.txt").write_text("\n".join(report) + "\n")
    after_checks = {p: sha(Path(p)) == expected for p, expected in protected.items()}
    assert all(after_checks.values())
    dump(OUT / "verification.json", {
        "case_count": len(cases), "unique_case_count": len({r["sample_id"] for r in cases}),
        "matches_all_24_always_wrong_ids": sorted(r["sample_id"] for r in cases) == sorted(ids),
        "all_70_image_paths_exist": True, "category_counts_sum_to_35": sum(counts.values()) == 35,
        "protected_file_checks": after_checks, "protected_files_unchanged": all(after_checks.values()),
        "reported_metrics_recomputed": False, "original_labels_modified": False,
    })
    print(json.dumps({"cases": len(cases), "counts": counts, "status_ids": status_ids,
                      "protected_files": len(after_checks), "all_unchanged": all(after_checks.values())}, indent=2))


if __name__ == "__main__":
    main()
