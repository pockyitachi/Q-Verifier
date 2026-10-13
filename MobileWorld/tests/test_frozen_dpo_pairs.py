"""Validate the one canonical preference dataset without GPUs or API calls."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

DATA = Path(__file__).resolve().parents[2] / "training" / "Rejection_Sampling_DPO"
spec = importlib.util.spec_from_file_location("frozen_dpo", DATA / "freeze_dpo_pairs.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def accepted():
    return dict(precondition="The query field is editable.", postcondition="The requested text appears in the field.",
                reason="Precondition check: The field is focused. Postcondition check: The text did not change. Task progress: No useful progress.",
                answer="No")


def test_frozen_dataset_contract_and_source_coverage():
    summary = frozen.validate(DATA, check_images=False)
    assert summary["pairs"] == 559
    assert summary["unique_training_instances"] == 263
    assert summary["input_decisions"] == dict(retained=559, excluded_quarantine=25, deduplicated=23)
    manifest = frozen.parse((DATA / "DPO_MANIFEST.json").read_text())
    assert manifest["pairs_by_input_batch"] == dict(legacy_488=440, post_distillation_119=119)
    assert manifest["label_direction_corrected_inputs"] == 57


def test_changed_labels_select_the_former_chosen_as_negative():
    pair = dict(chosen=json.dumps(dict(answer="Yes", reason="The query was entered.")),
                rejected=json.dumps(dict(answer="No", reason="The query remains blank.")), expected="yes")
    before = copy.deepcopy(pair)
    side, negative = frozen.select_legacy_negative(pair, "No")
    assert side == "chosen" and negative["answer"] == "Yes"
    assert frozen.select_legacy_negative(pair, "Yes")[0] == "rejected"
    assert pair == before


def test_numbered_conversion_preserves_all_old_claims():
    old = dict(answer="Yes", reason="1) BEFORE: OLD_BEFORE.\n2) AFTER: OLD_AFTER.\n3) CHANGE: OLD_CHANGE.\n4) EVALUATION: OLD_EVAL.\n5) FINAL: OLD_FINAL.")
    result, mode = frozen.legacy_negative(old, accepted())
    assert mode == "rehead_numbered_rationale"
    assert frozen.response(frozen.completion(result)) == result
    for marker in ("OLD_BEFORE", "OLD_AFTER", "OLD_CHANGE", "OLD_EVAL", "OLD_FINAL"):
        assert result["reason"].count(marker) == 1
    assert result["precondition"] == accepted()["precondition"]
    assert result["postcondition"] == accepted()["postcondition"]
    assert result["answer"] == "Yes"


def test_unsectioned_conversion_does_not_invent_visual_observations():
    old = dict(answer="Yes", reason="Action succeeded: the query was entered.")
    result, _ = frozen.legacy_negative(old, accepted())
    assert old["reason"] in result["reason"]
    assert "Precondition check: The field is focused." in result["reason"]
    assert "Task progress: The claimed effect makes valid progress" in result["reason"]


def copy_dataset(tmp_path):
    for name in ("dpo_pairs.jsonl", "DPO_MANIFEST.json", "dpo_pair_audit.jsonl"):
        shutil.copyfile(DATA / name, tmp_path / name)
    return frozen.rows(tmp_path / "dpo_pairs.jsonl")


def reseal_for_semantic_test(tmp_path, pairs):
    frozen.write_rows(tmp_path / "dpo_pairs.jsonl", pairs)
    manifest = frozen.parse((tmp_path / "DPO_MANIFEST.json").read_text())
    manifest["file_sha256"]["dpo_pairs.jsonl"] = frozen.digest(tmp_path / "dpo_pairs.jsonl")
    (tmp_path / "DPO_MANIFEST.json").write_text(json.dumps(manifest))


def test_checksum_rejects_even_a_one_byte_change(tmp_path):
    copy_dataset(tmp_path)
    with (tmp_path / "dpo_pairs.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(AssertionError, match="Frozen file changed"):
        frozen.validate(tmp_path, check_images=False)


def test_semantic_validation_rejects_a_correct_label_as_negative(tmp_path):
    pairs = copy_dataset(tmp_path)
    wrong = frozen.response(pairs[0]["rejected"])
    wrong["answer"] = pairs[0]["expected"]
    pairs[0]["rejected"] = frozen.completion(wrong)
    pairs[0]["pair_id"] = frozen.pair_id(pairs[0])
    reseal_for_semantic_test(tmp_path, pairs)
    with pytest.raises(AssertionError):
        frozen.validate(tmp_path, check_images=False)


def test_prompt_cannot_include_a_reference_label(tmp_path):
    pairs = copy_dataset(tmp_path)
    item = pairs[0]["prompt"][0]["content"][0]
    item["text"] = item["text"].replace('"goal":', '"reference_answer": "No", "goal":', 1)
    pairs[0]["pair_id"] = frozen.pair_id(pairs[0])
    reseal_for_semantic_test(tmp_path, pairs)
    with pytest.raises(AssertionError):
        frozen.validate(tmp_path, check_images=False)


def test_freeze_refuses_to_overwrite_before_reading_sources():
    with pytest.raises(FileExistsError, match="Refusing to overwrite canonical"):
        frozen.freeze(None, None, None, None, None, output=DATA)


def test_legacy_generator_cannot_overwrite_the_frozen_dataset():
    before = frozen.digest(DATA / "dpo_pairs.jsonl")
    result = subprocess.run([sys.executable, str(DATA / "create_dpo_pairs.py"),
                             "--output_path", str(DATA / "dpo_pairs.jsonl")],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "frozen" in result.stderr
    assert frozen.digest(DATA / "dpo_pairs.jsonl") == before
