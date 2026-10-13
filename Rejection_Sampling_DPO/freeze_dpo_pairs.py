"""Build or validate the sole reviewed v9 DPO dataset. CPU/filesystem only.

Legacy completions are format-adapted, not new model rollouts. Positive targets
come from accepted v9 supervision; the wrong side of each legacy pair supplies
the negative rationale. Existing frozen output cannot be overwritten.
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

FIELDS = ("precondition", "postcondition", "reason", "answer")
CHECKS = ("Precondition check:", "Postcondition check:", "Task progress:")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def rows(path):
    return [parse(line) for line in Path(path).read_text().splitlines() if line.strip()]


def serialize(value):
    return json.dumps(value, ensure_ascii=False)


def response(value):
    assert len(value) == 1 and value[0]["role"] == "assistant"
    obj = parse(value[0]["content"])
    assert tuple(obj) == FIELDS
    assert all(isinstance(obj[k], str) and obj[k].strip() for k in FIELDS)
    assert obj["answer"] in ("Yes", "No")
    offsets = [obj["reason"].find(k) for k in CHECKS]
    assert offsets[0] == 0 and offsets == sorted(set(offsets))
    return obj


def completion(obj):
    result = [{"role": "assistant", "content": serialize(obj)}]
    response(result)
    return result


def select_legacy_negative(pair, expected):
    chosen, rejected = parse(pair["chosen"]), parse(pair["rejected"])
    assert expected in ("Yes", "No")
    assert chosen["answer"].lower() == pair["expected"].lower()
    assert {chosen["answer"], rejected["answer"]} == {"Yes", "No"}
    side = "rejected" if chosen["answer"] == expected else "chosen"
    return side, chosen if side == "chosen" else rejected


def legacy_negative(obj, accepted):
    """Preserve the selected old rationale; borrow only reviewed conditions.

    Numbered BEFORE/AFTER/CHANGE/evaluation sections are reheaded. For unsectioned
    rationales, reuse the accepted starting-state check and keep the complete
    old rationale in the outcome check. No new visual observation is invented.
    """
    reason = obj["reason"].strip()
    headers = list(re.finditer(r"(?m)^\s*\d[).]\s*([A-Z][A-Z _/-]+):\s*", reason))
    sections = []
    for i, match in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(reason)
        sections.append((match.group(1), reason[match.end():end].strip()))
    if headers and reason[:headers[0].start()].strip():
        raise ValueError("Legacy rationale has an unclassified preamble")
    pre = " ".join(text for key, text in sections if key == "BEFORE")
    post = " ".join(text for key, text in sections if key in ("AFTER", "CHANGE"))
    progress = " ".join(text for key, text in sections if key not in ("BEFORE", "AFTER", "CHANGE"))
    if sections:
        assert pre and post and progress, "Do not silently drop legacy rationale text"
        mode = "rehead_numbered_rationale"
    else:
        pre = accepted["reason"].split(CHECKS[0], 1)[1].split(CHECKS[1], 1)[0].strip()
        post = reason
        progress = ("The claimed effect makes valid progress toward the task."
                    if obj["answer"] == "Yes" else
                    "The claimed effect does not make valid progress toward the task.")
        mode = "preserve_unsectioned_rationale_with_reviewed_starting_check"
    converted = dict(precondition=accepted["precondition"], postcondition=accepted["postcondition"],
                     reason=f"{CHECKS[0]} {pre} {CHECKS[1]} {post} {CHECKS[2]} {progress}",
                     answer=obj["answer"])
    return converted, mode


def pair_id(pair):
    content = {k: pair[k] for k in ("sample_id", "prompt", "image_paths", "chosen", "rejected", "expected")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_rows(path, data):
    Path(path).write_text("".join(serialize(row) + "\n" for row in data))


def validate(directory=HERE, check_images=True):
    directory = Path(directory)
    manifest = parse((directory / "DPO_MANIFEST.json").read_text())
    assert manifest["status"] == "frozen" and manifest["source_training_instances"] == 882
    for name, expected in manifest["file_sha256"].items():
        assert digest(directory / name) == expected, f"Frozen file changed: {name}"
    data = rows(directory / "dpo_pairs.jsonl")
    assert len(data) == manifest["pairs"]
    seen = set()
    for pair in data:
        sid = pair["sample_id"]
        assert sid in manifest["covered_sample_labels"] and sid not in manifest["quarantined_sample_ids"]
        assert pair["expected"] == manifest["covered_sample_labels"][sid]
        assert pair["sample_idx"] == int(sid.split("-")[1]) - 1
        assert 0 <= pair["sample_idx"] < 882
        assert pair["pair_id"] == pair_id(pair) and pair["pair_id"] not in seen
        seen.add(pair["pair_id"])
        assert response(pair["chosen"])["answer"] == pair["expected"]
        assert response(pair["rejected"])["answer"] != pair["expected"]
        assert len(pair["prompt"]) == 1 and pair["prompt"][0]["role"] == "user"
        content = pair["prompt"][0]["content"]
        assert [c["type"] for c in content] == ["text", "image", "image"]
        text = content[0]["text"]
        assert text.count("=== Image Geometry ===") == text.count("=== Context ===") == 1
        # The criterion template must match v9 exactly after removing dynamic data.
        without_geometry = text.split("=== Image Geometry ===", 1)[0] + text.split("=== End Image Geometry ===\n\n", 1)[1]
        after_marker = without_geometry.index("=== Context ===") + len("=== Context ===")
        rest = without_geometry[after_marker:]
        context_start = after_marker + len(rest) - len(rest.lstrip())
        _, context_length = json.JSONDecoder().raw_decode(without_geometry[context_start:])
        template = without_geometry[:context_start] + "{context}" + without_geometry[context_start + context_length:]
        assert hashlib.sha256(template.encode()).hexdigest() == manifest["prompt_sha256"]
        context = parse(text.split("=== Context ===", 1)[1].split("=== Image Reference ===", 1)[0])
        assert set(context) == {"goal", "history", "action"}
        assert [c["image"] for c in content[1:]] == pair["image_paths"]
        assert all(not Path(p).is_absolute() for p in pair["image_paths"])
        assert all(p in manifest["image_sha256"] for p in pair["image_paths"])
        assert pair["origins"]
    assert len({p["sample_id"] for p in data}) == manifest["unique_training_instances"]
    if check_images:
        for path, expected in manifest["image_sha256"].items():
            assert digest(directory / path) == expected, f"Image changed or absent: {path}"
    audit = rows(directory / "dpo_pair_audit.jsonl")
    assert len(audit) == manifest["input_pairs"]
    assert Counter(r["decision"] for r in audit) == manifest["input_decisions"]
    for record in audit:
        if record["decision"] != "excluded_quarantine":
            assert record["pair_id"] in seen
    return {k: manifest[k] for k in ("status", "pairs", "unique_training_instances", "input_decisions")}


def freeze(legacy_path, recent_path, run, data, source882, output=HERE):
    output = Path(output)
    for name in ("dpo_pairs.jsonl", "DPO_MANIFEST.json", "dpo_pair_audit.jsonl"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite canonical data: {output / name}")
    legacy, recent = rows(legacy_path), rows(recent_path)
    assert (len(legacy), len(recent)) == (488, 119)
    originals = rows(source882 / "train.jsonl")
    assert len(originals) == 882
    image_index = {}
    for i, row in enumerate(originals, 1):
        key = tuple(Path(c["image"]).name for c in row["messages"][0]["content"] if c["type"] == "image")
        assert key not in image_index
        image_index[key] = f"train-{i}"
    accepted = {r["sample_id"]: r for r in rows(run / "corrected_train.jsonl")}
    quarantine = {r["sample_id"] for f in ("prior_quarantine.jsonl", "new_quarantine.jsonl") for r in rows(run / f)}
    assert len(accepted) == 849 and len(quarantine) == 33 and not set(accepted) & quarantine
    assert set(accepted) | quarantine == {f"train-{i}" for i in range(1, 883)}
    labels = {r["sample_id"]: r["answer"] for r in rows(data / "train.labels.jsonl")}
    image_contract = parse((data / "image_sha256.json").read_text())
    pairs, audit, images, conversions = {}, [], {}, Counter()
    reversed_inputs = 0
    for batch, inputs in (("legacy_488", legacy), ("post_distillation_119", recent)):
        for number, original in enumerate(inputs, 1):
            key = tuple(Path(p).name for p in original["image_paths"])
            sid = image_index[key]
            assert sid == f"train-{original['sample_idx']+1}"
            record = dict(input_batch=batch, input_row=number, sample_id=sid)
            if sid in quarantine:
                audit.append({**record, "decision": "excluded_quarantine"})
                continue
            sample = accepted[sid]
            assert sample["expected"] == labels[sid]
            target = parse(sample["row"]["messages"][1]["content"][0]["text"])
            assert target["answer"] == sample["expected"]
            user = copy.deepcopy(sample["row"]["messages"][0])
            paths = []
            for item in user["content"]:
                if item["type"] == "image":
                    original_path = item["image"]
                    path = (data / original_path).resolve()
                    path.relative_to(PROJECT)
                    assert digest(path) == image_contract[original_path]["sha256"]
                    item["image"] = os.path.relpath(path, output)
                    images[item["image"]] = image_contract[original_path]["sha256"]
                    paths.append(item["image"])
            if batch == "legacy_488":
                # Verify source identity independently of its legacy row number.
                old_action = parse(original["prompt"].split("=== Current Transition ===", 1)[1].split("=== Previous Steps", 1)[0])["action"]
                context = parse(user["content"][0]["text"].split("=== Context ===", 1)[1].split("=== Image Reference ===", 1)[0])
                assert old_action == context["action"]
                old_goal = original["prompt"].split("=== Global Task ===", 1)[1].split("=== Current Subgoal ===", 1)[0].strip()
                assert old_goal == context["goal"]
                side, old_negative = select_legacy_negative(original, target["answer"])
                reversed_inputs += side == "chosen"
                negative, mode = legacy_negative(old_negative, target)
                conversions[mode] += 1
                rejected = completion(negative)
                origin = {**record, "negative_original_side": side, "conversion": mode,
                          "original_negative_sha256": hashlib.sha256(original[side].encode()).hexdigest()}
            else:
                assert original["sample_id"] == sid and original["expected"] == target["answer"]
                assert response(original["chosen"]) == target
                assert original["prompt"][0]["content"][0]["text"] == user["content"][0]["text"]
                rejected = copy.deepcopy(original["rejected"])
                origin = {**record, "negative_original_side": "rejected", "conversion": "unchanged",
                          "source_adapter_sha256": original["source_adapter_sha256"]}
            pair = dict(sample_id=sid, sample_idx=int(sid.split("-")[1])-1, prompt=[user],
                        image_paths=paths, chosen=completion(target), rejected=rejected, expected=target["answer"])
            assert response(rejected)["answer"] != pair["expected"]
            fingerprint = pair_id(pair)
            duplicate = fingerprint in pairs
            if not duplicate:
                pairs[fingerprint] = {"pair_id": fingerprint, **pair, "origins": []}
            pairs[fingerprint]["origins"].append(origin)
            audit.append({**record, "decision": "deduplicated" if duplicate else "retained", "pair_id": fingerprint,
                          "negative_original_side": origin["negative_original_side"]})
    final = sorted(pairs.values(), key=lambda p: (p["sample_idx"], p["pair_id"]))
    covered = {p["sample_id"]: p["expected"] for p in final}
    template = (data / "prompt_template.txt").read_text()
    for pair in final:
        text = pair["prompt"][0]["content"][0]["text"]
        without_geometry = text.split("=== Image Geometry ===", 1)[0] + text.split("=== End Image Geometry ===\n\n", 1)[1]
        prefix, suffix = template.split("{context}")
        assert without_geometry.startswith(prefix) and without_geometry.endswith(suffix)
    report = parse((run / "completion_report.json").read_text())
    manifest = dict(status="frozen", canonical_file="dpo_pairs.jsonl", created_utc=datetime.now(timezone.utc).isoformat(),
        source_training_instances=882, accepted_v9_sources=849, quarantined_sample_ids=sorted(quarantine),
        input_pairs=607, input_batch_sizes={"legacy_488":488,"post_distillation_119":119},
        pairs=len(final), unique_training_instances=len(covered),
        input_decisions=dict(Counter(r["decision"] for r in audit)),
        label_direction_corrected_inputs=reversed_inputs, legacy_conversion_counts=dict(conversions),
        pairs_by_input_batch={batch:sum(any(o["input_batch"]==batch for o in p["origins"]) for p in final)
                             for batch in ("legacy_488","post_distillation_119")},
        pair_labels=dict(Counter(p["expected"] for p in final)), covered_sample_labels=covered,
        image_max_side=512, image_resize="RGB; aspect-preserving LANCZOS; no upscaling", image_sha256=images,
        prompt_sha256=digest(data / "prompt_template.txt"), prompt_geometry="original dimensions and normalized coordinates",
        response_fields=list(FIELDS), preference_basis="chosen matches accepted current v9 label; rejected contradicts it",
        chosen_source="accepted_v9_luna_target", legacy_rejected_is_format_adapted_not_new_rollout=True,
        legacy_expected_conditions="copied from the accepted target; old conditions are not model-generated four-field predictions",
        no_teacher_calls=True, no_model_generation=True, no_training_launched=True,
        evaluation_examples_used_as_pairs=False, source_matching="ordered training image names + stable ID + exact goal/action for legacy pairs",
        reference_model_sha256=report["final_adapter_sha256"],
        required_future_loss="standard DPO with a frozen reference; summed full-response token log-probabilities, excluding prompt/padding and including first response token and EOS",
        sources={"legacy_pairs_sha256":digest(legacy_path), "recent_pairs_sha256":digest(recent_path),
                 "accepted_targets_sha256":digest(run / "corrected_train.jsonl"),
                 "reviewed_labels_sha256":digest(data / "train.labels.jsonl"),
                 "source882_sha256":digest(source882 / "train.jsonl")},
        builder_sha256=digest(__file__), file_sha256={})
    output.parent.mkdir(parents=True, exist_ok=True)
    # A sibling directory keeps relative image paths identical while validating
    # before publication. The manifest is installed last as the freeze marker.
    with tempfile.TemporaryDirectory(prefix=".dpo-freeze-", dir=output.parent) as temp:
        staging = Path(temp)
        write_rows(staging / "dpo_pairs.jsonl", final)
        write_rows(staging / "dpo_pair_audit.jsonl", audit)
        manifest["file_sha256"] = {name:digest(staging / name) for name in ("dpo_pairs.jsonl", "dpo_pair_audit.jsonl")}
        (staging / "DPO_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        validate(staging)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("dpo_pairs.jsonl", "dpo_pair_audit.jsonl", "DPO_MANIFEST.json"):
            os.replace(staging / name, output / name)
    return validate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--directory", type=Path, default=HERE)
    build = sub.add_parser("freeze")
    for name in ("legacy-pairs", "recent-pairs", "run", "data", "source882"):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate(args.directory)
    else:
        result = freeze(args.legacy_pairs, args.recent_pairs, args.run, args.data, args.source882, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
