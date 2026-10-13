"""Make English single-action prerequisites specific to the before interface.

Writes a new dataset version. Postconditions, reasons, labels and model inputs
are immutable. Before-state annotations are used as evidence, not previous
condition fields. Visually reviewed replacements are recorded separately.
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

from convert_qreflector_condition_generation import parse_source
from repair_qreflector_condition_semantics import context_of, tap_target, named_field


TARGET_RULES = {
    "explicit_action_effect", "intended_text_entry", "focus_field_only",
    "single_backspace", "long_press_selection_step", "open_clipboard_panel",
    "paste_current_content", "open_menu_only", "open_sort_filter_options",
    "scroll_intermediate_progress", "single_keypress",
}
INPUT_RULES = {"intended_text_entry", "focus_field_only", "single_backspace",
               "open_clipboard_panel", "paste_current_content", "single_keypress"}
GENERIC_PHRASES = ("intended input field", "intended tap target", "intended content area",
                   "intended numeric/text entry", "the intended text or item",
                   "indicated navigation or overflow-menu control",
                   "indicated sorting or filtering control")
VISUAL_FILE = Path(__file__).with_name("qreflector_precondition_visual_review.json")
ENGLISH_UI_TEXT = {
    "被引用次数": "Cited by", "賈伯斯傳": "Steve Jobs biography",
    "全部": "All", "图片": "Images", "购物": "Shopping",
    "越式嫩牛炒": "Vietnamese tender beef stir-fry",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences(text):
    # Protect quoted filenames, titles, abbreviations and UI strings from
    # sentence splitting, while retaining their original positions and text.
    def mask_quote(match):
        value = match.group()
        return ("_" * (len(value)-2) + value[-2:]) if value[-2] in ".!?" else "_" * len(value)
    mask = re.sub(r'"[^"\n]*"|`[^`\n]*`', mask_quote, text)
    cuts = [0] + [m.end() for m in re.finditer(r'[.!?]["`]?\s+(?=[A-Z])', mask)] + [len(text)]
    return [text[a:b].strip() for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


def remove_future_clauses(sentence):
    sentence = re.split(r",? (?:allowing|enabling|indicating that|suggesting that|ready to)\b|\bso that\b",
                        sentence, maxsplit=1)[0]
    return sentence.strip().rstrip(".,;:") + "."


def before_facts(before, action, rule):
    description = action["description"]
    if before.strip() == description.strip() or re.match(r"I will\b|This action\b|Swipe\b|Tap\b", before):
        raise ValueError("An action description cannot serve as before-screen evidence")
    parts = sentences(before)
    first = remove_future_clauses(parts[0])
    if re.search(r"\b(?:must|should|will)\b", first, re.I):
        raise ValueError("The first before-state sentence is not an observation")
    selected = [first]
    stop = {"this", "that", "screen", "action", "will", "with", "from", "open", "current",
            "allow", "allowing", "tap", "tapping", "the", "and", "for", "then", "which", "visible"}
    keywords = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", description)} - stop
    for part in parts[1:]:
        if part.startswith("The user "):
            continue
        if re.search(r"\b(?:must|should|will|needs? to|can then|can now|ready to|expected)\b", part, re.I):
            continue
        if not re.search(r"\b(?:is|are|has|contains?|shows?|displays?|includes?|appears?|remains?|have)\b", part, re.I):
            continue
        words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", part)}
        input_state = rule in INPUT_RULES and re.search(r"keyboard|focus|cursor|caret|selected|selection|highlight", part, re.I)
        if input_state or len(keywords & words) >= 2:
            fact = remove_future_clauses(part)
            if len(" ".join(selected + [fact]).split()) <= 100:
                selected.append(fact)
        if len(selected) == 3:
            break
    return " ".join(selected)


def field_for(before, action):
    label = re.search(r'["`]([^"`\n]{1,70})["`]\s+(?:text|input)\s+(?:field|box)', action["description"])
    if label:
        return json.dumps(label.group(1), ensure_ascii=False) + " field"
    field = named_field(action["description"])
    if field != "intended input field":
        return field
    if re.search(r"departure (?:location )?field", action["description"], re.I):
        return "departure airport/city selector"
    if re.search(r"destination (?:location )?field", action["description"], re.I):
        return "destination airport/city selector"
    if re.search(r"description (?:field|text)", action["description"], re.I) and "recipe" in before.lower():
        return '"Description" field of the recipe form'
    # Prioritize an explicitly focused/selected labeled field over arbitrary
    # labels elsewhere in a form. Preserve exact UI names, not their values.
    fields = list(re.finditer(r'["`]([^"`\n]{1,70})["`]\s+(?:input\s+)?(?:field|box)', before))
    if fields:
        focus = [m for m in fields if re.search(r"focus|active|filled|contains|selected|highlight|populated",
                                               before[m.end():m.end()+100], re.I)]
        return json.dumps((focus or fields)[0].group(1), ensure_ascii=False) + " field"
    if re.search(r"search (?:field|bar|box)|address bar", before, re.I):
        return "search field" if "address bar" not in before.lower() else "browser address bar"
    if re.search(r"filename|file name|note name|renam", before, re.I):
        return "filename entry"
    if re.search(r"passcode|PIN", before):
        return "passcode entry"
    if re.search(r"timer|duration", before, re.I):
        return "timer's numeric entry"
    if re.search(r"note|Markor|Keep", before, re.I):
        return "note-body editor"
    if re.search(r"message|chat", before, re.I):
        return "message composer"
    # This fallback must be manually reviewed, not shipped as another generic
    # field placeholder.
    raise ValueError("Cannot identify the input destination from before-state evidence")


def specific_precondition(before, action, rule):
    facts = before_facts(before, action, rule)
    description = action["description"]
    if rule == "explicit_action_effect":
        target = tap_target(description)
        if not target:
            raise ValueError("Tap target could not be identified")
        need = f"On this screen, {target} must be present in the active interface layer, with its touch area unobstructed."
    elif rule in INPUT_RULES:
        field = field_for(before, action)
        if rule == "focus_field_only":
            # Opening a search icon is different from focusing an existing box.
            if re.search(r"magnifying glass|search icon", description, re.I):
                need = "The search-opening icon in this interface must receive the tap before a query editor can gain focus."
            else:
                need = f"The {field} must accept the tap for text focus or open its dedicated editor."
        elif rule == "single_backspace":
            need = f"The {field} must retain keyboard focus, with the caret immediately after deletable text or a text selection to remove."
        elif rule == "open_clipboard_panel":
            need = f"The {field} must be the active paste destination, with the keyboard's clipboard control exposed."
        elif rule == "paste_current_content":
            entry = re.search(r'clipboard (?:entry|content)\s+["`]([^"`]+)["`]', description, re.I)
            clip = json.dumps(entry.group(1), ensure_ascii=False) if entry else "the clipboard entry specified by this action"
            need = f"The {field} must have input focus and access to {clip}; replacing existing text additionally requires that text to be selected or cleared."
        elif rule == "single_keypress":
            key = re.search(r'["`]([0-9A-Za-z.,]|00)["`]', description)
            if not key:
                raise ValueError("Key identity was not explicit")
            need = f"The {field} must accept the {key.group(1)} key at its current entry position."
        else:
            if action["type"] == "Type":
                need = f"The {field} must already have text focus with the caret or selection at the insertion or replacement location."
            else:
                need = f"The {field} must accept focus and text from this input shortcut."
            if re.search(r"replace|overwrite|clear (?:the )?(?:current|existing|incorrect)|correct the typo", description, re.I):
                need += " The text to replace must be selected or cleared so new typing does not merely append to it."
    elif rule == "long_press_selection_step":
        target = re.search(r'["`]([^"`]{1,100})["`]', description)
        if target:
            need = f"The long press must address {json.dumps(target.group(1), ensure_ascii=False)} within this content, where selection or contextual actions are supported."
        else:
            need = "The long press must land on the displayed document text or list entry, rather than the blank space around it, to initiate selection."
    elif rule == "open_menu_only":
        location = "upper-right overflow menu" if re.search(r"top.right|three.dot", description, re.I) else "navigation menu control named in the action"
        need = f"The {location} for this page or item must be exposed outside any blocking dialog."
    elif rule == "open_sort_filter_options":
        need = "The results shown here must expose their sorting/filtering control; opening it does not require any filter to be applied yet."
    elif rule == "scroll_intermediate_progress":
        direction = re.search(r"(?:swipe|scroll)(?:s|ing)?\s+(up|down|left|right)", description, re.I)
        direction = direction.group(1).lower() if direction else "in the requested direction"
        need = f"The gesture must act on this content's scrollable region and permit movement {direction}, with further material beyond the current viewport."
    else:
        raise ValueError(f"Unsupported rule {rule}")
    result = facts + " " + need
    for original, english in ENGLISH_UI_TEXT.items():
        result = result.replace(original, english)
    if any(p in result.lower() for p in GENERIC_PHRASES):
        raise ValueError("A generic placeholder remains")
    if re.search(r"[\u3400-\u9fff]", result):
        raise ValueError("New annotation must be English")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    old_meta = json.loads((source / "metadata.json").read_text())
    assert old_meta["dataset_version"] == "condition_generation_v2_semantic_pass1"
    manual = json.loads(VISUAL_FILE.read_text())
    protected = dict(old_meta["protected_source_sha256"])
    protected.update({str(p): sha(p) for p in source.iterdir() if p.is_file()})
    products, changes, decisions, unresolved, stats = {}, [], [], [], {}
    for split, count in (("train", 882), ("val", 71), ("test", 1082)):
        raw_rows = (source / f"{split}.jsonl").read_text().splitlines(keepends=True)
        refs = [json.loads(x) for x in (source / f"{split}.references.jsonl").read_text().splitlines()]
        assert len(raw_rows) == len(refs) == count
        new_rows, new_refs, counts = [], [], Counter()
        for line, (raw, reference) in enumerate(zip(raw_rows, refs), 1):
            row = json.loads(raw)
            target = json.loads(row["messages"][1]["content"][0]["text"])
            rule = reference["semantic_repair"]["rule"]
            identity = f"{split}:{line}"
            counts["rows"] += 1
            if rule not in TARGET_RULES and identity not in manual:
                new_rows.append(raw); new_refs.append(reference)
                continue
            counts["targeted_rows"] += 1
            _, _, transition, _, _, _ = parse_source(reference["original_sample"])
            action = context_of(row)["action"]
            before = transition["from_state"]["description"]
            try:
                pre = manual.get(identity) or specific_precondition(before, action, rule)
            except ValueError as exc:
                unresolved.append({"split": split, "line": line, "issue": str(exc), "before_description": before,
                                   "action": action, "old_precondition": target["precondition"]})
                new_rows.append(raw); new_refs.append(reference)
                continue
            assert pre != target["precondition"]
            assert not re.search(r"[\u3400-\u9fff]", pre)
            method = "before_image_visually_reviewed" if identity in manual else "before_description_and_action"
            change = {"split": split, "line": line, "field": "precondition", "old": target["precondition"],
                      "new": pre, "method": method, "before_image": reference["converted_images"][0],
                      "source_before_description": before if method == "before_description_and_action" else None}
            changes.append(change)
            decision = {"split": split, "line": line, "method": method, "previous_rule": rule,
                        "independent_before_image_review": identity in manual}
            decisions.append(decision)
            reference["precondition_refinement"] = decision
            reference["previous_precondition_review_status"] = reference["condition_review_status"]
            # Do not upgrade the review status of the whole target: this pass
            # reviews only preconditions, not postconditions, reasons or labels.
            target["precondition"] = pre
            row["messages"][1]["content"][0]["text"] = json.dumps(target, ensure_ascii=False)
            new_rows.append(json.dumps(row, ensure_ascii=False) + "\n")
            new_refs.append(reference)
            counts[method] += 1
            counts["changed_rows"] += 1
        stats[split] = dict(counts)
        products[f"{split}.jsonl"] = "".join(new_rows)
        products[f"{split}.references.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_refs)
    print(json.dumps({"splits": stats, "unresolved": unresolved}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if unresolved:
        raise RuntimeError("Resolve all targeted prerequisites before writing the new version")
    assert args.output is not None
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    assert all(sha(Path(p)) == expected for p, expected in protected.items())
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for p in (source / "images").iterdir():
        (output / "images" / p.name).symlink_to(os.path.relpath(p.resolve(), output / "images"), target_is_directory=True)
    # Keep the earlier audit logs and pending review queue, identified as prior
    # pass artifacts in metadata, so their history is not silently discarded.
    for p in source.iterdir():
        if p.is_file() and p.name not in {"metadata.json", "validation_report.json"}:
            (output / p.name).write_bytes(p.read_bytes())
    for name, contents in products.items():
        (output / name).write_text(contents)
    for name, records in (("precondition_changes.jsonl", changes), ("precondition_decisions.jsonl", decisions)):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "precondition_visual_review.json").write_bytes(VISUAL_FILE.read_bytes())
    report = validate(source, output, protected, changes, stats)
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    metadata = copy.deepcopy(old_meta)
    metadata.update({
        "dataset_version": "condition_generation_v3_specific_preconditions",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_version_directory": str(source),
        "protected_source_sha256": protected,
        "precondition_refinement_script": str(Path(__file__).resolve()),
        "precondition_refinement_script_sha256": sha(Path(__file__)),
        "precondition_refinement_statistics": stats,
        "precondition_annotation_language": "English",
        "precondition_refinement_scope": "Only precondition targets change. Model inputs, postconditions, reasons, Yes/No labels, split/order and images remain identical to v2.",
        "precondition_evidence_policy": "Use before-screen description and current action only; manually review before screenshots where descriptions are missing or copied from action intent. Never use after-screen descriptions, reasons or labels to formulate these prerequisites.",
        "condition_mapping": "Latest preconditions are in precondition_changes.jsonl. Earlier semantic_changes.jsonl and the original sample references remain historical provenance, not the latest target values.",
        "condition_targets_ready_for_unreviewed_training": False,
        "limitations": [
            "Direct visual review in this pass covers the before screenshot for each row listed in precondition_visual_review.json; it does not claim after-image or whole-target validation.",
            "Other refined prerequisites use original before-state descriptions and current action intent; accuracy of all those original descriptions has not been visually reaudited.",
            "Postconditions, reasons and labels are unchanged from v2, including previously flagged inconsistencies. The inherited review_required.jsonl describes the earlier pass; compare with current targets when reviewing it.",
        ],
    })
    metadata["output_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    assert all(sha(Path(p)) == expected for p, expected in protected.items())
    print(json.dumps({"output": str(output), "validation_passed": report["passed"]}))


def validate(source, output, protected, changes, stats):
    logged = {(r["split"], r["line"]): r for r in changes}
    images, source_images, counts = 0, set(), {}
    for split, count in (("train", 882), ("val", 71), ("test", 1082)):
        old = [json.loads(x) for x in (source / f"{split}.jsonl").read_text().splitlines()]
        new = [json.loads(x) for x in (output / f"{split}.jsonl").read_text().splitlines()]
        old_refs = [json.loads(x) for x in (source / f"{split}.references.jsonl").read_text().splitlines()]
        new_refs = [json.loads(x) for x in (output / f"{split}.references.jsonl").read_text().splitlines()]
        assert len(old) == len(new) == len(old_refs) == len(new_refs) == count
        assert (source / f"{split}.inputs.jsonl").read_bytes() == (output / f"{split}.inputs.jsonl").read_bytes()
        labels, changed = Counter(), 0
        for line, (a, b, ar, br) in enumerate(zip(old, new, old_refs, new_refs), 1):
            at = json.loads(a["messages"][1]["content"][0]["text"])
            bt = json.loads(b["messages"][1]["content"][0]["text"])
            assert list(at) == list(bt) == ["precondition", "postcondition", "reason", "answer"]
            assert all(isinstance(v, str) and v.strip() for v in bt.values())
            labels[bt["answer"]] += 1
            restored = copy.deepcopy(b)
            restored["messages"][1]["content"][0]["text"] = a["messages"][1]["content"][0]["text"]
            assert restored == a, (split, line, "Non-target fields changed")
            assert ar["original_sample"] == br["original_sample"]
            if at != bt:
                change = logged[(split, line)]
                assert change["old"] == at["precondition"] and change["new"] == bt["precondition"]
                assert not any(p in bt["precondition"].lower() for p in GENERIC_PHRASES)
                assert not re.search(r"[\u3400-\u9fff]", bt["precondition"])
                bt["precondition"] = at["precondition"]
                assert bt == at, (split, line, "Field other than precondition changed")
                changed += 1
            for value in b["messages"][0]["content"]:
                if value["type"] == "image":
                    path = output / value["image"]
                    assert path.samefile(source / value["image"])
                    images += 1
                    source_images.add(str(path.resolve()))
        assert changed == stats[split]["changed_rows"]
        counts[split] = {"rows": count, "changed_preconditions": changed, "labels": dict(labels)}
    assert images == len(source_images) == 4070
    assert (output / "prompt_template.txt").read_bytes() == (source / "prompt_template.txt").read_bytes()
    assert all(sha(Path(p)) == expected for p, expected in protected.items())
    return {"passed": True, "splits": counts, "inputs_byte_identical": True,
            "postconditions_reasons_labels_unchanged": True, "only_precondition_targets_changed": True,
            "all_changes_reversible_from_audit": True, "all_protected_sources_unchanged": True,
            "new_annotation_language": "English", "new_annotations_contain_no_cjk": True,
            "targeted_generic_phrases_remaining": 0, "image_references": images,
            "unique_original_image_files": len(source_images), "copied_image_files": 0,
            "scope": "Structural, language and provenance checks; not exhaustive validation of all before-state descriptions."}


if __name__ == "__main__":
    main()
