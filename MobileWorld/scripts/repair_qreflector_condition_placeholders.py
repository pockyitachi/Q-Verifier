"""Build a separate dataset with reviewed initial-state placeholders repaired.

This is a narrow annotation repair, not the proposed condition-generation SFT
conversion. Labels, actions, image paths, splits, and non-placeholder fields stay
unchanged. The original dataset is never overwritten.
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


INITIAL = {
    "description": "The initial screen state before the task begins.",
    "precondition": "The device is ready for interaction.",
    "postcondition": "The user can perform actions on the current screen.",
}
APP_TARGETS = {
    "4_x_back_myth_wukong": "X",
    "4_x_trending_news": "X",
    "2_recipes_chinese": "Google Chrome",
    "5_palo_alto_tour": "Tripadvisor",
    "General_Tool_1_ENG": "LinkedIn",
    "5_things_to_do_la": "Tripadvisor",
    "Social_Sharing_0_ENG": "X",
    "2_llm_agents_survey": "Google Chrome",
    "Multi_Apps_3_ENG": "Zoom",
    "3_oled_tv": "Best Buy",
    "Information_Management_1_ENG": "Google Chrome",
    "Media_Entertainment_0_ENG": "YouTube",
    "2_mcdonalds_deals": "McDonald's",
    "1_chinese_ohare": "Google Maps",
    "4_movie_trending": "Fandango",
    "3_walmart_sale_items": "Walmart",
    "Multi_Apps_0_ENG": "Google Chrome",
    "Social_Sharing_2_ENG": "Spotify",
    "Information_Management_0_ENG": "Facebook",
    "4_horror_movie_reviews": "Fandango",
    "1_nearest_bakery": "Google Maps",
    "5_cheap_flights_newyork": "Booking.com",
    "Information_Management_2_ENG": "Google Chrome",
    "General_Tool_0_ENG": "Google Play Store",
    "Multi_Apps_1_ENG": "Google Chrome",
    "Social_Sharing_1_ENG": "CBC News",
    "Media_Entertainment_1_ENG": "Google Chrome",
    "3_laptop_nvidia_gpu": "Amazon Shopping",
    "2_segment_anything_cited": "Google Chrome",
    "2_headphones_reviews": "Amazon Shopping",
    "1_late_night_korean_food": "Google Maps",
}
DRAWER_TARGETS = {
    "Multi_Apps_4_ENG": "Expedia",
    "Web_Shopping_0_ENG": "Facebook",
    "General_Tool_2_ENG": "Settings",
    "Multi_Apps_2_ENG": "Google Play Store",
    "Media_Entertainment_2_ENG": "Clock",
}
HOME_TARGETS = {
    "3_nintendo_switch_joy_con": "Google Chrome",
    "5_hotel_champaign": "Booking.com",
    "5_local_food_chicago": "Tripadvisor",
}
REVIEW_NOTES = {
    "2_recipes_chinese": {
        "category": "label_conflicts_with_local_action_effect",
        "note": "Original label is No because Chrome restored an unrelated Scholar tab. "
        "The action and subgoal only specify opening Chrome, which the screenshot "
        "supports. The original label and reason were retained for separate review.",
    },
    "Web_Shopping_1_ENG": {
        "category": "app_launch_intent_vs_folder_opening",
        "note": "The action description requests launching Amazon Shopping, but the "
        "tap targets the collapsed SPA4 folder and the after image shows the expanded "
        "folder, not Amazon Shopping. Original label Yes may reflect preparatory "
        "progress. The repair retains app launch as the stated expected effect; "
        "the original action, label, and reason were not rewritten to fit the outcome.",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(text: str):
    decoder = json.JSONDecoder()
    start = text.index("=== Current Transition ===") + len("=== Current Transition ===")
    start += len(text[start:]) - len(text[start:].lstrip())
    transition, size = decoder.raw_decode(text[start:])
    match = re.search(r"=== Previous Steps \(latest \d+\) ===\s*", text)
    assert match is not None
    hstart = match.end()
    history, hsize = decoder.raw_decode(text[hstart:])
    return transition, history, [(start, start + size), (hstart, hstart + hsize)]


def get_user_text(sample: dict):
    items = [c for m in sample["messages"] if m["role"] == "user"
             for c in m["content"] if c.get("type") == "text"]
    assert len(items) == 1
    return items[0]


def get_task(sample: dict) -> str:
    images = [c["image"] for m in sample["messages"] if m["role"] == "user"
              for c in m["content"] if c.get("type") == "image"]
    assert len(images) == 2
    return re.sub(r"_\d+_before\.[^.]+$", "", Path(images[0]).name)


def annotate(task: str, record: dict) -> dict:
    action_type = record["action"]["type"]
    if task in APP_TARGETS:
        assert action_type == "Tap"
        app = APP_TARGETS[task]
        description = f"The Android launcher is visible with an unobstructed {app} app icon."
        pre = f"The {app} launcher icon must be directly accessible for the intended tap, with no dialog covering it."
        post = f"{app} is brought to the foreground and its application interface is visible."
        if app == "Google Chrome":
            description = "An Android launcher page is visible with the Google Chrome shortcut in the bottom dock."
            post = "Google Chrome is brought to the foreground with its browser interface visible. A restored tab or a new-tab page can both satisfy this app-opening action."
        if app == "Zoom":
            post = "Zoom is brought to the foreground and displays its own interface or launch dialog. Scheduling a meeting is a later step."
    elif task in DRAWER_TARGETS:
        assert action_type == "Swipe"
        app = DRAWER_TARGETS[task]
        description = f"The Android home screen is visible; the {app} icon is not on this launcher page, and the app drawer is closed."
        pre = "The Android home screen must be in the foreground with no modal overlay intercepting the upward swipe used to open the app drawer."
        post = f"The app drawer opens and displays its app grid so the user can continue locating {app}. This swipe does not itself need to launch {app}."
    elif task in HOME_TARGETS:
        assert action_type == "Home"
        app = HOME_TARGETS[task]
        description = "Google Keep is displaying its note list, with the Android Home navigation control visible at the bottom."
        pre = "Google Keep is in the foreground and Android Home navigation is available to return to the launcher."
        post = f"The Android home screen is displayed instead of the Google Keep note list, allowing the user to locate {app} next."
    elif task == "Web_Shopping_2_ENG":
        assert action_type == "Tap"
        description = "The Android home screen shows the closed SPA4 folder, whose preview includes Instagram."
        pre = "The SPA4 folder must be visible and unobstructed on the launcher so it can be opened by tapping its icon."
        post = "The SPA4 folder is expanded and the Instagram app icon is available for the next tap. Opening Instagram itself is not required by this folder-opening action."
    elif task == "Web_Shopping_1_ENG":
        assert action_type == "Tap"
        description = "The Android home screen shows the closed SPA4 folder with a miniature Amazon Shopping icon in its preview; an individual Amazon Shopping icon is not yet exposed."
        pre = "To launch Amazon Shopping with one tap, its individual launcher icon must be exposed and directly selectable. A closed folder preview alone does not meet this prerequisite."
        post = "Amazon Shopping is brought to the foreground and displays its application interface."
    else:
        raise ValueError(f"No reviewed annotation for {task}")
    return {"description": description, "precondition": pre, "postcondition": post}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reviewed-transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    assert source != output and not output.exists(), "Use a new output directory."
    records = json.loads(args.reviewed_transitions.read_text())
    known = set(APP_TARGETS) | set(DRAWER_TARGETS) | set(HOME_TARGETS) | {
        "Web_Shopping_1_ENG", "Web_Shopping_2_ENG"
    }
    assert set(records) == known and len(records) == 41
    hashes = {name: sha256(source / name) for name in ["train.jsonl", "val.jsonl", "metadata.json"]}
    annotations = {}
    for task, record in records.items():
        annotations[task] = {
            "source_split": record["split"], "source_line": record["line"],
            "action": record["action"], "replacement": annotate(task, record),
            "evidence_images": {key: {"path": record[key], "sha256": sha256(Path(record[key]))}
                                for key in ["before", "after"]},
            "review_note": REVIEW_NOTES.get(task),
        }
    changes, split_stats, products = [], {}, {}
    for split in ["train", "val"]:
        lines = (source / f"{split}.jsonl").read_text().splitlines(keepends=True)
        result, row_changes, current_changes, history_changes = [], 0, 0, 0
        labels_before, labels_after = Counter(), Counter()
        for number, line in enumerate(lines, 1):
            original = json.loads(line)
            sample = copy.deepcopy(original)
            text_item = get_user_text(sample)
            text = text_item["text"]
            transition, history, spans = sections(text)
            task = get_task(sample)
            updated = False
            for scope, step in [("current", transition)] + [(f"history[{i}]", h) for i, h in enumerate(history)]:
                state = step["from_state"]
                matches = [key for key, value in INITIAL.items() if state.get(key) == value]
                if not matches:
                    continue
                assert set(matches) == set(INITIAL), (split, number, scope, "partial placeholder")
                assert task in annotations, (split, number, task)
                annotation = annotations[task]
                assert step["action"] == annotation["action"], (split, number, scope, "action mismatch")
                if scope.startswith("history"):
                    assert step.get("step") == 1
                replacement = annotation["replacement"]
                for field in INITIAL:
                    changes.append({"split": split, "line": number, "task": task,
                                    "field": f"{scope}.from_state.{field}",
                                    "old": state[field], "new": replacement[field],
                                    "annotation_source": f"{annotation['source_split']}:{annotation['source_line']}"})
                    state[field] = replacement[field]
                updated = True
                current_changes += scope == "current"
                history_changes += scope.startswith("history")
            if updated:
                row_changes += 1
                for (start, end), value in reversed(list(zip(spans, [transition, history]))):
                    text = text[:start] + json.dumps(value, indent=2) + text[end:]
                text_item["text"] = text
                result.append(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                result.append(line)
            assert all(value not in get_user_text(sample)["text"] for value in INITIAL.values())
            assert original["messages"][1:] == sample["messages"][1:]
            old_user = original["messages"][0]["content"]
            new_user = sample["messages"][0]["content"]
            assert [c for c in old_user if c.get("type") != "text"] == [c for c in new_user if c.get("type") != "text"]
            label = json.loads(original["messages"][1]["content"][0]["text"])["answer"]
            labels_before[label] += 1
            labels_after[json.loads(sample["messages"][1]["content"][0]["text"])["answer"]] += 1
        assert labels_before == labels_after
        split_stats[split] = {"rows": len(lines), "changed_rows": row_changes,
                              "current_transitions_repaired": current_changes,
                              "history_transitions_repaired": history_changes,
                              "label_counts": dict(labels_after), "known_placeholders_remaining": 0}
        products[f"{split}.jsonl"] = "".join(result)
    assert split_stats["train"]["changed_rows"] == 60
    assert split_stats["val"]["changed_rows"] == 4
    assert len(changes) == 192
    output.mkdir(parents=True)
    for filename, contents in products.items():
        (output / filename).write_text(contents)
    (output / "images").symlink_to(os.path.relpath(source / "images", output), target_is_directory=True)
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["annotation_repair"] = {
        "version": "initial_condition_placeholders_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source), "source_sha256": hashes,
        "scope": "Replace the reviewed initial from_state description/precondition/postcondition triplet in current and history entries.",
        "condition_semantics": "Preconditions are requirements of the current action. Postconditions are its expected local effect, not observed success or completion of the entire subgoal.",
        "review_method": "Assistant visual review of all 41 unique before/after image pairs; repeated history references use the same annotation and must match the original action exactly.",
        "splits": split_stats, "unique_transitions_reviewed": len(annotations),
        "changed_fields": len(changes),
        "labels_actions_splits_and_image_paths_preserved": True,
        "images_linked_to_original": True,
        "pending_review": REVIEW_NOTES,
        "limitations": [
            "This repairs confirmed placeholders only; existing non-placeholder conditions were not normalized.",
            "Two original labels or intent interpretations require separate review; no label was changed.",
            "This keeps the original input/output schema and does not create condition-generation training targets.",
            "All evaluation comparisons should identify this annotation version; old model training provenance is unchanged."
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (output / "condition_annotations.json").write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n")
    (output / "repair_changes.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in changes))
    (output / "review_required.json").write_text(json.dumps(REVIEW_NOTES, ensure_ascii=False, indent=2) + "\n")
    assert hashes == {name: sha256(source / name) for name in hashes}
    print(json.dumps({"output": str(output), "splits": split_stats, "field_changes": len(changes)}, indent=2))


if __name__ == "__main__":
    main()
