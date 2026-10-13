"""CPU-only regressions for annotation isolation and structured verdict parsing."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_module("condition_converter", "scripts/convert_qreflector_condition_generation.py")
parser = load_module("verifier_output", '../SFT/verifier_output.py')
repair = load_module("condition_repair", "scripts/repair_qreflector_condition_semantics.py")


def fixture_sample():
    transition = {
        "from_state": {"description": "before annotation", "precondition": "source pre",
                       "postcondition": "previous post"},
        "action": {"name": "Tap", "x": 12, "y": 34, "description": "Open the app folder"},
        "to_state": {"description": "after annotation", "precondition": "next pre",
                     "postcondition": "source post"},
    }
    history = [dict(copy.deepcopy(transition), step=1)]
    history[0]["to_state"]["valid"] = False
    text = ("=== Global Task ===\nFind a product\n\n=== Current Subgoal ===\nOpen the app\n\n"
            f"=== Current Transition ===\n{json.dumps(transition)}\n\n"
            f"=== Previous Steps (latest 1) ===\n{json.dumps(history)}\n")
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": text},
                                     {"type": "image", "image": "images/task_before.jpg"},
                                     {"type": "image", "image": "images/task_after.jpg"}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(
            {"answer": "Yes", "reason": "The folder opened."})}]},
    ]}


class ConversionTests(unittest.TestCase):
    def test_reference_annotations_cannot_change_model_input(self):
        original = fixture_sample()
        poisoned = copy.deepcopy(original)
        text = poisoned["messages"][0]["content"][0]["text"]
        for value in ["before annotation", "after annotation", "source pre", "previous post", "next pre", "source post"]:
            text = text.replace(value, "HIDDEN_ANNOTATION_" + value)
        text = text.replace('"valid": false', '"valid": true')
        poisoned["messages"][0]["content"][0]["text"] = text
        poisoned["messages"][1]["content"][0]["text"] = json.dumps({"answer": "No", "reason": "HIDDEN_REASON"})
        first, ref = converter.convert_sample(original, "test", 1, {})
        second, _ = converter.convert_sample(poisoned, "test", 1, {})
        self.assertEqual(first["messages"][0], second["messages"][0])
        self.assertNotEqual(first["messages"][1], second["messages"][1])
        self.assertEqual(ref["original_sample"], original)

    def test_flat_and_nested_actions_retain_the_same_arguments(self):
        flat = {"name": "Tap", "x": 10, "y": 20, "description": "Tap target"}
        nested = {"type": "Tap", "arguments": {"x": 10, "y": 20}, "description": "Tap target"}
        self.assertEqual(converter.normalize_action(flat), converter.normalize_action(nested))
        unknown = {"name": "Unknown", "raw": {"unparsed": [1, 2]}, "description": "Unknown action"}
        self.assertEqual(converter.normalize_action(unknown)["arguments"], {"raw": {"unparsed": [1, 2]}})


class VerdictTests(unittest.TestCase):
    def test_answer_after_long_conditions_is_read(self):
        response = json.dumps({"precondition": "A prerequisite. " * 60,
                               "postcondition": "A useful intermediate outcome.",
                               "reason": "No full subgoal completion is required.", "answer": "Yes"})
        self.assertEqual(parser.parse_verdict(response), "yes")

    def test_yes_in_conditions_does_not_override_no(self):
        response = json.dumps({"precondition": "The Yes button is available.",
                               "postcondition": "The dialog closes.", "answer": "No"})
        self.assertEqual(parser.parse_verdict(response), "no")

    def test_missing_or_truncated_verdict_is_invalid(self):
        for response in [None, "", '{"precondition": "Yes button"}', '{"answer": "No"',
                         "The action did not work.", '{"answer": true}']:
            with self.subTest(response=response):
                self.assertEqual(parser.parse_verdict(response), "invalid")

    def test_fenced_json_and_legacy_verdicts(self):
        self.assertEqual(parser.parse_verdict('```json\n{"answer":"Yes"}\n```'), "yes")
        self.assertEqual(parser.parse_verdict("Answer: No. The folder did not open."), "no")


class SemanticRepairTests(unittest.TestCase):
    def context(self, kind, description, arguments=None, subgoal="Find the requested information"):
        return {"goal": "Complete the requested task", "subgoal": subgoal, "history": [],
                "action": {"type": kind, "description": description, "arguments": arguments}}

    def test_corrupted_launch_does_not_make_wrong_app_the_expected_result(self):
        context = self.context("Open_App", "Open the Chrome app to search for the event.", {"app_name": "Clock"})
        before = copy.deepcopy(context)
        pre, post, _ = repair.propose(context, corrupted=True)
        self.assertIn("Chrome", pre + post)
        self.assertNotIn("Clock", pre + post)
        self.assertEqual(context, before)

    def test_corrupted_search_uses_intent_and_task_supported_punctuation(self):
        context = self.context("Tap_Type_and_Enter", 'Type "birthday cake bakery near me," and press Enter.',
                               {"text": "incorrect value"}, "Search for 'birthday cake bakery near me'")
        _, post, _ = repair.propose(context, corrupted=True)
        self.assertIn('"birthday cake bakery near me"', post)
        self.assertNotIn("incorrect value", post)
        context["subgoal"] = "Unspecified search"
        self.assertIsNone(repair.propose(context, corrupted=True))

    def test_app_drawer_swipe_accepts_intermediate_progress(self):
        context = self.context("Swipe", "Scroll the app drawer to find Google Play Store.")
        _, post, _ = repair.propose(context)
        self.assertIn("additional icons", post)
        self.assertIn("need not already be located or launched", post)

    def test_single_backspace_does_not_require_full_replacement(self):
        _, post, _ = repair.propose(self.context("Tap", 'Tap the backspace key to replace the whole "Title" field.'))
        self.assertIn("One character", post)
        self.assertIn("need not clear the entire field", post)

    def test_clipboard_button_can_open_panel_before_paste(self):
        _, post, _ = repair.propose(self.context("Tap", 'Tap the clipboard icon to paste the intended "Title" field.'))
        self.assertIn("clipboard panel opens", post)
        self.assertIn("another tap is needed to paste", post)

    def test_quoted_target_does_not_become_a_purpose_clause(self):
        description = 'Tap the "Add to a track" option to open the track-selection dialog.'
        self.assertIn('"Add to a track"', repair.tap_target(description))
        self.assertEqual(repair.explicit_tap_effect(description), "open the track-selection dialog")

    def test_opening_date_editor_does_not_require_later_date_selection(self):
        effect = repair.explicit_tap_effect('Tap the date field to open the date picker and adjust the date to October 2.')
        self.assertEqual(effect, "open the date picker")

    def test_speculative_nested_purpose_is_not_a_promised_effect(self):
        description = 'Tap the icon to check if it provides options to enable multi-select mode.'
        self.assertIsNone(repair.explicit_tap_effect(description))
        self.assertIsNone(repair.propose(self.context("Tap", description)))

    def test_target_position_next_to_does_not_hide_the_actual_purpose(self):
        description = 'Tap the heart icon next to "Hoover Tower" to save it as a favorite.'
        self.assertIn('"Hoover Tower"', repair.tap_target(description))
        self.assertEqual(repair.explicit_tap_effect(description), "save it as a favorite")

    def test_attempt_to_open_still_defines_an_intended_immediate_effect(self):
        description = 'Tap the "Chicken Alfredo" entry to attempt to open the recipe and read its ingredients.'
        effect = repair.explicit_tap_effect(description)
        self.assertEqual(effect, "open the recipe")

    def test_missing_action_is_left_for_review(self):
        self.assertIsNone(repair.propose(self.context("Unknown", "None")))


if __name__ == "__main__":
    unittest.main()
