"""CPU regressions for concrete, English, before-state prerequisites."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import refine_qreflector_preconditions as repair


def action(kind, description):
    return {"type": kind, "description": description, "arguments": {}}


class SpecificPreconditionTests(unittest.TestCase):
    def test_action_text_cannot_be_treated_as_observed_before_state(self):
        text = 'I will tap the "Save" button to save the note.'
        with self.assertRaises(ValueError):
            repair.specific_precondition(text, action("Tap", text), "explicit_action_effect")

    def test_conditions_are_tied_to_the_specific_page_and_control(self):
        before = "The Android share sheet is open over a CBC News article. Gmail appears among the share destinations."
        pre = repair.specific_precondition(before, action("Tap", "Tap the Gmail icon to share the article."), "explicit_action_effect")
        self.assertIn("CBC News", pre)
        self.assertIn("share sheet", pre)
        self.assertIn("Gmail", pre)
        self.assertNotIn("intended tap target", pre)

    def test_backspace_tracks_description_instead_of_an_unrelated_title(self):
        before = ('The New Recipe form is open in Broccoli. The "Title" field contains "Salad". '
                  'The description text contains an extra "w". The keyboard is open.')
        pre = repair.specific_precondition(before, action("Tap", "Tap backspace to remove the extra w from the description text."), "single_backspace")
        self.assertIn('"Description" field of the recipe form must retain keyboard focus', pre)
        self.assertIn("caret", pre)
        self.assertNotIn('"Title" field must retain', pre)

    def test_type_requires_existing_focus_but_shortcut_can_acquire_it(self):
        before = 'The Keep Notes editor is open. The "Title" field is empty.'
        a = 'Type "Shopping" into the "Title" field.'
        direct = repair.specific_precondition(before, action("Type", a), "intended_text_entry")
        shortcut = repair.specific_precondition(before, action("Tap_and_Type", a), "intended_text_entry")
        self.assertIn("must already have text focus", direct)
        self.assertIn("must accept focus", shortcut)

    def test_sentence_splitting_preserves_filenames_and_drops_future_requirements(self):
        before = ('The Markor editor displays "draft.v2.txt." The keyboard is open. '
                  'The user must save the completed note and send it later.')
        facts = repair.before_facts(before, action("Type", "Type into the note body."), "intended_text_entry")
        self.assertIn("draft.v2.txt", facts)
        self.assertIn("keyboard is open", facts)
        self.assertNotIn("send it later", facts)

    def test_missing_input_identity_does_not_get_a_new_generic_placeholder(self):
        with self.assertRaises(ValueError):
            repair.specific_precondition("An unspecified interface is displayed.", action("Type", "Type the text."), "intended_text_entry")

    def test_known_non_english_ui_label_is_explained_in_english(self):
        before = 'Google Scholar shows the Segment Anything search result and its citation count.'
        pre = repair.specific_precondition(before, action("Tap", 'Tap the citation link ("被引用次数:16348") to open citing papers.'), "explicit_action_effect")
        self.assertIn("Cited by:16348", pre)
        self.assertNotIn("被引用次数", pre)

    def test_departure_selector_is_distinguished_from_destination(self):
        before = 'Booking.com shows Montreal as departure and New York as destination.'
        field = repair.field_for(before, action("Tap", "Tap the departure location field to enable editing."))
        self.assertEqual(field, "departure airport/city selector")


if __name__ == "__main__":
    unittest.main()
