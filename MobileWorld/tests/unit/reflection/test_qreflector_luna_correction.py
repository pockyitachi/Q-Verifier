"""CPU tests for teacher/student isolation, label preservation and API caching."""
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import qreflector_luna_correction as correction  # noqa: E402
from PIL import Image  # noqa: E402


def fixture(root):
    Image.new("RGB", (12, 20), "red").save(root / "before.png")
    Image.new("RGB", (12, 20), "blue").save(root / "after.png")
    target = {"precondition": "REFERENCE_PRE_ONLY", "postcondition": "REFERENCE_POST_ONLY",
              "reason": "REFERENCE_REASON_ONLY", "answer": "Yes"}
    row = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Task: find a note. Action: tap search."},
                                     {"type": "image", "image": "before.png"},
                                     {"type": "image", "image": "after.png"}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]}]}
    cfg = {"data": str(root), "image_max_side": 512, "teacher_model": "gpt-6-luna",
           "teacher_reasoning": "medium", "teacher_max_output_tokens": 2048, "max_api_calls": 20}
    return row, cfg


def valid_correction():
    return {"training_usable": True, "review_note": "The reference agrees with the images.",
            "corrected_response": {"precondition": "The search icon is visible.",
                                   "postcondition": "The note search field opens.",
                                   "reason": "The search field opened, enabling the note lookup.",
                                   "answer": "Yes"}}


def api_reply(status="completed"):
    return {"status": status, "model": "gpt-6-luna", "id": "resp_fixture",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(valid_correction())}]}]}


class IsolationTests(unittest.TestCase):
    def test_teacher_sees_draft_reference_and_images_but_student_input_stays_exact(self):
        import base64

        with tempfile.TemporaryDirectory() as d:
            row, cfg = fixture(Path(d))
            original = copy.deepcopy(row)
            student = correction.student_messages(row)
            request = correction.teacher_request(row, "STUDENT_DRAFT_ONLY", cfg)
            self.assertEqual(row, original)
            self.assertEqual(student, [original["messages"][0]])
            for secret in ["REFERENCE_PRE_ONLY", "REFERENCE_POST_ONLY", "REFERENCE_REASON_ONLY", "STUDENT_DRAFT_ONLY"]:
                self.assertNotIn(secret, json.dumps(student))
                self.assertIn(secret, json.dumps(request))
            pictures = [c for c in request["input"][0]["content"] if c["type"] == "input_image"]
            colors = [Image.open(io.BytesIO(base64.b64decode(c["image_url"].split(",", 1)[1]))).getpixel((0, 0))
                      for c in pictures]
            self.assertEqual(colors, [(255, 0, 0), (0, 0, 255)])
            self.assertFalse(request["store"])

    def test_teacher_label_disagreement_and_flag_are_not_relabelled(self):
        answer = valid_correction()
        self.assertEqual(correction.validate_correction(answer, "Yes"), (True, "accepted"))
        original = copy.deepcopy(answer)
        self.assertEqual(correction.validate_correction(answer, "No")[1], "teacher_label_disagreement")
        self.assertEqual(answer, original)
        answer["training_usable"] = False
        self.assertEqual(correction.validate_correction(answer, "Yes")[1], "teacher_flagged_annotation")

    def test_invalid_or_control_bearing_targets_rejected(self):
        for malformed in [None, [], {}, {"training_usable": "true"}]:
            self.assertFalse(correction.validate_correction(malformed, "Yes")[0])
        for value in ["", "中文", "<|im_end|>", correction.CORRECTION_MARKER]:
            answer = valid_correction()
            answer["corrected_response"]["reason"] = value
            self.assertFalse(correction.validate_correction(answer, "Yes")[0])


class ApiTests(unittest.TestCase):
    def test_cache_depends_on_exact_draft_and_never_serializes_credentials(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"OPENAI_API_KEY": "secret_fixture"}):
            root = Path(d)
            row, cfg = fixture(root)
            client = correction.LunaClient(root, cfg, root / "missing.env")
            with patch.object(correction, "urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps(api_reply()).encode())) as call:
                first, hit = client.correct(row, "draft A")
                self.assertFalse(hit)
                same, hit = client.correct(row, "draft A")
                self.assertTrue(hit)
                other, hit = client.correct(row, "draft B")
                self.assertFalse(hit)
                self.assertEqual(call.call_count, 2)
            self.assertEqual(first, same)
            self.assertNotEqual(first["request_sha256"], other["request_sha256"])
            self.assertEqual(client.input_tokens, 200)
            self.assertEqual(client.output_tokens, 100)
            for path in root.rglob("*.json*"):
                self.assertNotIn("secret_fixture", path.read_text())
                self.assertNotIn("data:image", path.read_text())
            restored = correction.LunaClient(root, cfg, root / "missing.env")
            self.assertEqual(restored.calls, 2)
            self.assertEqual(restored.output_tokens, 100)

    def test_incomplete_response_is_rejected_even_if_its_json_is_valid(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"OPENAI_API_KEY": "secret_fixture"}):
            root = Path(d)
            row, cfg = fixture(root)
            client = correction.LunaClient(root, cfg, root / "missing.env")
            with patch.object(correction, "urlopen", return_value=io.BytesIO(json.dumps(api_reply("incomplete")).encode())):
                answer, _ = client.correct(row, "draft")
            self.assertFalse(answer["accepted"])
            self.assertEqual(answer["decision"], "teacher_response_incomplete")

    def test_call_cap_stops_before_sending_another_request(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"OPENAI_API_KEY": "secret_fixture"}):
            root = Path(d)
            row, cfg = fixture(root)
            cfg["max_api_calls"] = 0
            client = correction.LunaClient(root, cfg, root / "missing.env")
            with patch.object(correction, "urlopen") as call, self.assertRaisesRegex(RuntimeError, "call limit"):
                client.correct(row, "draft")
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
