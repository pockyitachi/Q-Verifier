import base64
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from eval_qreflector_luna import build_request, response_record, score_response


class LunaEvaluationTests(unittest.TestCase):
    def test_request_preserves_image_order_and_excludes_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in [("before", "red"), ("after", "blue")]:
                Image.new("RGB", (100, 200), color).save(root / f"{name}.png")
            user = {"role": "user", "content": [
                {"type": "text", "text": "EXACT_NEW_PROMPT"},
                {"type": "image", "image": "before.png"},
                {"type": "image", "image": "after.png"}]}
            with patch("eval_qreflector_luna.DATA", root):
                payload, hashes = build_request(user, {"model": "gpt-6-luna", "image_max_side": 50,
                    "reasoning_effort": "medium", "max_output_tokens": 4096})
            self.assertEqual(payload["input"][0]["content"][0]["text"], "EXACT_NEW_PROMPT")
            self.assertEqual(set(payload), {"model", "store", "reasoning", "max_output_tokens", "input"})
            for item, pixel in zip(payload["input"][0]["content"][1:], [(255, 0, 0), (0, 0, 255)]):
                image = Image.open(BytesIO(base64.b64decode(item["image_url"].split(",", 1)[1])))
                self.assertEqual(image.size, (25, 50))
                self.assertEqual(image.getpixel((0, 0)), pixel)
            self.assertNotEqual(*hashes)

    def test_only_final_answer_is_scored(self):
        text = json.dumps({"precondition": "Yes appears here.", "postcondition": "Expected local change.",
                           "reason": "Yes in reasoning must not override answer.", "answer": "No"})
        record = score_response({"full_response": text, "response_status": "completed"}, "no")
        self.assertTrue(record["correct"])
        self.assertTrue(record["valid_four_field_json"])

    def test_incomplete_refused_and_missing_answer_are_invalid(self):
        for record in [
            {"full_response": '{"answer":"Yes"}', "response_status": "incomplete"},
            {"full_response": '{"answer":"Yes"}', "response_status": "completed", "refusals": ["refusal"]},
            {"full_response": '{"reason":"Yes"}', "response_status": "completed"},
        ]:
            scored = score_response(record, "yes")
            self.assertIsNone(scored["predicted"])
            self.assertFalse(scored["correct"])

    def test_reasoning_is_not_mistaken_for_output_and_usage_is_preserved(self):
        response = {"status": "completed", "model": "gpt-6-luna", "usage": {"output_tokens": 99},
                    "output": [{"type": "reasoning", "content": [{"type": "output_text", "text": "Yes"}]},
                               {"type": "message", "content": [{"type": "output_text", "text": '{"answer":"No"}'}]}]}
        record = response_record(response)
        self.assertEqual(record["full_response"], '{"answer":"No"}')
        self.assertEqual(record["usage"]["output_tokens"], 99)


if __name__ == "__main__":
    unittest.main()
