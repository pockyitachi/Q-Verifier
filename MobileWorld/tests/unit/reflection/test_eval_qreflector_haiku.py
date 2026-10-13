import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from eval_qreflector_haiku import build_request, read_key, response_record
from eval_qreflector_luna import build_request as luna_request
from eval_qreflector_luna import score_response


class HaikuEvaluationTests(unittest.TestCase):
    def test_identical_prompt_and_resized_images_to_luna(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in (("before", "red"), ("after", "blue")):
                Image.new("RGB", (100, 200), color).save(root / f"{name}.png")
            user = {"role": "user", "content": [{"type": "text", "text": "EXACT_PROMPT"},
                {"type": "image", "image": "before.png"}, {"type": "image", "image": "after.png"}]}
            config = {"model": "test-model", "max_output_tokens": 4096, "thinking_budget_tokens": 2048,
                      "reasoning_effort": "medium", "image_max_side": 50}
            with patch("eval_qreflector_haiku.DATA", root), patch("eval_qreflector_luna.DATA", root):
                request, hashes = build_request(user, config)
                other, other_hashes = luna_request(user, config)
            self.assertEqual(hashes, other_hashes)
            self.assertEqual(set(request), {"model", "max_tokens", "thinking", "messages"})
            self.assertEqual(request["messages"][0]["content"][0]["text"], "EXACT_PROMPT")
            for a, b in zip(request["messages"][0]["content"][1:], other["input"][0]["content"][1:]):
                self.assertEqual(a["source"]["data"], b["image_url"].split(",", 1)[1])

    def test_thinking_text_never_becomes_the_verdict(self):
        record = response_record({"stop_reason": "end_turn", "content": [
            {"type": "thinking", "thinking": "Yes"},
            {"type": "text", "text": json.dumps({"answer": "No"})}],
            "usage": {"input_tokens": 7, "output_tokens": 9}})
        self.assertTrue(score_response(record, "no")["correct"])
        self.assertTrue(record["thinking_block_present"])
        self.assertIsNone(record["thinking_tokens"])
        self.assertEqual(record["usage"]["total_tokens"], 16)

    def test_nonfinal_stops_are_invalid_even_with_visible_answer(self):
        for stop in ("max_tokens", "refusal", "tool_use", "pause_turn"):
            response = {"stop_reason": stop, "content": [{"type": "text", "text": '{"answer":"Yes"}'}]}
            self.assertIsNone(score_response(response_record(response), "yes")["predicted"])

    def test_key_reader_never_falls_back_to_openai_or_executes_env_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env = Path(directory) / ".env"
            env.write_text("OPENAI_API_KEY=not-an-anthropic-key\n")
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY is not configured"):
                read_key(env)
            env.write_text("export ANTHROPIC_API_KEY = 'test-only-literal'\n")
            self.assertEqual(read_key(env), "test-only-literal")


if __name__ == "__main__":
    unittest.main()
