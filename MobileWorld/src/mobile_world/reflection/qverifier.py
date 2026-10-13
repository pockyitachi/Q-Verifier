"""Local Q-Verifier reflector: load LoRA adapter via unsloth, run greedy decode."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from loguru import logger
from PIL import Image

from mobile_world.reflection.base import (
    BaseReflector,
    ReflectionResult,
    build_prompt,
    parse_verdict,
)
from mobile_world.runtime.utils.models import JSONAction

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / '../GRPO/qwen3_vl_distilled_dpo_grpo_8b/final'
)


def _resize_for_verifier(img: Image.Image, max_size: int = 512) -> Image.Image:
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        return img.resize(
            (int(img.size[0] * ratio), int(img.size[1] * ratio)),
            Image.LANCZOS,
        )
    return img


class QVerifierReflector(BaseReflector):
    name = "qverifier"

    def __init__(
        self,
        model_path: str | None = None,
        max_new_tokens: int = 256,
        max_seq_length: int = 4096,
        load_in_4bit: bool = True,
    ):
        self.model_path = str(
            model_path or os.getenv("QVERIFIER_MODEL_PATH") or DEFAULT_MODEL_PATH
        )
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit

        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            logger.info(f"Loading Q-Verifier from {self.model_path}")
            from unsloth import FastModel

            model, processor = FastModel.from_pretrained(
                self.model_path,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
            )
            model.for_inference()
            self._model = model
            self._processor = processor
            logger.info("Q-Verifier loaded")

    def verify(
        self,
        before: Image.Image,
        after: Image.Image,
        action: JSONAction,
        subgoal: str,
        global_task: str,
    ) -> ReflectionResult:
        self._ensure_loaded()

        prompt_text = build_prompt(global_task=global_task, subgoal=subgoal, action=action)
        before_small = _resize_for_verifier(before.convert("RGB"))
        after_small = _resize_for_verifier(after.convert("RGB"))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": before_small},
                    {"type": "image", "image": after_small},
                ],
            }
        ]

        t0 = time.perf_counter()
        with self._inference_lock:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {
                k: v.to(self._model.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[0][input_len:]
            raw_output = self._processor.decode(gen_ids, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        verdict, reasoning = parse_verdict(raw_output)
        return ReflectionResult(
            verdict=verdict,
            reasoning=reasoning,
            latency_ms=latency_ms,
            raw_output=raw_output,
            prompt_tokens=input_len,
            completion_tokens=len(gen_ids),
            token_usage_source="model_token_ids",
        )
