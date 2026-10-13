"""OpenAI-compatible reflector with explicit legacy and v9 input profiles."""

from __future__ import annotations

import base64
import io
import os
import time

from loguru import logger
from openai import OpenAI
from PIL import Image

from mobile_world.reflection.base import (
    BaseReflector,
    ReflectionResult,
)
from mobile_world.reflection.protocol import (
    parse_response,
    protocol_configuration,
    reflection_prompt,
    validate_protocol,
)
from mobile_world.runtime.utils.models import JSONAction


def _pil_to_b64_data_url(img: Image.Image, max_size: int = 512) -> str:
    img = img.convert("RGB")
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize(
            (int(img.size[0] * ratio), int(img.size[1] * ratio)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class GPTReflector(BaseReflector):
    name = "gpt"

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        min_latency_ms: float = 10000.0,
        protocol: str = "legacy_shared",
    ):
        self.protocol = validate_protocol(protocol)
        self.uses_history = protocol == "v9_geometry"
        self.feedback_format = "evidence" if self.uses_history else "legacy"
        self.model = model
        self.temperature = temperature
        self.max_tokens = 1024 if self.uses_history else max_tokens
        self.image_max_side = 1024 if self.uses_history else 512
        # When the API responds faster than min_latency_ms, sleep the
        # difference. Used to slow GPT verifier pacing back to Q-Verifier
        # cadence so the AVD/KVM gets the same breathing room between
        # screenshots, isolating "verifier model" from "verifier speed".
        self.min_latency_ms = min_latency_ms
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "empty",
            timeout=60.0,
            max_retries=0,
        )

    @property
    def configuration(self):
        return {**protocol_configuration(self.protocol), "model": self.model,
                "image_max_side": self.image_max_side, "max_new_tokens": self.max_tokens,
                "min_latency_ms": self.min_latency_ms, "decoding": "provider_default"}

    def verify(
        self,
        before: Image.Image,
        after: Image.Image,
        action: JSONAction,
        subgoal: str,
        global_task: str,
        *, history=(),
    ) -> ReflectionResult:
        prompt_text = reflection_prompt(self.protocol, global_task, subgoal, action, before, after, history)
        before_url = _pil_to_b64_data_url(before, self.image_max_side)
        after_url = _pil_to_b64_data_url(after, self.image_max_side)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": before_url}},
                    {"type": "image_url", "image_url": {"url": after_url}},
                ],
            }
        ]

        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_tokens,
            )
            raw_output = resp.choices[0].message.content or ""
        except Exception as e:
            # Fail-open: return Yes so a transient API error doesn't trigger
            # early-termination or retry. The experimental comparison treats this
            # as a non-event; check logs for the warning if stats look off.
            logger.warning(f"GPTReflector API call failed: {e}")
            return ReflectionResult(
                verdict=None if self.uses_history else True,
                reasoning=(f"[verifier API error] {type(e).__name__}" if self.uses_history
                           else f"[verifier API error, defaulting to Yes] {e}"),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                raw_output="",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if self.min_latency_ms > 0 and latency_ms < self.min_latency_ms:
            time.sleep((self.min_latency_ms - latency_ms) / 1000.0)
            latency_ms = (time.perf_counter() - t0) * 1000.0

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        ctd = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(ctd, "reasoning_tokens", 0) or 0
        ptd = getattr(usage, "prompt_tokens_details", None)
        cached_prompt_tokens = getattr(ptd, "cached_tokens", 0) or 0

        verdict, reasoning = parse_response(self.protocol, raw_output)
        return ReflectionResult(
            verdict=verdict,
            reasoning=reasoning,
            latency_ms=latency_ms,
            raw_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            token_usage_source=(
                "api_usage" if getattr(usage, "prompt_tokens", None) is not None
                and getattr(usage, "completion_tokens", None) is not None else None
            ),
        )
