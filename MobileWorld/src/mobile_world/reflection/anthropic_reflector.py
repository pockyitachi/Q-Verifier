"""Anthropic Messages adapter for the shared MobileWorld reflector protocol."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mobile_world.reflection.base import BaseReflector, ReflectionResult
from mobile_world.reflection.gpt_reflector import _pil_to_b64_data_url
from mobile_world.reflection.protocol import (
    parse_response,
    protocol_configuration,
    reflection_prompt,
    validate_protocol,
)


def read_api_key(env_file: str | Path | None) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and env_file is not None:
        for line in Path(env_file).read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "ANTHROPIC_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key or key.lower() in {"changeme", "your-api-key", "your_anthropic_api_key"}:
        raise RuntimeError("Anthropic key is not configured")
    return key


def normalize_usage(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    required = (raw.get("input_tokens"), raw.get("output_tokens"))
    optional = (raw.get("cache_creation_input_tokens", 0) or 0,
                raw.get("cache_read_input_tokens", 0) or 0)
    if not all(type(x) is int and x >= 0 for x in (*required, *optional)):
        return None
    return {"prompt_tokens": required[0] + sum(optional), "completion_tokens": required[1],
            "cached_tokens": optional[1], "reasoning_tokens": 0, "source": "anthropic_api_usage"}


class AnthropicReflector(BaseReflector):
    name = "anthropic"

    def __init__(self, env_file, ledger_path, model="claude-haiku-4-5-20251001",
                 max_tokens=256, temperature=0.0, min_latency_ms=10000.0,
                 timeout=120.0, max_calls=8000, protocol="legacy_shared"):
        self.protocol = validate_protocol(protocol)
        self.uses_history = protocol == "v9_geometry"
        self.feedback_format = "evidence" if self.uses_history else "legacy"
        self.key = read_api_key(env_file)
        self.model = model
        self.max_tokens = 1024 if self.uses_history else max_tokens
        self.image_max_side = 1024 if self.uses_history else 512
        self.temperature = temperature
        self.min_latency_ms = min_latency_ms
        self.timeout = timeout
        self.max_calls = max_calls
        self.endpoint = "https://api.anthropic.com/v1/messages"
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.calls = 0
        if self.ledger_path.exists():
            self.calls = sum(json.loads(line).get("event") == "request_started"
                             for line in self.ledger_path.read_text().splitlines())

    def event(self, **record):
        with self.lock, self.ledger_path.open("a") as handle:
            handle.write(json.dumps({"time_unix": time.time(), **record}) + "\n")

    @property
    def configuration(self):
        return {**protocol_configuration(self.protocol), "model": self.model,
                "image_max_side": self.image_max_side, "max_new_tokens": self.max_tokens,
                "temperature": self.temperature, "min_latency_ms": self.min_latency_ms}

    def build_request(self, before, after, action, subgoal, global_task, *, history=()):
        content = [{"type": "text", "text": reflection_prompt(
            self.protocol, global_task, subgoal, action, before, after, history)}]
        for image in (before, after):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": _pil_to_b64_data_url(image, self.image_max_side).split(",", 1)[1]}})
        return {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "messages": [{"role": "user", "content": content}]}

    def verify(self, before, after, action, subgoal, global_task, *, history=()):
        payload = self.build_request(before, after, action, subgoal, global_task, history=history)
        encoded = json.dumps(payload).encode()
        call_id = uuid.uuid4().hex
        with self.lock:
            if self.calls >= self.max_calls:
                raise RuntimeError("Anthropic run call limit reached")
            self.calls += 1
        self.event(event="request_started", call_id=call_id, model=self.model,
                   request_sha256=hashlib.sha256(encoded).hexdigest())
        started = time.perf_counter()
        try:
            request = Request(self.endpoint, data=encoded, headers={
                "x-api-key": self.key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"})
            # One network attempt per verify; no hidden SDK retries.
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            raw_usage = result.get("usage")
            usage = normalize_usage(raw_usage)
            text = "".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")
            self.event(event="response_received", call_id=call_id, response_id=result.get("id"),
                       response_model=result.get("model"), stop_reason=result.get("stop_reason"),
                       provider_usage=raw_usage, normalized_usage=usage, raw_output=text)
            if result.get("type") != "message" or result.get("model") != self.model:
                raise ValueError("Unexpected Anthropic response type/model")
        except Exception as exc:
            error = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
            self.event(event="request_error", call_id=call_id, error=error)
            # Match the archived GPT adapter's fail-open policy; usage stays unknown.
            return ReflectionResult(None if self.uses_history else True,
                                    f"[verifier API error] {error}" if self.uses_history else f"[verifier API error, defaulting to Yes] {error}",
                                    (time.perf_counter() - started) * 1000)
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed < self.min_latency_ms:
            time.sleep((self.min_latency_ms - elapsed) / 1000)
        verdict, reason = parse_response(self.protocol, text)
        return ReflectionResult(verdict, reason, (time.perf_counter() - started) * 1000,
                                raw_output=text,
                                prompt_tokens=usage["prompt_tokens"] if usage else 0,
                                completion_tokens=usage["completion_tokens"] if usage else 0,
                                cached_prompt_tokens=usage["cached_tokens"] if usage else 0,
                                token_usage_source=usage["source"] if usage else None)
