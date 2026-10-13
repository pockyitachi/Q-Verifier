"""GUI-Reflection adapter with a lazy worker in its isolated InternVL environment.

The default legacy shared prompt and 512px screenshots match the old Q-Verifier.
The opt-in v9_geometry profile uses the frozen v9 prompt and 1024px screenshots.
``native`` changes only the text to the authors' action-purpose prompt; its
purpose is the actor's extracted subgoal, not a benchmark-provided annotation.
Neither construction nor importing this module loads a model or touches CUDA.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from PIL import Image

from mobile_world.reflection.base import BaseReflector, ReflectionResult
from mobile_world.reflection.protocol import (
    parse_response,
    protocol_configuration,
    reflection_prompt,
    validate_protocol,
)
from mobile_world.runtime.utils.models import JSONAction

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT / "artifacts/models/gui_reflection/GUI_Reflection_8b_pretrain"
DEFAULT_PYTHON = ROOT / "artifacts/envs/gui-reflection/bin/python"
DEFAULT_OFFICIAL_CODE = ROOT / "artifacts/vendor/GUI_Reflection"
# Physical GPU 0 on this server, independent of CUDA's logical device numbering.
DEFAULT_GPU_UUID = "GPU-991ac45f-e9e9-1c25-590c-fb49ca752965"

# From internvl_chat/eval/eval_tasks/reflection_action_verification_infer.py.
NATIVE_PROMPT = (
    "<image>\n<image>\nGiven an action purpose and two screenshots (the first screenshot "
    "corresponds to the step before a certain action while the second one is the outcome "
    "screenshot after the execution of the action). You need to judge whether the action "
    "purpose has been satisfied by the action executed between these screenshots based on "
    "the screenshots content. Directly answer Yes or No. \n"
    "\n## Input\naction purpose: {action_purpose}\n\n## Output:\n"
)


def parse_gui_verdict(raw: str) -> tuple[bool, str]:
    """Accept an explicit verdict only; a worker/format failure is not a No."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        answer = str(obj.get("answer", "")).strip().lower()
        if answer in ("yes", "no"):
            reason = obj.get("reason")
            return answer == "yes", reason.strip() if isinstance(reason, str) and reason.strip() else text
    match = re.match(r"^(?:answer\s*:\s*)?[\s*\"']*(yes|no)\b", text, re.I)
    if match:
        # A bare No stays a bare No. Do not invent a model explanation.
        return match.group(1).lower() == "yes", text
    raise ValueError(f"GUI-Reflection returned no explicit Yes/No verdict: {raw[:300]!r}")


def _encode_image(image: Image.Image, max_side: int) -> str:
    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class GUIReflectionReflector(BaseReflector):
    name = "gui_reflection"

    def __init__(
        self,
        model_path: str | None = None,
        python_path: str | None = None,
        official_code: str | None = None,
        gpu_uuid: str | None = None,
        prompt_mode: str = "shared",
        image_max_side: int = 512,
        max_new_tokens: int = 256,
        memory_cap_gib: float = 32,
        startup_timeout: float = 300,
        request_timeout: float = 180,
        protocol: str = "legacy_shared",
    ):
        self.protocol = validate_protocol(protocol)
        self.uses_history = protocol == "v9_geometry"
        self.feedback_format = "evidence" if self.uses_history else "legacy"
        if self.uses_history and prompt_mode != "shared":
            raise ValueError("v9_geometry requires shared prompt mode; native is a separate baseline")
        if self.uses_history and not gpu_uuid:
            raise ValueError("v9_geometry requires an explicit physical GPU UUID for a local worker")
        if self.uses_history:
            image_max_side, max_new_tokens = 1024, 1024
        if prompt_mode not in ("shared", "native"):
            raise ValueError("GUI-Reflection prompt must be 'shared' or 'native'")
        if image_max_side < 0 or min(max_new_tokens, memory_cap_gib, startup_timeout, request_timeout) <= 0:
            raise ValueError("Invalid GUI-Reflection image, generation, memory or timeout setting")
        self.model_path = str(Path(model_path or DEFAULT_MODEL_PATH).expanduser().absolute())
        # Do not resolve the venv symlink: that would select the base interpreter.
        self.python_path = str(Path(python_path or DEFAULT_PYTHON).expanduser().absolute())
        self.official_code = str(Path(official_code or DEFAULT_OFFICIAL_CODE).expanduser().absolute())
        self.gpu_uuid = gpu_uuid or DEFAULT_GPU_UUID
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", self.gpu_uuid):
            raise ValueError("Select one physical GPU by UUID, not a device list or logical index")
        self.prompt_mode = prompt_mode
        self.image_max_side = image_max_side
        self.max_new_tokens = max_new_tokens
        self.memory_cap_gib = memory_cap_gib
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._lock = threading.Lock()
        self._process = None
        self._reader = None
        self._messages = Queue()
        self._request_id = 0
        self._failure = None
        self._closed = False

    @property
    def configuration(self) -> dict:
        return {
            **protocol_configuration(self.protocol),
            "model_path": self.model_path,
            "python_path": self.python_path,
            "official_code": self.official_code,
            "gpu_uuid": self.gpu_uuid,
            "prompt_mode": self.prompt_mode,
            "image_max_side": self.image_max_side,
            "max_new_tokens": self.max_new_tokens,
            "max_tiles": 6,
            "memory_cap_gib": self.memory_cap_gib,
        }

    def _worker_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # Scope all dependency/GPU settings to our child, leaving the actor intact.
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTORCH_CUDA_ALLOC_CONF"):
            env.pop(key, None)
        env.update({
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid,
            "HF_HOME": str(ROOT / "artifacts/cache/gui-reflection-hf"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        })
        return env

    @staticmethod
    def _read_messages(stream, messages: Queue) -> None:
        try:
            for line in stream:
                messages.put(line)
        finally:
            messages.put(None)

    def _receive(self, timeout: float) -> dict:
        try:
            line = self._messages.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError(f"{self.name} worker timed out after {timeout}s") from exc
        if line is None:
            raise RuntimeError(f"{self.name} worker exited; see its stderr above")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise RuntimeError(f"Invalid {self.name} worker response")
        if message.get("type") == "error":
            raise RuntimeError(f"{self.name} worker error: {message.get('error')}")
        return message

    def _worker_command(self) -> list[str]:
        for path in (self.python_path, Path(self.model_path) / "config.json",
                     Path(self.official_code) / "internvl_chat/eval/infer.py"):
            if not Path(path).is_file():
                raise FileNotFoundError(f"GUI-Reflection dependency missing: {path}")
        return [self.python_path, "-u", str(Path(__file__).with_name("gui_reflection_worker.py")),
                "--model-path", self.model_path, "--official-code", self.official_code,
                "--memory-cap-gib", str(self.memory_cap_gib),
                "--max-new-tokens", str(self.max_new_tokens)]

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.name} reflector is closed")
        if self._failure:
            raise RuntimeError(self._failure)
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            self._worker_command(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1, env=self._worker_environment(),
        )
        atexit.register(self.close)
        self._reader = threading.Thread(
            target=self._read_messages, args=(self._process.stdout, self._messages), daemon=True,
        )
        self._reader.start()
        if self._receive(self.startup_timeout).get("type") != "ready":
            raise RuntimeError(f"{self.name} worker did not become ready")

    def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            # Only the Popen child owned by this adapter is ever signalled.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if self._reader is not None:
                self._reader.join(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        atexit.unregister(self.close)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop_worker()

    def verify(self, before: Image.Image, after: Image.Image, action: JSONAction,
               subgoal: str, global_task: str, *, history=()) -> ReflectionResult:
        prompt = (
            reflection_prompt(self.protocol, global_task, subgoal, action, before, after, history)
            if self.prompt_mode == "shared"
            else NATIVE_PROMPT.format(action_purpose=subgoal or "(no explicit subgoal)")
        )
        with self._lock:
            try:
                self._ensure_started()
                started = time.monotonic()
                self._request_id += 1
                request = {
                    "id": self._request_id, "prompt": prompt,
                    "images": [_encode_image(before, self.image_max_side),
                               _encode_image(after, self.image_max_side)],
                }
                self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                message = self._receive(self.request_timeout)
                if (message.get("type") != "result" or message.get("id") != self._request_id
                        or not isinstance(message.get("raw_output"), str)):
                    raise RuntimeError(f"Unexpected {self.name} worker response")
            except BaseException as exc:
                self._failure = f"{self.name} worker unavailable: {exc}"
                self._stop_worker()
                raise
            raw = message["raw_output"]
            verdict, reasoning = (parse_response(self.protocol, raw) if self.uses_history
                                  else parse_gui_verdict(raw))
            usage = message.get("token_usage") or {}
            return ReflectionResult(
                verdict=verdict, reasoning=reasoning, raw_output=raw,
                latency_ms=(time.monotonic() - started) * 1000,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                token_usage_source=usage.get("source"),
            )
