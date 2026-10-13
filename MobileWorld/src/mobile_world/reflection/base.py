"""
Reflection layer: verify that each agent step makes valid progress toward the task.

A Reflector is called from the runner between env.execute_action() and the next
agent.predict(). It compares BEFORE (pre-action screenshot) and AFTER
(post-action screenshot) and emits a binary verdict plus reasoning.

Shared prompt format and parsing utilities live here so QVerifierReflector and
GPTReflector can use the exact same input/output contract.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from mobile_world.runtime.utils.models import JSONAction

VERIFIABLE_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "click",
        "long_press",
        "double_tap",
        "drag",
        "scroll",
        "input_text",
        "keyboard_enter",
        "navigate_back",
        "navigate_home",
        "open_app",
    }
)


def is_verifiable(action: JSONAction) -> bool:
    return action.action_type in VERIFIABLE_ACTION_TYPES


@dataclass
class ReflectionResult:
    verdict: bool | None  # None is an explicit unavailable/invalid verification.
    reasoning: str
    latency_ms: float
    raw_output: str = ""
    # API usage or actual local model token IDs (including visual input tokens).
    # Zero defaults are retained for compatibility; source=None means UNKNOWN.
    # reasoning_tokens is OpenAI's hidden chain-of-thought count; it is
    # billed at the completion rate but not visible in raw_output.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    token_usage_source: str | None = None

    def token_usage(self) -> dict | None:
        if self.token_usage_source is None:
            return None
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "source": self.token_usage_source,
        }


PROMPT_TEMPLATE = """You are a mobile GUI reflector.
Your job is to determine whether the current state transition is valid — meaning whether it aligns with both the user's global task and the current subgoal.

=== Global Task ===
{global_task}

=== Current Subgoal ===
{subgoal}

=== Current Transition ===
{transition_json}

=== Previous Steps (latest 0) ===
[]

=== Image Reference ===
<image_0>: Image BEFORE the action
<image_1>: Image AFTER the action

=== Question ===
Is this transition valid and aligned with the current subgoal?
Answer with "Yes" or "No" and provide a brief reason explaining why."""


def _action_to_training_format(action: JSONAction) -> dict:
    d = action.model_dump(exclude_none=True)
    name = d.pop("action_type", "unknown")
    d["name"] = name
    return d


def build_transition_json(action: JSONAction) -> str:
    transition = {
        "from_state": {"description": "Screen before the action"},
        "action": _action_to_training_format(action),
        "to_state": {"description": "Current screen state"},
    }
    return json.dumps(transition, ensure_ascii=False, indent=2)


def build_prompt(global_task: str, subgoal: str, action: JSONAction) -> str:
    return PROMPT_TEMPLATE.format(
        global_task=global_task,
        subgoal=subgoal or "(no explicit subgoal)",
        transition_json=build_transition_json(action),
    )


_MAI_UI_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
_GENERAL_E2E_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?:\n\s*Action:|$)", re.DOTALL)
_GUI_OWL_ACTION_RE = re.compile(r"(?:^|\n)\s*Action:\s*(.*?)\s*<tool_call>", re.DOTALL)


def extract_subgoal(prediction: str, fallback_len: int = 500) -> str:
    """Pull a one-step description from the agent's raw prediction.

    MAI-UI format: <thinking>...</thinking><tool_call>...</tool_call>
    General E2E format: "Thought: ...\\nAction: {json}"
    GUI-Owl format: Action: <local intent> followed by <tool_call>...</tool_call>.
    This is an action-intent proxy, not an independently supplied subgoal.
    """
    if not prediction:
        return "(no subgoal available)"

    m = _MAI_UI_THINKING_RE.search(prediction)
    if m:
        return m.group(1).strip()

    m = _GENERAL_E2E_THOUGHT_RE.search(prediction)
    if m:
        return m.group(1).strip()

    m = _GUI_OWL_ACTION_RE.search(prediction)
    if m and m.group(1).strip():
        return m.group(1).strip().strip('"')

    return prediction.strip()[:fallback_len]


def parse_verdict(raw_output: str) -> tuple[bool, str]:
    """Parse verifier output into (verdict, reasoning).

    Training output shape: {"answer": "Yes"|"No", "reason": "..."}.
    Falls back to substring matching so a drifted model still produces a verdict.
    """
    text = (raw_output or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```"))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            answer = str(obj.get("answer", "")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            if answer in ("yes", "no"):
                return (answer == "yes", reason or text[:500])
        except json.JSONDecodeError:
            pass

    head = text.lower()[:100]
    yes_idx = head.find("yes")
    no_idx = head.find("no")
    if yes_idx >= 0 and (no_idx < 0 or yes_idx < no_idx):
        return (True, text[:500])
    return (False, text[:500])


class BaseReflector(ABC):
    name: str = "base"

    def close(self) -> None:
        """Release resources owned by this reflector, if any."""

    @abstractmethod
    def verify(
        self,
        before: Image.Image,
        after: Image.Image,
        action: JSONAction,
        subgoal: str,
        global_task: str,
    ) -> ReflectionResult:
        raise NotImplementedError
