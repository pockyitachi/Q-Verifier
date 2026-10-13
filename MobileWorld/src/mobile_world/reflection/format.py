"""Format the verifier verdict into a structured hint for the agent.

Two responsibilities:

1. ``REFLECTION_PROMPT_ADDENDUM`` — text appended to the agent's system prompt
   when reflection is enabled, telling the agent that hints will be injected
   between turns inside ``<reflection_hint>`` tags and how to respond to them.

2. ``format_reflection_hint(...)`` — turn a verifier ``ReflectionResult`` plus
   the action that was flagged into the actual hint string injected into the
   next user message.

Design notes (see also docs/reflection-v2.md):

- The verifier's ``reason`` field follows a 5-section schema (BEFORE / AFTER /
  CHANGE / SUBGOAL ALIGNMENT / FINAL). Sections 1, 2, 4 mostly re-describe
  state the agent already has access to; only sections 3 (CHANGE) and 5
  (FINAL) carry novel information. Distill keeps just those two by default.
- The hint always names the previous action explicitly so the agent does not
  need to look back into history to associate the hint with what failed.
"""

from __future__ import annotations

import re
from html import escape

from mobile_world.runtime.utils.models import JSONAction

REFLECTION_PROMPT_ADDENDUM = """
## Reflection Hints
After some of your actions, an automated verifier inspects the resulting
screen and may inject a hint into the next user message. These hints look
like:

  <reflection_hint attempt="N">
  Previous action: <action description>
  <one or two sentences explaining why it likely failed>
  Avoid repeating the same coordinates or text.
  </reflection_hint>

When you see a hint:
- Do NOT repeat the exact same coordinates or text
- Re-read the current screenshot before deciding
- Prefer a different UI element, or switch action type
  (e.g., long_press instead of click, scroll first to reveal a target,
  drag instead of click for sliders)
""".strip()


_CHANGE_RE = re.compile(r"3\)\s*CHANGE:\s*(.+?)(?=\n\s*\d\)|\Z)", re.DOTALL)
_ALIGNMENT_RE = re.compile(r"4\)\s*SUBGOAL ALIGNMENT:\s*(.+?)(?=\n\s*\d\)|\Z)", re.DOTALL)
_FINAL_RE = re.compile(r"5\)\s*FINAL:\s*(.+?)(?=\n\s*\d\)|\Z)", re.DOTALL)


def _action_brief(action: JSONAction) -> str:
    """Render a JSONAction as a compact human-readable string."""
    name = action.action_type
    if name in ("click", "long_press", "double_tap"):
        return f"{name}({action.x}, {action.y})"
    if name == "drag":
        return f"drag(({action.start_x},{action.start_y}) -> ({action.end_x},{action.end_y}))"
    if name == "scroll":
        direction = getattr(action, "direction", None) or "?"
        return f"scroll({direction})"
    if name == "input_text":
        text = (action.text or "")[:60]
        return f'input_text("{text}")'
    if name == "open_app":
        return f'open_app("{action.text or ""}")'
    if name in ("navigate_back", "navigate_home", "keyboard_enter"):
        return name
    return name


def _distill(reason: str, include_alignment: bool = False) -> str:
    """Pull section 3 (CHANGE) + optionally 4 (SUBGOAL ALIGNMENT) + 5 (FINAL).

    Falls back to the first 200 chars of the raw reason if the schema is
    missing — handles the rare case where the verifier did not produce the
    expected 5-section structure.
    """
    parts: list[str] = []
    if m := _CHANGE_RE.search(reason):
        parts.append(_clean(m.group(1)))
    if include_alignment:
        if m := _ALIGNMENT_RE.search(reason):
            parts.append(_clean(m.group(1)))
    if m := _FINAL_RE.search(reason):
        parts.append(_clean(m.group(1)))
    if not parts:
        return reason[:200].strip()
    return " ".join(parts)


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def _bounded(text: str, limit: int) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    prefix = text[:limit - 1]
    # Prefer a sentence boundary, but do not throw away most of the evidence.
    ends = list(re.finditer(r"[.!?](?=\s)", prefix))
    if ends and ends[-1].end() >= limit // 2:
        prefix = prefix[:ends[-1].end()]
    else:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix + "…"


def _evidence(reason: str) -> str:
    sections = {}
    pattern = re.compile(
        r"(Precondition check|Postcondition check|Task progress)\s*:\s*(.*?)"
        r"(?=(?:Precondition check|Postcondition check|Task progress)\s*:|\Z)", re.I | re.S)
    for match in pattern.finditer(reason):
        sections[match.group(1).lower()] = match.group(2)
    parts = [f"{label}: {_bounded(sections[label.lower()], 360)}"
             for label in ("Postcondition check", "Task progress")
             if sections.get(label.lower(), "").strip()]
    if parts:
        return "\n".join(parts)
    # Also handle historical numbered explanations without losing the conclusion.
    parts = [_clean(m.group(1)) for p in (_CHANGE_RE, _FINAL_RE)
             if (m := p.search(reason))]
    return _bounded(" ".join(parts) if parts else reason, 760)


def format_reflection_hint(
    attempt: int,
    prev_action: JSONAction,
    verifier_reasoning: str,
    include_alignment: bool = False,
    feedback_format: str = "legacy",
) -> str:
    """Build the ``<reflection_hint>`` string that gets injected into the
    next user turn after a NO verdict.

    ``attempt`` is the count of consecutive NO verdicts so far (1, 2, 3, …).
    """
    if feedback_format not in ("legacy", "evidence"):
        raise ValueError(f"Unknown feedback format: {feedback_format}")
    if feedback_format == "evidence":
        action = _action_brief(prev_action)
        if prev_action.action_type == "open_app":
            action = f'open_app("{prev_action.app_name or prev_action.text or ""}")'
        return (
            f'<reflection_hint attempt="{attempt}">\n'
            f"Previous action: {escape(action, quote=False)}.\n"
            f"{escape(_evidence(verifier_reasoning), quote=False)}\n"
            "Avoid repeating the same coordinates or text.\n</reflection_hint>"
        )
    distilled = _distill(verifier_reasoning, include_alignment=include_alignment)
    return (
        f'<reflection_hint attempt="{attempt}">\n'
        f"Previous action: {_action_brief(prev_action)}.\n"
        f"{distilled}\n"
        f"Avoid repeating the same coordinates or text.\n"
        f"</reflection_hint>"
    )
