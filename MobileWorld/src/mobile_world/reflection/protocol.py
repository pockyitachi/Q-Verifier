"""Versioned online inputs; legacy runs keep their original prompt and policy."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from mobile_world.reflection.base import build_prompt, parse_verdict

PROTOCOLS = ("legacy_shared", "v9_geometry")
V9_SHA256 = "e7606164a8dd466dccf3a02b7413d1f8f6fc386fec372ca7d5af11d698a72ea2"
HISTORY_STEPS = 3
GEOMETRY_INSTRUCTIONS = """The images may be resized. Current-action coordinates are pixels in the
original BEFORE screenshot, with origin at top-left, x increasing rightward
and y increasing downward. Normalized coordinates are (x / original_width,
y / original_height): (0, 0) is top-left and (1, 1) is bottom-right.
Use the normalized positions to locate action targets regardless of resizing.
Do not interpret original-image pixel coordinates as resized-image pixels.
This is geometric metadata, not evidence of success or the identity of a control.
"""


def validate_protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown reflection protocol: {protocol!r}")
    return protocol


def protocol_configuration(protocol):
    validate_protocol(protocol)
    v9 = protocol == "v9_geometry"
    return {
        "protocol": protocol,
        "prompt_sha256": V9_SHA256 if v9 else None,
        "image_max_side": 1024 if v9 else 512,
        "max_new_tokens": 1024 if v9 else 256,
        "history_steps": HISTORY_STEPS if v9 else 0,
        "feedback_format": "evidence" if v9 else "legacy",
        "invalid_output_policy": "log_unknown_and_continue_without_hint_or_streak_reset" if v9 else "legacy_backend_specific",
    }


def action_context(action, intent):
    """Preserve the executed operation, including text/clear flags and pixel coordinates.

    Never turn a standalone input_text into the training set's compound
    Tap_Type_and_Enter. Descriptions are actor claims, not verified facts.
    """
    arguments = action.model_dump(exclude_none=True)
    name = arguments.pop("action_type")
    names = {"click": "Tap", "input_text": "Type", "navigate_back": "Back",
             "navigate_home": "Home", "open_app": "Open_App", "wait": "Wait",
             "long_press": "Long_Press", "double_tap": "Double_Tap",
             "keyboard_enter": "Enter", "drag": "Drag", "scroll": "Scroll"}
    for old, new in (("start_x", "x1"), ("start_y", "y1"), ("end_x", "x2"), ("end_y", "y2")):
        if old in arguments:
            arguments[new] = arguments.pop(old)
    return {"type": names.get(name, name), "arguments": arguments,
            "description": (intent or "")[:500]}


def image_geometry(before, after, current_action, max_side=1024):
    images = []
    for role, image in (("BEFORE", before), ("AFTER", after)):
        w, h = image.size
        scale = min(1., max_side / max(w, h))
        images.append({"role": role, "original_width": w, "original_height": h,
                       "input_width": max(1, int(w * scale)), "input_height": max(1, int(h * scale))})
    points = []
    args = current_action["arguments"]
    for xk, yk in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
        if xk not in args and yk not in args:
            continue
        if xk not in args or yk not in args:
            raise ValueError("Incomplete coordinate pair")
        x, y = args[xk], args[yk]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)):
            raise ValueError("Coordinates must be finite numbers")
        nx, ny = x / before.width, y / before.height
        points.append({"keys": [xk, yk], "original_pixels": [x, y],
                       "normalized_xy": [round(nx, 6), round(ny, 6)],
                       "inside_original_image": 0 <= nx < 1 and 0 <= ny < 1})
    return {"images": images, "current_action_points": points}


def build_v9_prompt(global_task, current_action, before, after, history=()):
    template = Path(__file__).with_name("prompts").joinpath("with_condition_generation.txt").read_text()
    if hashlib.sha256(template.encode()).hexdigest() != V9_SHA256:
        raise ValueError("Frozen v9 prompt checksum mismatch")
    # Copy only the documented history fields; no previous verdicts, labels or hints.
    past = [{"step": h["step"], "action": copy.deepcopy(h["action"])}
            for h in list(history)[-HISTORY_STEPS:]]
    context = {"goal": global_task, "history": past, "action": current_action}
    text = template.replace("{context}", json.dumps(context, ensure_ascii=False, indent=2))
    geometry = json.dumps(image_geometry(before, after, current_action), ensure_ascii=False, separators=(",", ":"))
    block = ("=== Image Geometry ===\n" + GEOMETRY_INSTRUCTIONS + geometry
             + "\n=== End Image Geometry ===\n\n")
    return text.replace("=== Context ===", block + "=== Context ===", 1)


def reflection_prompt(protocol, global_task, intent, action, before, after, history=()):
    validate_protocol(protocol)
    if protocol == "legacy_shared":
        return build_prompt(global_task, intent, action)
    return build_v9_prompt(global_task, action_context(action, intent), before, after, history)


def parse_response(protocol, raw):
    """New protocol requires an explicit final answer; missing is not a No.

    Accept the frozen evaluation's unambiguous answer-only recovery if otherwise
    valid JSON is damaged/truncated. Conflicting answer fields stay unknown.
    """
    if protocol == "legacy_shared":
        return parse_verdict(raw)
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    matches = re.findall(r'"answer"\s*:\s*"(Yes|No)"', text, re.I)
    if len(matches) == 1:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        reason = obj.get("reason") if isinstance(obj, dict) else None
        return matches[0].lower() == "yes", reason if isinstance(reason, str) and reason.strip() else text
    # Binary native models sometimes ignore the requested JSON schema.
    match = re.fullmatch(r"(?:Answer\s*:\s*)?(Yes|No)[.!]?", text, re.I)
    if match:
        return match.group(1).lower() == "yes", text
    return None, "[no unambiguous final Yes/No answer] " + text
