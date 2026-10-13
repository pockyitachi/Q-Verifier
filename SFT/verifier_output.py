"""Parse the explicit verdict without mistaking conditions or reasons for it."""

import json
import re


def parse_verdict(response: str | None) -> str:
    if not isinstance(response, str):
        return "invalid"
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        # Retain explicit plain-text verdict support for older prompts.
        match = re.match(r"^(?:answer\s*:\s*)?[\s*\"']*(yes|no)\b", text, re.I)
        return match.group(1).lower() if match else "invalid"
    if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
        answer = obj["answer"].strip().lower()
        if answer in {"yes", "no"}:
            return answer
    return "invalid"
