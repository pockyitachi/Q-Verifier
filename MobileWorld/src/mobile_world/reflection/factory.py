"""Factory for constructing reflectors from CLI mode name."""

from __future__ import annotations

from mobile_world.reflection.base import BaseReflector
from mobile_world.reflection.protocol import validate_protocol

REFLECTION_MODES = (
    "none", "qverifier_early_stop", "qverifier_retry", "gpt_retry", "gui_reflection_retry",
    "anthropic_retry", "qreflector_v9_retry",
)


def create_reflector(
    mode: str,
    qverifier_path: str | None = None,
    gpt_model: str = "gpt-4o",
    gpt_base_url: str | None = None,
    gpt_api_key: str | None = None,
    gui_model_path: str | None = None,
    gui_python: str | None = None,
    gui_code_path: str | None = None,
    gui_gpu_uuid: str | None = None,
    gui_prompt: str = "shared",
    anthropic_model: str = "claude-haiku-4-5-20251001",
    anthropic_env_file: str | None = None,
    anthropic_ledger_path: str | None = None,
    protocol: str = "legacy_shared",
    qreflector_gpu_uuid: str | None = None,
    qreflector_python: str | None = None,
) -> tuple[BaseReflector | None, str]:
    """Return (reflector, policy).

    policy is one of: "none", "early_stop", "retry".
    """
    validate_protocol(protocol)
    if protocol == "v9_geometry" and mode in ("qverifier_retry", "qverifier_early_stop"):
        raise ValueError("Use qreflector_v9_retry for the current BF16 model; legacy Q-Verifier is 4-bit")
    if mode == "qreflector_v9_retry":
        from mobile_world.reflection.qreflector_v9 import QReflectorV9

        if protocol != "v9_geometry":
            raise ValueError("qreflector_v9_retry requires --reflection-protocol v9_geometry")
        return QReflectorV9(gpu_uuid=qreflector_gpu_uuid, adapter_path=qverifier_path,
                            python_path=qreflector_python), "retry"
    if mode == "none":
        return None, "none"
    if mode == "qverifier_early_stop":
        from mobile_world.reflection.qverifier import QVerifierReflector

        return QVerifierReflector(model_path=qverifier_path), "early_stop"
    if mode == "qverifier_retry":
        from mobile_world.reflection.qverifier import QVerifierReflector

        return QVerifierReflector(model_path=qverifier_path), "retry"
    if mode == "gpt_retry":
        from mobile_world.reflection.gpt_reflector import GPTReflector

        return GPTReflector(model=gpt_model, base_url=gpt_base_url, api_key=gpt_api_key,
                            protocol=protocol), "retry"
    if mode == "anthropic_retry":
        from mobile_world.reflection.anthropic_reflector import AnthropicReflector

        if not anthropic_ledger_path:
            raise ValueError("anthropic_retry requires a per-run anthropic_ledger_path")
        return AnthropicReflector(
            env_file=anthropic_env_file, ledger_path=anthropic_ledger_path,
            model=anthropic_model, protocol=protocol,
        ), "retry"
    if mode == "gui_reflection_retry":
        from mobile_world.reflection.gui_reflection import GUIReflectionReflector

        return GUIReflectionReflector(
            model_path=gui_model_path, python_path=gui_python, official_code=gui_code_path,
            gpu_uuid=gui_gpu_uuid, prompt_mode=gui_prompt, protocol=protocol,
        ), "retry"
    raise ValueError(f"Unknown reflection mode: {mode!r}. Valid: {REFLECTION_MODES}")
