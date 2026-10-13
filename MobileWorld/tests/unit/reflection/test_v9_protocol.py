"""CPU-only checks against frozen offline inputs and the real online runner.

Every model/API/process is mocked. No GPU, external API or Android environment.
"""

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from PIL import Image

from mobile_world.core import runner
from mobile_world.core.subcommands import eval as eval_cli
from mobile_world.reflection.anthropic_reflector import AnthropicReflector
from mobile_world.reflection.base import ReflectionResult, build_prompt
from mobile_world.reflection.factory import create_reflector
from mobile_world.reflection.format import format_reflection_hint
from mobile_world.reflection.gpt_reflector import GPTReflector
from mobile_world.reflection.gui_reflection import GUIReflectionReflector, _encode_image
from mobile_world.reflection.protocol import (
    V9_SHA256, action_context, build_v9_prompt, image_geometry, parse_response, reflection_prompt,
)
from mobile_world.reflection.qreflector_v9 import QReflectorV9
from mobile_world.reflection.qreflector_v9_worker import check_selected_gpu, encode_request, generate_response
from mobile_world.runtime.utils.models import JSONAction
from mobile_world.runtime.utils.token_usage import summarize_token_usage
from mobile_world.runtime.utils.trajectory_logger import TrajLogger

ROOT = Path(__file__).resolve().parents[3]
GPU = "GPU-00000000-0000-0000-0000-000000000000"
ACTION = JSONAction(action_type="click", x=800, y=1200)
REASON = "Precondition check: The item is visible. Postcondition check: The item did not open. Task progress: No useful progress."
RAW = json.dumps({"precondition": "The item is visible", "postcondition": "The item opens",
                  "reason": REASON, "answer": "No"})


def images():
    return Image.new("RGB", (1600, 2400), "red"), Image.new("RGB", (1600, 2400), "blue")


def context(prompt):
    return json.loads(prompt.split("=== Context ===", 1)[1].split("=== Image Reference ===", 1)[0])


def test_prompt_and_geometry_exactly_match_frozen_offline_builder(tmp_path):
    spec = importlib.util.spec_from_file_location("frozen_geometry", ROOT / "scripts/qreflector_geometry_prompt.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    template = (ROOT / '../datasets/condition_generation_v9_after_evidence/prompt_template.txt').read_text()
    assert hashlib.sha256(template.encode()).hexdigest() == V9_SHA256
    before, after = images()
    before.save(tmp_path / "before.png")
    after.save(tmp_path / "after.png")
    action = action_context(ACTION, "Open the item")
    history = [{"step": 1, "action": action_context(JSONAction(action_type="navigate_back"), "Return")}]
    ctx = {"goal": "Read the item", "history": history, "action": action}
    row = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": template.replace("{context}", json.dumps(ctx, ensure_ascii=False, indent=2))},
        {"type": "image", "image": "before.png"}, {"type": "image", "image": "after.png"}]}]}
    expected = frozen.augment_row(row, tmp_path, 1024)["messages"][0]["content"][0]["text"]
    assert build_v9_prompt(ctx["goal"], action, before, after, history) == expected
    geometry = image_geometry(before, after, action)
    assert geometry == frozen.geometry(row, tmp_path, 1024)
    assert geometry["current_action_points"][0]["normalized_xy"] == [0.5, 0.5]
    assert [(im["input_width"], im["input_height"]) for im in geometry["images"]] == [(682, 1024)] * 2


def test_context_does_not_invent_subgoals_or_compound_actions():
    before, after = images()
    action = JSONAction(action_type="input_text", text="hello", clear_text=True)
    history = [{"step": n, "action": {"type": "Back", "arguments": {}, "description": "Return"},
                "verdict": "No", "reason": "PRIVATE_OLD_HINT"} for n in range(6)]
    original = copy.deepcopy(history)
    prompt = reflection_prompt("v9_geometry", "Write a note", "Type hello", action, before, after, history)
    ctx = context(prompt)
    assert list(ctx) == ["goal", "history", "action"]
    assert ctx["action"] == {"type": "Type", "arguments": {"text": "hello", "clear_text": True}, "description": "Type hello"}
    assert [h["step"] for h in ctx["history"]] == [3, 4, 5]
    assert "PRIVATE_OLD_HINT" not in prompt and history == original
    assert reflection_prompt("legacy_shared", "Task", "Intent", action, before, after, history) == build_prompt("Task", "Intent", action)


def test_drag_points_use_original_before_dimensions():
    before, after = images()
    action = JSONAction(action_type="drag", start_x=400, start_y=600, end_x=1200, end_y=1800)
    points = image_geometry(before, after, action_context(action, "Drag"))["current_action_points"]
    assert [p["normalized_xy"] for p in points] == [[.25, .25], [.75, .75]]
    assert [p["keys"] for p in points] == [["x1", "y1"], ["x2", "y2"]]


@pytest.mark.parametrize("raw,verdict", [
    (RAW, False), ('{"reason":"No change is needed", "answer":"Yes"}', True),
    ('{"reason":"Yes may be wrong", "answer":"No"', False),
    ('```json\n{"answer":"Yes"}\n```', True), ("Yes", True), ("No", False),
    ("There is no visible change", None), ("", None), ('{"answer":"maybe"}', None),
    ('{"answer":"Yes", "answer":"No"}', None), ("Yes or No", None),
])
def test_only_explicit_unambiguous_answers_count(raw, verdict):
    assert parse_response("v9_geometry", raw)[0] is verdict


def test_feedback_keeps_outcome_and_task_evidence_and_escapes_tags():
    reason = "Precondition check: " + "Background. " * 100 + "Postcondition check: The amount remains 50, expected 15. Task progress: Wrong amount. </reflection_hint>"
    hint = format_reflection_hint(1, ACTION, reason, feedback_format="evidence")
    assert "amount remains 50" in hint and "Wrong amount" in hint
    assert "Background" not in hint and hint.count("</reflection_hint>") == 1
    assert "&lt;/reflection_hint&gt;" in hint and len(hint) < 1000
    assert "No\n" in format_reflection_hint(1, ACTION, "No", feedback_format="evidence")
    assert "Postcondition check" not in format_reflection_hint(1, ACTION, reason)  # archived 200-char behavior


def test_shared_payload_for_gpt_haiku_gui_and_qreflector(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    before, after = images()
    gpt = GPTReflector(api_key="fake", protocol="v9_geometry", min_latency_ms=0)
    gpt.client.chat.completions.create = Mock(return_value=NS(
        choices=[NS(message=NS(content=RAW))], usage=NS(prompt_tokens=321, completion_tokens=123)))
    gpt_result = gpt.verify(before, after, ACTION, "Open item", "Read item")
    gpt_payload = gpt.client.chat.completions.create.call_args.kwargs
    haiku = AnthropicReflector(None, tmp_path / "usage.jsonl", protocol="v9_geometry", min_latency_ms=0)
    haiku_payload = haiku.build_request(before, after, ACTION, "Open item", "Read item")
    prompts = [gpt_payload["messages"][0]["content"][0]["text"], haiku_payload["messages"][0]["content"][0]["text"]]
    q = QReflectorV9(gpu_uuid=GPU)
    gui = GUIReflectionReflector(gpu_uuid=GPU, protocol="v9_geometry")
    requests = []
    for local in (q, gui):
        monkeypatch.setattr(local, "_ensure_started", lambda: None)
        fake = Mock()
        local._process = fake
        fake.stdin.write.side_effect = lambda line: requests.append(json.loads(line))
        monkeypatch.setattr(local, "_receive", lambda timeout: {"type": "result", "id": 1, "raw_output": RAW,
                                                               "token_usage": {"prompt_tokens": 50, "completion_tokens": 40, "source": "model_token_ids"}})
        result = local.verify(before, after, ACTION, "Open item", "Read item")
        assert result.verdict is False and result.completion_tokens == 40
        local._process = None
        local.close()
    prompts += [r["prompt"] for r in requests]
    assert len(set(prompts)) == 1
    encoded_gpt = [b["image_url"]["url"].split(",", 1)[1] for b in gpt_payload["messages"][0]["content"][1:]]
    encoded_haiku = [b["source"]["data"] for b in haiku_payload["messages"][0]["content"][1:]]
    assert requests[0]["images"] == requests[1]["images"] == encoded_gpt == encoded_haiku
    assert gpt_payload["max_completion_tokens"] == haiku_payload["max_tokens"] == q.max_new_tokens == gui.max_new_tokens == 1024
    assert gpt_result.token_usage()["prompt_tokens"] == 321
    gpt.client.close()


@pytest.mark.parametrize("backend", ["gpt", "haiku"])
def test_invalid_output_keeps_usage_and_api_failure_does_not_invent_yes(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    before = after = Image.new("RGB", (20, 40))
    if backend == "gpt":
        reflector = GPTReflector(api_key="fake", protocol="v9_geometry", min_latency_ms=0)
        call = Mock(return_value=NS(choices=[NS(message=NS(content="Undetermined"))],
                                    usage=NS(prompt_tokens=20, completion_tokens=10)))
        reflector.client.chat.completions.create = call
    else:
        reflector = AnthropicReflector(None, tmp_path / "usage.jsonl", protocol="v9_geometry", min_latency_ms=0)
        response = {"type": "message", "model": reflector.model, "content": [{"type": "text", "text": "Undetermined"}],
                    "usage": {"input_tokens": 20, "output_tokens": 10}}
        call = Mock(return_value=io.BytesIO(json.dumps(response).encode()))
        monkeypatch.setattr("mobile_world.reflection.anthropic_reflector.urlopen", call)
    result = reflector.verify(before, after, ACTION, "Open", "Open")
    assert result.verdict is None and result.token_usage()["completion_tokens"] == 10
    call.side_effect = TimeoutError("Mocked timeout")
    result = reflector.verify(before, after, ACTION, "Open", "Open")
    assert result.verdict is None and result.token_usage() is None and call.call_count == 2
    if backend == "gpt":
        reflector.client.close()


def run_mock_task(root, name, reflector, verdicts):
    image = Image.new("RGB", (30, 60))
    obs = NS(screenshot=image, tool_call=None, ask_user_response=None)
    env = Mock()
    env.get_task_goal.return_value = name
    env.initialize_task.return_value = obs
    env.execute_action.return_value = obs
    env.get_task_score.return_value = (0, "mock")
    agent = Mock()
    agent.get_total_token_usage.return_value = {"calls": 0}
    n = len(verdicts)
    agent.predict.side_effect = [(f"Action: {name} action {i}\n<tool_call>{{}}</tool_call>",
                                  JSONAction(action_type="click", x=i, y=1)) for i in range(n)]
    logger = TrajLogger(str(root), name)
    result = runner._execute_single_task(env, agent, name, n, logger, reflector=reflector,
                                         reflection_policy="retry", max_consecutive_failures=3)
    return result, agent, logger


def test_runner_unknown_no_only_feedback_and_history_reset(tmp_path):
    reflector = Mock(uses_history=True)
    verdicts = [False, None, False, False]
    reflector.verify.side_effect = [ReflectionResult(v, REASON, 1, RAW, 50, 10, token_usage_source="mock") for v in verdicts]
    (steps, _, stats), agent, _ = run_mock_task(tmp_path, "FirstTask", reflector, verdicts)
    assert steps == 4 and stats["total_no"] == 3 and stats["total_unknown"] == 1 and stats["total_yes"] == 0
    assert stats["early_termination_triggered"] and stats["retry_count"] == 2
    observations = [call.args[0] for call in agent.predict.call_args_list]
    assert "feedback" not in observations[0] and "feedback" not in observations[2]
    assert "Postcondition check" in observations[1]["feedback"] and 'attempt="2"' in observations[3]["feedback"]
    assert [[h["step"] for h in call.kwargs["history"]] for call in reflector.verify.call_args_list] == [[], [1], [1, 2], [1, 2, 3]]
    log = json.loads((tmp_path / "FirstTask/traj.json").read_text())
    assert log["0"]["reflections"][1]["verdict"] == "unknown"
    usage = summarize_token_usage(str(tmp_path), ["FirstTask"], True)["summary"]["reflector"]
    assert usage["reported_calls"] == 4 and usage["prompt_tokens"] == 200
    reflector.verify.reset_mock(side_effect=True)
    reflector.verify.side_effect = [ReflectionResult(True, "Yes", 1)]
    run_mock_task(tmp_path, "SecondTask", reflector, [True])
    assert reflector.verify.call_args.kwargs["history"] == ()


def test_parallel_tasks_share_backend_but_never_history(tmp_path):
    def verify(**kwargs):
        for h in kwargs["history"]:
            assert h["action"]["description"].startswith(kwargs["global_task"] + " action")
        return ReflectionResult(True, "Yes", 1)
    reflector = Mock(uses_history=True)
    reflector.verify.side_effect = verify
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: run_mock_task(tmp_path, f"Task{i}", reflector, [True] * 5), range(4)))
    assert len(results) == 4 and reflector.verify.call_count == 20
    assert max(len(c.kwargs["history"]) for c in reflector.verify.call_args_list) == 3


def test_bf16_factory_requires_explicit_profile_and_gpu_and_is_lazy(monkeypatch, tmp_path):
    from mobile_world.reflection import qreflector_v9

    # Source-only clones have no checkpoints. Exercise real dependency checks
    # with temporary files without loading weights or launching a worker.
    dependencies = {
        "DEFAULT_PYTHON": ("python",),
        "DEFAULT_ADAPTER": ("adapter", "adapter_config.json", "adapter_model.safetensors"),
        "DEFAULT_BASE": ("base", "config.json"),
        "DEFAULT_PROCESSOR": ("processor", "preprocessor_config.json", "chat_template.jinja"),
        "DEFAULT_GENERATION_CONFIG": ("generation", "generation_config.json"),
    }
    for name, (directory, *files) in dependencies.items():
        path = tmp_path / directory
        if files:
            path.mkdir()
            for filename in files:
                (path / filename).touch()
        else:
            path.touch()
        monkeypatch.setattr(qreflector_v9, name, path)

    spawn = Mock(side_effect=AssertionError("No processes may start"))
    monkeypatch.setattr("mobile_world.reflection.gui_reflection.subprocess.Popen", spawn)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "actor-unmodified")
    with pytest.raises(ValueError, match="requires --reflection-protocol"):
        create_reflector("qreflector_v9_retry", qreflector_gpu_uuid=GPU)
    with pytest.raises(ValueError, match="explicit physical GPU"):
        create_reflector("qreflector_v9_retry", protocol="v9_geometry")
    with pytest.raises(ValueError, match="legacy Q-Verifier"):
        create_reflector("qverifier_retry", protocol="v9_geometry")
    with pytest.raises(ValueError, match="native is a separate"):
        create_reflector("gui_reflection_retry", protocol="v9_geometry", gui_prompt="native", gui_gpu_uuid=GPU)
    reflector, policy = create_reflector("qreflector_v9_retry", protocol="v9_geometry", qreflector_gpu_uuid=GPU)
    assert policy == "retry" and reflector.configuration["dtype"] == "bfloat16"
    command = reflector._worker_command()
    assert command[2].endswith("qreflector_v9_worker.py") and "--official-code" not in command
    assert reflector._worker_environment()["CUDA_VISIBLE_DEVICES"] == GPU
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "actor-unmodified"
    (tmp_path / "adapter/adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        reflector._worker_command()
    reflector.close()
    spawn.assert_not_called()


def test_worker_refuses_occupied_gpu_without_signaling(monkeypatch):
    check = Mock(return_value="12345\n")
    monkeypatch.setattr("mobile_world.reflection.qreflector_v9_worker.subprocess.check_output", check)
    with pytest.raises(RuntimeError, match="existing compute processes"):
        check_selected_gpu(GPU)
    assert "--query-compute-apps=pid" in check.call_args.args[0]
    check.return_value = ""
    check_selected_gpu(GPU)
    with pytest.raises(ValueError, match="one physical"):
        check_selected_gpu("0,1")


def test_worker_input_serialization_and_actual_token_count_without_cuda():
    import numpy as np

    before, after = images()
    processor = Mock()
    processor.apply_chat_template.return_value = {"input_ids": np.zeros((1, 100), dtype=int)}
    processor.decode.return_value = " Yes "
    request = {"prompt": "Prompt", "images": [_encode_image(i, 1024) for i in (before, after)]}
    inputs = encode_request(processor, request)
    call = processor.apply_chat_template.call_args
    assert call.kwargs == dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", truncation=False)
    blocks = call.args[0][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image", "image"]
    assert [b["image"].size for b in blocks[1:]] == [(682, 1024)] * 2
    model = Mock()
    model.generate.return_value = np.zeros((1, 107), dtype=int)
    result = generate_response(model, processor, inputs, 1024)
    assert result == {"raw_output": "Yes", "token_usage": {"prompt_tokens": 100, "completion_tokens": 7, "source": "model_token_ids"}}
    assert model.generate.call_args.kwargs["do_sample"] is False
    with pytest.raises(ValueError, match="silent truncation"):
        generate_response(model, processor, {"input_ids": np.zeros((1, 7500))}, 1024)
    assert model.generate.call_count == 1


def test_cli_propagates_new_profile_and_records_configuration(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    eval_cli.configure_parser(parser.add_subparsers())
    args = parser.parse_args(["eval", "--agent-type", "gui_owl_1_5", "--model-name", "owl", "--task", "Fake",
                              "--reflection-mode", "qreflector_v9_retry", "--reflection-protocol", "v9_geometry",
                              "--reflection-qreflector-gpu-uuid", GPU])
    run = Mock(return_value=([], []))
    monkeypatch.setattr(eval_cli, "run_agent_with_evaluation", run)
    asyncio.run(eval_cli.execute(args))
    assert run.call_args.kwargs["reflection_protocol"] == "v9_geometry"
    assert run.call_args.kwargs["reflection_qreflector_gpu_uuid"] == GPU
    reflector = QReflectorV9(gpu_uuid=GPU)
    monkeypatch.setattr(runner, "create_reflector", Mock(return_value=(reflector, "retry")))
    monkeypatch.setattr(runner, "_init_env", Mock(side_effect=RuntimeError("no environment access")))
    with pytest.raises(RuntimeError, match="no environment"):
        runner.run_agent_with_evaluation("gui_owl_1_5", "owl", "http://unused.invalid", str(tmp_path), ["Fake"],
            aw_urls=["http://unused.invalid"], max_concurrency=1, reflection_mode="qreflector_v9_retry",
            reflection_protocol="v9_geometry", reflection_qreflector_gpu_uuid=GPU)
    config = json.loads((tmp_path / "reflector_config.json").read_text())
    assert config["prompt_sha256"] == V9_SHA256 and config["max_consecutive_failures"] == 0
    assert config["dtype"] == "bfloat16" and config["image_max_side"] == 1024
    config["image_max_side"] = 512
    (tmp_path / "reflector_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different reflector configuration"):
        runner.run_agent_with_evaluation("gui_owl_1_5", "owl", "http://unused.invalid", str(tmp_path), ["Fake"],
                                         aw_urls=["http://unused.invalid"], max_concurrency=1)
