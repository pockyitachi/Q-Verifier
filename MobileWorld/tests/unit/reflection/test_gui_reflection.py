"""CPU-only integration checks; all model workers and environments are mocked.

Run: PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit/reflection
"""

import argparse
import ast
import asyncio
import base64
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from mobile_world.reflection import REFLECTION_MODES, create_reflector
from mobile_world.reflection.base import ReflectionResult, build_prompt, extract_subgoal, is_verifiable
from mobile_world.reflection.format import format_reflection_hint
from mobile_world.reflection.protocol import PROTOCOLS
from mobile_world.reflection.gui_reflection import (
    DEFAULT_GPU_UUID,
    GUIReflectionReflector,
    NATIVE_PROMPT,
    parse_gui_verdict,
)
from mobile_world.runtime.utils.models import ANSWER, ENV_FAIL, FINISHED, UNKNOWN, JSONAction

ROOT = Path(__file__).resolve().parents[3]
ACTION = JSONAction(action_type="click", x=10, y=20)


class FakeStdout:
    def __init__(self):
        self.lines = Queue()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.lines.get()
        if line is None:
            raise StopIteration
        return line

    def close(self):
        pass


class FakeProcess:
    def __init__(self, response="No"):
        self.stdout = FakeStdout()
        self.stdout.lines.put('{"type":"ready"}\n')
        self.stdin = Mock(write=self.write)
        self.requests = []
        self.response = response
        self.returncode = None
        self.terminate = Mock(side_effect=self.stop)
        self.kill = Mock(side_effect=self.stop)
        self.wait = Mock(return_value=0)

    def write(self, line):
        request = json.loads(line)
        self.requests.append(request)
        if self.response is None:
            return  # Simulate a worker that never answers.
        result = (self.response if isinstance(self.response, dict) else
                  {"type": "result", "id": request["id"], "raw_output": self.response})
        self.stdout.lines.put(json.dumps(result) + "\n")

    def poll(self):
        return self.returncode

    def stop(self):
        self.returncode = -15
        self.stdout.lines.put(None)


@pytest.fixture
def worker(monkeypatch, tmp_path):
    paths = [tmp_path / "python", tmp_path / "model/config.json",
             tmp_path / "official/internvl_chat/eval/infer.py"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    process = FakeProcess()
    popen = Mock(return_value=process)
    monkeypatch.setattr("mobile_world.reflection.gui_reflection.subprocess.Popen", popen)
    reflector = GUIReflectionReflector(
        python_path=str(paths[0]), model_path=str(paths[1].parent),
        official_code=str(tmp_path / "official"), request_timeout=0.05,
    )
    yield reflector, process, popen
    reflector.close()


def verify(reflector):
    return reflector.verify(Image.new("RGB", (600, 1200), "red"),
                            Image.new("RGB", (600, 1200), "blue"),
                            ACTION, "Open settings", "Change the wallpaper")


def test_lazy_factory_and_child_gpu_isolation(monkeypatch):
    popen = Mock(side_effect=AssertionError("No processes may start in this test"))
    monkeypatch.setattr("mobile_world.reflection.gui_reflection.subprocess.Popen", popen)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "actor-gpu")
    monkeypatch.setenv("PYTHONPATH", "/actor/dependencies")
    modules_before = set(sys.modules)
    reflector, policy = create_reflector("gui_reflection_retry")
    assert policy == "retry"
    assert reflector.configuration["prompt_mode"] == "shared"
    assert reflector.configuration["image_max_side"] == 512
    env = reflector._worker_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == DEFAULT_GPU_UUID
    assert "PYTHONPATH" not in env
    assert env["HF_HUB_OFFLINE"] == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "actor-gpu"
    assert os.environ["PYTHONPATH"] == "/actor/dependencies"
    assert not ({"torch", "transformers", "unsloth"} & (set(sys.modules) - modules_before))
    reflector.close()
    popen.assert_not_called()
    assert create_reflector("none") == (None, "none")


@pytest.mark.parametrize("response,verdict,reason", [
    ("Yes", True, "Yes"), ("No", False, "No"),
    ("No. The menu stayed closed.", False, "No. The menu stayed closed."),
    ('```json\n{"answer":"No","reason":"Menu is closed"}\n```', False, "Menu is closed"),
])
def test_transport_preserves_evidence_and_reason(worker, response, verdict, reason):
    reflector, process, popen = worker
    process.response = response
    result = verify(reflector)
    assert (result.verdict, result.reasoning, result.raw_output) == (verdict, reason, response)
    request = process.requests[0]
    assert request["prompt"] == build_prompt("Change the wallpaper", "Open settings", ACTION)
    images = [Image.open(io.BytesIO(base64.b64decode(data))) for data in request["images"]]
    assert [image.size for image in images] == [(256, 512), (256, 512)]
    assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]
    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == DEFAULT_GPU_UUID
    reflector.close()
    reflector.close()
    process.terminate.assert_called_once()
    process.kill.assert_not_called()


def test_native_prompt_retains_image_policy(worker):
    reflector, process, _ = worker
    reflector.prompt_mode = "native"
    verify(reflector)
    assert process.requests[0]["prompt"] == NATIVE_PROMPT.format(action_purpose="Open settings")
    assert "Change the wallpaper" not in process.requests[0]["prompt"]
    assert reflector.image_max_side == 512


@pytest.mark.parametrize("raw", ["", "The screen has no changes.", "not sure", '{"answer":"maybe"}'])
def test_invalid_model_output_is_not_a_no(raw):
    with pytest.raises(ValueError, match="no explicit"):
        parse_gui_verdict(raw)


@pytest.mark.parametrize("response,error", [
    (None, TimeoutError),
    ({"type": "error", "error": "CUDA out of memory"}, RuntimeError),
    ({"type": "result", "id": 999, "raw_output": "Yes"}, RuntimeError),
])
def test_failed_worker_is_closed_and_not_restarted(worker, response, error):
    reflector, process, popen = worker
    process.response = response
    with pytest.raises(error):
        verify(reflector)
    process.terminate.assert_called_once()
    with pytest.raises(RuntimeError, match="unavailable"):
        verify(reflector)
    assert popen.call_count == 1


def test_shared_worker_serializes_concurrent_tasks(worker):
    reflector, process, popen = worker
    process.response = "Yes"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: verify(reflector), range(4)))
    assert all(result.verdict for result in results)
    assert [r["id"] for r in process.requests] == [1, 2, 3, 4]
    assert popen.call_count == 1


def load_functions(relative_path, names, namespace):
    """Exercise real function bodies without importing Docker/Android/model SDKs."""
    source = ROOT / relative_path
    tree = ast.parse(source.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names)
    assert len(body) == len(names) + 1
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), namespace)
    return namespace


def test_cli_passes_gui_settings_to_runner():
    runner = Mock(return_value=([], []))
    scope = load_functions("src/mobile_world/core/subcommands/eval.py",
                           {"_add_common_arguments", "configure_parser", "execute"},
                           {"argparse": argparse, "REFLECTION_MODES": REFLECTION_MODES, "PROTOCOLS": PROTOCOLS,
                            "os": os, "logger": Mock(), "run_agent_with_evaluation": runner})
    parser = argparse.ArgumentParser()
    scope["configure_parser"](parser.add_subparsers())
    args = parser.parse_args([
        "eval", "--agent-type", "mai_ui_agent", "--model-name", "Tongyi-MAI/MAI-UI-8B",
        "--task", "FakeTask", "--reflection-mode", "gui_reflection_retry",
        "--reflection-gui-python", "/isolated/python", "--reflection-gui-prompt", "native",
        "--reflection-gui-model-path", "/model", "--reflection-gui-code-path", "/code",
        "--reflection-gui-gpu-uuid", DEFAULT_GPU_UUID,
    ])
    asyncio.run(scope["execute"](args))
    kwargs = runner.call_args.kwargs
    assert kwargs["model_name"] == "Tongyi-MAI/MAI-UI-8B"
    assert kwargs["reflection_mode"] == "gui_reflection_retry"
    assert kwargs["reflection_gui_python"] == "/isolated/python"
    assert kwargs["reflection_gui_model_path"] == "/model"
    assert kwargs["reflection_gui_code_path"] == "/code"
    assert kwargs["reflection_gui_prompt"] == "native"
    assert kwargs["reflection_gui_gpu_uuid"] == DEFAULT_GPU_UUID
    assert kwargs["reflection_max_consecutive_failures"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_runner_always_closes_reflector_and_keeps_actor_options(fail, tmp_path):
    reflector = Mock()
    factory = Mock(return_value=(reflector, "retry"))
    init = Mock(side_effect=RuntimeError("fake env error")) if fail else Mock()
    task = Mock(return_value={"reflection_stats": {}})
    scope = load_functions("src/mobile_world/core/runner.py", {"run_agent_with_evaluation"}, {
        "logger": Mock(), "create_reflector": factory, "_init_env": init,
        "Parallel": lambda **kwargs: lambda jobs: list(jobs),
        "delayed": lambda fn: fn, "Queue": Queue, "AndroidEnvClient": object,
        "_process_task_on_env": task,
        "scan_finished_tasks": Mock(side_effect=[([], []), (["FakeTask"], [1])]),
    })
    kwargs = dict(agent_type="mai_ui_agent", model_name="actor", llm_base_url="http://actor/v1",
                  log_file_root=str(tmp_path), tasks=["FakeTask"], aw_urls=["http://fake-env"],
                  auto_retry=0, reflection_mode="gui_reflection_retry",
                  reflection_gui_model_path="/model", reflection_gui_python="/python",
                  reflection_gui_code_path="/code", reflection_gui_gpu_uuid=DEFAULT_GPU_UUID,
                  reflection_gui_prompt="native", scale_factor=1000)
    if fail:
        with pytest.raises(RuntimeError, match="fake env error"):
            scope["run_agent_with_evaluation"](**kwargs)
    else:
        result, missing = scope["run_agent_with_evaluation"](**kwargs)
        assert result[0]["score"] == 1 and missing == []
        passed = task.call_args.kwargs
        assert passed["model_name"] == "actor" and passed["llm_base_url"] == "http://actor/v1"
        assert passed["reflection_policy"] == "retry" and passed["scale_factor"] == 1000
        assert not any(key.startswith("reflection_gui_") for key in passed)
        token_summary = json.loads((tmp_path / "token_usage_summary.json").read_text())["summary"]
        assert token_summary["combined"]["prompt_tokens"] is None  # Mock task was unmetered.
    assert factory.call_args.kwargs["gui_python"] == "/python"
    assert factory.call_args.kwargs["gui_prompt"] == "native"
    reflector.close.assert_called_once()


@pytest.mark.parametrize("verdicts,early_stop,retries", [
    ([False, False, False], True, 2),
    ([False, True, False], False, 1),
])
def test_existing_feedback_and_stop_policy(verdicts, early_stop, retries):
    scope = load_functions("src/mobile_world/core/runner.py", {"_execute_single_task"}, {
        "logger": Mock(), "ENV_FAIL": ENV_FAIL, "FINISHED": FINISHED,
        "UNKNOWN": UNKNOWN, "ANSWER": ANSWER, "is_verifiable": is_verifiable,
        "extract_subgoal": extract_subgoal, "format_reflection_hint": format_reflection_hint,
    })
    observation = SimpleNamespace(screenshot=Image.new("RGB", (2, 2)), tool_call=None, ask_user_response=None)
    env = Mock()
    env.get_task_goal.return_value = "Open settings"
    env.initialize_task.return_value = observation
    env.execute_action.return_value = observation
    env.get_task_score.return_value = (0, "fake score")
    agent = Mock()
    agent.predict.return_value = ("<thinking>Open settings</thinking>", ACTION)
    reflector = Mock()
    reflector.verify.side_effect = [ReflectionResult(v, "Yes" if v else "No", 1, "Yes" if v else "No")
                                     for v in verdicts]
    traj = Mock()
    steps, _, stats = scope["_execute_single_task"](
        env, agent, "FakeTask", 3, traj, reflector=reflector, reflection_policy="retry",
        max_consecutive_failures=3,
    )
    assert steps == 3
    assert stats["early_termination_triggered"] == early_stop
    assert stats["retry_count"] == retries
    second_observation = agent.predict.call_args_list[1].args[0]
    assert "No" in second_observation["feedback"]
    assert "Avoid repeating" in second_observation["feedback"]
    assert traj.log_reflection.call_args_list[0].kwargs["raw_output"] == "No"


def test_worker_usage_reaches_reflection_result(worker):
    reflector, process, _ = worker
    process.response = {"type": "result", "id": 1, "raw_output": "Yes",
                        "token_usage": {"prompt_tokens": 1025, "completion_tokens": 8,
                                        "source": "model_token_ids"}}
    result = verify(reflector)
    assert result.token_usage()["prompt_tokens"] == 1025
    assert result.token_usage()["completion_tokens"] == 8


def test_old_worker_without_usage_is_unknown(worker):
    reflector, _, _ = worker
    assert verify(reflector).token_usage() is None


def test_legacy_reflection_summary_does_not_report_zero_usage_or_cost():
    cost = Mock()
    scope = load_functions("src/mobile_world/core/subcommands/eval.py",
                           {"_aggregate_reflection_stats"}, {"_compute_cost_usd": cost})
    result = scope["_aggregate_reflection_stats"]([
        {"reflection_stats": {"total_verification_calls": 4, "total_prompt_tokens": 0}},
    ], model="fake")
    assert result["total_prompt_tokens"] is None
    assert result["total_completion_tokens"] is None
    assert result["total_cost_usd"] is None
    cost.assert_not_called()
