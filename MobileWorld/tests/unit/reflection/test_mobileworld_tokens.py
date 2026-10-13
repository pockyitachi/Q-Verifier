"""CPU-only usage checks: real accounting paths, fake API/model responses."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import numpy as np
from PIL import Image

from mobile_world.agents.base import BaseAgent
from mobile_world.reflection.base import ReflectionResult
from mobile_world.reflection.gui_reflection_worker import install_token_counter
from mobile_world.reflection.gpt_reflector import GPTReflector
from mobile_world.reflection.qverifier import QVerifierReflector
from mobile_world.runtime.utils.models import JSONAction
from mobile_world.runtime.utils.token_usage import (
    TaskTokenRecorder,
    summarize_token_usage,
    write_token_usage_report,
)
from mobile_world.runtime.utils.trajectory_logger import TrajLogger


class Agent(BaseAgent):
    def predict(self, observation):
        raise NotImplementedError


def response(prompt=100, completion=20, content="done"):
    return NS(usage=NS(prompt_tokens=prompt, completion_tokens=completion,
                       prompt_tokens_details=NS(cached_tokens=40)),
              choices=[NS(message=NS(content=content))])


def test_api_retries_count_each_response_once(tmp_path, monkeypatch):
    recorder = TaskTokenRecorder(str(tmp_path), "task")
    agent = Agent()
    agent.token_usage_callback = lambda usage: recorder.record("actor", usage)
    agent.openai_client = Mock()
    # Parsing fails after usage was returned: paid tokens still count exactly once.
    agent.openai_client.chat.completions.create.side_effect = [
        response(content=None), response(prompt=200),
    ]
    monkeypatch.setattr("mobile_world.agents.base.time.sleep", lambda _: None)
    assert agent.openai_chat_completions_create("actor", []) == "done"
    summary = summarize_token_usage(str(tmp_path), ["task"], False)["summary"]
    assert summary["actor"]["reported_calls"] == 2
    assert summary["actor"]["missing_usage_calls"] == 0
    assert summary["combined"]["prompt_tokens"] == 300  # Cached tokens already included.
    assert summary["combined"]["completion_tokens"] == 40


def test_api_error_is_unknown_not_zero(tmp_path, monkeypatch):
    recorder = TaskTokenRecorder(str(tmp_path), "task")
    agent = Agent()
    agent.token_usage_callback = lambda usage: recorder.record("actor", usage)
    agent.openai_client = Mock()
    agent.openai_client.chat.completions.create.side_effect = [TimeoutError(), response()]
    monkeypatch.setattr("mobile_world.agents.base.time.sleep", lambda _: None)
    assert agent.openai_chat_completions_create("actor", []) == "done"
    summary = summarize_token_usage(str(tmp_path), ["task"], False)["summary"]
    assert summary["actor"]["missing_usage_calls"] == 1
    assert summary["actor"]["known_prompt_tokens"] == 100
    assert summary["combined"]["mean_prompt_tokens_per_task"] is None


@pytest.mark.parametrize("with_usage", [True, False])
def test_stream_records_one_call_with_or_without_usage(with_usage):
    agent = Agent()
    callback = Mock()
    agent.token_usage_callback = callback
    chunks = [NS(usage=None)] + ([response()] if with_usage else [])
    assert list(agent._wrap_stream_with_usage_logging(iter(chunks))) == chunks
    callback.assert_called_once()
    assert (callback.call_args.args[0] is not None) == with_usage


def test_internvl_counts_expanded_inputs_and_continuation_without_changing_output():
    generated = NS(shape=(1, 17))
    model = NS(generate=Mock(return_value=generated))
    generate = model.generate
    counts = install_token_counter(model)
    inputs = NS(shape=(1, 2048))  # Includes expanded visual tokens.
    assert model.generate(input_ids=inputs, do_sample=False) is generated
    generate.assert_called_once_with(input_ids=inputs, do_sample=False)
    assert counts == {"prompt_tokens": 2048, "completion_tokens": 17, "source": "model_token_ids"}
    assert ReflectionResult(True, "ok", 0).token_usage() is None


def test_qverifier_counts_only_generated_suffix_without_loading_a_model():
    reflector = QVerifierReflector.__new__(QVerifierReflector)
    reflector._ensure_loaded = Mock()
    reflector._inference_lock = threading.Lock()
    reflector.max_new_tokens = 256
    reflector._processor = Mock()
    reflector._processor.apply_chat_template.return_value = {"input_ids": np.zeros((1, 123), dtype=int)}
    reflector._processor.decode.return_value = "Yes"
    reflector._model = Mock()
    reflector._model.generate.return_value = np.zeros((1, 130), dtype=int)
    picture = Image.new("RGB", (2, 2))
    result = reflector.verify(picture, picture, JSONAction(action_type="click", x=1, y=1), "Open", "Open")
    assert result.token_usage()["prompt_tokens"] == 123
    assert result.token_usage()["completion_tokens"] == 7
    assert result.verdict is True


@pytest.mark.parametrize("has_usage", [True, False])
def test_gpt_reflector_uses_provider_usage_and_preserves_missing(has_usage):
    reflector = GPTReflector.__new__(GPTReflector)
    reflector.protocol = "legacy_shared"
    reflector.uses_history = False
    reflector.image_max_side = 512
    reflector.model = "fake"
    reflector.max_tokens = 256
    reflector.min_latency_ms = 0
    reflector.client = Mock()
    reply = response(content="Yes")
    if not has_usage:
        reply.usage = None
    reflector.client.chat.completions.create.return_value = reply
    picture = Image.new("RGB", (2, 2))
    result = reflector.verify(picture, picture, JSONAction(action_type="click", x=1, y=1), "Open", "Open")
    if has_usage:
        assert result.token_usage()["prompt_tokens"] == 100
        assert result.token_usage()["cached_tokens"] == 40
        assert result.token_usage()["completion_tokens"] == 20
    else:
        assert result.token_usage() is None


def test_all_tasks_retries_and_reflector_usage_survive_trajectory_backups(tmp_path):
    first = TrajLogger(str(tmp_path), "success")
    first.log_model_token_usage("actor", {"prompt_tokens": 100, "completion_tokens": 10})
    first.reset_traj()  # Device retry must not lose or duplicate earlier usage.
    first.log_model_token_usage("actor", {"prompt_tokens": 200, "completion_tokens": 20})
    # A later task retry creates a fresh agent/logger and backs up the directory.
    second = TrajLogger(str(tmp_path), "success")
    second.log_model_token_usage("actor", {"prompt_tokens": 300, "completion_tokens": 30})
    second.log_model_token_usage("reflector", {
        "prompt_tokens": 20, "completion_tokens": 8, "reasoning_tokens": 5,
    })
    failed = TrajLogger(str(tmp_path), "failed")
    failed.log_model_token_usage("actor", {"prompt_tokens": 100, "completion_tokens": 10})
    summary = write_token_usage_report(str(tmp_path), ["success", "failed"], True)
    assert summary["total_tasks"] == 2
    assert summary["actor"]["mean_prompt_tokens_per_task"] == 350
    assert summary["reflector"]["mean_prompt_tokens_per_task"] == 10
    assert summary["combined"]["mean_prompt_tokens_per_task"] == 360
    assert summary["combined"]["mean_completion_tokens_per_task"] == 39
    assert len((tmp_path / "token_usage_by_task.csv").read_text().splitlines()) == 3


def test_missing_call_or_missing_task_blocks_combined_average(tmp_path):
    recorder = TaskTokenRecorder(str(tmp_path), "measured")
    recorder.record("actor", {"prompt_tokens": 100, "completion_tokens": 5})
    recorder.record("reflector", None)
    summary = summarize_token_usage(str(tmp_path), ["measured"], True)["summary"]
    assert summary["actor"]["mean_prompt_tokens_per_task"] == 100
    assert summary["reflector"]["prompt_tokens"] is None
    assert summary["combined"]["prompt_tokens"] is None
    summary = summarize_token_usage(str(tmp_path), ["measured", "not_started"], True)["summary"]
    assert summary["actor"]["complete_tasks"] == 1
    assert summary["actor"]["mean_prompt_tokens_per_task"] is None


def test_legacy_logs_are_not_silently_treated_as_free(tmp_path):
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "traj.json").write_text(json.dumps({"0": {"token_usage": {"prompt_tokens": 10}}}))
    summary = summarize_token_usage(str(tmp_path), ["old"], True)["summary"]
    assert summary["combined"]["prompt_tokens"] is None
    recorder = TaskTokenRecorder(str(tmp_path), "old")
    recorder.record("actor", {"prompt_tokens": 20, "completion_tokens": 2})
    summary = summarize_token_usage(str(tmp_path), ["old"], True)["summary"]
    assert summary["actor"]["known_prompt_tokens"] == 20
    assert summary["actor"]["prompt_tokens"] is None  # Earlier calls were unmetered.


def test_parallel_tasks_write_valid_independent_counts(tmp_path):
    def run(task):
        recorder = TaskTokenRecorder(str(tmp_path), task)
        for _ in range(20):
            recorder.record("actor", {"prompt_tokens": 10, "completion_tokens": 1})

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, ["a", "b"]))
    summary = summarize_token_usage(str(tmp_path), ["a", "b"], False)["summary"]
    assert summary["actor"]["reported_calls"] == 40
    assert summary["combined"]["mean_prompt_tokens_per_task"] == 200
    assert summary["reflector"]["mean_prompt_tokens_per_task"] == 0
