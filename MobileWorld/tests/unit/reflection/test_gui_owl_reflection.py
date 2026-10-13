"""GUI-Owl's actual prompt/runner/accounting paths, without GPU or network calls."""

import argparse
import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from PIL import Image

from mobile_world.agents.implementations.gui_owl_1_5 import GUIOWL15AgentMCP
from mobile_world.agents.utils.prompts import (
    GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE,
    GUI_OWL_1_5_USER_PROMPT_TEMPLATE,
)
from mobile_world.core import runner
from mobile_world.core.subcommands import eval as eval_cli
from mobile_world.reflection.base import ReflectionResult, extract_subgoal
from mobile_world.reflection.factory import create_reflector
from mobile_world.reflection.format import REFLECTION_PROMPT_ADDENDUM
from mobile_world.runtime.utils.token_usage import summarize_token_usage
from mobile_world.runtime.utils.trajectory_logger import TrajLogger


PREDICTION = 'Action: Open Settings.\n<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>'
HINT = '<reflection_hint attempt="1">The Settings app did not open.</reflection_hint>'


def observation(**extra):
    return {"screenshot": Image.new("RGB", (1080, 2400)), **extra}


@pytest.fixture
def make_agent():
    agents = []

    def make(history_n=1, runtime_conf=None):
        config = {"history_n": history_n} if runtime_conf is None else runtime_conf
        agent = GUIOWL15AgentMCP("GUI-Owl-1.5-8B-Instruct", "http://unused.invalid/v1",
                                tools=[], runtime_conf=config)
        agents.append(agent)
        messages = []

        def complete(**kwargs):
            messages.append(copy.deepcopy(kwargs["messages"]))
            return NS(choices=[NS(message=NS(content=PREDICTION))],
                      usage=NS(prompt_tokens=100, completion_tokens=20))

        agent.openai_client.chat.completions.create = Mock(side_effect=complete)
        agent.initialize("Open Settings")
        return agent, messages

    yield make
    for agent in agents:
        agent.openai_client.close()


def user_text(messages):
    return "\n".join(b.get("text", "") for m in messages if m["role"] == "user"
                     for b in m["content"])


def test_no_reflection_preserves_native_prompt_and_action(make_agent):
    agent, messages = make_agent()
    prediction, action = agent.predict(observation(reflection_enabled=False))
    assert prediction == PREDICTION
    assert action.action_type == "click" and (action.x, action.y) == (541, 601)
    assert messages[0][0] == {"role": "system", "content": GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE.render(tools="")}
    assert messages[0][1]["content"][0]["text"] == GUI_OWL_1_5_USER_PROMPT_TEMPLATE.format(instruction="Open Settings")
    assert [b["type"] for b in messages[0][1]["content"]] == ["text", "image_url"]
    assert "reflection_hint" not in json.dumps(messages)
    assert agent.openai_client.max_retries == 0


@pytest.mark.parametrize("history_n", [1, 3])
@pytest.mark.parametrize("position", ["before", "after"])
def test_hint_on_current_turn_only_across_history_windows(make_agent, history_n, position):
    agent, messages = make_agent(history_n)
    agent.predict(observation(reflection_enabled=True))
    assert REFLECTION_PROMPT_ADDENDUM in messages[0][0]["content"]
    assert "width=1080, height=2400" in messages[0][0]["content"]
    assert "0-999" in messages[0][0]["content"]
    for _ in range(3):
        agent.predict(observation())
    agent.predict(observation(feedback=HINT, feedback_position=position))
    current = messages[-1]
    assert user_text(current).count(HINT) == 1
    assert HINT not in user_text(current[:-1])
    content = current[-1]["content"]
    hint_idx = next(i for i,b in enumerate(content) if b.get("text") == HINT)
    image_idx = next(i for i,b in enumerate(content) if b["type"] == "image_url")
    assert (hint_idx < image_idx) == (position == "before")
    agent.predict(observation())
    assert HINT not in user_text(messages[-1])
    assert HINT not in repr(agent.history_user_content)
    agent.initialize("A different task")
    agent.predict(observation())
    assert "Reflection Hints" not in messages[-1][0]["content"]
    assert "Previous actions" not in user_text(messages[-1])


def test_parallel_agent_configuration_is_not_consumed(make_agent):
    config = {"history_n": 3, "temperature": 0.2, "max_tokens": 1024}
    first, _ = make_agent(runtime_conf=config)
    second, _ = make_agent(runtime_conf=config)
    assert config == {"history_n": 3, "temperature": 0.2, "max_tokens": 1024}
    assert first.history_n == second.history_n == 3
    assert first.temperature == second.temperature == 0.2


@pytest.mark.parametrize("prediction,intent", [
    (PREDICTION, "Open Settings."),
    ('Action: "Go back"\n<tool_call>{}</tool_call>', "Go back"),
    ('<thinking>Open settings</thinking><tool_call>{}</tool_call>', "Open settings"),
    ('Thought: Find Settings\nAction: {"action_type":"click"}', "Find Settings"),
])
def test_action_intent_extraction_does_not_leak_tool_json(prediction, intent):
    assert extract_subgoal(prediction) == intent


@pytest.mark.parametrize("verdicts,early_stop", [([False, True, False], False), ([False, False, False], True)])
def test_real_runner_delivers_feedback_and_records_both_roles(make_agent, tmp_path, verdicts, early_stop):
    agent, messages = make_agent(history_n=2)
    images = [Image.new("RGB", (20, 40), color) for color in ["black", "red", "blue", "green"]]
    observations = [NS(screenshot=img, tool_call=None, ask_user_response=None) for img in images]
    env = Mock()
    env.get_task_goal.return_value = "Open Settings"
    env.initialize_task.return_value = observations[0]
    env.execute_action.side_effect = observations[1:]
    env.get_task_score.return_value = (0, "synthetic environment")
    reflector = Mock()
    reflector.verify.side_effect = [ReflectionResult(v, "Observed local effect", 1, "Yes" if v else "No",
                                                      30, 5, token_usage_source="fake_api_usage") for v in verdicts]
    logger = TrajLogger(str(tmp_path), "FakeTask")
    agent.token_usage_callback = lambda usage: logger.log_model_token_usage("actor", usage)
    steps, _, stats = runner._execute_single_task(env, agent, "FakeTask", 3, logger, max_consecutive_failures=3,
                                                  reflector=reflector, reflection_policy="retry")
    assert steps == 3 and stats["early_termination_triggered"] == early_stop
    assert "<reflection_hint" not in user_text(messages[0])
    assert "<reflection_hint" in user_text(messages[1])
    assert ("<reflection_hint" in user_text(messages[2])) == (not verdicts[1])
    for i, call in enumerate(reflector.verify.call_args_list):
        assert call.kwargs["before"] is images[i] and call.kwargs["after"] is images[i+1]
        assert call.kwargs["subgoal"] == "Open Settings."
        assert call.kwargs["global_task"] == "Open Settings"
    report = summarize_token_usage(str(tmp_path), ["FakeTask"], True)["summary"]
    assert report["actor"]["reported_calls"] == report["reflector"]["reported_calls"] == 3
    assert report["combined"]["mean_prompt_tokens_per_task"] == 390
    assert report["combined"]["mean_completion_tokens_per_task"] == 75


def test_cli_selects_haiku_without_changing_host(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    eval_cli.configure_parser(parser.add_subparsers())
    args = parser.parse_args(["eval", "--agent-type", "gui_owl_1_5", "--model-name", "owl-host",
                             "--llm-base-url", "http://owl.invalid/v1", "--task", "FakeTask",
                             "--reflection-mode", "anthropic_retry", "--reflection-anthropic-model", "haiku-test",
                             "--reflection-anthropic-env-file", str(tmp_path/'key.env')])
    run = Mock(return_value=([], []))
    monkeypatch.setattr(eval_cli, "run_agent_with_evaluation", run)
    asyncio.run(eval_cli.execute(args))
    kwargs = run.call_args.kwargs
    assert kwargs["model_name"] == "owl-host" and kwargs["agent_type"] == "gui_owl_1_5"
    assert kwargs["reflection_anthropic_model"] == "haiku-test"
    assert kwargs["reflection_anthropic_env_file"] == str(tmp_path/'key.env')


def test_runner_routes_haiku_credentials_and_ledger(tmp_path, monkeypatch):
    reflector = Mock()
    factory = Mock(return_value=(reflector, "retry"))
    monkeypatch.setattr(runner, "create_reflector", factory)
    monkeypatch.setattr(runner, "_init_env", Mock(side_effect=RuntimeError("stop before any environment access")))
    with pytest.raises(RuntimeError, match="stop before"):
        runner.run_agent_with_evaluation("gui_owl_1_5", "owl-host", "http://owl.invalid/v1", str(tmp_path),
                                         ["FakeTask"], aw_urls=["http://unused.invalid"], max_concurrency=1,
                                         reflection_mode="anthropic_retry", reflection_anthropic_model="haiku-test",
                                         reflection_anthropic_env_file="key.env")
    assert factory.call_args.kwargs["anthropic_model"] == "haiku-test"
    assert factory.call_args.kwargs["anthropic_env_file"] == "key.env"
    assert factory.call_args.kwargs["anthropic_ledger_path"] == str(tmp_path/'reflector_provider_usage.jsonl')
    reflector.close.assert_called_once()


def test_factory_selects_haiku_with_environment_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    reflector, policy = create_reflector("anthropic_retry", anthropic_ledger_path=str(tmp_path/'usage.jsonl'))
    assert reflector.model == "claude-haiku-4-5-20251001" and policy == "retry"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="key is not configured"):
        create_reflector("anthropic_retry", anthropic_ledger_path=str(tmp_path/'usage2.jsonl'))
    assert create_reflector("none") == (None, "none")


def test_parse_retries_keep_all_actor_token_usage(make_agent, tmp_path):
    agent, _ = make_agent()
    agent.openai_client.chat.completions.create.side_effect = [
        NS(choices=[NS(message=NS(content=text))], usage=NS(prompt_tokens=100, completion_tokens=20))
        for text in ["Malformed action response", PREDICTION]
    ]
    logger = TrajLogger(str(tmp_path), "FakeTask")
    agent.token_usage_callback = lambda usage: logger.log_model_token_usage("actor", usage)
    _, action = agent.predict(observation())
    assert action.action_type == "click"
    usage = summarize_token_usage(str(tmp_path), ["FakeTask"], False)["summary"]
    assert usage["actor"]["reported_calls"] == 2
    assert usage["combined"]["prompt_tokens"] == 200
    assert usage["combined"]["completion_tokens"] == 40
