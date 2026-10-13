"""Exercise real task execution with mocked models/environment, without GPU/API calls."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from mobile_world.core import runner
from mobile_world.reflection.base import ReflectionResult
from mobile_world.runtime.utils.models import JSONAction


@pytest.fixture
def execute_task():
    # Also verify the exact frozen runner used by each experiment.
    source = Path(os.environ.get("RUNNER_SOURCE_UNDER_TEST", runner.__file__))
    tree = ast.parse(source.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "_execute_single_task"]
    assert len(body) == 2
    namespace = dict(vars(runner))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), namespace)
    return namespace["_execute_single_task"]


def task_fixture(verdicts, finish=False):
    obs = SimpleNamespace(screenshot=Image.new("RGB", (20, 40)), tool_call=None, ask_user_response=None)
    env = Mock()
    env.get_task_goal.return_value = "Send the message"
    env.initialize_task.return_value = obs
    env.execute_action.return_value = obs
    env.get_task_score.return_value = (1.0, "Task evaluator decides success")
    agent = Mock()
    agent.get_total_token_usage.return_value = {}
    actions = [(f"<thinking>Perform step {i}</thinking>", JSONAction(action_type="click", x=i, y=1))
               for i in range(len(verdicts))]
    if finish:
        actions.append(("Task complete", JSONAction(action_type=runner.FINISHED)))
    agent.predict.side_effect = actions
    reflector = Mock(uses_history=False)
    reflector.verify.side_effect = [ReflectionResult(v, "Evidence for this action", 1,
        "Yes" if v else "No", 100, 10, token_usage_source="mock") for v in verdicts]
    return env, agent, reflector, Mock()


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_no_feedback_continues_after_three_nos_until_budget(execute_task, explicit_zero):
    env, agent, reflector, log = task_fixture([False] * 6)
    options = {"max_consecutive_failures": 0} if explicit_zero else {}
    steps, score, stats = execute_task(env, agent, "MockTask", 6, log,
        reflector=reflector, reflection_policy="retry", **options)
    assert steps == 6 and score == 1.0 and not stats["early_termination_triggered"]
    assert stats["total_no"] == 6 and stats["retry_count"] == 5
    observations = [c.args[0] for c in agent.predict.call_args_list]
    assert "feedback" not in observations[0]
    assert all(f'attempt="{i}"' in observations[i]["feedback"] for i in range(1, 6))
    assert stats["total_prompt_tokens"] == 600 and stats["total_completion_tokens"] == 60
    assert log.log_reflection.call_count == 6
    assert not any(c.kwargs["triggered_early_stop"] for c in log.log_reflection.call_args_list)
    env.get_task_score.assert_called_once()
    env.tear_down_task.assert_called_once()


def test_explicit_threshold_reproduces_legacy_stopping(execute_task):
    env, agent, reflector, log = task_fixture([False] * 6)
    steps, score, stats = execute_task(env, agent, "MockTask", 6, log,
        reflector=reflector, reflection_policy="retry", max_consecutive_failures=3)
    assert steps == 3 and stats["early_termination_triggered"] and stats["total_no"] == 3
    assert score == 1.0  # Stopping must never replace the environment's score with failure.
    assert log.log_reflection.call_args.kwargs["triggered_early_stop"]


def test_yes_resets_consecutive_no_count(execute_task):
    env, agent, reflector, log = task_fixture([False, False, True, False, False, False])
    steps, _, stats = execute_task(env, agent, "MockTask", 6, log,
        reflector=reflector, reflection_policy="retry", max_consecutive_failures=3)
    assert steps == 6 and stats["early_termination_triggered"]
    observations = [c.args[0] for c in agent.predict.call_args_list]
    assert "feedback" not in observations[3] and 'attempt="1"' in observations[4]["feedback"]


def test_actor_can_finish_after_three_nos(execute_task):
    env, agent, reflector, log = task_fixture([False] * 3, finish=True)
    steps, _, stats = execute_task(env, agent, "MockTask", 50, log,
        reflector=reflector, reflection_policy="retry", max_consecutive_failures=0)
    assert steps == 4 and not stats["early_termination_triggered"]
    assert reflector.verify.call_count == 3 and env.execute_action.call_count == 3


def test_negative_threshold_rejected_before_environment_access(execute_task):
    env, agent, reflector, log = task_fixture([False])
    with pytest.raises(ValueError, match="must be >= 0"):
        execute_task(env, agent, "MockTask", 50, log,
            reflector=reflector, reflection_policy="retry", max_consecutive_failures=-1)
    env.initialize_task.assert_not_called()
