import json
import os
import random
import threading
import time
from pathlib import Path
from queue import Queue

from dotenv import load_dotenv
from joblib import Parallel, delayed
from loguru import logger

from mobile_world.agents.base import BaseAgent, MCPAgent
from mobile_world.agents.registry import create_agent
from mobile_world.reflection import (
    BaseReflector,
    create_reflector,
    extract_subgoal,
    format_reflection_hint,
    is_verifiable,
)
from mobile_world.reflection.protocol import HISTORY_STEPS, action_context
from mobile_world.runtime.client import (
    AndroidEnvClient,
    AndroidMCPEnvClient,
    scan_finished_tasks,
)
from mobile_world.runtime.utils.docker import (
    discover_backends,
)
from mobile_world.runtime.utils.models import ANSWER, ENV_FAIL, FINISHED, UNKNOWN
from mobile_world.runtime.utils.trajectory_logger import TrajLogger

load_dotenv()


def _execute_single_task(
    env: AndroidEnvClient,
    agent: BaseAgent,
    task_name: str,
    max_step: int,
    traj_logger: TrajLogger,
    enable_mcp: bool = False,
    reflector: BaseReflector | None = None,
    reflection_policy: str = "none",
    max_consecutive_failures: int = 0,
    feedback_position: str = "after",
    feedback_include_alignment: bool = False,
) -> tuple[int, float, dict]:
    """Execute a single task.

    Returns:
        tuple[int, float, dict]: (number of steps, score, reflection_stats)
    """

    if max_consecutive_failures < 0:
        raise ValueError("max_consecutive_failures must be >= 0; 0 disables early termination")
    logger.debug(f"max_step: {max_step}")

    if enable_mcp and not isinstance(agent, MCPAgent):
        logger.error(
            "MCP is enabled but agent type is not a MCP agent. Please use a MCP agent type."
        )

    if enable_mcp:
        traj_logger.log_tools(env.tools)
    task_goal = env.get_task_goal(task_type=task_name)

    logger.debug(f"task_goal: {task_goal}")

    reflection_stats: dict = {
        "mode_policy": reflection_policy,
        "total_verification_calls": 0,
        "total_yes": 0,
        "total_no": 0,
        "total_unknown": 0,
        "early_termination_triggered": False,
        "retry_count": 0,
        "total_verification_latency_ms": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_reasoning_tokens": 0,
        "total_cached_prompt_tokens": 0,
        "token_usage_reported_calls": 0,
        "token_usage_missing_calls": 0,
    }
    consecutive_failures = 0
    pending_feedback: str | None = None
    # Per-task state, never stored on the reflector shared by parallel workers.
    reflector_history: list[dict] = []
    uses_history = getattr(reflector, "uses_history", False) is True

    step = 0
    obs = env.initialize_task(task_name=task_name)
    agent.initialize(task_goal)

    while True:
        step += 1

        logger.debug(f"Screenshot captured in step {step}")

        observation = {
            "screenshot": obs.screenshot,
            "tool_call": obs.tool_call,
            "ask_user_response": obs.ask_user_response,
            # Tell the agent that hints may arrive — so the system prompt
            # advertises the channel from step 1 (not only after the first NO).
            "reflection_enabled": reflector is not None and reflection_policy == "retry",
        }
        if pending_feedback:
            observation["feedback"] = pending_feedback
            observation["feedback_position"] = feedback_position
            reflection_stats["retry_count"] += 1
            logger.info(f"[reflection] injecting hint at step {step}")
            pending_feedback = None

        calls_before = agent.get_total_token_usage().get("calls")
        try:
            prediction, action = agent.predict(observation)
        finally:
            # Custom agents that bypass the shared API wrapper are unmetered.
            if calls_before == agent.get_total_token_usage().get("calls"):
                traj_logger.log_model_token_usage("actor", None)
        traj_logger.log_traj(
            task_name,
            task_goal,
            step,
            prediction,
            action.model_dump(exclude_none=True),
            obs,
            agent.get_total_token_usage(),
        )
        if prediction is None:
            logger.warning(f"Agent prediction failed in step {step}")
            break

        terminate = False
        logger.debug(f"current step {step}")

        if action.action_type in [ENV_FAIL, FINISHED, UNKNOWN]:
            logger.debug(f"task terminated in step {step} with action {action.action_type}")
            terminate = True
        elif action.action_type in [ANSWER]:
            logger.debug(f"answer triggered, execution action {action}")
            obs = env.execute_action(action)
            terminate = True
        else:
            logger.debug(f"execution action {action}")
            before_screenshot = obs.screenshot
            obs = env.execute_action(action)

            if reflector is not None and is_verifiable(action):
                subgoal = extract_subgoal(prediction)
                try:
                    result = reflector.verify(
                        before=before_screenshot,
                        after=obs.screenshot,
                        action=action,
                        subgoal=subgoal,
                        global_task=task_goal,
                        **({"history": tuple(reflector_history)} if uses_history else {}),
                    )
                except Exception:
                    traj_logger.log_model_token_usage("reflector", None)
                    raise
                traj_logger.log_model_token_usage("reflector", result.token_usage())
                reflection_stats["total_verification_calls"] += 1
                reflection_stats["total_verification_latency_ms"] += result.latency_ms
                reflection_stats["total_prompt_tokens"] += result.prompt_tokens
                reflection_stats["total_completion_tokens"] += result.completion_tokens
                reflection_stats["total_reasoning_tokens"] += result.reasoning_tokens
                reflection_stats["total_cached_prompt_tokens"] += result.cached_prompt_tokens
                usage_key = "token_usage_reported_calls" if result.token_usage() is not None else "token_usage_missing_calls"
                reflection_stats[usage_key] += 1

                triggered_retry = False
                triggered_early_stop = False

                if result.verdict is None:
                    reflection_stats["total_unknown"] += 1
                    logger.warning(f"[reflection] step {step} verification unavailable/invalid")
                    # Continue without a hint. No synthetic Yes/No or streak reset.
                elif result.verdict:
                    reflection_stats["total_yes"] += 1
                    consecutive_failures = 0
                    logger.info(
                        f"[reflection] step {step} verdict=YES ({result.latency_ms:.0f}ms)"
                    )
                else:
                    reflection_stats["total_no"] += 1
                    consecutive_failures += 1
                    logger.info(
                        f"[reflection] step {step} verdict=NO "
                        f"(consec={consecutive_failures}, {result.latency_ms:.0f}ms): "
                        f"{result.reasoning[:200]}"
                    )
                    # Inject a structured hint on every NO so the agent gets a
                    # chance to correct on the very next turn. (retry policy.)
                    if reflection_policy == "retry":
                        pending_feedback = format_reflection_hint(
                            attempt=consecutive_failures,
                            prev_action=action,
                            verifier_reasoning=result.reasoning,
                            include_alignment=feedback_include_alignment,
                            feedback_format="evidence" if uses_history else "legacy",
                        )
                        triggered_retry = True
                    # Optional stopping-policy ablation. Zero keeps No feedback
                    # active and lets the actor use the normal task-step budget.
                    if max_consecutive_failures > 0 and consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"[reflection] early termination at step {step}: "
                            f"{consecutive_failures} consecutive failures"
                        )
                        reflection_stats["early_termination_triggered"] = True
                        triggered_early_stop = True
                        terminate = True

                traj_logger.log_reflection(
                    step=step,
                    verdict=result.verdict,
                    reasoning=result.reasoning,
                    latency_ms=result.latency_ms,
                    triggered_retry=triggered_retry,
                    triggered_early_stop=triggered_early_stop,
                    raw_output=result.raw_output,
                    token_usage=result.token_usage(),
                )
            if uses_history:
                reflector_history.append({"step": step, "action": action_context(action, extract_subgoal(prediction))})
                reflector_history = reflector_history[-HISTORY_STEPS:]
        if terminate:
            break

        if step >= max_step:
            logger.debug("task steps reach max step, terminate")
            break

    score, reason = env.get_task_score(task_type=task_name)
    logger.debug(f"task_score: {score}, reason: {reason}")
    traj_logger.log_score(score=score, reason=reason)
    if reflector is not None:
        traj_logger.log_reflection_stats(reflection_stats)

    res = env.tear_down_task(task_type=task_name)
    agent.done()
    logger.debug(f"tear_down_task response: {res}")

    return step, score, reflection_stats


def _process_task_on_env(
    task_name: str,
    env_queue: Queue,
    agent_type: str,
    model_name: str,
    llm_base_url: str,
    api_key: str | None,
    log_file_root: str,
    max_step: int,
    retry_on_device_unhealthy: int = 5,
    enable_mcp: bool = False,
    reflector: BaseReflector | None = None,
    reflection_policy: str = "none",
    reflection_max_consecutive_failures: int = 0,
    reflection_feedback_position: str = "after",
    reflection_feedback_include_alignment: bool = False,
    **kwargs,
) -> dict:
    """Process a single task on a specific environment.

    Args:
        task_name: Name of the task to execute
        env_url: URL of the environment to use
        agent_type: Type of agent to create
        model_name: Model name for the agent
        llm_base_url: LLM service base URL
        api_key: API key for LLM service
        log_file_root: Root directory for log files
        max_step: Maximum steps for task execution
        **kwargs: Additional kwargs for agent creation

    Returns:
        dict: Task result containing task_name, success, score, steps, duration_seconds
    """
    # Create thread-specific log file
    thread_id = threading.current_thread().ident
    thread_log_file = os.path.join(log_file_root, task_name, f"thread_{thread_id}.log")
    os.makedirs(os.path.dirname(thread_log_file), exist_ok=True)
    traj_logger = TrajLogger(log_file_root, task_name)

    def thread_filter(record):
        return record["extra"].get("thread_id") == thread_id

    thread_handler_id = logger.add(
        thread_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | container: {extra[container_name]} | {message}",
        level="DEBUG",
        enqueue=True,
        filter=thread_filter,
    )
    env, container_name = env_queue.get()

    try:
        with logger.contextualize(thread_id=thread_id, container_name=container_name):
            logger.info("Processing task '{}' on environment {}", task_name, env.base_url)
            if enable_mcp:
                assert isinstance(env, AndroidMCPEnvClient), (
                    f"env must be a AndroidMCPEnvClient, but got {type(env)}"
                )
                try:
                    env.reset_tools(task_type=task_name)
                except Exception as e:
                    logger.exception(f"Error resetting tools for task {task_name}: {e}")
                    return None

            agent = create_agent(agent_type, model_name, llm_base_url, api_key, env=env, **kwargs)
            agent.token_usage_callback = lambda usage: traj_logger.log_model_token_usage("actor", usage)

            task_start_time = time.time()
            while True:
                try:
                    task_steps, task_score, reflection_stats = _execute_single_task(
                        env,
                        agent,
                        task_name,
                        max_step,
                        traj_logger=traj_logger,
                        enable_mcp=enable_mcp,
                        reflector=reflector,
                        reflection_policy=reflection_policy,
                        max_consecutive_failures=reflection_max_consecutive_failures,
                        feedback_position=reflection_feedback_position,
                        feedback_include_alignment=reflection_feedback_include_alignment,
                    )
                    break
                except Exception as e:
                    err_msg = str(e)
                    # Treat "500 Server Error" / "Internal Server Error" the
                    # same as "Device is not healthy" — both mean the broker
                    # is in trouble and might recover with a wait. Without
                    # this, a single 500 from a half-stuck broker drops the
                    # task into tasks_with_no_results immediately.
                    transient = (
                        "Device is not healthy" in err_msg
                        or "500 Server Error" in err_msg
                        or "Internal Server Error" in err_msg
                    )
                    if transient and retry_on_device_unhealthy > 0:
                        logger.warning(
                            "Transient broker error ({}), retrying after 30s...",
                            err_msg[:120],
                        )
                        time.sleep(30)
                        retry_on_device_unhealthy -= 1
                        traj_logger.reset_traj()
                        continue
                    else:
                        logger.exception(f"Error executing task {task_name}")
                        return None

            task_duration = time.time() - task_start_time
            task_success = task_score > 0.0

            logger.info(
                "Task '{}' completed on {}: success={}, score={}, steps={}, duration={:.1f}s",
                task_name,
                env.base_url,
                task_success,
                task_score,
                task_steps,
                task_duration,
            )

            return {
                "task_name": task_name,
                "score": task_score,
                "reflection_stats": reflection_stats,
            }
    finally:
        # Remove the thread-specific handler
        logger.remove(thread_handler_id)
        env_queue.put((env, container_name))


def _init_env(
    env_url: str, device: str, step_wait_time: float, suite_family: str, enable_mcp: bool
) -> AndroidEnvClient:
    """Initialize the environment."""
    if enable_mcp:
        env = AndroidMCPEnvClient(env_url, device, step_wait_time=step_wait_time)
    else:
        env = AndroidEnvClient(env_url, device, step_wait_time=step_wait_time)
    env.switch_suite_family(suite_family)
    return env


def run_agent_with_evaluation(
    agent_type: str,
    model_name: str,
    llm_base_url: str,
    log_file_root: str,
    tasks: list[str],
    max_step: int = -1,
    aw_urls: list[str] | None = None,
    api_key: str | None = None,
    device: str = "emulator-5554",
    step_wait_time: float = 1.0,
    suite_family: str = "mobile_world",
    env_name_prefix: str = "mobile_world_env",
    env_image: str = "mobile_world",
    dry_run: bool = False,
    enable_mcp: bool = False,
    enable_user_interaction: bool = False,
    max_concurrency: int | None = None,
    shuffle_tasks: bool = False,
    auto_retry: int = 10,
    reflection_mode: str = "none",
    reflection_gpt_model: str = "gpt-4o",
    reflection_gpt_base_url: str | None = None,
    reflection_gpt_api_key: str | None = None,
    reflection_qverifier_path: str | None = None,
    reflection_gui_model_path: str | None = None,
    reflection_gui_python: str | None = None,
    reflection_gui_code_path: str | None = None,
    reflection_gui_gpu_uuid: str | None = None,
    reflection_gui_prompt: str = "shared",
    reflection_max_consecutive_failures: int = 0,
    reflection_feedback_position: str = "after",
    reflection_feedback_include_alignment: bool = False,
    reflection_anthropic_model: str = "claude-haiku-4-5-20251001",
    reflection_anthropic_env_file: str | None = None,
    reflection_protocol: str = "legacy_shared",
    reflection_qreflector_gpu_uuid: str | None = None,
    reflection_qreflector_python: str | None = None,
    **kwargs,
) -> list[dict]:
    """Run the agent and return the evaluation results.

    Args:
        agent_type: Type of agent to use
        model_name: Model name for the agent
        llm_base_url: LLM service base URL
        log_file_root: Root directory for log files
        tasks: List of task names to execute (empty list for all tasks)
        max_step: Maximum steps for task execution
        aw_urls: List of Android World backend URLs. If None, auto-discover from containers
        api_key: API key for LLM service
        device: Android device ID
        step_wait_time: Wait time after each step
        suite_family: Suite family to use
        **kwargs: Additional kwargs for agent creation

    Returns:
        list[dict]: The evaluation results for each task, containing task_name, success, score, steps, duration_seconds, env_url
    """

    if reflection_max_consecutive_failures < 0:
        raise ValueError("reflection_max_consecutive_failures must be >= 0; 0 disables early termination")
    container_names = None
    if aw_urls is None or len(aw_urls) == 0:
        logger.info("No backend URLs specified, auto-discovering from containers...")
        aw_urls, container_names = discover_backends(image_filter=env_image, prefix=env_name_prefix)
        logger.info("Container names: {}", container_names)
        if not aw_urls:
            logger.error("No backend URLs found. Please start containers or specify --aw-host")
            return [], []

    logger.info("Using {} backend URL(s): {}", len(aw_urls), aw_urls)

    reflector, reflection_policy = create_reflector(
        mode=reflection_mode,
        qverifier_path=reflection_qverifier_path,
        gpt_model=reflection_gpt_model,
        gpt_base_url=reflection_gpt_base_url,
        gpt_api_key=reflection_gpt_api_key,
        gui_model_path=reflection_gui_model_path,
        gui_python=reflection_gui_python,
        gui_code_path=reflection_gui_code_path,
        gui_gpu_uuid=reflection_gui_gpu_uuid,
        gui_prompt=reflection_gui_prompt,
        anthropic_model=reflection_anthropic_model,
        anthropic_env_file=reflection_anthropic_env_file,
        protocol=reflection_protocol,
        qreflector_gpu_uuid=reflection_qreflector_gpu_uuid,
        qreflector_python=reflection_qreflector_python,
        anthropic_ledger_path=(
            os.path.join(log_file_root, "reflector_provider_usage.jsonl")
            if reflection_mode == "anthropic_retry" else None
        ),
    )
    logger.info(
        "Reflection: mode={} policy={} reflector={}",
        reflection_mode,
        reflection_policy,
        type(reflector).__name__ if reflector is not None else "None",
    )
    if reflection_mode == "gui_reflection_retry":
        logger.info("GUI-Reflection configuration: {}", reflector.configuration)

    try:
        configuration = getattr(reflector, "configuration", None)
        if isinstance(configuration, dict):
            configuration = {**configuration, "mode": reflection_mode, "policy": reflection_policy,
                             "max_consecutive_failures": reflection_max_consecutive_failures,
                             "feedback_position": reflection_feedback_position,
                             "feedback_include_alignment": reflection_feedback_include_alignment}
            config_path = Path(log_file_root) / "reflector_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists() and json.loads(config_path.read_text()) != configuration:
                raise ValueError("Run directory contains a different reflector configuration; use a fresh directory")
            config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n")
        envs = Parallel(
            n_jobs=min(max_concurrency if max_concurrency is not None else len(aw_urls), len(aw_urls)),
            backend="threading",
        )(
            delayed(_init_env)(env_url, device, step_wait_time, suite_family, enable_mcp)
            for env_url in aw_urls
        )

        if len(tasks) != 0:
            task_list = tasks
        else:
            task_list = envs[0].get_suite_task_list(enable_mcp=enable_mcp, enable_user_interaction=enable_user_interaction)

        logger.info("Task list: {} ({} tasks)", task_list, len(task_list))

        num_envs = len(envs)
        max_attempts = min(1 + auto_retry, 10)  # Cap at 10 to prevent infinite loops
        reflection_stats_by_task: dict[str, dict] = {}

        for attempt in range(max_attempts):
            # Scan finished tasks each iteration (picks up results from previous attempts)
            finished_task_list, finished_scores = scan_finished_tasks(log_file_root, task_list)
            logger.info("Finished task list: {} ({} tasks)", finished_task_list, len(finished_task_list))

            pending_tasks = [task for task in task_list if task not in finished_task_list]
            logger.info(
                "Attempt {}/{}: {} remaining tasks to execute",
                attempt + 1, max_attempts, len(pending_tasks),
            )

            if not pending_tasks:
                logger.info("All tasks finished, no retry needed")
                break

            env_queue = Queue[tuple[AndroidEnvClient, str | None]](maxsize=num_envs)
            for i, env in enumerate(envs):
                env_queue.put((env, container_names[i] if container_names else None))

            if shuffle_tasks:
                random.shuffle(pending_tasks)

            if not dry_run:
                task_results = Parallel(
                    n_jobs=min(max_concurrency if max_concurrency is not None else num_envs, num_envs),
                    backend="threading",
                )(
                    delayed(_process_task_on_env)(
                        task_name=task_name,
                        env_queue=env_queue,
                        agent_type=agent_type,
                        model_name=model_name,
                        llm_base_url=llm_base_url,
                        api_key=api_key,
                        log_file_root=log_file_root,
                        max_step=max_step,
                        enable_mcp=enable_mcp,
                        reflector=reflector,
                        reflection_policy=reflection_policy,
                        reflection_max_consecutive_failures=reflection_max_consecutive_failures,
                        reflection_feedback_position=reflection_feedback_position,
                        reflection_feedback_include_alignment=reflection_feedback_include_alignment,
                        **kwargs,
                    )
                    for task_name in pending_tasks
                )
            else:
                logger.info("Dry run mode, skipping task execution")
                task_results = []
                break

            # Capture reflection stats from this round's results before they are discarded.
            for task_name, task_result in zip(pending_tasks, task_results):
                if task_result is not None and "reflection_stats" in task_result:
                    reflection_stats_by_task[task_name] = task_result["reflection_stats"]

            # Identify failed tasks for potential retry
            failed_this_round = [
                task_name for task_name, task_result in zip(pending_tasks, task_results)
                if task_result is None
            ]

            logger.info(
                "Attempt {}/{} done: {} succeeded, {} failed/stale",
                attempt + 1, max_attempts,
                len(pending_tasks) - len(failed_this_round), len(failed_this_round),
            )

            if not failed_this_round or attempt >= max_attempts - 1:
                break

            logger.info("Auto-retrying {} failed tasks (retry {}/{})", len(failed_this_round), attempt + 1, auto_retry)

        # Final scan to get all finished results (including from retries)
        finished_task_list, finished_scores = scan_finished_tasks(log_file_root, task_list)
        # Build final results from scan (authoritative source); merge in reflection
        # stats captured during this run.
        success_task_results = []
        for task_name, score in zip(finished_task_list, finished_scores):
            entry = {"task_name": task_name, "score": score}
            if task_name in reflection_stats_by_task:
                entry["reflection_stats"] = reflection_stats_by_task[task_name]
            success_task_results.append(entry)

        task_list_with_no_results = [task for task in task_list if task not in finished_task_list]
        logger.info(f"Final: {len(success_task_results)} tasks with results, {len(task_list_with_no_results)} with no results")

        if not dry_run:
            from mobile_world.runtime.utils.token_usage import write_token_usage_report

            write_token_usage_report(log_file_root, task_list, reflector is not None)

        return (success_task_results, task_list_with_no_results)
    finally:
        if reflector is not None:
            reflector.close()
