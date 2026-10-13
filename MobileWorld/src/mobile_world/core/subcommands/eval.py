"""Eval subcommand for MobileWorld CLI - Run benchmark evaluation suite."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mobile_world.reflection import REFLECTION_MODES
from mobile_world.reflection.protocol import PROTOCOLS

from ..runner import run_agent_with_evaluation


# Per-1M-token USD prices for API-backed reflectors.
# `completion_tokens` from OpenAI already includes reasoning tokens for reasoning
# models, so do NOT double-count reasoning_tokens when computing cost.
# Cached input is billed at a discount (~50%) when the API reports cache hits.
# Update this table when new GPT models are added or prices change.
GPT_PRICING_PER_M_TOKENS = {
    "gpt-5.2": {"input": 1.25, "cached_input": 0.625, "output": 10.00},
    "gpt-5.2-2025-12-11": {"input": 1.25, "cached_input": 0.625, "output": 10.00},
    "gpt-5.2-pro": {"input": 15.00, "cached_input": 7.50, "output": 120.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.625, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.625, "output": 10.00},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
}


def _compute_cost_usd(model: str, prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost given a model id and token counts. 0.0 if model unknown."""
    prices = GPT_PRICING_PER_M_TOKENS.get(model)
    if not prices:
        return 0.0
    fresh_input = max(prompt_tokens - cached_prompt_tokens, 0)
    return (
        fresh_input * prices["input"]
        + cached_prompt_tokens * prices["cached_input"]
        + completion_tokens * prices["output"]
    ) / 1_000_000


def _aggregate_reflection_stats(task_results: list[dict], model: str | None = None) -> dict:
    """Sum per-task reflection stats into a run-level summary."""
    agg = {
        "tasks_with_reflection": 0,
        "total_verification_calls": 0,
        "total_yes": 0,
        "total_no": 0,
        "total_unknown": 0,
        "total_retry_count": 0,
        "early_termination_tasks": 0,
        "total_verification_latency_ms": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_reasoning_tokens": 0,
        "total_cached_prompt_tokens": 0,
        "token_usage_reported_calls": 0,
        "token_usage_missing_calls": 0,
    }
    for result in task_results:
        stats = result.get("reflection_stats")
        if not stats:
            continue
        agg["tasks_with_reflection"] += 1
        agg["total_verification_calls"] += stats.get("total_verification_calls", 0)
        agg["total_yes"] += stats.get("total_yes", 0)
        agg["total_no"] += stats.get("total_no", 0)
        agg["total_unknown"] += stats.get("total_unknown", 0)
        agg["total_retry_count"] += stats.get("retry_count", 0)
        if stats.get("early_termination_triggered"):
            agg["early_termination_tasks"] += 1
        agg["total_verification_latency_ms"] += stats.get("total_verification_latency_ms", 0.0)
        agg["total_prompt_tokens"] += stats.get("total_prompt_tokens", 0)
        agg["total_completion_tokens"] += stats.get("total_completion_tokens", 0)
        agg["total_reasoning_tokens"] += stats.get("total_reasoning_tokens", 0)
        agg["total_cached_prompt_tokens"] += stats.get("total_cached_prompt_tokens", 0)
        reported = stats.get("token_usage_reported_calls", 0)
        agg["token_usage_reported_calls"] += reported
        agg["token_usage_missing_calls"] += stats.get("total_verification_calls", 0) - reported
    if agg["token_usage_missing_calls"]:
        # Older local-reflector logs used zero placeholders; never present those
        # (or partial measurements) as a complete total or a zero-dollar cost.
        agg["total_prompt_tokens"] = None
        agg["total_completion_tokens"] = None
    if model:
        agg["model"] = model
        agg["total_cost_usd"] = None if agg["token_usage_missing_calls"] else _compute_cost_usd(
            model,
            agg["total_prompt_tokens"],
            agg["total_cached_prompt_tokens"],
            agg["total_completion_tokens"],
        )
    return agg


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between eval and test commands."""
    parser.add_argument(
        "--agent-type",
        "--agent_type",
        required=True,
        dest="agent_type",
        help="Type of agent to use (registered name or path to Python file containing agent class)",
    )
    parser.add_argument("--model-name", "--model_name", dest="model_name", help="Model name to use")
    parser.add_argument(
        "--llm-base-url",
        "--llm_base_url",
        dest="llm_base_url",
        help="LLM service base URL",
    )
    parser.add_argument(
        "--api-key",
        "--api_key",
        dest="api_key",
        help="API key for LLM service",
    )
    parser.add_argument(
        "--log-file-root",
        "--log_file_root",
        dest="log_file_root",
        help="Root directory for log files",
    )
    parser.add_argument(
        "--max-round",
        "--max_round",
        "--max-step",
        "--max_step",
        dest="max_round",
        type=int,
        help="Maximum number of steps (-1 for unlimited)",
    )
    parser.add_argument(
        "--aw-host", "--aw_host", dest="aw_host", help="Android World server host", default=None
    )
    parser.add_argument("--timeout", type=int, help="Task timeout in seconds")
    parser.add_argument("--output", dest="output", help="Output directory for results")

    # Executor settings for planner-executor agents
    parser.add_argument(
        "--executor-llm-base-url",
        "--executor_llm_base_url",
        dest="executor_llm_base_url",
        help="Executor LLM service base URL",
    )
    parser.add_argument(
        "--executor-model-name",
        "--executor_model_name",
        dest="executor_model_name",
        help="Executor model name",
    )
    parser.add_argument(
        "--executor-agent-class",
        "--executor_agent_class",
        dest="executor_agent_class",
        help="Executor agent class name",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="Android device ID (default: get via adb devices)",
    )
    parser.add_argument(
        "--step-wait-time",
        "--step_wait_time",
        dest="step_wait_time",
        type=float,
        default=1.0,
        help="Wait time in seconds after each step (default: 1.0)",
    )
    parser.add_argument(
        "--suite-family",
        "--suite_family",
        dest="suite_family",
        choices=["mobile_world"],
        default="mobile_world",
        help="Suite family to use (default: mobile_world)",
    )
    parser.add_argument(
        "--env-name-prefix",
        "--env_name_prefix",
        "--env-prefix",
        "--env_prefix",
        dest="env_name_prefix",
        default="mobile_world_env",
        help="Name prefix for containers (default: mobile_world_env)",
    )
    parser.add_argument(
        "--env-image",
        "--env_image",
        dest="env_image",
        default="mobile_world",
        help="Image name for containers (default: mobile_world)",
    )
    parser.add_argument(
        "--enable-mcp",
        "--enable_mcp",
        dest="enable_mcp",
        action="store_true",
        help="Enable MCP server",
    )
    parser.add_argument(
        "--enable-user-interaction",
        "--enable_user_interaction",
        dest="enable_user_interaction",
        action="store_true",
        help="Enable user interaction tasks (agent-user-interaction). Default: only GUI-only tasks",
    )
    parser.add_argument(
        "--scale-factor",
        "--scale_factor",
        dest="scale_factor",
        type=int,
        default=1000,
        help="Scale factor for coordinate conversion (default: 1000)",
    )


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    """Configure the eval subcommand parser."""
    # Create eval parser with 'run' as an alias for backward compatibility
    eval_parser = subparsers.add_parser(
        "eval",
        aliases=["run"],
        help="Run benchmark evaluation suite",
    )

    _add_common_arguments(eval_parser)

    # Eval-specific arguments
    eval_parser.add_argument(
        "--task",
        "--tasks",
        dest="task",
        help='Specific task(s) to run (comma-separated) or "ALL" to run all tasks and generate statistics',
    )
    eval_parser.add_argument(
        "--auto-retry",
        "--auto_retry",
        dest="auto_retry",
        type=int,
        default=10,
        help="Number of automatic retry rounds for failed/stale tasks (default: 10)",
    )
    eval_parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Dry run the command, print final results only without executing tasks",
    )
    eval_parser.add_argument(
        "--max-concurrency",
        "--max_concurrency",
        dest="max_concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent tasks to run, Note: min(max_concurrency, number of tasks, number of docker envs)",
    )
    eval_parser.add_argument(
        "--shuffle-tasks",
        "--shuffle_tasks",
        dest="shuffle_tasks",
        action="store_true",
        help="Shuffle the order of tasks before running",
    )

    # Reflection / verifier arguments
    eval_parser.add_argument(
        "--reflection-mode",
        "--reflection_mode",
        dest="reflection_mode",
        choices=REFLECTION_MODES,
        default="none",
        help="Reflection mode. 'none' disables verification (baseline). "
        "'qverifier_early_stop' terminates a task after N consecutive Q-Verifier No verdicts. "
        "'qverifier_retry' / 'qreflector_v9_retry' / 'gpt_retry' / 'gui_reflection_retry' / 'anthropic_retry' inject feedback into the next "
        "observation on a No verdict.",
    )
    eval_parser.add_argument(
        "--reflection-protocol", choices=PROTOCOLS, default="legacy_shared",
        help="legacy_shared preserves archived runs; v9_geometry uses the frozen v9 prompt, "
        "3 prior actions, original/normalized coordinates, 1024px images and 1024 output tokens.",
    )
    eval_parser.add_argument(
        "--reflection-qreflector-gpu-uuid", default=None,
        help="Required physical GPU UUID for qreflector_v9_retry; refuses an occupied GPU.",
    )
    eval_parser.add_argument(
        "--reflection-qreflector-python", default=None,
        help="Isolated BF16 worker Python (default: artifacts/envs/qreflector-native/bin/python).",
    )
    eval_parser.add_argument(
        "--reflection-gpt-model",
        "--reflection_gpt_model",
        dest="reflection_gpt_model",
        default="gpt-4o",
        help="Model name for an OpenAI-compatible reflector (GPT or a local served model; default: gpt-4o)",
    )
    eval_parser.add_argument(
        "--reflection-gpt-base-url",
        "--reflection_gpt_base_url",
        dest="reflection_gpt_base_url",
        default=None,
        help="Base URL for GPT reflector (default: OpenAI)",
    )
    eval_parser.add_argument(
        "--reflection-anthropic-model", "--reflection_anthropic_model",
        default="claude-haiku-4-5-20251001",
        help="Pinned Anthropic reflector model (default: claude-haiku-4-5-20251001)",
    )
    eval_parser.add_argument(
        "--reflection-anthropic-env-file", "--reflection_anthropic_env_file",
        default=None,
        help="Optional file containing ANTHROPIC_API_KEY; otherwise read the environment. "
        "Provider usage is saved separately under the run log directory.",
    )
    eval_parser.add_argument(
        "--reflection-qverifier-path",
        "--reflection_qverifier_path",
        dest="reflection_qverifier_path",
        default=None,
        help="LoRA adapter path. Legacy modes use QVERIFIER_MODEL_PATH/repo default; "
        "qreflector_v9_retry defaults to the selected Sol round-2 final adapter.",
    )
    eval_parser.add_argument(
        "--reflection-gui-model-path", "--reflection_gui_model_path",
        default=None,
        help="GUI-Reflection checkpoint (default: artifacts/models/gui_reflection/GUI_Reflection_8b_pretrain)",
    )
    eval_parser.add_argument(
        "--reflection-gui-python", "--reflection_gui_python",
        default=None,
        help="Isolated InternVL Python (default: artifacts/envs/gui-reflection/bin/python)",
    )
    eval_parser.add_argument(
        "--reflection-gui-code-path", "--reflection_gui_code_path",
        default=None,
        help="Official GUI-Reflection repository (default: artifacts/vendor/GUI_Reflection)",
    )
    eval_parser.add_argument(
        "--reflection-gui-gpu-uuid", "--reflection_gui_gpu_uuid",
        default=None,
        help="Single physical GPU UUID for the isolated worker (default: this server's GPU 0)",
    )
    eval_parser.add_argument(
        "--reflection-gui-prompt", "--reflection_gui_prompt",
        choices=["shared", "native"], default="shared",
        help="shared: existing Q-Reflector text; native: authors' purpose-only prompt using the "
        "actor's extracted intent. Native requires legacy_shared (512px). Shared follows --reflection-protocol.",
    )
    eval_parser.add_argument(
        "--reflection-max-consecutive-failures",
        "--reflection_max_consecutive_failures",
        dest="reflection_max_consecutive_failures",
        type=int,
        default=0,
        help="Consecutive No verdicts before forced termination; 0 disables this rule (default: 0). "
        "No feedback remains active in retry mode. Set a positive threshold for a stopping-policy ablation.",
    )
    eval_parser.add_argument(
        "--reflection-feedback-position",
        "--reflection_feedback_position",
        dest="reflection_feedback_position",
        choices=["before", "after"],
        default="after",
        help="Where the reflection hint text is placed relative to the screenshot in the next user "
        "turn. 'after' (default) puts it after the image (recency bias); 'before' puts it first.",
    )
    eval_parser.add_argument(
        "--reflection-feedback-include-alignment",
        "--reflection_feedback_include_alignment",
        dest="reflection_feedback_include_alignment",
        action="store_true",
        help="Also include the verifier's SUBGOAL ALIGNMENT section (sec 4) in the hint. "
        "Default keeps only CHANGE (sec 3) + FINAL (sec 5).",
    )


async def execute(args: argparse.Namespace) -> None:
    """Execute the eval command."""
    log_file_root = args.log_file_root or args.output or "./traj_logs"

    # Check if running all tasks
    run_all_tasks = args.task and args.task.upper() == "ALL"
    if run_all_tasks:
        final_tasks = []
        logger.info("Running ALL tasks with statistics generation")
    else:
        final_tasks = args.task.split(",") if args.task else []

    start_time = time.time() if run_all_tasks else None

    # Parse aw_host URLs - if None, will auto-discover; if provided, split by comma
    aw_urls = None if args.aw_host is None else args.aw_host.split(",")

    task_results, task_list_with_no_results = run_agent_with_evaluation(
        agent_type=args.agent_type,
        model_name=args.model_name,
        llm_base_url=args.llm_base_url,
        log_file_root=log_file_root,
        tasks=final_tasks,
        max_step=args.max_round or -1,
        aw_urls=aw_urls,
        api_key=args.api_key or os.getenv("API_KEY"),
        executor_llm_base_url=args.executor_llm_base_url,
        executor_model_name=args.executor_model_name,
        executor_agent_class=args.executor_agent_class,
        device=args.device or "emulator-5554",
        step_wait_time=args.step_wait_time or 1.0,
        suite_family=args.suite_family or "mobile_world",
        env_name_prefix=args.env_name_prefix,
        env_image=args.env_image,
        dry_run=args.dry_run,
        enable_mcp=args.enable_mcp,
        enable_user_interaction=args.enable_user_interaction,
        max_concurrency=args.max_concurrency,
        shuffle_tasks=args.shuffle_tasks,
        scale_factor=getattr(args, "scale_factor", 1000),
        auto_retry=args.auto_retry,
        reflection_mode=args.reflection_mode,
        reflection_protocol=args.reflection_protocol,
        reflection_qreflector_gpu_uuid=args.reflection_qreflector_gpu_uuid,
        reflection_qreflector_python=args.reflection_qreflector_python,
        reflection_gpt_model=args.reflection_gpt_model,
        reflection_gpt_base_url=args.reflection_gpt_base_url,
        reflection_anthropic_model=args.reflection_anthropic_model,
        reflection_anthropic_env_file=args.reflection_anthropic_env_file,
        reflection_qverifier_path=args.reflection_qverifier_path,
        reflection_gui_model_path=args.reflection_gui_model_path,
        reflection_gui_python=args.reflection_gui_python,
        reflection_gui_code_path=args.reflection_gui_code_path,
        reflection_gui_gpu_uuid=args.reflection_gui_gpu_uuid,
        reflection_gui_prompt=args.reflection_gui_prompt,
        reflection_max_consecutive_failures=args.reflection_max_consecutive_failures,
        reflection_feedback_position=args.reflection_feedback_position,
        reflection_feedback_include_alignment=args.reflection_feedback_include_alignment,
    )
    if run_all_tasks and task_results:
        total_duration = time.time() - start_time

        total_tasks = len(task_results)

        successful_tasks = sum(1 for result in task_results if result["score"] > 0.99)
        overall_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        reflection_summary = _aggregate_reflection_stats(
            task_results,
            model=args.reflection_gpt_model if args.reflection_mode == "gpt_retry" else None,
        )

        report = {
            "summary": {
                "total_tasks_assigned": total_tasks + len(task_list_with_no_results),
                "total_tasks_with_results": total_tasks,
                "successful_tasks": successful_tasks,
                "total_tasks_with_no_results": len(task_list_with_no_results),
                "overall_success_rate": overall_success_rate,
                "total_duration_seconds": total_duration,
            },
            "reflection_summary": reflection_summary,
            "metadata": {
                "agent_type": args.agent_type,
                "model_name": args.model_name,
                "reflection_mode": args.reflection_mode,
                "reflection_protocol": args.reflection_protocol,
                "reflector_model": (
                    args.reflection_anthropic_model if args.reflection_mode == "anthropic_retry"
                    else args.reflection_gpt_model if args.reflection_mode == "gpt_retry" else None
                ),
                "timestamp": datetime.now().isoformat(),
                "log_file_root": log_file_root,
            },
            "tasks_with_results": task_results,
            "tasks_with_no_results": task_list_with_no_results,
        }

        token_report = Path(log_file_root) / "token_usage_summary.json"
        if token_report.exists():
            report["token_usage_summary"] = json.loads(token_report.read_text())["summary"]

        reflector_config = Path(log_file_root) / "reflector_config.json"
        if reflector_config.exists():
            report["metadata"]["reflector_configuration"] = json.loads(reflector_config.read_text())

        if args.reflection_mode == "gui_reflection_retry":
            from mobile_world.reflection.gui_reflection import GUIReflectionReflector

            # Construction is lazy; record resolved defaults without starting a worker.
            report["metadata"]["gui_reflection"] = GUIReflectionReflector(
                model_path=args.reflection_gui_model_path,
                python_path=args.reflection_gui_python,
                official_code=args.reflection_gui_code_path,
                gpu_uuid=args.reflection_gui_gpu_uuid,
                prompt_mode=args.reflection_gui_prompt,
                protocol=args.reflection_protocol,
            ).configuration

        output_path = Path(log_file_root)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"eval_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Pretty print results using Rich
        console = Console()

        # Create summary panel
        summary_text = Text()
        summary_text.append("Evaluation Complete!\n\n", style="bold green")
        summary_text.append(f"Overall Success Rate: {overall_success_rate:.1%}\n", style="cyan")
        summary_text.append(
            f"Successful Tasks: {successful_tasks}/{total_tasks}\n", style="magenta"
        )
        summary_text.append(f"Total Duration: {total_duration:.1f} seconds\n", style="yellow")

        summary_panel = Panel(
            summary_text,
            title="[bold blue]📊 Evaluation Summary",
            border_style="blue",
            padding=(1, 2),
        )

        console.print(summary_panel)

        # Create detailed stats table
        stats_table = Table(
            title="[bold]📈 Detailed Statistics", show_header=True, header_style="bold blue"
        )
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", style="magenta", justify="right")

        stats_table.add_row("Total Tasks Assigned", str(report["summary"]["total_tasks_assigned"]))
        stats_table.add_row(
            "Tasks with Results", str(report["summary"]["total_tasks_with_results"])
        )
        stats_table.add_row("Successful Tasks", str(report["summary"]["successful_tasks"]))
        stats_table.add_row(
            "Tasks with No Results", str(report["summary"]["total_tasks_with_no_results"])
        )
        stats_table.add_row("Success Rate", f"{report['summary']['overall_success_rate']:.1%}")

        console.print(stats_table)

        # Create metadata panel
        metadata_text = Text()
        metadata_text.append(f"Agent Type: {report['metadata']['agent_type']}\n", style="green")
        metadata_text.append(f"Model: {report['metadata']['model_name'] or 'N/A'}\n", style="green")
        metadata_text.append(f"Timestamp: {report['metadata']['timestamp']}\n", style="green")
        metadata_text.append(f"Log Root: {report['metadata']['log_file_root']}\n", style="green")

        metadata_panel = Panel(
            metadata_text, title="[bold]🔧 Configuration", border_style="green", padding=(1, 2)
        )

        console.print(metadata_panel)

        # Show task results if available
        if task_results:
            results_table = Table(
                title="[bold]📋 Task Results", show_header=True, header_style="bold magenta"
            )
            results_table.add_column("Task", style="cyan", width=30)
            results_table.add_column("Score", style="green", justify="center")
            results_table.add_column("Status", style="yellow", justify="center")

            for result in task_results:
                status = "✅ Success" if result["score"] > 0.99 else "❌ Failed"
                status_style = "green" if result["score"] > 0.99 else "red"
                results_table.add_row(
                    result.get("task_name", "Unknown"),
                    f"{result['score']:.3f}",
                    f"[{status_style}]{status}[/{status_style}]",
                )

            console.print(results_table)

        # Show tasks with no results if any
        if task_list_with_no_results:
            no_results_text = Text()
            no_results_text.append("Tasks with no results:\n", style="bold red")
            for task in task_list_with_no_results[:5]:  # Show first 5
                no_results_text.append(f"• {task}\n", style="red")
            if len(task_list_with_no_results) > 5:
                no_results_text.append(
                    f"... and {len(task_list_with_no_results) - 5} more", style="red"
                )

            no_results_panel = Panel(
                no_results_text,
                title="[bold red]⚠️  Tasks with No Results",
                border_style="red",
                padding=(1, 2),
            )
            console.print(no_results_panel)

        # File locations panel
        files_text = Text()
        files_text.append(f"Results JSON: {report_file}\n", style="blue")
        files_text.append(f"Trajectory Logs: {log_file_root}", style="blue")

        files_panel = Panel(
            files_text, title="[bold]💾 Output Files", border_style="cyan", padding=(1, 2)
        )

        console.print(files_panel)
