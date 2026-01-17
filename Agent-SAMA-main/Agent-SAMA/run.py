from infer import run_task, REASONING_MODEL
from Agents.screen_parser import ScreenParser
from constant import DEFAULT_PERCEPTION_ARGS, ADB_PATH, INIT_TIPS, INIT_SHORTCUTS
from AgentsUtils.tool_utils import reset_to_home
import torch
import os
import json
import shutil
import pickle
import time

# Use environment variable for ADB_PATH, fallback to constant
adb_path = os.environ.get("ADB_PATH", ADB_PATH)


def prepare_log_paths(args):
    """
    Prepare persistent paths for cross-task knowledge transfer.
    
    Returns:
        Tuple of (tips_path, shortcuts_path, graph_path, patterns_path)
    """
    run_log_dir = f"{args.log_root}/{args.run_name}"
    os.makedirs(run_log_dir, exist_ok=True)

    if args.setting == "single":
        return None, None, None, None

    persistent_tips_path = os.path.join(run_log_dir, "persistent_tips.txt")
    persistent_shortcuts_path = os.path.join(run_log_dir, "persistent_shortcuts.json")
    persistent_graph_path = os.path.join(run_log_dir, "graph.pkl")
    persistent_patterns_path = os.path.join(run_log_dir, "persistent_patterns.json")  # FSM patterns

    if args.specified_tips_path:
        shutil.copy(args.specified_tips_path, persistent_tips_path)
    elif not os.path.exists(persistent_tips_path):
        with open(persistent_tips_path, "w") as f:
            f.write(INIT_TIPS)

    if args.specified_shortcuts_path:
        shutil.copy(args.specified_shortcuts_path, persistent_shortcuts_path)
    elif not os.path.exists(persistent_shortcuts_path):
        with open(persistent_shortcuts_path, "w") as f:
            json.dump(INIT_SHORTCUTS, f, indent=4)

    if args.specified_graph_path:
        shutil.copy(args.specified_graph_path, persistent_graph_path)
    elif not os.path.exists(persistent_graph_path):
        with open(persistent_graph_path, "wb") as f:
            pickle.dump({}, f)

    # Initialize FSM patterns file if needed
    if args.specified_patterns_path:
        shutil.copy(args.specified_patterns_path, persistent_patterns_path)
    elif not os.path.exists(persistent_patterns_path):
        with open(persistent_patterns_path, "w") as f:
            json.dump({"learned_navigation_patterns": [], "common_flows": {}}, f, indent=4)

    return persistent_tips_path, persistent_shortcuts_path, persistent_graph_path, persistent_patterns_path


def run_single_task(args):
    run_task(
        args.instruction,
        run_name=args.run_name,
        log_root=args.log_root,
        tips_path=args.specified_tips_path,
        shortcuts_path=args.specified_shortcuts_path,
        graph_path=args.specified_graph_path,
        patterns_path=args.specified_patterns_path,  # FSM patterns
        persistent_tips_path=None,
        persistent_shortcuts_path=None,
        persistent_graph_path=None,
        persistent_patterns_path=None,
        screen_parser=None,
        perception_args=DEFAULT_PERCEPTION_ARGS,
        max_itr=args.max_itr,
        max_consecutive_failures=args.max_consecutive_failures,
        max_repetitive_actions=args.max_repetitive_actions,
        overwrite_log_dir=args.overwrite_task_log_dir,
        temperature=args.temperature,
        collect_trajectory=args.collect_trajectory,  # Trajectory collection
        trajectory_output_dir=args.trajectory_output_dir
    )


def run_multiple_tasks(args):
    with open(args.tasks_json, "r") as f:
        task_json = json.load(f)
    tasks = task_json.get("tasks", task_json)

    screen_parser = ScreenParser(adb_path, perception_args=DEFAULT_PERCEPTION_ARGS)
    persistent_paths = prepare_log_paths(args)
    # Unpack persistent paths: (tips, shortcuts, graphs, patterns)
    persistent_tips_path, persistent_shortcuts_path, persistent_graph_path, persistent_patterns_path = persistent_paths

    error_tasks = []
    print(f"INFO: Running tasks from {args.tasks_json} using {args.setting} setting ...")
    print(f"INFO: Persistent patterns path: {persistent_patterns_path}")

    for i, task in enumerate(tasks):
        instruction = task["instruction"]
        future_tasks = [t['instruction'] for t in tasks[i + 1:]]
        task_id = task.get("task_id", f"{args.tasks_json.split('/')[-1].split('.')[0]}_{args.setting}_{i}")

        try:
            run_task(
                instruction,
                future_tasks=future_tasks,
                run_name=args.run_name,
                log_root=args.log_root,
                task_id=task_id,
                tips_path=args.specified_tips_path,
                shortcuts_path=args.specified_shortcuts_path,
                graph_path=args.specified_graph_path,
                patterns_path=args.specified_patterns_path,  # FSM patterns (single task init)
                persistent_tips_path=persistent_tips_path,
                persistent_shortcuts_path=persistent_shortcuts_path,
                persistent_graph_path=persistent_graph_path,
                persistent_patterns_path=persistent_patterns_path,  # FSM patterns (cross-task)
                screen_parser=screen_parser,
                perception_args=DEFAULT_PERCEPTION_ARGS,
                max_itr=args.max_itr,
                max_consecutive_failures=args.max_consecutive_failures,
                max_repetitive_actions=args.max_repetitive_actions,
                overwrite_log_dir=args.overwrite_task_log_dir,
                temperature=args.temperature,
                collect_trajectory=args.collect_trajectory,  # Trajectory collection
                trajectory_output_dir=args.trajectory_output_dir
            )
            print(f"\n\nDONE: {instruction}")
            # Automatically reset to home screen and close all apps before next task
            print("Resetting phone to home screen and closing all apps...")
            reset_to_home(adb_path, close_apps=True, wait_time=5.0)
            print("Ready for next task.\n")
        except Exception as e:
            print(f"Failed when doing task: {instruction}\nERROR: {e}")
            error_tasks.append(task_id)

    with open(f"{args.log_root}/{args.run_name}/error_tasks.json", "w") as f:
        json.dump(error_tasks, f, indent=4)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--tasks_json", type=str, default=None)
    parser.add_argument("--specified_tips_path", type=str, default=None)
    parser.add_argument("--specified_shortcuts_path", type=str, default=None)
    parser.add_argument("--specified_graph_path", type=str, default=None)
    parser.add_argument("--specified_patterns_path", type=str, default=None, 
                        help="Path to FSM patterns JSON file for cross-task learning")
    parser.add_argument("--setting", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--max_itr", type=int, default=40)
    parser.add_argument("--max_consecutive_failures", type=int, default=5)
    parser.add_argument("--max_repetitive_actions", type=int, default=5)
    parser.add_argument("--overwrite_task_log_dir", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    
    # Trajectory data collection
    parser.add_argument("--collect_trajectory", action="store_true", default=False,
                        help="Enable trajectory data collection for training reflection models")
    parser.add_argument("--trajectory_output_dir", type=str, default=None,
                        help="Custom output directory for trajectory data (default: log_dir/trajectory)")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.log_root is None:
        args.log_root = f"logs/{REASONING_MODEL}/Agent-SAMA"

    if args.instruction and args.tasks_json:
        raise ValueError("You cannot provide both instruction and tasks_json.")
    elif not args.instruction and not args.tasks_json:
        raise ValueError("You must provide either instruction or tasks_json.")

    if args.instruction:
        run_single_task(args)
    else:
        run_multiple_tasks(args)


if __name__ == "__main__":
    main()
