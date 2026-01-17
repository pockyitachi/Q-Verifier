import os
import json
import copy
import pickle
import time
import shutil
from Agents.info_pool import InfoPool
from Agents.planner import Planner
from Agents.reflection_agent import ReflectionAgent
from Agents.actor import Actor
from Agents.mentor import Mentor
from Agents.state_agent import StateAgent
from Agents.screen_parser import ScreenParser
from AgentsUtils.chat_utils import inference_chat, add_response, add_response_with_images
from AgentsUtils.common_utils import serialize_graph
from AgentsUtils.trajectory_collector import TrajectoryCollector, create_transition_from_info_pool
from AgentsUtils.tool_utils import get_screenshot
from dataclasses import asdict
from PIL import Image
from time import sleep
from constant import (ADB_PATH, BACKBONE_TYPE, QWEN_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY,
                      OPENAI_API_URL, GEMINI_API_URL, CLAUDE_API_URL, USAGE_TRACKING_JSONL, DEFAULT_PERCEPTION_ARGS,
                      INIT_SHORTCUTS, INIT_TIPS, SCREENSHOT_DIR, TEMP_DIR, LLM_Judge, ATOMIC_ACTION_SIGNATURES,
                      MAX_RECOVERY_RETRIES, SLEEP_BETWEEN_STEPS)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Settings
adb_path = os.environ.get("ADB_PATH", default=ADB_PATH)
backbone_type = os.environ.get("BACKBONE_TYPE", default=BACKBONE_TYPE)

assert BACKBONE_TYPE in ["OpenAI", "Gemini", "Claude"], "Unknown BACKBONE_TYPE"
print("### Using BACKBONE_TYPE:", backbone_type)

openai_key = os.environ.get("OPENAI_API_KEY", default=OPENAI_API_KEY)
gemini_key = os.environ.get("GEMINI_API_KEY", default=GEMINI_API_KEY)
claude_key = os.environ.get("CLAUDE_API_KEY", default=CLAUDE_API_KEY)

if BACKBONE_TYPE == "OpenAI":
    REASONING_MODEL = "gpt-4o-2024-11-20"
    KNOWLEDGE_REFLECTION_MODEL = "gpt-4o-2024-11-20"
elif BACKBONE_TYPE == "Gemini":
    REASONING_MODEL = "gemini-1.5-pro-latest"
    KNOWLEDGE_REFLECTION_MODEL = "gemini-1.5-pro-latest"
elif BACKBONE_TYPE == "Claude":
    REASONING_MODEL = "claude-3-5-sonnet-20241022"
    KNOWLEDGE_REFLECTION_MODEL = "claude-3-5-sonnet-20241022"

qwen_key = os.environ.get("QWEN_API_KEY", default=QWEN_API_KEY)


# Helper function to detect task completion with multiple keywords
def is_task_finished(current_subgoal: str) -> bool:
    """
    Check if the current subgoal indicates TASK completion (not subgoal completion).
    
    We need to distinguish between:
    - "Tap the search icon" (action subgoal, NOT finished)
    - "Step 1 done, now do step 2" (progress note, NOT finished)  
    - "Finished" or "Task completed" (task IS finished)
    
    Strategy: Only trigger on short, definitive finish statements.
    """
    subgoal_clean = current_subgoal.strip().lower()
    # Remove common punctuation for comparison
    subgoal_clean = subgoal_clean.rstrip('.!,;:')
    
    # Exact matches - the subgoal IS essentially just the finish indicator
    exact_finish_phrases = [
        "finished", "done", "complete", "completed",
        "task finished", "task completed", "task done", "task complete",
        "goal achieved", "goal accomplished", "mission accomplished",
        "all done", "all finished", "all complete", "all completed",
        "no further actions required", "no more actions needed",
        "the task is finished", "the task is completed", "the task is done",
        "user instruction completed", "instruction completed",
        "task is complete", "task is done", "task is finished",
    ]
    
    if subgoal_clean in exact_finish_phrases:
        return True
    
    # Also check the original case-sensitive "Finished" for backward compatibility
    # This is what the prompt explicitly asks the LLM to output
    if "Finished" in current_subgoal.strip():
        return True
    
    return False


# When finish tasks
def finish(
        info_pool: InfoPool,
        persistent_tips_path=None,
        persistent_shortcuts_path=None,
        persistent_graph_path=None,
        persistent_patterns_path=None,  # NEW: FSM patterns persistence
):
    print("\n" + "="*60)
    print("TASK COMPLETED")
    print("="*60)
    print("Plan:", info_pool.plan)
    print("\nProgress Logs:")
    for i, p in enumerate(info_pool.progress_status_history):
        print(f"  Step {i}: {p}")
    print("\nImportant Notes:", info_pool.important_notes)
    print("Finish Thought:", info_pool.finish_thought)
    
    # Recovery statistics summary
    if info_pool.recovery_attempts > 0:
        recovery_rate = 100 * info_pool.recovery_successes / info_pool.recovery_attempts
        print(f"\n[Recovery Statistics]")
        print(f"  Attempts: {info_pool.recovery_attempts}")
        print(f"  Successes: {info_pool.recovery_successes}")
        print(f"  Failures: {info_pool.recovery_failures}")
        print(f"  Recovery Rate: {recovery_rate:.1f}%")
    else:
        print("\n[Recovery Statistics] No recovery attempts needed")
    
    # FSM Graph summary
    print(f"\n[FSM Graph Statistics]")
    print(f"  Apps tracked: {len(info_pool.state_graphs)}")
    total_states = sum(g.number_of_nodes() for g in info_pool.state_graphs.values())
    total_transitions = sum(g.number_of_edges() for g in info_pool.state_graphs.values())
    print(f"  Total states: {total_states}")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Cross-app transitions: {len(info_pool.app_transition_history)}")
    for app_name, graph in info_pool.state_graphs.items():
        beacons = [graph.nodes[n].get('beacon', '?') for n in graph.nodes]
        print(f"  [{app_name}] {graph.number_of_nodes()} states: {', '.join(beacons[:4])}{'...' if len(beacons) > 4 else ''}")
    
    # FSM Patterns summary
    print(f"\n[FSM Patterns (Cross-Task Knowledge)]")
    print(f"  Navigation patterns: {len(info_pool.learned_navigation_patterns)}")
    print(f"  Common flows: {len(info_pool.common_flows)}")
    if info_pool.learned_navigation_patterns:
        for pattern in info_pool.learned_navigation_patterns[:3]:
            print(f"    - {pattern.get('pattern_name', '?')} ({pattern.get('app', '?')})")
    if info_pool.common_flows:
        for flow_name in list(info_pool.common_flows.keys())[:3]:
            print(f"    - {flow_name}")
    
    # Persist tips
    if persistent_tips_path:
        print("Update persistent tips:", persistent_tips_path)
        with open(persistent_tips_path, "w") as f:
            f.write(info_pool.tips)
    
    # Persist shortcuts
    if persistent_shortcuts_path:
        print("Update persistent shortcuts:", persistent_shortcuts_path)
        with open(persistent_shortcuts_path, "w") as f:
            json.dump(info_pool.shortcuts, f, indent=4)
    
    # Persist state graphs
    if persistent_graph_path:
        print("Update persistent state graphs:", persistent_graph_path)
        with open(persistent_graph_path, "wb") as f:
            pickle.dump(info_pool.state_graphs, f)
    
    # Persist FSM patterns (NEW)
    if persistent_patterns_path:
        print("Update persistent FSM patterns:", persistent_patterns_path)
        patterns_data = {
            "learned_navigation_patterns": info_pool.learned_navigation_patterns,
            "common_flows": info_pool.common_flows
        }
        with open(persistent_patterns_path, "w") as f:
            json.dump(patterns_data, f, indent=4)


def get_reasoning_model_api_response(chat, model_type=BACKBONE_TYPE, model=None, temperature=0.0):
    # chat messages in openai format
    model = REASONING_MODEL if model is None else model
    if model_type == "OpenAI":
        return inference_chat(chat, model, OPENAI_API_URL, openai_key, usage_tracking_jsonl=USAGE_TRACKING_JSONL,
                              temperature=temperature)
    elif model_type == "Gemini":
        return inference_chat(chat, model, GEMINI_API_URL, gemini_key, usage_tracking_jsonl=USAGE_TRACKING_JSONL,
                              temperature=temperature)
    elif model_type == "Claude":
        return inference_chat(chat, model, CLAUDE_API_URL, claude_key, usage_tracking_jsonl=USAGE_TRACKING_JSONL,
                              temperature=temperature)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_task(
        instruction,
        future_tasks=[],
        run_name="test",
        log_root=f"logs/{REASONING_MODEL}/agent_sama",
        task_id=None,
        tips_path=None,
        shortcuts_path=None,
        graph_path=None,
        patterns_path=None,  # FSM patterns path (single task)
        persistent_tips_path=None,  # cross tasks
        persistent_shortcuts_path=None,  # cross tasks
        persistent_graph_path=None,
        persistent_patterns_path=None,  # cross tasks: FSM navigation patterns
        screen_parser: ScreenParser = None,
        perception_args=DEFAULT_PERCEPTION_ARGS,
        max_itr=40,
        max_consecutive_failures=3,
        max_repetitive_actions=3,
        overwrite_log_dir=False,
        err_to_planner_thresh=2,
        temperature=0.0,
        collect_trajectory=False,  # NEW: Enable trajectory data collection
        trajectory_output_dir=None  # NEW: Custom output directory for trajectory data
):
    ### set up log dir ###
    if task_id is None:
        task_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_root}/{run_name}/{task_id}"
    if os.path.exists(log_dir) and not overwrite_log_dir:
        print("The log dir already exists. And overwrite_log_dir is set to False. Skipping...")
        return
    os.makedirs(f"{log_dir}/screenshots", exist_ok=True)
    os.makedirs(f"{log_dir}/perception", exist_ok=True)
    log_json_path = f"{log_dir}/steps.json"

    # local experience save paths
    local_shortcuts_save_path = f"{log_dir}/shortcuts.json"  # single-task setting
    local_tips_save_path = f"{log_dir}/tips.txt"  # single-task setting
    local_graphs_save_path = f"{log_dir}/graphs.pkl"
    local_patterns_save_path = f"{log_dir}/patterns.json"  # FSM patterns (single-task)

    ### Init Information Pool ###
    if shortcuts_path is not None and persistent_shortcuts_path is not None and shortcuts_path != persistent_shortcuts_path:
        raise ValueError("You cannot specify different shortcuts_path and persistent_shortcuts_path.")
    if tips_path is not None and persistent_tips_path is not None and tips_path != persistent_tips_path:
        raise ValueError("You cannot specify different tips_path and persistent_tips_path.")

    if shortcuts_path:
        initial_shortcuts = json.load(open(shortcuts_path, "r"))  # load agent collected shortcuts
    elif persistent_shortcuts_path:
        initial_shortcuts = json.load(open(persistent_shortcuts_path, "r"))
    else:
        initial_shortcuts = copy.deepcopy(INIT_SHORTCUTS)
    print("INFO: Initial shortcuts:", initial_shortcuts)

    if tips_path:
        tips = open(tips_path, "r").read() # load agent updated tips
    elif persistent_tips_path:
        tips = open(persistent_tips_path, "r").read()
    else:
        tips = copy.deepcopy(INIT_TIPS) # user provided initial tips
    print("INFO: Initial tips:", tips)

    if graph_path:
        with open(graph_path, "rb") as f:
            initial_state_graphs = pickle.load(f)
    elif persistent_graph_path and os.path.exists(persistent_graph_path):
        with open(persistent_graph_path, "rb") as f:
            initial_state_graphs = pickle.load(f)
    else:
        initial_state_graphs = {}
    print("INFO: Initial state graphs:", list(initial_state_graphs.keys()))
    
    # Load FSM patterns (cross-task knowledge transfer)
    initial_navigation_patterns = []
    initial_common_flows = {}
    if patterns_path and os.path.exists(patterns_path):
        with open(patterns_path, "r") as f:
            patterns_data = json.load(f)
            initial_navigation_patterns = patterns_data.get("learned_navigation_patterns", [])
            initial_common_flows = patterns_data.get("common_flows", {})
    elif persistent_patterns_path and os.path.exists(persistent_patterns_path):
        with open(persistent_patterns_path, "r") as f:
            patterns_data = json.load(f)
            initial_navigation_patterns = patterns_data.get("learned_navigation_patterns", [])
            initial_common_flows = patterns_data.get("common_flows", {})
    print(f"INFO: Initial FSM patterns: {len(initial_navigation_patterns)} navigation patterns, {len(initial_common_flows)} common flows")
    
    steps = []
    task_start_time = time.time()

    info_pool = InfoPool(
        instruction=instruction,
        shortcuts=initial_shortcuts,
        tips=tips,
        future_tasks=future_tasks,
        err_to_planner_thresh=err_to_planner_thresh,
        state_graphs=initial_state_graphs,
        learned_navigation_patterns=initial_navigation_patterns,
        common_flows=initial_common_flows
    )
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    else:
        shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)
    if not os.path.exists(SCREENSHOT_DIR):
        os.mkdir(SCREENSHOT_DIR)

    # build agents
    if screen_parser is None:
        screen_parser = ScreenParser(adb_path, perception_args=perception_args)
    planner = Planner()
    actor = Actor(adb_path)
    reflection_agent = ReflectionAgent()
    mentor = Mentor(mode="unified")  # Unified mode: generates both shortcuts and tips
    state_agent = StateAgent(adb_path=adb_path)  # Pass ADB path for reliable app detection
    
    # Initialize trajectory collector (optional)
    trajectory_collector = None
    if collect_trajectory:
        # Default to trajectory_data/ folder in project root
        traj_output = trajectory_output_dir or "trajectory_data"
        trajectory_collector = TrajectoryCollector(output_dir=traj_output, task_id=task_id)
        trajectory_collector.start_task(instruction)
        print(f"[Trajectory] Collecting trajectory data to: {traj_output}")

    with open(local_tips_save_path, "w") as f:
        f.write(tips)
    with open(local_shortcuts_save_path, "w") as f:
        json.dump(initial_shortcuts, f, indent=4)
    with open(local_graphs_save_path, "wb") as f:
        pickle.dump(initial_state_graphs, f)
    with open(local_patterns_save_path, "w") as f:
        json.dump({
            "learned_navigation_patterns": initial_navigation_patterns,
            "common_flows": initial_common_flows
        }, f, indent=4)

    steps.append({
        "step": 0,
        "operation": "init",
        "instruction": instruction,
        "task_id": task_id,
        "run_name": run_name,
        "max_itr": max_itr,
        "max_consecutive_failures": max_consecutive_failures,
        "max_repetitive_actions": max_repetitive_actions,
        "future_tasks": future_tasks,
        "log_root": log_root,
        "tips_path": tips_path,
        "shortcuts_path": shortcuts_path,
        "persistent_tips_path": persistent_tips_path,
        "persistent_shortcuts_path": persistent_shortcuts_path,
        "perception_args": perception_args
    })
    with open(log_json_path, "w") as f:
        json.dump(steps, f, indent=4)

    # Capture initial screenshot from phone before starting the task
    print("\n### Capturing initial screenshot... ###\n")
    get_screenshot(adb_path)
    print("Initial screenshot captured at ./screenshot/screenshot.jpg")

    iter = 0
    while True:
        iter += 1

        ## max iteration stop ##
        if max_itr is not None and iter >= max_itr:
            print("Max iteration reached. Stopping...")
            task_end_time = time.time()
            final_info_pool_dict = asdict(info_pool)
            final_info_pool_dict["state_graphs"] = {
                app: serialize_graph(graph) for app, graph in info_pool.state_graphs.items()
            }
            final_info_pool_dict["beacon_hist"] = list(info_pool.beacon_hist)
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "max_iteration",
                "max_itr": max_itr,
                "final_info_pool": final_info_pool_dict,
                "task_duration": task_end_time - task_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            
            # Save trajectory data even on max iteration
            if trajectory_collector:
                trajectory_collector.save()
                print(f"[Trajectory] Stats: {trajectory_collector.get_stats()}")
            return

        ## consecutive failures stop ##
        if len(info_pool.action_outcomes) >= max_consecutive_failures:
            last_k_action_outcomes = info_pool.action_outcomes[-max_consecutive_failures:]
            err_flags = [1 if outcome in ["B", "C"] else 0 for outcome in last_k_action_outcomes]
            if sum(err_flags) == max_consecutive_failures:
                print("Consecutive failures reaches the limit. Stopping...")
                task_end_time = time.time()
                final_info_pool_dict = asdict(info_pool)
                final_info_pool_dict["state_graphs"] = {
                    app: serialize_graph(graph) for app, graph in info_pool.state_graphs.items()
                }
                final_info_pool_dict["beacon_hist"] = list(info_pool.beacon_hist)
                steps.append({
                    "step": iter,
                    "operation": "finish",
                    "finish_flag": "max_consecutive_failures",
                    "max_consecutive_failures": max_consecutive_failures,
                    "final_info_pool": final_info_pool_dict,
                    "task_duration": task_end_time - task_start_time,
                })
                with open(log_json_path, "w") as f:
                    json.dump(steps, f, indent=4)
                
                # Save trajectory data even on consecutive failures
                if trajectory_collector:
                    trajectory_collector.save()
                    print(f"[Trajectory] Stats: {trajectory_collector.get_stats()}")
                return

        ## max repetitive actions stop ##
        if len(info_pool.action_history) >= max_repetitive_actions:
            last_k_actions = info_pool.action_history[-max_repetitive_actions:]
            last_k_actions_set = set()
            try:
                for act_obj in last_k_actions:
                    if "name" in act_obj:
                        hash_key = act_obj['name']
                    else:
                        hash_key = json.dumps(act_obj)
                    if "arguments" in act_obj:
                        if act_obj['arguments'] is not None:
                            for arg, value in act_obj['arguments'].items():
                                hash_key += f"-{arg}-{value}"
                        else:
                            hash_key += "-None"
                    print("hashable action key:", hash_key)
                    last_k_actions_set.add(hash_key)
            except:
                last_k_actions_set = set()  # not stopping if there is any error
                pass
            if len(last_k_actions_set) == 1:
                repeated_action_key = last_k_actions_set.pop()
                if "Swipe" not in repeated_action_key and "Back" not in repeated_action_key:
                    print("Repetitive actions reaches the limit. Stopping...")
                    task_end_time = time.time()
                    final_info_pool_dict = asdict(info_pool)
                    final_info_pool_dict["state_graphs"] = {
                        app: serialize_graph(graph) for app, graph in info_pool.state_graphs.items()
                    }
                    final_info_pool_dict["beacon_hist"] = list(info_pool.beacon_hist)

                    steps.append({
                        "step": iter,
                        "operation": "finish",
                        "finish_flag": "max_repetitive_actions",
                        "max_repetitive_actions": max_repetitive_actions,
                        "final_info_pool": final_info_pool_dict,
                        "task_duration": task_end_time - task_start_time,
                    })
                    with open(log_json_path, "w") as f:
                        json.dump(steps, f, indent=4)
                    
                    # Save trajectory data even on repetitive actions
                    if trajectory_collector:
                        trajectory_collector.save()
                        print(f"[Trajectory] Stats: {trajectory_collector.get_stats()}")
                    return


        if iter == 1:  # first perception
            screenshot_file = "./screenshot/screenshot.jpg"
            print("\n### Screen Parser ... ###\n")
            perception_start_time = time.time()
            perception_infos, width, height = screen_parser.get_perception_infos(screenshot_file, temp_file=TEMP_DIR)
            shutil.rmtree(TEMP_DIR)
            os.mkdir(TEMP_DIR)

            keyboard = False
            keyboard_height_limit = 0.9 * height
            for perception_info in perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                if 'ADB Keyboard' in perception_info['text']:
                    keyboard = True
                    break

            info_pool.width = width
            info_pool.height = height

            ## log ##
            save_screen_shot_path = f"{log_dir}/screenshots/{iter}.jpg"
            Image.open(screenshot_file).save(save_screen_shot_path)
            info_pool.current_screenshot_path = save_screen_shot_path

            perception_end_time = time.time()
            # Save perception info separately to reduce steps.json size
            perception_save_path = f"{log_dir}/perception/{iter}.json"
            with open(perception_save_path, "w") as f:
                json.dump(perception_infos, f, indent=2)
            steps.append({
                "step": iter,
                "operation": "perception",
                "screenshot": save_screen_shot_path,
                "perception_file": perception_save_path,
                "perception_count": len(perception_infos),
                "duration": perception_end_time - perception_start_time,
            })
            print("Perception Infos:", perception_infos)
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)

        ### get perception infos ###
        info_pool.perception_infos_pre = copy.deepcopy(perception_infos)
        info_pool.keyboard_pre = keyboard

        if iter == 1:  # first state
            print("\n### State Agent First Look... ###\n")
            ### State Agent First Look ###
            info_pool.state_perception_infos = copy.deepcopy(perception_infos)
            # Ensure the current screenshot path is saved into the InfoPool (if not already done)
            state_prompt = state_agent.get_prompt(info_pool)
            # Initialize chat for state analysis
            chat_state = state_agent.init_chat()
            # Append the state prompt with the current screenshot (reusing the saved screenshot)
            chat_state = add_response("user", state_prompt, chat_state, image=info_pool.current_screenshot_path)
            # Get state analysis response from your reasoning model
            output_state = get_reasoning_model_api_response(chat_state, temperature=temperature)
            # Parse the state response
            parsed_state = state_agent.parse_response(output_state)
            state_description = parsed_state.get("state_description", "State description not provided")
            predicted_next_state = parsed_state.get("predicted_next_state", "No prediction provided")
            current_app_involved = parsed_state.get("current_app_name", "None")
            current_state_beacon = parsed_state.get("state_beacon", "Unknown")
            precondition = parsed_state.get("precondition", "Unknown")
            postcondition = parsed_state.get("postcondition", "Unknown")
            print("\nState Description:", state_description)
            print("\nPredicted Next State:", predicted_next_state)
            # create the node of the first state
            state_agent.update_state_graph(info_pool, state_description, current_app_involved, current_state_beacon,
                                       precondition, postcondition, predicted_next_state)
            # Log the state update in your steps
            info_pool.pre_condition = precondition
            info_pool.post_condition = postcondition
            info_pool.state_prediction = predicted_next_state
            # Store the actual state description for use in trajectory from_state
            info_pool._current_state_description = state_description
            info_pool.pre_state_description = state_description
            info_pool.pre_state_pre_condition = precondition
            info_pool.pre_state_post_condition = postcondition
            info_pool.pre_state_state_prediction = predicted_next_state
            info_pool.active_app_name = current_app_involved
            steps.append({
                "step": iter,
                "operation": "First state",
                "state_description": state_description,
                "predicted_next_state": predicted_next_state,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)

        if not info_pool.state_agent_recovery_mode:
            ### Manager: High-level Planning ###
            print("\n### Planner ... ###\n")
            ## check if stuck with errors for a long time ##
            # if so need to think about the high-level plan again
            plan_candidates = []
            single_plan = {}
            info_pool.error_flag_plan = False
            if len(info_pool.action_outcomes) >= err_to_planner_thresh:
                # check if the last err_to_manager_thresh actions are all errors
                latest_outcomes = info_pool.action_outcomes[-err_to_planner_thresh:]
                count = 0
                for outcome in latest_outcomes:
                    if outcome in ["B", "C"]:
                        count += 1
                if count == err_to_planner_thresh:
                    info_pool.error_flag_plan = True
            ##
            info_pool.prev_subgoal = info_pool.current_subgoal

            planning_start_time = time.time()
            prompt_planning = planner.get_prompt(info_pool)
            chat_planning = planner.init_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning, image=screenshot_file)
            output_planning = get_reasoning_model_api_response(chat_planning, temperature=temperature)
            if LLM_Judge:
                plan_candidates = planner.parse_response(output_planning).get("candidates", [])
            else:
                single_plan = planner.parse_response_single(output_planning)

            if len(plan_candidates) > 1:
                prompt_judging = planner.get_prompt_for_judging(info_pool, plan_candidates)
                chat_judging = planner.init_chat_for_judging()
                chat_judging = add_response("user", prompt_judging, chat_judging, image=screenshot_file)
                output_judging = get_reasoning_model_api_response(chat_judging, temperature=temperature)
                parsed_result_selected_planning = planner.parse_judge_response(output_judging)
                info_pool.plan = parsed_result_selected_planning['plan']
                info_pool.current_subgoal = parsed_result_selected_planning['current_subgoal']
                plan_log = parsed_result_selected_planning
            else:
                if LLM_Judge:
                    continuing_planning = plan_candidates[0]
                    info_pool.plan = continuing_planning['plan']
                    info_pool.current_subgoal = continuing_planning['current_subgoal']
                    plan_log = continuing_planning
                else:
                    info_pool.plan = single_plan['plan']
                    info_pool.current_subgoal = single_plan['current_subgoal']
                    plan_log = single_plan

            ## log ##
            planning_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "planning",
                "prompt_planning": prompt_planning,
                "error_flag_plan": info_pool.error_flag_plan,
                "raw_response": output_planning,
                "thought": plan_log['thought'],
                "plan": plan_log['plan'],
                "current_subgoal": plan_log['current_subgoal'],
                "duration": planning_end_time - planning_start_time,
            })
            print("Thought:", plan_log['thought'])
            print("Overall Plan:", info_pool.plan)
            print("Current Subgoal:", info_pool.current_subgoal)

            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)

            ###

            ### Experience Reflection: Update Tips & Shortcuts for Self-Evolving ###
            if len(info_pool.action_outcomes) > 0:
                # at the end of each task, update the tips and shortcuts
                if is_task_finished(info_pool.current_subgoal):
                    print("\n### Mentor Agent: Self-Evolution Reflection ... ###\n")
                    experience_reflection_start_time = time.time()
                    
                    # Unified self-evolution: generates both shortcuts and tips in one call
                    prompt_mentor = mentor.get_prompt(info_pool)
                    chat_mentor = mentor.init_chat()
                    chat_mentor = add_response("user", prompt_mentor, chat_mentor, image=None)
                    output_mentor = get_reasoning_model_api_response(
                        chat_mentor,
                        model=KNOWLEDGE_REFLECTION_MODEL,
                        temperature=temperature
                    )
                    
                    # Process the reflection results
                    reflection_results = mentor.reflect(info_pool, output_mentor)
                    
                    # Extract for logging
                    parsed_mentor = mentor.parse_response(output_mentor)
                    new_shortcut_str = parsed_mentor.get('new_shortcut', 'None')
                    updated_tips = parsed_mentor.get('updated_tips', info_pool.tips)
                    
                    print(f"[Mentor] Shortcuts added: {reflection_results['shortcuts_added']}")
                    print(f"[Mentor] Tips updated: {reflection_results['tips_updated']}")
                    if new_shortcut_str and new_shortcut_str.lower() != 'none':
                        print(f"New Shortcut: {new_shortcut_str[:200]}...")
                    print(f"Updated Tips: {updated_tips[:300]}..." if len(updated_tips) > 300 else f"Updated Tips: {updated_tips}")

                    prompt_knowledge = prompt_mentor
                    output_knowledge = output_mentor

                    experience_reflection_end_time = time.time()
                    steps.append({
                        "step": iter,
                        "operation": "mentor_self_evolution",
                        "prompt_knowledge": prompt_knowledge,
                        "raw_response": output_knowledge,
                        "new_shortcut": new_shortcut_str,
                        "updated_tips": updated_tips,
                        "shortcuts_added": reflection_results['shortcuts_added'],
                        "tips_updated": reflection_results['tips_updated'],
                        "total_shortcuts": len(info_pool.shortcuts),
                        "duration": experience_reflection_end_time - experience_reflection_start_time,
                    })
                    with open(log_json_path, "w") as f:
                        json.dump(steps, f, indent=4)
                    ## save the updated tips and shortcuts ##
                    with open(local_tips_save_path, "w") as f:
                        f.write(info_pool.tips)
                    with open(local_shortcuts_save_path, "w") as f:
                        json.dump(info_pool.shortcuts, f, indent=4)
                    with open(local_graphs_save_path, "wb") as f:
                        pickle.dump(info_pool.state_graphs, f)
                    # Save FSM patterns for cross-task learning
                    with open(local_patterns_save_path, "w") as f:
                        json.dump({
                            "learned_navigation_patterns": info_pool.learned_navigation_patterns,
                            "common_flows": info_pool.common_flows
                        }, f, indent=4)
                    print(f"\n[FSM Patterns] Saved {len(info_pool.learned_navigation_patterns)} patterns, {len(info_pool.common_flows)} flows")
                    # graph_data = json_graph.node_link_data(info_pool.state_graph)
                    print("\nFinal State Graph (JSON):")
                # print(json.dumps(graph_data, indent=4))

            ### Stopping by planner ###
            if is_task_finished(info_pool.current_subgoal):
                info_pool.finish_thought = plan_log['thought']
                task_end_time = time.time()
                final_info_pool_dict = asdict(info_pool)
                final_info_pool_dict["state_graphs"] = {
                    app: serialize_graph(graph) for app, graph in info_pool.state_graphs.items()
                }
                final_info_pool_dict["beacon_hist"] = list(info_pool.beacon_hist)
                steps.append({
                    "step": iter,
                    "operation": "finish",
                    "finish_flag": "success",
                    "final_info_pool": final_info_pool_dict,
                    "task_duration": task_end_time - task_start_time,
                })
                with open(log_json_path, "w") as f:
                    json.dump(steps, f, indent=4)
                
                # Save trajectory data if collecting
                if trajectory_collector:
                    trajectory_collector.save()
                    print(f"[Trajectory] Stats: {trajectory_collector.get_stats()}")
                
                finish(
                    info_pool,
                    persistent_tips_path=persistent_tips_path,
                    persistent_shortcuts_path=persistent_shortcuts_path,
                    persistent_graph_path=persistent_graph_path,
                    persistent_patterns_path=persistent_patterns_path
                )

                return

        ### Executor: Action Decision ###
        print("\n### Actor ... ###\n")
        action_decision_start_time = time.time()
        if info_pool.state_agent_recovery_mode:
            print("\n Processing Recovery Plan")
            prompt_action = actor.get_prompt_recovery(info_pool)
            chat_action = actor.init_chat_recovery()
            chat_action = add_response("user", prompt_action, chat_action, image=screenshot_file)
        else:
            prompt_action = actor.get_prompt(info_pool)
            chat_action = actor.init_chat()
            chat_action = add_response("user", prompt_action, chat_action, image=screenshot_file)
        output_action = get_reasoning_model_api_response(chat_action, temperature=temperature)
        parsed_result_action = actor.parse_response(output_action)
        action_thought, action_object_str, action_description = parsed_result_action['thought'], parsed_result_action[
            'action'], parsed_result_action['description']
        action_decision_end_time = time.time()
        if info_pool.state_agent_recovery_mode:
            info_pool.current_recovery_action = action_object_str
            info_pool.current_recovery_thought = action_thought
        info_pool.last_action_thought = action_thought
        
        ## IMPORTANT: Save current state info BEFORE action execution (for trajectory from_state) ##
        # These will become the "from_state" for the next transition
        info_pool.pre_state_description = getattr(info_pool, 'post_condition', 'Current screen state')
        # Try to get the actual state description from the last state analysis
        if hasattr(info_pool, '_current_state_description'):
            info_pool.pre_state_description = info_pool._current_state_description
        info_pool.pre_state_pre_condition = getattr(info_pool, 'pre_condition', '')
        info_pool.pre_state_post_condition = getattr(info_pool, 'post_condition', '')
        info_pool.pre_state_state_prediction = getattr(info_pool, 'state_prediction', '')
        
        ## execute the action ##
        action_execution_start_time = time.time()
        action_object, num_atomic_actions_executed, shortcut_error_message = actor.execute(action_object_str, info_pool,
                        screenshot_file=screenshot_file,
                        ocr_detection=screen_parser.ocr_detection,
                        ocr_recognition=screen_parser.ocr_recognition,
                        thought=action_thought,
                        screenshot_log_dir=os.path.join(log_dir,"screenshots"),
                        iter=str(iter))
        action_execution_end_time = time.time()
        if action_object is None:
            task_end_time = time.time()
            final_info_pool_dict = asdict(info_pool)
            final_info_pool_dict["beacon_hist"] = list(info_pool.beacon_hist)

            final_info_pool_dict["state_graphs"] = {
                app: serialize_graph(graph) for app, graph in info_pool.state_graphs.items()
            }
            steps.append({
                "step": iter,
                "operation": "finish",
                "finish_flag": "abnormal",
                "final_info_pool": final_info_pool_dict,
                "task_duration": task_end_time - task_start_time,
            })
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)
            
            # Save trajectory data if collecting
            if trajectory_collector:
                trajectory_collector.save()
                print(f"[Trajectory] Stats: {trajectory_collector.get_stats()}")
            
            finish(
                info_pool,
                persistent_tips_path=persistent_tips_path,
                persistent_shortcuts_path=persistent_shortcuts_path,
                persistent_graph_path=persistent_graph_path,
                persistent_patterns_path=persistent_patterns_path
            )
            print("WARNING!!: Abnormal finishing:", action_object_str)

            return

        info_pool.last_action = action_object
        info_pool.last_summary = action_description

        ## log ##
        steps.append({
            "step": iter,
            "operation": "action",
            "prompt_action": prompt_action,
            "raw_response": output_action,
            "action_object": action_object,
            "action_object_str": action_object_str,
            "action_thought": action_thought,
            "action_description": action_description,
            "duration": action_decision_end_time - action_decision_start_time,
            "execution_duration": action_execution_end_time - action_execution_start_time,
        })
        print("Action Thought:", action_thought)
        print("Action Description:", action_description)
        print("Action:", action_object)

        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)

        print("\n### Screen Parser ... ###\n")
        ## perception on the next step ##
        perception_start_time = time.time()
        # last_perception_infos = copy.deepcopy(perception_infos)
        last_screenshot_file = "./screenshot/last_screenshot.jpg"
        # last_keyboard = keyboard
        if os.path.exists(last_screenshot_file):
            os.remove(last_screenshot_file)
        os.rename(screenshot_file, last_screenshot_file)

        perception_infos, width, height = screen_parser.get_perception_infos(screenshot_file, temp_file=TEMP_DIR)
        shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)

        keyboard = False
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            if 'ADB Keyboard' in perception_info['text']:
                keyboard = True
                break

        info_pool.perception_infos_post = perception_infos
        info_pool.keyboard_post = keyboard
        assert width == info_pool.width and height == info_pool.height  # assert the screen size not changed

        ## log ##
        Image.open(screenshot_file).save(f"{log_dir}/screenshots/{iter + 1}.jpg")
        perception_end_time = time.time()
        # Save perception info separately to reduce steps.json size
        perception_save_path = f"{log_dir}/perception/{iter + 1}.json"
        with open(perception_save_path, "w") as f:
            json.dump(perception_infos, f, indent=2)
        steps.append({
            "step": iter + 1,
            "operation": "perception",
            "screenshot": f"{log_dir}/screenshots/{iter + 1}.jpg",
            "perception_file": perception_save_path,
            "perception_count": len(perception_infos),
            "duration": perception_end_time - perception_start_time
        })
        print("Perception Infos:", perception_infos)
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)

        ##
        print("\n### State Agent... ###\n")
        ### State Agent ###
        info_pool.current_screenshot_path = f"{log_dir}/screenshots/{iter + 1}.jpg"
        info_pool.state_perception_infos = info_pool.perception_infos_post
        state_prompt = state_agent.get_prompt(info_pool)
        chat_state = state_agent.init_chat()
        # Append the state prompt with the current screenshot (reusing the saved screenshot)
        chat_state = add_response("user", state_prompt, chat_state, image=info_pool.current_screenshot_path)
        # Get state analysis response from your reasoning model
        output_state = get_reasoning_model_api_response(chat_state, temperature=temperature)
        # Parse the state response
        parsed_state = state_agent.parse_response(output_state)
        state_description = parsed_state.get("state_description", "State description not provided")
        predicted_next_state = parsed_state.get("predicted_next_state", "No prediction provided")
        current_app_involved = parsed_state.get("current_app_name", "None")
        current_state_beacon = parsed_state.get("state_beacon", "Unknown")
        precondition = parsed_state.get("precondition", "Unknown")
        postcondition = parsed_state.get("postcondition", "Unknown")
        print("\nState Description: ", state_description)
        print("\nCurrent state beacon: ", current_state_beacon)
        print("\nPostcondition of current state: ", postcondition)
        print("\nPrecondition of next state: ", precondition)
        print("\nPredicted Next State: ", predicted_next_state)
        if info_pool.state_agent_recovery_mode:
            recovery_process_history = info_pool.recovery_process_history.get(info_pool.recovery_target_state_id, [])
            recovery_process_history.append(current_state_beacon)
            info_pool.recovery_process_history[info_pool.recovery_target_state_id] = recovery_process_history
            # In recovery mode, normalize app name for consistency
            from AgentsUtils.tool_utils import normalize_app_name
            info_pool.active_app_name = normalize_app_name(current_app_involved)
        else:
            # update_state_graph sets active_app_name to verified name internally
            state_agent.update_state_graph(info_pool, state_description, current_app_involved, current_state_beacon,
                                       precondition, postcondition, predicted_next_state)
            # active_app_name is now set inside update_state_graph with verified name
        info_pool.pre_condition = precondition
        info_pool.post_condition = postcondition
        info_pool.state_prediction = predicted_next_state
        # Store the actual state description for use in trajectory to_state (and next from_state)
        info_pool._current_state_description = state_description
        # Log the state update in your steps
        steps.append({
            "step": iter,
            "operation": "record new state after action",
            "previous action": info_pool.last_action,
            # "last state id": info_pool.state_id_list[-1],
            "state_description": state_description,
            "predicted_next_state": predicted_next_state,
        })
        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)

        ### END: State Agent Integration ###

        print("\n### Reflector ... ###\n")
        ### Action Reflection: Check whether the action works as expected ###
        action_reflection_start_time = time.time()
        if info_pool.state_agent_recovery_mode:
            prompt_action_reflect = reflection_agent.get_prompt_recovery_mode(info_pool)
            chat_action_reflect = reflection_agent.init_chat()
            chat_action_reflect = add_response_with_images("user", prompt_action_reflect, chat_action_reflect,
                                                         [last_screenshot_file, screenshot_file])
            output_action_reflect = get_reasoning_model_api_response(chat_action_reflect, temperature=temperature)
            parsed_result_action_reflect = reflection_agent.parse_response(output_action_reflect)

        else:
            prompt_action_reflect = reflection_agent.get_prompt(info_pool)
            chat_action_reflect = reflection_agent.init_chat()
            chat_action_reflect = add_response_with_images("user", prompt_action_reflect, chat_action_reflect,
                                                         [last_screenshot_file, screenshot_file])
            output_action_reflect = get_reasoning_model_api_response(chat_action_reflect, temperature=temperature)
            parsed_result_action_reflect = reflection_agent.parse_response(output_action_reflect)

        outcome, error_description, progress_status = (
            parsed_result_action_reflect['outcome'],
            parsed_result_action_reflect['error_description'],
            parsed_result_action_reflect['progress_status']
        )
        info_pool.progress_status_history.append(progress_status)
        # NOTE: pre_state_* values are now set BEFORE action execution (around line 745)
        # to correctly capture the state BEFORE the action was taken.
        # Do NOT overwrite them here with post-action values!
        action_reflection_end_time = time.time()

        if "A" in outcome:  # Successful. The result of the last action meets the expectation.
            action_outcome = "A"
        elif "G" in outcome and info_pool.state_agent_recovery_mode:
            action_outcome = "G"
        elif "B" in outcome:  # Failed. The last action results in a wrong page. I need to return to the previous state.
            action_outcome = "B"
            action_name = action_object['name']
            if action_name in ATOMIC_ACTION_SIGNATURES:
                # back(ADB_PATH) # back one step for atomic actions
                pass
            elif action_name in info_pool.shortcuts:
                # shortcut_object = info_pool.shortcuts[action_name]
                # num_of_atomic_actions = len(shortcut_object['atomic_action_sequence'])
                if shortcut_error_message is not None:
                    error_description += f"; Error occurred while executing the shortcut: {shortcut_error_message}"
                # for _ in range(num_atomic_actions_executed):
                #     back(ADB_PATH)
            else:
                raise ValueError("Invalid action name:", action_name)

        elif "C" in outcome:  # Failed. The last action produces no changes.
            action_outcome = "C"
        else:
            raise ValueError("Invalid outcome:", outcome)

        # update action history
        info_pool.action_history.append(action_object)
        info_pool.summary_history.append(action_description)
        info_pool.action_outcomes.append(action_outcome)
        info_pool.error_descriptions.append(error_description)
        info_pool.progress_status = progress_status

        ## log ##
        steps.append({
            "step": iter,
            "operation": "action_reflection",
            "prompt_action_reflect": prompt_action_reflect,
            "raw_response": output_action_reflect,
            "outcome": outcome,
            "error_description": error_description,
            "progress_status": progress_status,
            "duration": action_reflection_end_time - action_reflection_start_time,
        })
        print("Outcome:", action_outcome)
        print("Progress Status:", progress_status)
        print("Error Description:", error_description)

        with open(log_json_path, "w") as f:
            json.dump(steps, f, indent=4)

        ## Trajectory Data Collection ##
        if trajectory_collector and not info_pool.state_agent_recovery_mode:
            # Build transition data from current state
            transition_data = create_transition_from_info_pool(info_pool, action_outcome, error_description)
            
            # Update from_state with pre-action state info
            transition_data["from_state"] = {
                "description": getattr(info_pool, 'pre_state_description', 'Previous screen state'),
                "precondition": getattr(info_pool, 'pre_state_pre_condition', ''),
                "postcondition": getattr(info_pool, 'pre_state_post_condition', '')
            }
            
            # Update to_state with current state info
            transition_data["to_state"] = {
                "description": state_description if 'state_description' in dir() else 'Current screen state',
                "precondition": info_pool.pre_condition,
                "postcondition": info_pool.post_condition
            }
            
            # Update action with more details
            if isinstance(action_object, dict):
                transition_data["action"] = {
                    "type": action_object.get("name", "Unknown"),
                    "description": action_description,
                    **{k: v for k, v in action_object.items() if k not in ["name"]}
                }
            
            trajectory_collector.record_transition(
                from_state=transition_data["from_state"],
                action=transition_data["action"],
                to_state=transition_data["to_state"],
                before_screenshot=last_screenshot_file if 'last_screenshot_file' in dir() else "",
                after_screenshot=screenshot_file if 'screenshot_file' in dir() else "",
                current_subgoal=info_pool.current_subgoal,
                outcome=action_outcome,
                reason=error_description if action_outcome != "A" else f"Action succeeded: {action_description}"
            )

        ##
        # State agent recovery
        if action_outcome in ["B", "C"] and info_pool.state_agent_recovery_mode is False:
            print("\n### Recovery Mode: Invoking State Agent for recovery plan... ###\n")
            recovery_prompt = state_agent.get_prompt_for_recovery(info_pool)
            chat = state_agent.init_chat_for_recovery()
            chat = add_response("user", recovery_prompt, chat)
            recovery_response = get_reasoning_model_api_response(chat, temperature=temperature)
            
            # Parse recovery response with info_pool for beacon/state_id resolution
            parsed_recovery = state_agent.parse_recovery_response(recovery_response, info_pool)
            target_state = parsed_recovery.get("target_state_id", "")
            target_beacon = parsed_recovery.get("target_beacon", "")
            
            # Validate target state exists in FSM
            if not target_state:
                print(f"[Recovery] Warning: No valid target state resolved, using most recent valid state")
                visited_states = info_pool.state_id_lists.get(info_pool.active_app_name, [])
                if visited_states:
                    target_state = visited_states[-1]
                    print(f"[Recovery] Using fallback target: {target_state[:12]}...")
            else:
                app_graph = info_pool.state_graphs.get(info_pool.active_app_name, None)
                if app_graph and target_state not in app_graph.nodes:
                    print(f"[Recovery] Warning: Target state {target_state[:12]}... not found in FSM")
                    visited_states = info_pool.state_id_lists.get(info_pool.active_app_name, [])
                    if visited_states:
                        target_state = visited_states[-1]
                        print(f"[Recovery] Using fallback target: {target_state[:12]}...")
            
            info_pool.state_agent_recovery_mode = True
            info_pool.recovery_plan[target_state] = parsed_recovery
            info_pool.recovery_target_state_id = target_state
            info_pool.recovery_target_beacon = target_beacon  # Store beacon for verification
            info_pool.total_recovery_retries = 1
            info_pool.recovery_attempts += 1  # Track recovery attempt
            print(f"\n[Recovery] Target: {target_beacon or target_state[:12]}")
            print(f"[Recovery] Plan: {parsed_recovery.get('plan', [])}")
            print(f"[Recovery Stats] Attempt #{info_pool.recovery_attempts} initiated")
            
        elif action_outcome in ["B", "C"] and info_pool.state_agent_recovery_mode:
            print("\n### Recovery Mode: Revising recovery plan... ###\n")
            info_pool.total_recovery_retries += 1
            if info_pool.total_recovery_retries >= MAX_RECOVERY_RETRIES:
                print("Exceeded retry limit. Disabling recovery mode.")
                info_pool.state_agent_recovery_mode = False
                info_pool.recovery_failures += 1  # Track recovery failure
                print(f"[Recovery Stats] Failed (exceeded {MAX_RECOVERY_RETRIES} retries). Total: {info_pool.recovery_successes}/{info_pool.recovery_attempts} successful")
            else:
                revise_prompt = state_agent.get_prompt_revise_recovery_plan(info_pool)
                chat = state_agent.init_chat_for_recovery()
                chat = add_response("user", revise_prompt, chat)
                revise_recovery_response = get_reasoning_model_api_response(chat, temperature=temperature)
                
                # Parse with info_pool for resolution
                revised_recovery_plan = state_agent.parse_recovery_response(revise_recovery_response, info_pool)
                target_state = revised_recovery_plan.get("target_state_id", "")
                target_beacon = revised_recovery_plan.get("target_beacon", "")
                
                # Validate revised target state exists
                if not target_state:
                    print(f"[Recovery] Warning: Revised target not resolved, using fallback")
                    visited_states = info_pool.state_id_lists.get(info_pool.active_app_name, [])
                    if visited_states:
                        target_state = visited_states[-1]
                
                if info_pool.recovery_target_state_id in info_pool.recovery_process_history:
                    del info_pool.recovery_process_history[info_pool.recovery_target_state_id]
                info_pool.recovery_plan[target_state] = revised_recovery_plan
                info_pool.recovery_target_state_id = target_state
                info_pool.recovery_target_beacon = target_beacon
                print(f"[Recovery] Revised target: {target_beacon or target_state[:12]}")

        ### NoteTaker: Record Important Content ###
        if action_outcome == "A":
            if info_pool.state_agent_recovery_mode:
                recovery_plan_dict = info_pool.recovery_plan
                recovery_plan = recovery_plan_dict.get(info_pool.recovery_target_state_id, {})

                plan = recovery_plan.get("plan", [])
                current_subgoal = recovery_plan.get("current_subgoal", "")
                
                # Verify if we've reached the target state by comparing current beacon
                target_reached = False
                if current_state_beacon and info_pool.recovery_target_state_id:
                    target_reached = state_agent.verify_recovery_reached(
                        info_pool, 
                        info_pool.recovery_target_state_id, 
                        current_state_beacon
                    )
                    if target_reached:
                        print(f"[Recovery] ✓ Target state reached! Beacon: {current_state_beacon}")
                        info_pool.state_agent_recovery_mode = False
                        info_pool.recovery_successes += 1
                        print(f"[Recovery Stats] Success (target verified)! Total: {info_pool.recovery_successes}/{info_pool.recovery_attempts}")

                # If not yet at target, progress through plan
                if not target_reached and plan and current_subgoal in plan:
                    current_index = plan.index(current_subgoal)
                    if current_index + 1 < len(plan):
                        recovery_plan["current_subgoal"] = plan[current_index + 1]
                        print(f"[Recovery] Moving to next subgoal: {plan[current_index + 1]}")
                    else:
                        # Recovery plan is done - all subgoals completed
                        info_pool.state_agent_recovery_mode = False
                        info_pool.recovery_successes += 1
                        print(f"[Recovery Stats] Success (plan completed)! Total: {info_pool.recovery_successes}/{info_pool.recovery_attempts}")
                        
                elif not target_reached and plan:
                    # current_subgoal not in plan - check if this is a single-step plan that just completed
                    if len(plan) == 1:
                        # Single-step recovery plan completed successfully
                        info_pool.state_agent_recovery_mode = False
                        info_pool.recovery_successes += 1
                        print(f"[Recovery Stats] Success (single-step plan completed)! Total: {info_pool.recovery_successes}/{info_pool.recovery_attempts}")
                    else:
                        # Multi-step plan, subgoal not found - reset to first item
                        recovery_plan["current_subgoal"] = plan[0]
                        print(f"[Recovery] Subgoal mismatch, resetting to: {plan[0]}")
                        
            print("\n### NoteKeeper ... ###\n")
            # if previous action is successful, record the important content
            notetaking_start_time = time.time()
            prompt_note = reflection_agent.get_prompt_for_note(info_pool)
            chat_note = reflection_agent.init_chat()
            chat_note = add_response("user", prompt_note, chat_note, image=screenshot_file)  # new screenshot
            output_note = get_reasoning_model_api_response(chat_note, temperature=temperature)
            parsed_result_note = reflection_agent.parse_response_note(output_note)
            important_notes = parsed_result_note['important_notes']
            info_pool.important_notes = important_notes
            os.remove(last_screenshot_file)

            notetaking_end_time = time.time()
            steps.append({
                "step": iter,
                "operation": "notetaking",
                "prompt_note": prompt_note,
                "raw_response": output_note,
                "important_notes": important_notes,
                "duration": notetaking_end_time - notetaking_start_time,
            })
            print("Important Notes:", important_notes)
            with open(log_json_path, "w") as f:
                json.dump(steps, f, indent=4)

        elif action_outcome == "G":
            print("Recovery finished, returned to previous state")
            info_pool.state_agent_recovery_mode = False
            info_pool.recovery_successes += 1  # Track recovery success
            print(f"[Recovery Stats] Success! Total: {info_pool.recovery_successes}/{info_pool.recovery_attempts} ({100*info_pool.recovery_successes/max(1,info_pool.recovery_attempts):.1f}%)")

        elif action_outcome in ["B", "C"]:
            os.remove(last_screenshot_file)


        print("\n=========================================================")
        print(f"sleeping for {SLEEP_BETWEEN_STEPS} before next iteration ...\n\n")
        sleep(SLEEP_BETWEEN_STEPS)