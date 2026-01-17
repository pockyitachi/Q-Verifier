"""
Planner Agent for Agent-SAMA: Task Decomposition and Planning

The Planner Agent is responsible for decomposing user instructions into
high-level subgoals. It leverages:
1. FSM navigation patterns from past task executions
2. Learned shortcuts for efficient action sequences
3. Tips from past experiences to avoid common errors
4. LLM-as-Judge for selecting the best plan from candidates

Reference: Agent-SAMA paper (AAAI 2026) - Planning Phase
"""

from Agents.base_agent import BaseAgent
from Agents.info_pool import InfoPool
from constant import LLM_Judge
from AgentsUtils.common_utils import extract_json_object
import json
from typing import List, Dict, Optional


class Planner(BaseAgent):
    """
    Planner Agent for high-level task decomposition.
    
    The Planner generates subgoal sequences by considering:
    - User instruction and current screen state
    - FSM navigation patterns learned from past tasks
    - Available shortcuts and their preconditions
    - Tips for avoiding common errors
    
    When LLM_Judge is enabled, generates 5 candidate plans and uses
    a separate LLM call to select the best one.
    """

    def init_chat(self):
        operation_history = []
        system_prompt = (
            "You are a helpful AI assistant for operating mobile phones. "
            "Your goal is to generate high-level plans consisting of subgoals. "
            "You have access to learned navigation patterns from past tasks that show "
            "common state transitions in various apps. Use this knowledge to create "
            "efficient and robust plans. Think as if you are a human user operating the phone."
        )
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def init_chat_for_judging(self):
        operation_history = []
        system_prompt = (
            "You are an LLM judge. Your job is to evaluate 5 candidate plans that were generated to accomplish a specific user instruction on a mobile phone. "
            "Each plan consists of a sequence of subgoals aimed at completing the instruction. "
            "Select the best plan using a structured rubric that prioritizes relevance to the instruction, logical correctness, execution efficiency, robustness, and clarity. "
            "Think critically and justify your selection clearly."
        )
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        if info_pool.plan == "" and LLM_Judge:
            # first time planning
            prompt += "---\n"
            prompt += "Think step by step and make 5 high-level plans to achieve the user's instruction. For each plan, break it down into subgoals. If the request is complex, divide the plan into meaningful steps. If the request involves exploration, include concrete subgoals to quantify the investigation steps. The screenshot displays the starting state of the phone.\n\n"

            # Add FSM navigation patterns from past experience
            prompt += self._build_fsm_patterns_section(info_pool)
            
            # Add common flows knowledge
            prompt += self._build_common_flows_section(info_pool)

            if info_pool.shortcuts != {}:
                prompt += "### Available Shortcuts from Past Experience ###\n"
                prompt += "We additionally provide some shortcut functionalities based on past experience. These shortcuts are predefined sequences of operations that might make the plan more efficient. Each shortcut includes a precondition specifying when it is suitable for use. If your plan implies the use of certain shortcuts, ensure that the precondition is fulfilled before using them. Note that you don't necessarily need to include the names of these shortcuts in your high-level plan; they are provided as a reference.\n"
                for shortcut, value in info_pool.shortcuts.items():
                    prompt += f"- {shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
                prompt += "\n"
            
            # Add tips if available
            if info_pool.tips:
                prompt += "### Tips from Past Experience ###\n"
                prompt += f"{info_pool.tips}\n\n"
            
            prompt += "---\n"

            prompt += "Provide your output as a JSON list of five candidate plans. Each plan must be a dictionary with three keys:\n"
            prompt += "- \"thought\": a string explaining the reasoning behind the plan\n"
            prompt += "- \"plan\": a list of subgoals (each a string)\n"
            prompt += "- \"current_subgoal\": the first subgoal to work on\n\n"
            prompt += "Example format:\n"
            prompt += "```json\n"
            prompt += "[\n"
            prompt += "  {\n"
            prompt += "    \"thought\": \"This plan first opens the app, then searches for the item, then reviews it.\",\n"
            prompt += "    \"plan\": [\"Open the app\", \"Search for the item\", \"Review the item\"],\n"
            prompt += "    \"current_subgoal\": \"Open the app\"\n"
            prompt += "  },\n"
            prompt += "  {\n"
            prompt += "    \"thought\": \"This plan starts with searching directly, assuming the app is already open.\",\n"
            prompt += "    \"plan\": [\"Search for the item\", \"Read reviews\", \"Check availability\"],\n"
            prompt += "    \"current_subgoal\": \"Search for the item\"\n"
            prompt += "  },\n"
            prompt += "  { ... },\n"
            prompt += "  { ... },\n"
            prompt += "  { ... }\n"
            prompt += "]\n"
            prompt += "```\n"
        elif info_pool.plan == "":
            prompt += "---\n"
            prompt += "Think step by step and generate one high-level plan to achieve the user's instruction. Break it down into a sequence of meaningful subgoals. If the task is complex, ensure the subgoals reflect the logical steps needed to complete it. If the instruction involves exploration, include concrete subgoals that guide the investigation process. The screenshot shows the starting state of the phone.\n\n"

            if info_pool.shortcuts != {}:
                prompt += "### Available Shortcuts from Past Experience ###\n"
                prompt += "We provide some shortcut functionalities based on past experience. Each shortcut includes a precondition indicating when it can be used. You may use these shortcuts if they help make the plan more efficient. Ensure that any precondition is met before using them.\n"
                for shortcut, value in info_pool.shortcuts.items():
                    prompt += f"- {shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
                prompt += "\n"
            prompt += "---\n"

            prompt += "Provide your output as a JSON object with the following keys:\n"
            prompt += "- \"thought\": a string explaining the rationale behind the plan\n"
            prompt += "- \"plan\": a list of subgoals (each a string)\n"
            prompt += "- \"current_subgoal\": the first subgoal to work on\n\n"

            prompt += "Example format:\n"
            prompt += "```json\n"
            prompt += "{\n"
            prompt += "  \"thought\": \"This plan starts by opening the app, then navigates to the target screen, and finally performs the required action.\",\n"
            prompt += "  \"plan\": [\"Open the app\", \"Navigate to the target screen\", \"Perform the required action\"],\n"
            prompt += "  \"current_subgoal\": \"Open the app\"\n"
            prompt += "}\n"
            prompt += "```\n"
        else:
            # continue planning
            prompt += "### Current Plan ###\n"
            prompt += f"{info_pool.plan}\n\n"
            prompt += "### Previous Subgoal ###\n"
            prompt += f"{info_pool.current_subgoal}\n\n"
            prompt += f"### Progress Status ###\n"
            prompt += f"{info_pool.progress_status}\n\n"
            prompt += "### Important Notes ###\n"
            if info_pool.important_notes != "":
                prompt += f"{info_pool.important_notes}\n\n"
            else:
                prompt += "No important notes recorded.\n\n"
            if info_pool.error_flag_plan:
                prompt += "### Potentially Stuck! ###\n"
                prompt += "You have encountered several failed attempts. Here are some logs:\n"
                k = info_pool.err_to_planner_thresh
                recent_actions = info_pool.action_history[-k:]
                recent_summaries = info_pool.summary_history[-k:]
                recent_err_des = info_pool.error_descriptions[-k:]
                for i, (act, summ, err_des) in enumerate(zip(recent_actions, recent_summaries, recent_err_des)):
                    prompt += f"- Attempt: Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"

            prompt += "---\n"
            prompt += "The sections above provide an overview of the plan you are following, the current subgoal you are working on, the overall progress made, and any important notes you have recorded. The screenshot displays the current state of the phone.\n"
            prompt += "Carefully assess the current status to determine if the task has been fully completed. "
            prompt += "If the user's request involves exploration, ensure you have conducted sufficient investigation. If you are confident that no further actions are required, mark the task as \"Finished\" in your output. If the task is not finished, outline the next steps. If you are stuck with errors, think step by step about whether the overall plan needs to be revised to address the error.\n"
            prompt += "NOTE: If the current situation prevents proceeding with the original plan or requires clarification from the user, make reasonable assumptions and revise the plan accordingly. Act as though you are the user in such cases.\n\n"

            if info_pool.shortcuts != {}:
                prompt += "### Available Shortcuts from Past Experience ###\n"
                prompt += "We additionally provide some shortcut functionalities based on past experience. These shortcuts are predefined sequences of operations that might make the plan more efficient. Each shortcut includes a precondition specifying when it is suitable for use. If your plan implies the use of certain shortcuts, ensure that the precondition is fulfilled before using them. Note that you don't necessarily need to include the names of these shortcuts in your high-level plan; they are provided only as a reference.\n"
                for shortcut, value in info_pool.shortcuts.items():
                    prompt += f"- {shortcut}: {value['description']} | Precondition: {value['precondition']}\n"
                prompt += "\n"
            prompt += "If the current plan remains valid, you must still respond with a JSON object containing the following keys:\n"
            prompt += "If you are confident that no further actions are required, mark the task as \"Finished\" in your output."
            prompt += "- \"thought\": a brief reasoning of why the plan remains valid\n"
            prompt += "- \"plan\": a list of subgoals (same as the current one)\n"
            prompt += "- \"current_subgoal\": the next subgoal to work on\n"
            prompt += "\n"
            prompt += "Use this JSON format:\n"
            prompt += "```json\n"
            prompt += "{\n"
            prompt += "  \"thought\": \"The current plan is valid because...\",\n"
            prompt += "  \"plan\": [\"...\", \"...\"],\n"
            prompt += "  \"current_subgoal\": \"...\"\n"
            prompt += "}\n"
            prompt += "```\n"
            prompt += "---\n"
            prompt += "If an update is required for the high-level plan, generate five alternative plans in JSON format for evaluation. Otherwise, keep the current plan and copy it here.\n\n"
            prompt += "If you believe an update is required, your output must follow this format:\n"
            prompt += "```json\n"
            prompt += "[\n"
            prompt += "  {\n"
            prompt += "    \"thought\": \"Rationale for plan 1\",\n"
            prompt += "    \"plan\": [\"Subgoal 1\", \"Subgoal 2\"],\n"
            prompt += "    \"current_subgoal\": \"Subgoal 1\"\n"
            prompt += "  },\n"
            prompt += "  {\n"
            prompt += "    \"thought\": \"Rationale for plan 2\",\n"
            prompt += "    \"plan\": [\"...\"],\n"
            prompt += "    \"current_subgoal\": \"...\"\n"
            prompt += "  },\n"
            prompt += "  ... (3 more plans)\n"
            prompt += "]\n"
            prompt += "```\n"

        return prompt

    def parse_response_single(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ",
                                                                                                            " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ",
                                                                                                                 " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "plan": plan, "current_subgoal": current_subgoal}

    def parse_response(self, response: str) -> dict:
        parsed = extract_json_object(response)
        if isinstance(parsed, dict):
            # Wrap single plan dict in a list
            parsed = [parsed]
        if isinstance(parsed, list) and all(isinstance(p, dict) for p in parsed):
            formatted_candidates = []
            for p in parsed:
                formatted_candidates.append({
                    "thought": p.get("thought", "").strip(),
                    "plan": p.get("plan", []),
                    "current_subgoal": p.get("current_subgoal", "").strip()
                })
            return {"candidates": formatted_candidates}
        else:
            print("[Planner.parse_response] Failed to extract JSON list of plans.")
            return {"candidates": []}

    def get_prompt_for_judging(self, info_pool: InfoPool, plan_candidates: list) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        if info_pool.plan:
            prompt += "### Previous Plan ###\n"
            prompt += f"{info_pool.plan}\n\n"
        if info_pool.progress_status:
            prompt += "### Progress Status ###\n"
            prompt += f"{info_pool.progress_status}\n\n"
        if info_pool.important_notes:
            prompt += "### Important Notes ###\n"
            prompt += f"{info_pool.important_notes}\n\n"

        prompt += "### Plan Candidates ###\n"
        for i, pc in enumerate(plan_candidates):
            prompt += f"### Plan Candidate {i + 1} ###\n"
            prompt += json.dumps(pc, indent=2) + "\n\n"

        prompt += "---\n"
        prompt += "You are an LLM Judge. Select the best plan from the 5 candidates below based on the following criteria:\n\n"
        prompt += "**Judging Rubric**:\n"
        prompt += "- Relevance: Accurately reflects the instruction.\n"
        prompt += "- Correctness: Logical and valid subgoals.\n"
        prompt += "- Efficiency: Achieves the goal with minimal steps, avoids excessive app switching; fewer switches between apps is better\n"
        prompt += "- Use of Shortcuts: Uses helpful prior knowledge.\n"
        prompt += "- Robustness: Avoids likely errors.\n"
        prompt += "- Clarity: Plan is easy to follow.\n\n"

        prompt += "Provide your output in the following format:\n\n"
        prompt += "### Thought ###\n"
        prompt += "Explain your reasoning for selecting the best plan and a detailed explanation of your rationale for the plan and subgoals.\n\n"
        prompt += "### Plan ###\n"
        prompt += "Copy the full content of the best plan (including Thought / Plan / Subgoal) exactly as it appeared above.\n\n"
        prompt += "### Current Subgoal ###\n"
        prompt += "Write the `current_subgoal` of the selected plan.\n"
        return prompt

    def parse_judge_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].strip().replace("\n", " ").replace(
            "  ", " ")
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].strip().replace("\n", " ").replace("  ", " ")
        return {"thought": thought, "plan": plan, "current_subgoal": current_subgoal}
    
    def _build_fsm_patterns_section(self, info_pool: InfoPool) -> str:
        """
        Build the FSM navigation patterns section for the prompt.
        
        This includes:
        - Known app navigation structures from state_graphs
        - Learned navigation patterns from past tasks
        - Cross-app pattern similarities
        """
        prompt = ""
        
        # Extract patterns from existing state graphs
        if info_pool.state_graphs:
            prompt += "### Known App Navigation Patterns (from FSM) ###\n"
            prompt += "Based on past interactions, here are the known navigation structures:\n\n"
            
            for app_name, graph in info_pool.state_graphs.items():
                if graph.number_of_nodes() > 0:
                    prompt += f"**{app_name}**:\n"
                    
                    # List known states (beacons)
                    states = []
                    for node in graph.nodes:
                        beacon = graph.nodes[node].get('beacon', 'Unknown')
                        visit_count = graph.nodes[node].get('visit_count', 1)
                        states.append((beacon, visit_count))
                    
                    # Sort by visit count (most visited first)
                    states.sort(key=lambda x: x[1], reverse=True)
                    
                    prompt += "  Known states: "
                    prompt += " → ".join([s[0] for s in states[:5]])  # Top 5 states
                    prompt += "\n"
                    
                    # Show common transitions
                    if graph.number_of_edges() > 0:
                        prompt += "  Common transitions:\n"
                        for u, v, data in list(graph.edges(data=True))[:3]:  # Top 3 transitions
                            source_beacon = graph.nodes[u].get('beacon', 'Unknown')
                            target_beacon = graph.nodes[v].get('beacon', 'Unknown')
                            action = data.get('action', {})
                            action_name = action.get('name', 'Unknown') if isinstance(action, dict) else str(action)
                            prompt += f"    - {source_beacon} --[{action_name}]--> {target_beacon}\n"
                    prompt += "\n"
            
            prompt += "Use these patterns to inform your plan. If a task involves an app you've seen before, "
            prompt += "you can expect similar navigation flow.\n\n"
        
        # Include learned navigation patterns
        if info_pool.learned_navigation_patterns:
            prompt += "### Learned Navigation Patterns ###\n"
            for pattern in info_pool.learned_navigation_patterns[:5]:  # Top 5 patterns
                prompt += f"- **{pattern.get('pattern_name', 'Unknown')}** ({pattern.get('app', 'General')}):\n"
                prompt += f"  {pattern.get('description', 'No description')}\n"
                if 'states' in pattern:
                    prompt += f"  States: {' → '.join(pattern['states'][:4])}\n"
            prompt += "\n"
        
        return prompt
    
    def _build_common_flows_section(self, info_pool: InfoPool) -> str:
        """
        Build the common flows section that shows reusable patterns across apps.
        
        Common flows are abstract patterns like "search_flow", "checkout_flow"
        that apply across multiple apps.
        """
        prompt = ""
        
        if info_pool.common_flows:
            prompt += "### Common Task Flows (Cross-App Patterns) ###\n"
            prompt += "These are reusable flow patterns that work across similar apps:\n\n"
            
            for flow_name, flow_info in info_pool.common_flows.items():
                prompt += f"**{flow_name}**:\n"
                if isinstance(flow_info, dict):
                    prompt += f"  Description: {flow_info.get('description', 'N/A')}\n"
                    if 'steps' in flow_info:
                        prompt += f"  Typical steps: {' → '.join(flow_info['steps'])}\n"
                    if 'apps' in flow_info:
                        prompt += f"  Works in: {', '.join(flow_info['apps'])}\n"
                else:
                    prompt += f"  {flow_info}\n"
                prompt += "\n"
        
        return prompt
    
    @staticmethod
    def extract_app_from_instruction(instruction: str) -> List[str]:
        """
        Extract likely app names from the instruction.
        
        This helps identify which FSM patterns might be relevant.
        """
        common_apps = [
            "Maps", "Google Maps", "Chrome", "Notes", "Calendar", "Gmail",
            "Messages", "Phone", "Camera", "Settings", "Walmart", "Amazon",
            "YouTube", "Twitter", "X", "Instagram", "Facebook", "WhatsApp"
        ]
        
        found_apps = []
        instruction_lower = instruction.lower()
        for app in common_apps:
            if app.lower() in instruction_lower:
                found_apps.append(app)
        
        return found_apps
