from Agents.base_agent import BaseAgent
from Agents.info_pool import InfoPool


class ReflectionAgent(BaseAgent):
    def init_chat(self) -> list:
        operation_history = []
        system_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to verify whether the last action produced the expected behavior and to keep track of the overall progress and take notes on important contents related to user's instruction."
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"
        prompt += "---\n"
        prompt += f"The attached two images are two phone screenshots before and after your last action. "
        prompt += f"The width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        prompt += (
            "To help you better perceive the content in these screenshots, we have extracted positional information for the text elements and icons. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom).\n"
        )
        prompt += (
            "Note that these information might not be entirely accurate. "
            "You should combine them with the screenshots to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "### Screen Information Before the Action ###\n"
        for clickable_info in info_pool.perception_infos_pre:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info[
                'coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status before the action: "
        if info_pool.keyboard_pre:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"
        # Add PRE/POST/PREDICT information here
        if hasattr(info_pool, "pre_state_pre_condition") and info_pool.pre_state_pre_condition:
            prompt += "### Precondition ###\n"
            prompt += f"{info_pool.pre_state_pre_condition}\n\n"
        if hasattr(info_pool, "pre_state_post_condition") and info_pool.pre_state_post_condition:
            prompt += "### Postcondition ###\n"
            prompt += f"{info_pool.pre_state_post_condition}\n\n"
        if hasattr(info_pool, "pre_state_state_prediction") and info_pool.pre_state_state_prediction:
            prompt += "### Predicted Next Screen Description ###\n"
            prompt += f"{info_pool.pre_state_state_prediction}\n\n"

        prompt += "### Screen Information After the Action ###\n"
        for clickable_info in info_pool.perception_infos_post:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info[
                'coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status after the action: "
        if info_pool.keyboard_post:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"

        prompt += "---\n"
        prompt += "### Latest Action ###\n"
        # assert info_pool.last_action != ""
        prompt += f"Action: {info_pool.last_action}\n"
        prompt += f"Expectation: {info_pool.last_summary}\n\n"

        prompt += "---\n"
        prompt += "Carefully examine the information provided above to determine whether the last action produced the expected behavior. If the action was successful, update the progress status accordingly. If the action failed, identify the failure mode and provide reasoning on the potential reason causing this failure. Note that for the “Swipe” action, it may take multiple attempts to display the expected content. Thus, for a \"Swipe\" action, if the screen shows new content, it usually meets the expectation.\n\n"
        prompt += "---\n"
        prompt += "🧠 To improve your judgment, additional predictive information is provided below the 'Screen Information Before the Action' section:\n"
        prompt += "- Precondition: What should to be true before the current subgoal can start.\n"
        prompt += "- Postcondition: What should be true after successfully completing the current subgoal.\n"
        prompt += "- Predicted Next Screen: A natural-language description of what the next screen is expected to display after the next subgoal. Use this to compare with the actual post-screen.\n"
        prompt += "Use these expectations to guide your judgment on whether the last action result aligns with the plan.\n\n"
        prompt += "Provide your output in the following format containing three parts:\n\n"
        prompt += "### Outcome ###\n"
        prompt += "Choose from the following options. Give your answer as \"A\", \"B\" or \"C\":\n"
        prompt += "A: Successful or Partially Successful. The result of the last action meets the expectation.\n"
        prompt += "B: Failed. The last action results in a wrong page. I need to return to the previous state.\n"
        prompt += "C: Failed. The last action produces no changes.\n\n"

        prompt += "### Error Description ###\n"
        prompt += "If the action failed, provide a detailed description of the error and the potential reason causing this failure. If the action succeeded, put \"None\" here.\n\n"

        prompt += "### Progress Status ###\n"
        prompt += "If the action was successful or partially successful, update the progress status. If the action failed, copy the previous progress status.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n",
                                                                                                      " ").replace("  ",
                                                                                                                   " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Progress Status ###")[0].replace(
            "\n", " ").replace("  ", " ").strip()
        progress_status = response.split("### Progress Status ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"outcome": outcome, "error_description": error_description, "progress_status": progress_status}

    def get_prompt_recovery_mode(self, info_pool: InfoPool) -> str:
        recovery_plan_dict = info_pool.recovery_plan
        recovery_plan = recovery_plan_dict.get(info_pool.recovery_target_state_id, {})
        if not recovery_plan:
            print("Recovery plan is empty. Skipping execution.")
            info_pool.state_agent_recovery_mode = False
            return ""
        current_sub_task = recovery_plan["current_subgoal"]
        recovery_goal = recovery_plan["recovery_goal"]
        prompt = "### Recovery Mode: Current Execution Step ###\n"
        prompt += "You are verifying whether the most recent recovery action successfully moved the agent closer to the target state for the recovery goal.\n\n"

        prompt += "\n ### Recovery Goal###\n"
        prompt += f"{recovery_goal}\n\n"
        prompt += "\n ### Target Recovery State ID ###\n"
        prompt += f"{info_pool.recovery_target_state_id}\n\n"
        prompt += "### Current Recovery Subtask ###\n"
        prompt += f"{current_sub_task}\n\n"
        prompt += "### Current Recovery Action ###\n"
        prompt += f"{info_pool.current_recovery_action}\n\n"

        prompt += "### Action Rationale ###\n"
        prompt += f"{info_pool.current_recovery_thought}\n\n"
        prompt += "### Full Recovery Plan ###\n"
        prompt += f"{recovery_plan}\n\n"

        prompt += "---\n"
        prompt += "The attached two images are screenshots before and after the current recovery action. "
        prompt += f"The screen resolution is {info_pool.width} × {info_pool.height} pixels.\n"
        prompt += (
            "For each screen, we provide extracted elements (text or icons) with their positions in the format: (coordinates; content).\n"
            "Note: The extracted information may be imperfect—use it alongside the screenshots for judgment.\n\n"
        )

        prompt += "### Screen Information Before the Action ###\n"
        for item in info_pool.perception_infos_pre:
            if item['text'] and item['text'] != "icon: None" and item['coordinates'] != (0, 0):
                prompt += f"{item['coordinates']}; {item['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status before the action: "
        prompt += "Activated.\n" if info_pool.keyboard_pre else "Not activated.\n"

        prompt += "\n"

        prompt += "### Screen Information After the Action ###\n"
        for item in info_pool.perception_infos_post:
            if item['text'] and item['text'] != "icon: None" and item['coordinates'] != (0, 0):
                prompt += f"{item['coordinates']}; {item['text']}\n"
        prompt += "\n"
        prompt += "Keyboard status after the action: "
        prompt += "Activated.\n" if info_pool.keyboard_post else "Not activated.\n"

        prompt += "\n"
        prompt += "---\n"
        prompt += "Carefully examine whether the current recovery action has successfully progressed toward the target state. "
        prompt += "Use the visual and textual screen changes, and consider if the transition aligns with the expected effect described in the action rationale.\n\n"

        prompt += "Provide your output in the following format:\n\n"

        prompt += "### Outcome ###\n"
        prompt += "A: Successful or Partially Successful. The action helped move closer to the recovery goal.\n"
        prompt += "G: Successful achieved recovery goal.\n"
        prompt += "B: Failed. The action led to the wrong page or state.\n"
        prompt += "C: Failed. The action produced no observable change.\n\n"

        prompt += "### Error Description ###\n"
        prompt += "Explain what went wrong if the action failed. If successful, return 'None'.\n\n"

        prompt += "### Progress Status ###\n"
        prompt += "Update if the agent made meaningful progress toward the target. Otherwise, retain the previous status.\n"

        return prompt

    def section(self, title, content) -> str:
        if isinstance(content, list):
            content_str = "\n".join(f"- {item}" for item in content)
        else:
            content_str = str(content).strip()
        return f"### {title} ###\n{content_str}\n"

    def screen_info(self, info_pool: InfoPool) -> str:
        lines = [
            "### Current Screen Information ###",
            f"The attached image is a screenshot showing the current state of the phone.",
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.",
            "To help you better perceive the content in this screenshot, we have extracted positional information for the text elements and icons.",
            "The format is: (coordinates; content). The coordinates are [x, y], where x is horizontal and y is vertical.",
            "The extracted information is as follows:"
        ]
        for info in info_pool.perception_infos_post:
            if info['text'] and info['text'] != "icon: None" and info['coordinates'] != (0, 0):
                lines.append(f"{info['coordinates']}; {info['text']}")
        lines.append(
            "\nNote: The extracted information may not be fully accurate. Refer to the screenshot to verify.\n")
        return "\n".join(lines)

    def instructions(self) -> str:
        return (
            "---\n"
            "Carefully examine the information above to identify any important content that needs to be recorded. "
            "IMPORTANT: Do not take notes on low-level actions; only keep track of significant textual or visual information relevant to the user's request.\n\n"
            "Provide your output in the following format:\n"
            "### Important Notes ###\n"
            "The updated important notes, combining the old and new ones. If nothing new to record, copy the existing important notes.\n"
        )

    def get_prompt_for_note(self, info_pool: InfoPool) -> str:
        sections = [
            self.section("User Instruction", info_pool.instruction),
            self.section("Overall Plan", info_pool.plan),
            self.section("Current Subgoal", info_pool.current_subgoal),
            self.section("Progress Status", info_pool.progress_status),
            self.section("Existing Important Notes", info_pool.important_notes or "No important notes recorded."),
            self.screen_info(info_pool),
            self.instructions(),
        ]
        return "\n".join(sections)

    def parse_response_note(self, response: str) -> dict:
        important_notes = response.split("### Important Notes ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"important_notes": important_notes}
