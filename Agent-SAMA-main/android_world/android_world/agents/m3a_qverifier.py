#!/usr/bin/env python3
"""
M3A with Q-Verifier Integration

Keeps ALL M3A logic exactly the same.
ONLY replaces the GPT summarization with Q-Verifier.
"""

import time
from absl import logging
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import m3a_utils
from android_world.agents.qverifier import QVerifier
from android_world.env import interface
from android_world.env import json_action


class M3AWithQVerifier(m3a.M3A):
    """M3A agent with Q-Verifier replacing GPT summarization.
    
    This is M3A with ONE change: instead of calling GPT to summarize
    the action result, we use Q-Verifier to verify and explain.
    """
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        qverifier_model_path: str = "/home/ubuntu/Q-Verifier/GRPO/qwen3_vl_sft_64_grpo_8b/final",
        name: str = 'M3A+QVerifier',
        wait_after_action_seconds: float = 2.0,
        advisory_mode: bool = False,
        replace_gpt_summary: bool = True,
    ):
        """Initialize M3A with Q-Verifier."""
        super().__init__(env, llm, name, wait_after_action_seconds)
        self.qverifier = QVerifier(model_path=qverifier_model_path)
        self.advisory_mode = advisory_mode
        self.replace_gpt_summary = replace_gpt_summary
        self._current_task_name = None
    
    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Perform a step - EXACTLY like M3A, but with Q-Verifier for summarization."""
        
        # ========================================
        # STEP 1-6: EXACTLY COPY FROM M3A
        # ========================================
        step_data = {
            'raw_screenshot': None,
            'before_screenshot_with_som': None,
            'before_ui_elements': [],
            'after_screenshot_with_som': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
        }
        logging.info('----------step %s----------', str(len(self.history) + 1))

        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        before_ui_elements = state.ui_elements
        step_data['before_ui_elements'] = before_ui_elements
        before_ui_elements_list = m3a._generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )
        step_data['raw_screenshot'] = state.pixels.copy()
        before_screenshot_raw = state.pixels.copy()  # Keep raw for Q-Verifier
        before_screenshot = state.pixels.copy()
        for index, ui_element in enumerate(before_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
        step_data['before_screenshot_with_som'] = before_screenshot.copy()

        action_prompt = m3a._action_selection_prompt(
            goal,
            [
                'Step ' + str(i + 1) + '- ' + step_info['summary']
                for i, step_info in enumerate(self.history)
            ],
            before_ui_elements_list,
            self.additional_guidelines,
        )
        step_data['action_prompt'] = action_prompt
        action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt,
            [
                step_data['raw_screenshot'],
                before_screenshot,
            ],
        )

        if is_safe == False:
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

        if not raw_response:
            raise RuntimeError('Error calling LLM in action selection phase.')
        step_data['action_output'] = action_output
        step_data['action_raw_response'] = raw_response

        reason, action = m3a_utils.parse_reason_action_output(action_output)

        if (not reason) or (not action):
            logging.info('Action prompt output is not in the correct format.')
            step_data['summary'] = (
                'Output for action selection is not in the correct format, so no'
                ' action is performed.'
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        logging.info('Action: %s', action)
        logging.info('Reason: %s', reason)
        step_data['action_reason'] = reason

        try:
            converted_action = json_action.JSONAction(
                **agent_utils.extract_json(action),
            )
            step_data['action_output_json'] = converted_action
        except Exception as e:
            logging.info('Failed to convert the output to a valid action.')
            logging.info(str(e))
            step_data['summary'] = (
                'Can not parse the output to a valid action. Please make sure to pick'
                ' the action from the list with required parameters (if any) in the'
                ' correct JSON format!'
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        action_index = converted_action.index
        num_ui_elements = len(before_ui_elements)
        if (
            converted_action.action_type
            in ['click', 'long_press', 'input_text', 'scroll']
            and action_index is not None
        ):
            if action_index >= num_ui_elements:
                logging.info(
                    'Index out of range, prediction index is %s, but the'
                    ' UI element list only has %d elements.',
                    action_index,
                    num_ui_elements,
                )
                step_data['summary'] = (
                    'The parameter index is out of range. Remember the index must be in'
                    ' the UI element list!'
                )
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

            m3a_utils.add_ui_element_mark(
                step_data['raw_screenshot'],
                before_ui_elements[action_index],
                action_index,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )

        if converted_action.action_type == 'status':
            if converted_action.goal_status == 'infeasible':
                logging.info('Agent stopped since it thinks mission impossible.')
            step_data['summary'] = 'Agent thinks the request has been completed.'
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted_action.action_type == 'answer':
            logging.info('Agent answered with: %s', converted_action.text)

        try:
            self.env.execute_action(converted_action)
        except Exception as e:
            logging.info('Failed to execute action.')
            logging.info(str(e))
            step_data['summary'] = (
                'Can not execute the action, make sure to select the action with'
                ' the required parameters (if any) in the correct JSON format!'
            )
            return base_agent.AgentInteractionResult(False, step_data)

        time.sleep(self.wait_after_action_seconds)

        state = self.env.get_state(wait_to_stabilize=False)
        after_screenshot_raw = state.pixels.copy()  # Keep raw for Q-Verifier
        after_ui_elements = state.ui_elements
        after_ui_elements_list = m3a._generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size
        )
        after_screenshot = state.pixels.copy()
        for index, ui_element in enumerate(after_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    after_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )

        m3a_utils.add_screenshot_label(
            step_data['before_screenshot_with_som'], 'before'
        )
        m3a_utils.add_screenshot_label(after_screenshot, 'after')
        step_data['after_screenshot_with_som'] = after_screenshot.copy()

        # ========================================
        # STEP 7: THIS IS THE ONLY CHANGE!
        # Replace GPT summarization with Q-Verifier
        # ========================================
        if self.replace_gpt_summary:
            # Use Q-Verifier instead of GPT for summarization
            print(f'[Q-Verifier] Running verification...')
            
            try:
                before_desc = self._build_state_description(before_ui_elements)
                after_desc = self._build_state_description(after_ui_elements)
                action_desc = self._build_action_description(converted_action, before_ui_elements, reason)
                
                verification_result = self.qverifier.verify(
                    before_screenshot=before_screenshot_raw,
                    after_screenshot=after_screenshot_raw,
                    global_task=goal,
                    subgoal=reason,
                    action_type=converted_action.action_type,
                    action_description=action_desc,
                    before_description=before_desc,
                    after_description=after_desc,
                    previous_steps=[s.get('summary', '')[:100] for s in self.history[-3:]],
                )
                
                step_num = len(self.history) + 1
                if verification_result.success:
                    print(f'[Q-Verifier #{step_num}] ✅ Yes | {verification_result.reason[:80]}...')
                else:
                    print(f'[Q-Verifier #{step_num}] ❌ No | {verification_result.reason[:80]}...')
                
                # Use Q-Verifier output as summary (same format as M3A)
                qv_reason = verification_result.reason[:200]
                if verification_result.success:
                    summary = f'SUCCESS: {qv_reason}'
                else:
                    summary = f'FAILED: {qv_reason}'
                    
            except Exception as e:
                print(f'[Q-Verifier] ERROR: {e}')
                import traceback
                traceback.print_exc()
                summary = f'Q-Verifier error, assuming success: {e}'
            
            step_data['summary_prompt'] = None
            step_data['summary'] = f'Action selected: {action}. {summary}'
            step_data['summary_raw_response'] = None
            
        else:
            # Original M3A: Use GPT for summarization
            summary_prompt = m3a._summarize_prompt(
                action,
                reason,
                goal,
                before_ui_elements_list,
                after_ui_elements_list,
            )
            summary, is_safe, raw_response = self.llm.predict_mm(
                summary_prompt,
                [before_screenshot, after_screenshot],
            )

            if is_safe == False:
                summary = "Summary triggered LLM safety classifier."

            if not raw_response:
                logging.info('Error calling LLM in summarization phase.')
                step_data['summary'] = 'Some error occurred calling LLM during summarization phase.'
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

            step_data['summary_prompt'] = summary_prompt
            step_data['summary'] = f'Action selected: {action}. {summary}'
            step_data['summary_raw_response'] = raw_response

        logging.info('Summary: %s', step_data['summary'])
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
    
    # ========================================
    # HELPER METHODS FOR Q-VERIFIER
    # ========================================
    def _build_state_description(self, ui_elements: list) -> str:
        """Build screen description from UI elements."""
        if not ui_elements:
            return "The screen is empty or loading."
        
        screen_context = self._identify_screen_context(ui_elements)
        buttons = []
        visible_text = []
        
        for elem in ui_elements[:20]:
            text = elem.text or ""
            if elem.is_clickable and text and len(text) < 25:
                buttons.append(text)
            elif text and len(text) < 40:
                visible_text.append(text)
        
        parts = [f"The screen displays {screen_context}."]
        if buttons:
            parts.append(f"Available actions: {', '.join(buttons[:6])}.")
        if visible_text:
            unique_text = list(dict.fromkeys(visible_text[:4]))
            parts.append(f"Visible text: {', '.join(unique_text)}.")
        
        return " ".join(parts)
    
    def _build_action_description(self, action, ui_elements: list, reason: str) -> str:
        """Build action description."""
        action_type = action.action_type
        
        if action_type == "click" and action.index is not None:
            elem = ui_elements[action.index] if action.index < len(ui_elements) else None
            if elem:
                elem_text = elem.text or elem.content_description or f"element {action.index}"
                return f"I will tap on '{elem_text}'. {reason}"
        elif action_type == "input_text":
            return f"I will type '{action.text}'. {reason}"
        elif action_type == "scroll":
            return f"I will scroll {getattr(action, 'direction', 'down')}. {reason}"
        elif action_type == "open_app":
            return f"I will open {getattr(action, 'app_name', 'the app')}. {reason}"
        
        return reason
    
    def _identify_screen_context(self, ui_elements: list) -> str:
        """Identify what screen/app we're on."""
        all_text = " ".join([
            elem.text or getattr(elem, 'content_description', '') or ""
            for elem in ui_elements[:15]
        ]).lower()
        
        home_apps = ['gmail', 'photos', 'youtube', 'chrome', 'phone', 'messages']
        if sum(1 for app in home_apps if app in all_text) >= 3:
            return "the Android home screen with app icons"
        
        if "markor" in all_text: return "the Markor app"
        elif "calendar" in all_text: return "a calendar app"
        elif "expense" in all_text: return "the Pro Expense app"
        elif "message" in all_text or "sms" in all_text: return "a messaging app"
        elif "recipe" in all_text: return "a recipe app"
        elif "task" in all_text: return "a task app"
        elif "vlc" in all_text: return "VLC media player"
        elif "camera" in all_text: return "the Camera app"
        elif "file" in all_text: return "a file manager"
        elif "settings" in all_text: return "Settings"
        
        return f"an app screen with {len(ui_elements)} elements"
    
    def set_task(self, task_name: str):
        """Set current task for logging."""
        self._current_task_name = task_name
        self.qverifier.set_task(task_name)
    
    def mark_task_complete(self, success: bool):
        """Mark task as completed."""
        task_name = self._current_task_name or "unknown"
        self.qverifier.mark_task_complete(task_name, success)
