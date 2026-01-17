# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent-SAMA with Q-Verifier: Replaces GPT ReflectionAgent with local Qwen3-VL.

This module extends Agent-SAMA to use Q-Verifier for action verification,
replacing the GPT-based ReflectionAgent with a local Qwen3-VL model.

Benefits:
- Faster verification (~0.5s vs ~2s)
- No API cost for reflection step
- Q-Verifier trained specifically for verification tasks
"""

import logging
import os
from typing import Dict

import numpy as np
from PIL import Image

from android_world.agents import agent_sama
from android_world.agents.qverifier import QVerifier
from android_world.env import interface


class AgentSAMAWithQVerifier(agent_sama.AgentSAMA):
    """
    Agent-SAMA with Q-Verifier for action verification.
    
    Extends the base Agent-SAMA adapter but replaces the GPT ReflectionAgent
    with a local Qwen3-VL Q-Verifier model.
    """
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        model_name: str = "gpt-4o-2024-11-20",
        name: str = "AgentSAMA+QVerifier",
        wait_after_action_seconds: float = 5.0,
        use_perception: bool = True,
        use_fsm: bool = True,
        use_recovery: bool = True,
        collect_trajectory: bool = False,
        trajectory_output_dir: str = None,
        adb_path: str = None,
        qverifier_model_path: str = "/home/ubuntu/Q-Verifier/GRPO/qwen3_vl_sft_64_grpo_8b/final",
        advisory_mode: bool = False,
        replace_gpt_reflection: bool = True,
    ):
        """
        Initialize Agent-SAMA with Q-Verifier.
        
        Args:
            qverifier_model_path: Path to the Q-Verifier model
            advisory_mode: If True, Q-Verifier logs but doesn't affect outcomes
            replace_gpt_reflection: If True, Q-Verifier fully replaces GPT ReflectionAgent
        """
        super().__init__(
            env=env,
            model_name=model_name,
            name=name,
            wait_after_action_seconds=wait_after_action_seconds,
            use_perception=use_perception,
            use_fsm=use_fsm,
            use_recovery=use_recovery,
            collect_trajectory=collect_trajectory,
            trajectory_output_dir=trajectory_output_dir,
            adb_path=adb_path,
        )
        
        self.advisory_mode = advisory_mode
        self.replace_gpt_reflection = replace_gpt_reflection
        self.qverifier_model_path = qverifier_model_path
        
        # Initialize Q-Verifier
        print(f"[Q-Verifier] Loading model from {qverifier_model_path}...")
        self.qverifier = QVerifier(model_path=qverifier_model_path)
        print("[Q-Verifier] Model loaded successfully!")
    
    def set_task_info(self, task_name: str, log_dir: str):
        """Set task info for logging."""
        super().set_task_info(task_name, log_dir)
        
        # Also set task for Q-Verifier logging
        if hasattr(self, 'qverifier') and self.qverifier:
            self.qverifier.set_task(task_name)
    
    def _run_reflection(self, before_screenshot: str, after_screenshot: str, action_str: str) -> Dict:
        """
        Run Q-Verifier for action verification.
        
        Replaces the GPT-based ReflectionAgent with local Qwen3-VL model.
        
        Args:
            before_screenshot: Path to screenshot before action
            after_screenshot: Path to screenshot after action
            action_str: The action that was executed
            
        Returns:
            Dict with 'outcome' (A/B/C), 'error_description', 'progress_status'
        """
        print("\n### Q-Verifier Reflection ... ###\n")
        
        # Load screenshots as numpy arrays
        try:
            before_img = np.array(Image.open(before_screenshot).convert('RGB'))
            after_img = np.array(Image.open(after_screenshot).convert('RGB'))
        except Exception as e:
            logging.error(f"Failed to load screenshots: {e}")
            # Fallback to GPT ReflectionAgent
            if self.advisory_mode:
                print("[Q-Verifier] Falling back to GPT ReflectionAgent due to image load error")
                return super()._run_reflection(before_screenshot, after_screenshot, action_str)
            return {
                'outcome': 'C',
                'error_description': f'Failed to load screenshots: {e}',
                'progress_status': ''
            }
        
        # Extract action details from action string
        action_type = "unknown"
        action_description = action_str
        try:
            from AgentsUtils.common_utils import extract_json_object
            action_obj = extract_json_object(action_str)
            if action_obj:
                action_type = action_obj.get("name", "unknown")
                args = action_obj.get("arguments", {}) or {}
                if action_type.lower() == "tap":
                    action_description = f"Tap at ({args.get('x', 0)}, {args.get('y', 0)})"
                elif action_type.lower() == "type":
                    action_description = f"Type text: '{args.get('text', '')}'"
                elif action_type.lower() == "swipe":
                    action_description = f"Swipe from ({args.get('x1', 0)}, {args.get('y1', 0)}) to ({args.get('x2', 0)}, {args.get('y2', 0)})"
                else:
                    action_description = f"{action_type} with args {args}"
        except Exception as e:
            logging.warning(f"Could not parse action: {e}")
        
        # Build state descriptions from info_pool
        before_description = self._build_state_description(self.info_pool.perception_infos_pre if self.info_pool else [])
        after_description = self._build_state_description(self.info_pool.perception_infos_post if self.info_pool else [])
        
        # Get previous steps for context
        previous_steps = []
        if self.info_pool and self.info_pool.summary_history:
            previous_steps = self.info_pool.summary_history[-5:]
        
        # Run Q-Verifier
        result = self.qverifier.verify(
            before_screenshot=before_img,
            after_screenshot=after_img,
            global_task=self.info_pool.instruction if self.info_pool else "",
            subgoal=self.info_pool.current_subgoal if self.info_pool else "",
            action_type=action_type,
            action_description=action_description,
            before_description=before_description,
            after_description=after_description,
            previous_steps=previous_steps,
        )
        
        # Map Q-Verifier output to Agent-SAMA format
        if result.success:
            outcome = "A"
            error_description = ""
            progress_status = result.reason
            print(f"Outcome: A ✓ Q-Verifier: Action verified successful")
        else:
            # Determine if it's B (wrong page) or C (no change)
            outcome = self._determine_failure_type(before_img, after_img)
            error_description = result.reason
            progress_status = ""
            if outcome == "B":
                print(f"Outcome: B ✗ Q-Verifier: Wrong page - {error_description[:100]}")
            else:
                print(f"Outcome: C ✗ Q-Verifier: No change - {error_description[:100]}")
        
        print(f"Q-Verifier Reason: {result.reason[:200]}...")
        
        # In advisory mode, also run GPT ReflectionAgent and compare
        if self.advisory_mode:
            print("\n[Advisory Mode] Also running GPT ReflectionAgent...")
            gpt_result = super()._run_reflection(before_screenshot, after_screenshot, action_str)
            print(f"[Advisory] GPT outcome: {gpt_result['outcome']}, Q-Verifier outcome: {outcome}")
            if gpt_result['outcome'] != outcome:
                print(f"[Advisory] DISAGREEMENT: GPT={gpt_result['outcome']}, QV={outcome}")
            # In advisory mode, return GPT's result (Q-Verifier is just logging)
            return gpt_result
        
        return {
            'outcome': outcome,
            'error_description': error_description,
            'progress_status': progress_status
        }
    
    def _build_state_description(self, perception_infos: list) -> str:
        """Build a rich state description from perception info."""
        if not perception_infos:
            return "No UI elements detected"
        
        # Count total elements
        total = len(perception_infos)
        
        # Extract key text elements
        key_elements = []
        for info in perception_infos[:10]:
            text = info.get('text', '')
            if text and text != "icon: None":
                # Clean up the text
                text_clean = text.strip()[:30]
                if text_clean:
                    key_elements.append(f"'{text_clean}'")
        
        if key_elements:
            elements_str = ", ".join(key_elements[:5])
            return f"Screen with {total} UI elements. Key elements: {elements_str}"
        else:
            return f"Screen with {total} UI elements"
    
    def _determine_failure_type(self, before_img: np.ndarray, after_img: np.ndarray) -> str:
        """
        Determine if failure is B (wrong page) or C (no change).
        
        Uses simple image comparison to detect if screen changed.
        """
        try:
            # Resize images for faster comparison
            from PIL import Image
            before_small = np.array(Image.fromarray(before_img).resize((100, 100)))
            after_small = np.array(Image.fromarray(after_img).resize((100, 100)))
            
            # Calculate mean absolute difference
            diff = np.abs(before_small.astype(float) - after_small.astype(float)).mean()
            
            # Threshold: if diff > 10, screen changed significantly
            if diff > 10:
                return "B"  # Wrong page (screen changed but action failed)
            else:
                return "C"  # No change
        except Exception as e:
            logging.warning(f"Could not compare images: {e}")
            return "C"  # Default to "no change" if comparison fails

