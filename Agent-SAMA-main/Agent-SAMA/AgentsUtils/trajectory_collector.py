"""
Trajectory Data Collector for Agent-SAMA

Collects state transition data during task execution for training reflection models.
Output format matches the sample.json structure with:
- Global task and current subgoal context
- State transitions (from_state, action, to_state)
- Pre/post conditions
- Before/after screenshots
- Validity labels (from reflection agent outcomes)

Usage:
    collector = TrajectoryCollector(output_dir="trajectory_data", task_id="task_001")
    collector.start_task(instruction="Open Google Maps and search for coffee shops")
    
    # During execution loop:
    collector.record_transition(
        from_state={...},
        action={...},
        to_state={...},
        before_screenshot="path/to/before.jpg",
        after_screenshot="path/to/after.jpg",
        current_subgoal="Search for coffee shops",
        outcome="A",  # A=success, B=wrong page, C=no change
        previous_steps=collector.get_previous_steps(k=2)
    )
    
    # At task end:
    collector.save()
"""

import os
import json
import shutil
from typing import Dict, List, Optional
from datetime import datetime


class TrajectoryCollector:
    """
    Collects trajectory data in the format required for training reflection models.
    """
    
    # Default output directory (can be changed)
    DEFAULT_OUTPUT_DIR = "trajectory_data"
    
    def __init__(self, output_dir: str = None, task_id: str = None, autosave: bool = True):
        """
        Initialize the trajectory collector.
        
        Args:
            output_dir: Directory to save trajectory data (default: trajectory_data/)
            task_id: Unique identifier for this task run
            autosave: If True, automatically save after each transition (prevents data loss on crash)
        """
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.task_dir = os.path.join(output_dir, self.task_id)
        self.images_dir = os.path.join(self.task_dir, "images")
        
        # Task metadata
        self.instruction = ""
        self.current_subgoal = ""
        
        # Collected transitions
        self.transitions: List[Dict] = []
        self.messages: List[Dict] = []  # Final output format
        
        # Step counter
        self.step_count = 0
        
        # Track if save() has been called to avoid double-saving
        self._saved = False
        
        # Autosave feature - saves after each transition to prevent data loss
        self.autosave = autosave
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
    
    def start_task(self, instruction: str):
        """
        Start collecting data for a new task.
        
        Args:
            instruction: The global task instruction
        """
        self.instruction = instruction
        self.transitions = []
        self.messages = []
        self.step_count = 0
        print(f"[TrajectoryCollector] Started collecting for task: {self.task_id}")
    
    def update_subgoal(self, subgoal: str):
        """Update the current subgoal."""
        self.current_subgoal = subgoal
    
    def record_transition(
        self,
        from_state: Dict,
        action: Dict,
        to_state: Dict,
        before_screenshot: str,
        after_screenshot: str,
        current_subgoal: str,
        outcome: str,
        reason: str = "",
        previous_steps: List[Dict] = None
    ) -> Dict:
        """
        Record a single state transition.
        
        Args:
            from_state: Dict with description, precondition, postcondition
            action: Dict with type, coordinates (if tap/swipe), description
            to_state: Dict with description, precondition, postcondition
            before_screenshot: Path to screenshot before action
            after_screenshot: Path to screenshot after action
            current_subgoal: Current subgoal being executed
            outcome: "A" (success), "B" (wrong page), "C" (no change)
            reason: Explanation for validity
            previous_steps: List of previous step dicts
            
        Returns:
            The recorded transition dict
        """
        self.step_count += 1
        self.current_subgoal = current_subgoal
        
        # Determine validity from outcome
        is_valid = outcome == "A"
        
        # Copy screenshots to output directory
        before_img_name = f"{self.step_count}_before.jpg"
        after_img_name = f"{self.step_count}_after.jpg"
        before_dest = os.path.join(self.images_dir, before_img_name)
        after_dest = os.path.join(self.images_dir, after_img_name)
        
        if os.path.exists(before_screenshot):
            shutil.copy(before_screenshot, before_dest)
        if os.path.exists(after_screenshot):
            shutil.copy(after_screenshot, after_dest)
        
        # Add validity to to_state
        to_state_with_valid = {**to_state, "valid": is_valid}
        
        # Create transition record
        transition = {
            "step": self.step_count,
            "from_state": from_state,
            "action": action,
            "to_state": to_state_with_valid,
            "before_image": f"images/{before_img_name}",
            "after_image": f"images/{after_img_name}",
            "outcome": outcome,
            "reason": reason
        }
        self.transitions.append(transition)
        
        # Generate message format
        message = self._create_message(
            transition=transition,
            previous_steps=previous_steps or self.get_previous_steps(k=2),
            is_valid=is_valid,
            reason=reason
        )
        self.messages.append(message)
        
        print(f"[TrajectoryCollector] Recorded step {self.step_count}: outcome={outcome}, valid={is_valid}")
        
        # Autosave after each transition to prevent data loss on crash/interrupt
        if self.autosave:
            self._incremental_save()
        
        return transition
    
    def _incremental_save(self):
        """
        Save current trajectory data incrementally (called after each transition if autosave=True).
        This ensures no data is lost if the task is interrupted.
        """
        filename = f"trajectory_{self.task_id}.json"
        output_path = os.path.join(self.task_dir, filename)
        jsonl_path = os.path.join(self.task_dir, f"trajectory_{self.task_id}.jsonl")
        
        # Save JSONL format
        with open(jsonl_path, "w") as f:
            for msg in self.messages:
                f.write(json.dumps(msg) + "\n")
        
        # Save full JSON format
        full_data = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "total_steps": self.step_count,
            "success_count": sum(1 for t in self.transitions if t.get("outcome") == "A"),
            "transitions": self.transitions,
            "messages": self.messages
        }
        
        with open(output_path, "w") as f:
            json.dump(full_data, f, indent=2)
    
    def _create_message(
        self,
        transition: Dict,
        previous_steps: List[Dict],
        is_valid: bool,
        reason: str
    ) -> Dict:
        """
        Create a message in the training data format.
        """
        # Build the prompt text
        prompt = self._build_prompt(transition, previous_steps)
        
        # Create user message
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image", "image": transition["before_image"]},
            {"type": "image", "image": transition["after_image"]}
        ]
        
        # Create assistant response
        answer = "Yes" if is_valid else "No"
        response_json = json.dumps({
            "answer": answer,
            "reason": reason or self._generate_default_reason(is_valid, transition)
        })
        
        assistant_content = [
            {"type": "text", "text": response_json}
        ]
        
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    
    def _build_prompt(self, transition: Dict, previous_steps: List[Dict]) -> str:
        """Build the prompt text for the reflection task."""
        prompt = """You are a mobile GUI reflector.
Your job is to determine whether the current state transition is valid — meaning whether it aligns with both the user's global task and the current subgoal.

=== Global Task ===
{instruction}

=== Current Subgoal ===
{subgoal}

=== Current Transition ===
{transition_json}

=== Previous Steps (latest {k}) ===
{previous_steps_json}

=== Image Reference ===
<image_0>: Image BEFORE the action
<image_1>: Image AFTER the action

=== Question ===
Is this transition valid and aligned with the current subgoal?
Answer with "Yes" or "No" and provide a brief reason explaining why."""

        # Format transition for display (exclude image paths)
        transition_display = {
            "from_state": transition["from_state"],
            "action": transition["action"],
            "to_state": {k: v for k, v in transition["to_state"].items() if k != "valid"}
        }
        
        # Format previous steps
        prev_steps_display = []
        for step in previous_steps:
            step_display = {
                "step": step.get("step", 0),
                "from_state": step.get("from_state", {}),
                "action": step.get("action", {}),
                "to_state": step.get("to_state", {})
            }
            prev_steps_display.append(step_display)
        
        return prompt.format(
            instruction=self.instruction,
            subgoal=self.current_subgoal,
            transition_json=json.dumps(transition_display, indent=2),
            k=len(previous_steps),
            previous_steps_json=json.dumps(prev_steps_display, indent=2)
        )
    
    def _generate_default_reason(self, is_valid: bool, transition: Dict) -> str:
        """Generate a default reason based on validity and action."""
        action_type = transition["action"].get("type", "Unknown")
        action_desc = transition["action"].get("description", "")
        
        if is_valid:
            return f"The {action_type} action successfully achieved the expected outcome, advancing toward the subgoal."
        else:
            outcome = transition.get("outcome", "")
            if outcome == "B":
                return f"The {action_type} action led to an unexpected screen state that does not match the subgoal."
            elif outcome == "C":
                return f"The {action_type} action had no effect; the screen remained unchanged and no progress was made."
            else:
                return f"The transition failed to achieve the current subgoal."
    
    def get_previous_steps(self, k: int = 2) -> List[Dict]:
        """
        Get the last k transitions for context.
        
        Args:
            k: Number of previous steps to return
            
        Returns:
            List of previous transition dicts
        """
        if len(self.transitions) <= 1:
            return []
        # Exclude the current step, get last k
        return self.transitions[-(k+1):-1] if len(self.transitions) > k else self.transitions[:-1]
    
    def save(self, filename: str = None):
        """
        Save collected trajectory data to JSON file.
        
        Args:
            filename: Optional custom filename
        """
        if not filename:
            filename = f"trajectory_{self.task_id}.json"
        
        output_path = os.path.join(self.task_dir, filename)
        
        # Save in JSONL format (one message per line) for training
        jsonl_path = os.path.join(self.task_dir, f"trajectory_{self.task_id}.jsonl")
        with open(jsonl_path, "w") as f:
            for msg in self.messages:
                f.write(json.dumps(msg) + "\n")
        
        # Also save full trajectory with metadata
        full_data = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "total_steps": self.step_count,
            "success_count": sum(1 for t in self.transitions if t.get("outcome") == "A"),
            "transitions": self.transitions,
            "messages": self.messages
        }
        
        with open(output_path, "w") as f:
            json.dump(full_data, f, indent=2)
        
        print(f"[TrajectoryCollector] Saved {len(self.messages)} samples to:")
        print(f"  - {output_path}")
        print(f"  - {jsonl_path}")
        
        # Mark as saved to avoid double-saving
        self._saved = True
        
        return output_path, jsonl_path
    
    def get_stats(self) -> Dict:
        """Get statistics about collected data."""
        return {
            "task_id": self.task_id,
            "total_steps": self.step_count,
            "success_steps": sum(1 for t in self.transitions if t.get("outcome") == "A"),
            "failure_steps": sum(1 for t in self.transitions if t.get("outcome") in ["B", "C"]),
            "images_saved": self.step_count * 2  # before + after for each step
        }


def create_transition_from_info_pool(info_pool, action_outcome: str, error_description: str = "") -> Dict:
    """
    Helper function to create a transition dict from InfoPool data.
    
    Args:
        info_pool: The InfoPool instance
        action_outcome: "A", "B", or "C"
        error_description: Description of error if any
        
    Returns:
        Dict with from_state, action, to_state structure
    """
    # Use the state description captured BEFORE the action was executed
    from_state = {
        "description": getattr(info_pool, 'pre_state_description', '') or 'Previous screen state',
        "precondition": getattr(info_pool, 'pre_state_pre_condition', ''),
        "postcondition": getattr(info_pool, 'pre_state_post_condition', '')
    }
    
    action = info_pool.last_action if isinstance(info_pool.last_action, dict) else {
        "type": "Unknown",
        "description": str(info_pool.last_action)
    }
    
    # Add coordinates if it's a tap/swipe
    if isinstance(action, dict):
        action_copy = action.copy()
        if "description" not in action_copy:
            action_copy["description"] = info_pool.last_summary if hasattr(info_pool, 'last_summary') else ""
    else:
        action_copy = {"type": "Unknown", "description": str(action)}
    
    to_state = {
        "description": getattr(info_pool, 'state_description', 'Current screen state'),
        "precondition": getattr(info_pool, 'pre_condition', ''),
        "postcondition": getattr(info_pool, 'post_condition', '')
    }
    
    return {
        "from_state": from_state,
        "action": action_copy,
        "to_state": to_state
    }

