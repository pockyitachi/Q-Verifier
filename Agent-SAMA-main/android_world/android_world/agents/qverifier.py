#!/usr/bin/env python3
"""
Q-Verifier Integration for AndroidWorld

Provides action verification using a trained Qwen3-VL model to determine
if an action successfully achieved its intended subgoal.
"""

import os
# Disable Unsloth compilation to avoid potential VLM detection issues
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from PIL import Image


@dataclass
class VerificationResult:
    """Result of Q-Verifier verification."""
    success: bool  # True if action achieved subgoal
    answer: str    # "Yes" or "No"
    reason: str    # Detailed reasoning
    confidence: float  # Confidence score (0-1)
    raw_response: str  # Full model response
    recovery_hint: str = ""  # Suggested recovery action when failed


class QVerifier:
    """Q-Verifier for action verification in mobile GUI agents.
    
    Uses a fine-tuned Qwen3-VL model to verify if screen transitions
    align with intended subgoals.
    """
    
    # Verification prompt template - simplified without precondition/postcondition
    # (M3A doesn't have these, and generic templates hurt more than help)
    PROMPT_TEMPLATE = """You are a mobile GUI reflector.
Your job is to determine whether the current state transition is valid — meaning whether it aligns with both the user's global task and the current subgoal.

=== Global Task ===
{global_task}

=== Current Subgoal ===
{subgoal}

=== Current Transition ===
{{
  "from_state": {{
    "description": "{before_description}"
  }},
  "action": {{
    "type": "{action_type}",
    "description": "{action_description}",
    "arguments": {action_arguments}
  }},
  "to_state": {{
    "description": "{after_description}"
  }}
}}

=== Previous Steps (latest {num_steps}) ===
{previous_steps}

=== Image Reference ===
<image_0>: Image BEFORE the action
<image_1>: Image AFTER the action

=== Question ===
Is this transition valid and aligned with the current subgoal?
Answer with "Yes" or "No" and provide a brief reason explaining why."""

    def __init__(
        self,
        model_path: str = "/home/ubuntu/Q-Verifier/GRPO/qwen3_vl_sft_64_grpo_8b/final",
        max_new_tokens: int = 512,
        device: str = "cuda",
        log_dir: str = "/home/ubuntu/Q-Verifier/Agent-SAMA-main/android_world/qverifier_logs",
        enable_logging: bool = True,
    ):
        """Initialize Q-Verifier with a trained model.
        
        Args:
            model_path: Path to the trained Q-Verifier model
            max_new_tokens: Maximum tokens to generate
            device: Device to run model on
            log_dir: Directory to save verification logs
            enable_logging: Whether to log inputs/outputs
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
        
        # Logging setup
        self.enable_logging = enable_logging
        self.base_log_dir = Path(log_dir)
        self.verification_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_task_name = None
        self.log_file = None
        
        # Session-level statistics for SR tracking
        self.session_stats = {
            "total_verifications": 0,
            "total_yes": 0,
            "total_no": 0,
            "tasks_completed": 0,      # Total tasks that finished
            "tasks_successful": 0,     # Tasks that achieved their goal
            "tasks": {},  # task_name -> {"verifications": n, "yes": n, "no": n, "completed": bool, "success": bool}
        }
        
        # Create session-specific folder for organized logs
        if self.enable_logging:
            self.log_dir = self.base_log_dir / f"session_{self.session_id}"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Q-Verifier] Session logs: {self.log_dir}/")
    
    def set_task(self, task_name: str):
        """Set current task name for per-task logging.
        
        Args:
            task_name: Name of the current task
        """
        self.current_task_name = task_name
        self.verification_count = 0  # Reset count for new task
        
        # Initialize task stats
        if task_name not in self.session_stats["tasks"]:
            self.session_stats["tasks"][task_name] = {
                "verifications": 0,
                "yes": 0,
                "no": 0,
                "completed": False,
                "success": None,  # True/False/None
            }
        
        if self.enable_logging:
            # Create per-task log file inside session folder
            safe_name = task_name.replace("/", "_").replace(" ", "_")
            self.log_file = self.log_dir / f"task_{safe_name}.jsonl"
            print(f"[Q-Verifier] Task log: {self.log_file}")
    
    def mark_task_complete(self, task_name: str = None, success: bool = False):
        """Mark a task as completed with success/failure status.
        
        Call this when AndroidWorld reports task completion.
        
        Args:
            task_name: Name of the task (uses current_task_name if None)
            success: Whether the task was successfully completed
        """
        task = task_name or self.current_task_name
        if not task:
            return
        
        if task in self.session_stats["tasks"]:
            task_stats = self.session_stats["tasks"][task]
            if not task_stats["completed"]:  # Only count first completion
                task_stats["completed"] = True
                task_stats["success"] = success
                self.session_stats["tasks_completed"] += 1
                if success:
                    self.session_stats["tasks_successful"] += 1
                
                # Update summary
                self._write_summary()
                
                # Print status
                status = "✅ SUCCESS" if success else "❌ FAILED"
                sr = self.session_stats["tasks_successful"]
                total = self.session_stats["tasks_completed"]
                rate = (sr / total * 100) if total > 0 else 0
                print(f"\n[Q-Verifier] Task '{task}' {status}")
                print(f"[Q-Verifier] Session SR: {sr}/{total} = {rate:.1f}%\n")
        
    def load_model(self):
        """Load the Q-Verifier model (lazy loading)."""
        if self._loaded:
            return
            
        print(f"[Q-Verifier] Loading model from {self.model_path}...")
        
        from unsloth import FastModel
        
        self.model, self.processor = FastModel.from_pretrained(
            self.model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        self.model.for_inference()
        self._loaded = True
        print("[Q-Verifier] Model loaded successfully!")
        
    def verify(
        self,
        before_screenshot: np.ndarray,
        after_screenshot: np.ndarray,
        global_task: str,
        subgoal: str,
        action_type: str,
        action_description: str,
        action_arguments: dict = None,
        before_description: str = "Screenshot before the action",
        after_description: str = "Screenshot after the action",
        previous_steps: list = None,
        # Deprecated - kept for backward compat but ignored
        before_precondition: str = "",
        before_postcondition: str = "",
        after_precondition: str = "",
        after_postcondition: str = "",
    ) -> VerificationResult:
        """Verify if an action achieved its subgoal.
        
        Args:
            before_screenshot: Screenshot before action (numpy array)
            after_screenshot: Screenshot after action (numpy array)
            global_task: The overall user goal/task
            subgoal: The intended subgoal of this specific action
            action_type: Type of action (e.g., "click", "input_text", "scroll")
            action_description: Description of what the action does
            action_arguments: Dict of action arguments (x, y, text, etc.)
            before_description: Description of the before state
            after_description: Description of the after state
            previous_steps: List of previous step summaries
            
        Returns:
            VerificationResult with success status and reasoning
        """
        if not self._loaded:
            self.load_model()
            
        # Convert numpy arrays to PIL Images
        before_img = Image.fromarray(before_screenshot).convert("RGB")
        after_img = Image.fromarray(after_screenshot).convert("RGB")
        
        # Resize images to reasonable size
        max_size = 512
        before_img_resized = before_img.copy()
        after_img_resized = after_img.copy()
        
        if max(before_img_resized.size) > max_size:
            before_img_resized.thumbnail((max_size, max_size), Image.LANCZOS)
        if max(after_img_resized.size) > max_size:
            after_img_resized.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # Format previous steps
        if previous_steps:
            steps_str = json.dumps(previous_steps[-5:], indent=2)  # Last 5 steps
            num_steps = len(previous_steps[-5:])
        else:
            steps_str = "[]"
            num_steps = 0
        
        # Format action arguments as JSON (matching training format)
        if action_arguments:
            args_str = json.dumps(action_arguments)
        else:
            args_str = "{}"
        
        # Build prompt - simplified without precondition/postcondition
        prompt_text = self.PROMPT_TEMPLATE.format(
            global_task=global_task,
            subgoal=subgoal,
            before_description=before_description,
            action_type=action_type,
            action_description=action_description,
            action_arguments=args_str,
            after_description=after_description,
            num_steps=num_steps,
            previous_steps=steps_str,
        )
        
        # Build message with images - matching training format order
        # Training format: text prompt first, then before image, then after image
        input_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image", "image": before_img_resized},
            {"type": "image", "image": after_img_resized},
        ]
        
        input_messages = [{"role": "user", "content": input_content}]
        
        # Process and generate
        inputs = self.processor.apply_chat_template(
            input_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                  for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        
        # Decode response
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
        elif "assistant" in generated_text.lower():
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text[-500:]
            
        # Parse response
        result = self._parse_response(response)
        
        # Log the verification
        if self.enable_logging:
            self._log_verification(
                global_task=global_task,
                subgoal=subgoal,
                action_type=action_type,
                action_description=action_description,
                before_description=before_description,
                after_description=after_description,
                previous_steps=previous_steps,
                prompt_text=prompt_text,
                result=result,
            )
        
        return result
    
    def _log_verification(
        self,
        global_task: str,
        subgoal: str,
        action_type: str,
        action_description: str,
        before_description: str,
        after_description: str,
        previous_steps: list,
        prompt_text: str,
        result: 'VerificationResult',
    ):
        """Log verification input and output to file."""
        self.verification_count += 1
        
        # Create default log file if not set (for backwards compatibility)
        if self.log_file is None:
            self.log_file = self.log_dir / f"qverifier_session_{self.session_id}.jsonl"
        
        log_entry = {
            "verification_id": self.verification_count,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "task_name": self.current_task_name,
            "input": {
                "global_task": global_task,
                "subgoal": subgoal,
                "action_type": action_type,
                "action_description": action_description,
                "before_description": before_description,
                "after_description": after_description,
                "previous_steps": previous_steps or [],
                "full_prompt": prompt_text,
            },
            "output": {
                "success": result.success,
                "answer": result.answer,
                "reason": result.reason,
                "confidence": result.confidence,
                "raw_response": result.raw_response,
            },
        }
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Update session stats
        self._update_stats(result)
        
        # Also print summary
        status = "✅" if result.success else "❌"
        print(f"[Q-Verifier #{self.verification_count}] {status} {result.answer} | {result.reason[:80]}...")
    
    def _update_stats(self, result: 'VerificationResult'):
        """Update session statistics."""
        self.session_stats["total_verifications"] += 1
        
        if result.success:
            self.session_stats["total_yes"] += 1
        else:
            self.session_stats["total_no"] += 1
        
        # Update per-task stats
        if self.current_task_name:
            task_stats = self.session_stats["tasks"].get(self.current_task_name)
            if task_stats:
                task_stats["verifications"] += 1
                if result.success:
                    task_stats["yes"] += 1
                else:
                    task_stats["no"] += 1
        
        # Write updated summary to file
        self._write_summary()
    
    def _write_summary(self):
        """Write session summary to a file."""
        if not self.enable_logging:
            return
        
        summary_file = self.log_dir / "SESSION_SUMMARY.txt"
        
        # Task success rate (the main metric!)
        tasks_completed = self.session_stats["tasks_completed"]
        tasks_successful = self.session_stats["tasks_successful"]
        task_sr = (tasks_successful / tasks_completed * 100) if tasks_completed > 0 else 0
        
        # Verification stats
        total = self.session_stats["total_verifications"]
        yes = self.session_stats["total_yes"]
        no = self.session_stats["total_no"]
        yes_rate = (yes / total * 100) if total > 0 else 0
        
        lines = [
            "=" * 60,
            "Q-VERIFIER SESSION SUMMARY",
            f"Session: {self.session_id}",
            "=" * 60,
            "",
            "╔══════════════════════════════════════════════════════════╗",
            f"║  TASK SUCCESS RATE: {tasks_successful}/{tasks_completed} = {task_sr:.1f}%".ljust(59) + "║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            "VERIFICATION STATS:",
            f"  Total Verifications: {total}",
            f"  Yes (Action OK):     {yes} ({yes_rate:.1f}%)",
            f"  No (Action Failed):  {no} ({100-yes_rate:.1f}%)",
            "",
            "-" * 60,
            "PER-TASK BREAKDOWN:",
            "-" * 60,
        ]
        
        for task_name, stats in self.session_stats["tasks"].items():
            t_total = stats["verifications"]
            t_yes = stats["yes"]
            t_no = stats["no"]
            t_completed = stats.get("completed", False)
            t_success = stats.get("success", None)
            
            # Task status icon
            if not t_completed:
                status = "🔄 Running"
            elif t_success:
                status = "✅ SUCCESS"
            else:
                status = "❌ FAILED"
            
            t_rate = (t_yes / t_total * 100) if t_total > 0 else 0
            lines.append(f"  {task_name}: {status}")
            lines.append(f"    Verifications: {t_total} | Yes: {t_yes} ({t_rate:.0f}%) | No: {t_no}")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"Last updated: {datetime.now().isoformat()}")
        lines.append("=" * 60)
        
        with open(summary_file, 'w') as f:
            f.write("\n".join(lines))
    
    def _parse_response(self, response: str) -> VerificationResult:
        """Parse model response into VerificationResult."""
        answer = "No"
        reason = response
        confidence = 0.5
        recovery_hint = ""
        
        # Try to parse JSON response
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
                answer = parsed.get("answer", "No")
                reason = parsed.get("reason", response)
                recovery_hint = parsed.get("recovery_hint", parsed.get("suggested_action", ""))
        except json.JSONDecodeError:
            # Fallback to simple parsing
            if "yes" in response.lower()[:50]:
                answer = "Yes"
            else:
                answer = "No"
                
        success = answer.lower() == "yes"
        
        # Extract recovery hint from reason if not explicitly provided
        if not success and not recovery_hint:
            recovery_hint = self._extract_recovery_hint(reason)
        
        # Note: This is a heuristic for response format quality, NOT a trained confidence
        # Q-Verifier was not trained with confidence scores
        if '"answer"' in response and ('"Yes"' in response or '"No"' in response):
            confidence = 1.0  # Proper JSON format
        else:
            confidence = 0.8  # Fallback parsing used
            
        return VerificationResult(
            success=success,
            answer=answer,
            reason=reason,
            confidence=confidence,
            raw_response=response,
            recovery_hint=recovery_hint,
        )
    
    def _extract_recovery_hint(self, reason: str) -> str:
        """Extract recovery hint from the reason text.
        
        Looks for patterns like:
        - "should try..."
        - "need to..."
        - "suggest..."
        - "recommend..."
        """
        import re
        
        patterns = [
            r"(?:should|could|might|try to|need to|must)\s+([^.]+)",
            r"(?:suggest|recommend)\s+([^.]+)",
            r"(?:instead|rather)\s+([^.]+)",
            r"(?:FINAL|RECOVERY|SUGGESTION|HINT)[:\s]+([^.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reason, re.IGNORECASE)
            if match:
                hint = match.group(1).strip()
                if len(hint) > 10:  # Only return meaningful hints
                    return hint
        
        # If no explicit hint, generate one based on common failure patterns
        reason_lower = reason.lower()
        if "no change" in reason_lower or "same screen" in reason_lower:
            return "Try a different tap location or scroll to reveal more elements"
        elif "wrong page" in reason_lower or "unexpected" in reason_lower:
            return "Press Back to return and try a different approach"
        elif "not found" in reason_lower or "missing" in reason_lower:
            return "Scroll or navigate to find the target element"
        
        return ""


class QVerifierWrapper:
    """Wrapper for Q-Verifier that matches the MultimodalLlmWrapper interface.
    
    Can be used as a drop-in replacement for GPT verification.
    """
    
    def __init__(
        self,
        model_path: str = "/home/ubuntu/Q-Verifier/GRPO/qwen3_vl_sft_64_grpo_8b/final",
    ):
        self.verifier = QVerifier(model_path=model_path)
        
    def verify_action(
        self,
        before_screenshot: np.ndarray,
        after_screenshot: np.ndarray,
        global_task: str,
        subgoal: str,
        action_type: str,
        action_description: str,
        action_arguments: dict = None,
        before_description: str = "Screenshot before the action",
        after_description: str = "Screenshot after the action",
        previous_steps: list = None,
    ) -> Tuple[bool, str]:
        """Verify an action and return (success, reason).
        
        Returns:
            Tuple of (success: bool, reason: str)
        """
        result = self.verifier.verify(
            before_screenshot=before_screenshot,
            after_screenshot=after_screenshot,
            global_task=global_task,
            subgoal=subgoal,
            action_type=action_type,
            action_description=action_description,
            action_arguments=action_arguments,
            before_description=before_description,
            after_description=after_description,
            previous_steps=previous_steps,
        )
        return result.success, result.reason


# Singleton instance for easy access
_verifier_instance: Optional[QVerifier] = None


def get_qverifier(model_path: str = None) -> QVerifier:
    """Get or create Q-Verifier singleton instance."""
    global _verifier_instance
    if _verifier_instance is None:
        if model_path:
            _verifier_instance = QVerifier(model_path=model_path)
        else:
            _verifier_instance = QVerifier()
    return _verifier_instance

