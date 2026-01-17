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

"""Agent-SAMA: State-Aware Mobile Assistant for AndroidWorld.

This module integrates the FULL Agent-SAMA framework with AndroidWorld:
- Uses Agent-SAMA's ScreenParser (OCR + GroundingDINO) for perception
- Uses Agent-SAMA's Actor with coordinate-based actions
- Uses Agent-SAMA's FSM StateAgent for state tracking and recovery
- Executes actions via ADB

Reference: Agent-SAMA paper (AAAI 2026)
"""

import copy
import json
import logging
import os
import re
import sys
import time
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import json_action

# Add Agent-SAMA path for imports
AGENT_SAMA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Agent-SAMA')
AGENT_SAMA_PATH = os.path.abspath(AGENT_SAMA_PATH)
if AGENT_SAMA_PATH not in sys.path:
    sys.path.insert(0, AGENT_SAMA_PATH)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Mock modelscope if not available (datasets version conflict)
try:
    import modelscope
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['modelscope'] = MagicMock()
    sys.modules['modelscope.pipelines'] = MagicMock()
    sys.modules['modelscope.utils'] = MagicMock()
    sys.modules['modelscope.utils.constant'] = MagicMock()
    sys.modules['modelscope.utils.constant'].Tasks = MagicMock()
    logging.warning("modelscope mocked due to datasets version conflict - perception will be limited")

# Import Agent-SAMA components
try:
    from Agents.planner import Planner
    from Agents.actor import Actor
    from Agents.reflection_agent import ReflectionAgent
    from Agents.state_agent import StateAgent
    from Agents.mentor import Mentor
    from Agents.info_pool import InfoPool
    from Agents.screen_parser import ScreenParser, load_perception_models
    from AgentsUtils.chat_utils import (
        inference_chat, add_response, add_response_with_images
    )
    from AgentsUtils.common_utils import extract_json_object
    from AgentsUtils.trajectory_collector import TrajectoryCollector, create_transition_from_info_pool
    from constant import (
        BACKBONE_TYPE, OPENAI_API_KEY, QWEN_API_KEY, GEMINI_API_KEY, CLAUDE_API_KEY,
        OPENAI_API_URL, GEMINI_API_URL, CLAUDE_API_URL, USAGE_TRACKING_JSONL,
        ATOMIC_ACTION_SIGNATURES, DEFAULT_PERCEPTION_ARGS, INIT_SHORTCUTS, INIT_TIPS,
        LLM_Judge
    )
    # Set KNOWLEDGE_REFLECTION_MODEL based on BACKBONE_TYPE (matches infer.py logic)
    if BACKBONE_TYPE == "OpenAI":
        KNOWLEDGE_REFLECTION_MODEL = "gpt-4o-2024-11-20"
    elif BACKBONE_TYPE == "Gemini":
        KNOWLEDGE_REFLECTION_MODEL = "gemini-1.5-pro-latest"
    elif BACKBONE_TYPE == "Claude":
        KNOWLEDGE_REFLECTION_MODEL = "claude-3-5-sonnet-20241022"
    else:
        KNOWLEDGE_REFLECTION_MODEL = "gpt-4o-2024-11-20"
    AGENT_SAMA_AVAILABLE = True
    logging.info("Agent-SAMA modules loaded successfully")
except ImportError as e:
    logging.warning(f"Agent-SAMA not available: {e}")
    AGENT_SAMA_AVAILABLE = False
    LLM_Judge = True
    INIT_SHORTCUTS = {}
    INIT_TIPS = ""


def get_reasoning_model_api_response(chat, model_type=None, model=None, temperature=0.0):
    """Get API response using the configured backbone model."""
    if not AGENT_SAMA_AVAILABLE:
        raise RuntimeError("Agent-SAMA modules not available")
    
    # Read API keys from environment variables (with fallback to constant.py)
    openai_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    gemini_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    claude_key = os.environ.get("CLAUDE_API_KEY", CLAUDE_API_KEY)
    
    # Default to BACKBONE_TYPE from constant.py
    if model_type is None:
        model_type = BACKBONE_TYPE
    
    # Default model based on backend type
    if model is None:
        if model_type == "OpenAI":
            model = "gpt-4o-2024-11-20"
        elif model_type == "Gemini":
            model = "gemini-1.5-pro-latest"
        elif model_type == "Claude":
            model = "claude-3-5-sonnet-20241022"
        else:
            model = "gpt-4o-2024-11-20"
    
    if model_type == "OpenAI":
        return inference_chat(chat, model, OPENAI_API_URL, openai_key, 
                              usage_tracking_jsonl=USAGE_TRACKING_JSONL, temperature=temperature)
    elif model_type == "Gemini":
        return inference_chat(chat, model, GEMINI_API_URL, gemini_key, 
                              usage_tracking_jsonl=USAGE_TRACKING_JSONL, temperature=temperature)
    elif model_type == "Claude":
        return inference_chat(chat, model, CLAUDE_API_URL, claude_key, 
                              usage_tracking_jsonl=USAGE_TRACKING_JSONL, temperature=temperature)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Alias for compatibility
get_model_api_response = get_reasoning_model_api_response


def is_task_finished(current_subgoal: str) -> bool:
    """Check if the current subgoal indicates task completion."""
    subgoal_clean = current_subgoal.strip().lower().rstrip('.!,;:')
    
    # Exact match phrases
    exact_finish_phrases = [
        "finished", "done", "complete", "completed",
        "task finished", "task completed", "task done", "task complete",
        "task is finished", "task is completed", "task is done", "task is complete",
        "goal achieved", "goal accomplished", "mission accomplished",
        "all done", "all finished", "all complete", "all completed",
        "no further actions required", "no more actions needed",
        "no action needed", "no actions needed", "nothing to do",
    ]
    
    if subgoal_clean in exact_finish_phrases:
        return True
    
    # Partial match patterns - check if any of these appear in the subgoal
    finish_patterns = [
        "finished", "task complete", "task done", "task is done",
        "no further action", "no more action", "nothing more to do",
        "successfully completed", "has been completed", "have been completed",
    ]
    
    for pattern in finish_patterns:
        if pattern in subgoal_clean:
            return True
    
    # Check for "Finished" with capital F (original check)
    if "Finished" in current_subgoal.strip():
        return True
    
    return False


class AgentSAMA(base_agent.EnvironmentInteractingAgent):
    """
    Agent-SAMA for AndroidWorld using the FULL Agent-SAMA framework.
    
    This implementation mirrors the original infer.py flow:
    1. ScreenParser (OCR + GroundingDINO) for perception
    2. StateAgent for FSM state tracking
    3. Planner for high-level planning (with LLM-as-Judge)
    4. Actor for action decisions
    5. ReflectionAgent for outcome analysis
    6. Recovery mode using FSM when errors occur
    """
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        model_name: str = "gpt-4o-2024-11-20",
        name: str = "AgentSAMA",
        wait_after_action_seconds: float = 5.0,  # Match original Agent-SAMA SLEEP_BETWEEN_STEPS
        use_perception: bool = True,
        use_fsm: bool = True,
        use_recovery: bool = True,
        collect_trajectory: bool = False,
        trajectory_output_dir: str = None,
        adb_path: str = None,
    ):
        super().__init__(env, name)
        self.model_name = model_name
        self.wait_after_action_seconds = wait_after_action_seconds
        self.use_perception = use_perception
        self.use_fsm = use_fsm
        self.use_recovery = use_recovery
        self.collect_trajectory = collect_trajectory
        self.trajectory_output_dir = trajectory_output_dir
        
        # Detect ADB path
        self.adb_path = adb_path or self._detect_adb_path()
        
        if not AGENT_SAMA_AVAILABLE:
            raise ImportError("Agent-SAMA modules not available. Check path.")
        
        print("Initializing agent...")
        
        # Initialize Agent-SAMA components
        self.planner = Planner()
        self.actor = Actor(self.adb_path)
        self.reflection_agent = ReflectionAgent()
        self.state_agent = StateAgent(self.adb_path)
        self.mentor = Mentor(mode="unified")  # Unified mode: generates both shortcuts and tips
        
        # Initialize perception models (OCR + GroundingDINO)
        try:
            logging.info("Loading perception models (OCR + GroundingDINO)...")
            # ScreenParser takes adb_path and perception_args, loads models internally
            perception_args = {**DEFAULT_PERCEPTION_ARGS, "device": "cpu"}
            self.screen_parser = ScreenParser(self.adb_path, perception_args)
            # Get references to the loaded models
            self.ocr_detection = self.screen_parser.ocr_detection
            self.ocr_recognition = self.screen_parser.ocr_recognition
            self.groundingdino_model = self.screen_parser.groundingdino_model
            self.vlm_model = self.screen_parser.vlm_model
            self.vlm_tokenizer = self.screen_parser.vlm_tokenizer
            self.perception_available = True
            logging.info("Perception models loaded successfully")
        except Exception as e:
            logging.warning(f"Failed to load perception models: {e}")
            self.perception_available = False
        
        # Screenshot directory - use the same path as original Agent-SAMA
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Agent-SAMA', 'screenshot')
        self.screenshot_dir = os.path.abspath(self.screenshot_dir)
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Temp directory for icon crops (matches original Agent-SAMA)
        self.temp_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Agent-SAMA', 'temp')
        self.temp_dir = os.path.abspath(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # State tracking
        self.info_pool = None
        self.history = []
        self.step_count = 0
        self.current_task_id = None
        
        # JSON logging (like original Agent-SAMA steps.json)
        self.steps_log = []
        self.current_task_name = None
        self.log_dir = None
        
        # Trajectory collector for training data
        self.trajectory_collector = None
        
        # Track task success for Mentor
        self.task_succeeded = False
        
        # Per-run FSM knowledge (accumulates within a run, isolated between runs)
        # Will be set up in set_run_knowledge_dir() or set_task_info()
        self.persistent_knowledge_dir = None
        self.persistent_graphs_path = None
        self.persistent_patterns_path = None
        self.persistent_shortcuts_path = None
        self.persistent_tips_path = None
        self._knowledge_loaded = False
        
        # Initialize empty knowledge (will be populated when knowledge dir is set)
        self._persistent_state_graphs = {}
        self._persistent_navigation_patterns = []
        self._persistent_common_flows = {}
        self._persistent_shortcuts = {}
        self._persistent_tips = INIT_TIPS
        
        # Disable debug overlays that pollute screenshots (pointer_location, show_touches)
        self._disable_debug_overlays()
    
    def _disable_debug_overlays(self):
        """Disable Android developer options that pollute screenshots."""
        try:
            import subprocess
            subprocess.run([self.adb_path, "shell", "settings", "put", "system", "pointer_location", "0"], 
                          capture_output=True, timeout=5)
            subprocess.run([self.adb_path, "shell", "settings", "put", "system", "show_touches", "0"], 
                          capture_output=True, timeout=5)
            logging.info("[Agent-SAMA] Disabled debug overlays (pointer_location, show_touches)")
        except Exception as e:
            logging.warning(f"[Agent-SAMA] Failed to disable debug overlays: {e}")
    
    def _detect_adb_path(self) -> str:
        """Detect ADB path."""
        # Check common ADB paths including Android SDK
        adb_paths = [
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),  # Linux default
            "/home/ubuntu/Android/Sdk/platform-tools/adb",  # Server setup
            "/usr/local/bin/adb",
            "/opt/homebrew/bin/adb",  # macOS Homebrew
            os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),  # macOS default
        ]
        for path in adb_paths:
            if os.path.exists(path):
                print(f"[Agent-SAMA] Using ADB path: {path}")
                return path
        print("[Agent-SAMA] Warning: ADB not found in standard paths, using 'adb' from PATH")
        return "adb"
    
    def set_run_knowledge_dir(self, run_dir: str):
        """Set up per-run knowledge directory.
        
        This creates a knowledge/ subdirectory within the run output directory,
        so each run has isolated knowledge that accumulates within the run.
        
        Args:
            run_dir: The run output directory (e.g., runs/run_20251229T...)
        """
        if self._knowledge_loaded:
            return  # Already loaded for this run
        
        # Create knowledge dir inside run dir
        self.persistent_knowledge_dir = os.path.join(run_dir, 'knowledge')
        os.makedirs(self.persistent_knowledge_dir, exist_ok=True)
        
        self.persistent_graphs_path = os.path.join(self.persistent_knowledge_dir, 'state_graphs.pkl')
        self.persistent_patterns_path = os.path.join(self.persistent_knowledge_dir, 'fsm_patterns.json')
        self.persistent_shortcuts_path = os.path.join(self.persistent_knowledge_dir, 'shortcuts.json')
        self.persistent_tips_path = os.path.join(self.persistent_knowledge_dir, 'tips.txt')
        
        print(f"[FSM] Using per-run knowledge directory: {self.persistent_knowledge_dir}")
        
        # Load any existing knowledge from this run (for resumed runs)
        self._load_persistent_knowledge()
    
    def _load_persistent_knowledge(self):
        """Load persistent FSM knowledge from previous tasks in this run."""
        import pickle
        
        if not self.persistent_knowledge_dir:
            return  # Knowledge dir not set yet
        
        self._knowledge_loaded = True
        
        # Load state graphs
        if os.path.exists(self.persistent_graphs_path):
            try:
                with open(self.persistent_graphs_path, 'rb') as f:
                    self._persistent_state_graphs = pickle.load(f)
                print(f"[FSM] Loaded {len(self._persistent_state_graphs)} app graphs from persistent storage")
            except Exception as e:
                logging.warning(f"Failed to load persistent graphs: {e}")
        
        # Load FSM patterns
        if os.path.exists(self.persistent_patterns_path):
            try:
                with open(self.persistent_patterns_path, 'r') as f:
                    patterns_data = json.load(f)
                    self._persistent_navigation_patterns = patterns_data.get('learned_navigation_patterns', [])
                    self._persistent_common_flows = patterns_data.get('common_flows', {})
                print(f"[FSM] Loaded {len(self._persistent_navigation_patterns)} navigation patterns, "
                      f"{len(self._persistent_common_flows)} common flows")
            except Exception as e:
                logging.warning(f"Failed to load persistent FSM patterns: {e}")
        
        # Load shortcuts
        if os.path.exists(self.persistent_shortcuts_path):
            try:
                with open(self.persistent_shortcuts_path, 'r') as f:
                    self._persistent_shortcuts = json.load(f)
                print(f"[FSM] Loaded {len(self._persistent_shortcuts)} shortcuts from persistent storage")
            except Exception as e:
                logging.warning(f"Failed to load persistent shortcuts: {e}")
        
        # Load tips
        if os.path.exists(self.persistent_tips_path):
            try:
                with open(self.persistent_tips_path, 'r') as f:
                    self._persistent_tips = f.read()
                print(f"[FSM] Loaded tips from persistent storage")
            except Exception as e:
                logging.warning(f"Failed to load persistent tips: {e}")
    
    def _save_persistent_knowledge(self):
        """Save FSM knowledge for future tasks."""
        import pickle
        
        if not self.info_pool:
            print("[FSM] No info_pool to save")
            return
        
        if not self.persistent_knowledge_dir:
            print("[FSM] Knowledge directory not set, cannot save")
            return
        
        print("\n### Saving Persistent Knowledge ... ###\n")
        
        try:
            # Save state graphs
            if self.info_pool.state_graphs:
                with open(self.persistent_graphs_path, 'wb') as f:
                    pickle.dump(self.info_pool.state_graphs, f)
                print(f"[FSM] Saved {len(self.info_pool.state_graphs)} app graphs to persistent storage")
            
            # Save FSM patterns
            patterns_data = {
                'learned_navigation_patterns': self.info_pool.learned_navigation_patterns,
                'common_flows': self.info_pool.common_flows
            }
            with open(self.persistent_patterns_path, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            print(f"[FSM] Saved {len(self.info_pool.learned_navigation_patterns)} navigation patterns, "
                  f"{len(self.info_pool.common_flows)} common flows")
            
            # Save shortcuts
            if self.info_pool.shortcuts:
                with open(self.persistent_shortcuts_path, 'w') as f:
                    json.dump(self.info_pool.shortcuts, f, indent=2)
                print(f"[FSM] Saved {len(self.info_pool.shortcuts)} shortcuts")
            
            # Save tips
            if self.info_pool.tips:
                with open(self.persistent_tips_path, 'w') as f:
                    f.write(self.info_pool.tips)
                print(f"[FSM] Saved tips")
            
            # IMPORTANT: Update in-memory variables so next task in this run can use them
            self._persistent_state_graphs = copy.deepcopy(self.info_pool.state_graphs)
            self._persistent_navigation_patterns = copy.deepcopy(self.info_pool.learned_navigation_patterns)
            self._persistent_common_flows = copy.deepcopy(self.info_pool.common_flows)
            self._persistent_shortcuts = copy.deepcopy(self.info_pool.shortcuts)
            self._persistent_tips = self.info_pool.tips
            print(f"[FSM] Updated in-memory knowledge for next task")
                
        except Exception as e:
            logging.error(f"Failed to save persistent knowledge: {e}")
            import traceback
            traceback.print_exc()
    
    def reset(self, go_home: bool = True):
        """Reset the agent state.
        
        Note: Does NOT reset log_dir, current_task_name, or trajectory_collector
        because set_task_info() is called BEFORE reset() in the AndroidWorld flow.
        """
        self.history = []
        self.step_count = 0
        self.info_pool = None
        self.steps_log = []  # For steps.json output
        # Do NOT reset: self.current_task_name, self.log_dir, self.trajectory_collector
        # These are set by set_task_info() which is called BEFORE reset() in suite_utils.py
        
        if self.collect_trajectory:
            import uuid
            self.current_task_id = str(uuid.uuid4())[:8]
    
    def set_task_info(self, task_name: str, log_dir: str):
        """Set task info for logging. Called before step() is invoked."""
        self.current_task_name = task_name
        self.log_dir = log_dir
        self.steps_log = []
        
        # Create subdirectories for this task
        if self.log_dir:
            os.makedirs(os.path.join(self.log_dir, "screenshots"), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, "perception"), exist_ok=True)
        
        # Set up per-run knowledge directory (shared across tasks in this run)
        # Extract run directory from log_dir (log_dir is typically runs/run_XXXXX/)
        if self.log_dir and not self._knowledge_loaded:
            # log_dir is the run directory, create knowledge/ inside it
            run_dir = self.log_dir
            # If log_dir contains task-specific path, go up to run dir
            if task_name and task_name in self.log_dir:
                run_dir = os.path.dirname(self.log_dir)
            self.set_run_knowledge_dir(run_dir)
        
        # Initialize trajectory collector if enabled
        if self.collect_trajectory and AGENT_SAMA_AVAILABLE:
            traj_dir = self.trajectory_output_dir or os.path.join(self.log_dir, "trajectory") if self.log_dir else "trajectory_data"
            # Use task_name + timestamp as task_id to prevent overwrites
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            task_name_clean = task_name.replace(" ", "_")[:40] if task_name else "unknown_task"
            task_id = f"{task_name_clean}_{timestamp}"
            self.trajectory_collector = TrajectoryCollector(
                output_dir=traj_dir,
                task_id=task_id,
                autosave=True
            )
            print(f"[Trajectory] Initialized collector for task: {task_name}")
            print(f"[Trajectory] Output directory: {traj_dir}/{task_id}")
    
    def _save_steps_json(self):
        """Save steps to JSON file like original Agent-SAMA."""
        if not self.log_dir:
            print(f"[Warning] log_dir not set, cannot save steps.json")
            return
        if not self.steps_log:
            print(f"[Warning] steps_log is empty, nothing to save")
            return
            
        steps_path = os.path.join(self.log_dir, "steps.json")
        try:
            with open(steps_path, "w") as f:
                json.dump(self.steps_log, f, indent=4, default=str)
            print(f"[Logging] Saved steps.json to {steps_path}")
        except Exception as e:
            print(f"[Error] Failed to save steps.json: {e}")
    
    def _save_screenshot_from_pixels(self, pixels: np.ndarray, filename: str) -> str:
        """Save screenshot pixels to a file.
        
        IMPORTANT: Original Agent-SAMA uses ./screenshot/screenshot.jpg as the
        canonical screenshot path. ScreenParser.get_perception_infos() calls
        get_screenshot() which saves to this path. We save to the same location
        to maintain consistency.
        """
        # Always save to the canonical screenshot path (matches original Agent-SAMA)
        canonical_path = os.path.join(self.screenshot_dir, "screenshot.jpg")
        img = Image.fromarray(pixels)
        img.save(canonical_path)
        
        # Also save a timestamped copy for logging if log_dir exists
        if self.log_dir:
            log_screenshots_dir = os.path.join(self.log_dir, "screenshots")
            os.makedirs(log_screenshots_dir, exist_ok=True)
            log_filepath = os.path.join(log_screenshots_dir, filename)
            img.save(log_filepath)
        
        # If collecting trajectory, save to trajectory folder too
        if self.collect_trajectory and self.trajectory_collector:
            traj_dir = self.trajectory_collector.images_dir
            os.makedirs(traj_dir, exist_ok=True)
            traj_filepath = os.path.join(traj_dir, filename)
            img.save(traj_filepath)
        
        # Return the canonical path that ScreenParser expects
        return canonical_path
    
    def _get_perception_infos(self, screenshot_file: str) -> List[Dict]:
        """Run OCR and icon detection on screenshot using ScreenParser.
        
        IMPORTANT: Original Agent-SAMA's ScreenParser.get_perception_infos() calls
        get_screenshot(adb_path) internally, taking its own fresh screenshot.
        We need to ensure our screenshot is saved to ./screenshot/screenshot.jpg
        BEFORE calling ScreenParser, since it reads from that path.
        """
        if not self.perception_available:
            return []
        
        try:
            # Clean temp directory before processing (matches original infer.py)
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # ScreenParser.get_perception_infos() will call get_screenshot() internally
            # which saves to ./screenshot/screenshot.jpg - this is the original behavior
            perception_infos, width, height = self.screen_parser.get_perception_infos(
                screenshot_file, temp_file=self.temp_dir
            )
            
            # Update info_pool dimensions
            if self.info_pool:
                self.info_pool.width = width
                self.info_pool.height = height
            
            # Clean up temp directory after processing (matches original infer.py)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)
            
            return perception_infos
            
        except Exception as e:
            logging.error(f"Perception failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _initialize_info_pool(self, goal: str, screen_size: tuple):
        """Initialize the InfoPool for a new task with persistent knowledge."""
        self.info_pool = InfoPool()
        self.info_pool.instruction = goal
        self.info_pool.width, self.info_pool.height = screen_size
        self.info_pool.adb_path = self.adb_path
        self.info_pool.current_subgoal = goal
        self.info_pool.progress_status = "Just started the task."
        self.info_pool.action_history = []
        self.info_pool.action_outcomes = []
        self.info_pool.summary_history = []
        self.info_pool.error_descriptions = []
        
        # Use persistent knowledge from previous tasks
        if self._persistent_shortcuts:
            self.info_pool.shortcuts = copy.deepcopy(self._persistent_shortcuts)
            print(f"[FSM] Using {len(self._persistent_shortcuts)} persistent shortcuts")
        else:
            self.info_pool.shortcuts = copy.deepcopy(INIT_SHORTCUTS)
        
        if self._persistent_tips:
            self.info_pool.tips = self._persistent_tips
            print(f"[FSM] Using persistent tips")
        else:
            self.info_pool.tips = INIT_TIPS
        
        # Load persistent state graphs
        if self._persistent_state_graphs:
            self.info_pool.state_graphs = copy.deepcopy(self._persistent_state_graphs)
            print(f"[FSM] Using {len(self._persistent_state_graphs)} persistent app graphs")
        
        # Load persistent FSM patterns
        if self._persistent_navigation_patterns:
            self.info_pool.learned_navigation_patterns = copy.deepcopy(self._persistent_navigation_patterns)
            print(f"[FSM] Using {len(self._persistent_navigation_patterns)} persistent navigation patterns")
        
        if self._persistent_common_flows:
            self.info_pool.common_flows = copy.deepcopy(self._persistent_common_flows)
            print(f"[FSM] Using {len(self._persistent_common_flows)} persistent common flows")
    
    def _log_perception_infos(self, perception_infos: List[Dict]):
        """Log detected UI elements in a readable format."""
        print(f"Detected {len(perception_infos)} UI elements:")
        
        for i, elem in enumerate(perception_infos[:15]):
            text = elem.get('text', '')
            coords = elem.get('coordinates', (0, 0))
            if text and text != "icon: None" and coords != (0, 0):
                print(f"  [{i+1}] \"{text}\" @ {coords}")
        
        if len(perception_infos) > 15:
            print(f"  ... and {len(perception_infos) - 15} more")
        print()
    
    def _run_state_agent(self, screenshot_file: str, is_first: bool = False) -> Dict:
        """Run StateAgent to analyze current state and update FSM."""
        if is_first:
            print("\n### State Agent (First Look) ... ###\n")
        else:
            print("\n### State Agent ... ###\n")
        
        self.info_pool.current_screenshot_path = screenshot_file
        
        prompt = self.state_agent.get_prompt(self.info_pool)
        chat = self.state_agent.init_chat()
        chat = add_response("user", prompt, chat, image=screenshot_file)
        
        response = get_reasoning_model_api_response(chat, temperature=0.0)
        
        parsed = self.state_agent.parse_response(response)
        state_description = parsed.get('state_description', 'No description')
        predicted_next_state = parsed.get('predicted_next_state', 'No prediction')
        current_app = parsed.get('current_app_name', 'Unknown')
        beacon = parsed.get('state_beacon', 'Unknown')
        precondition = parsed.get('precondition', '')
        postcondition = parsed.get('postcondition', '')
        
        print(f"Current App: {current_app}")
        print(f"State Beacon: {beacon}")
        print(f"State Description: {state_description[:100]}..." if len(state_description) > 100 else f"State Description: {state_description}")
        print(f"Predicted Next: {predicted_next_state[:80]}..." if len(predicted_next_state) > 80 else f"Predicted Next: {predicted_next_state}")
        
        # Update FSM graph
        try:
            self.state_agent.update_state_graph(
                self.info_pool, state_description, current_app, beacon,
                precondition, postcondition, predicted_next_state
            )
        except Exception as e:
            logging.warning(f"Failed to update state graph: {e}")
        
        # Store state info
        self.info_pool.pre_condition = precondition
        self.info_pool.post_condition = postcondition
        self.info_pool.state_prediction = predicted_next_state
        self.info_pool._current_state_description = state_description
        
        if is_first:
            self.info_pool.active_app_name = current_app
        
        return {
            'state_description': state_description,
            'predicted_next_state': predicted_next_state,
            'current_app': current_app,
            'beacon': beacon
        }
    
    def _run_planner(self, screenshot_file: str):
        """Run Planner with LLM-as-Judge."""
        print("\n### Planner ... ###\n")
        
        prompt = self.planner.get_prompt(self.info_pool)
        chat = self.planner.init_chat()
        chat = add_response("user", prompt, chat, image=screenshot_file)
        
        response = get_model_api_response(chat, temperature=0.0)
        
        try:
            if LLM_Judge and self.info_pool.plan == "":
                # First time planning - get 5 candidates
                parsed = self.planner.parse_response(response)
                plan_candidates = parsed.get("candidates", [])
                
                if len(plan_candidates) > 1:
                    print(f"Generated {len(plan_candidates)} candidate plans, judging...")
                    prompt_judging = self.planner.get_prompt_for_judging(self.info_pool, plan_candidates)
                    chat_judging = self.planner.init_chat_for_judging()
                    chat_judging = add_response("user", prompt_judging, chat_judging, image=screenshot_file)
                    output_judging = get_model_api_response(chat_judging, temperature=0.0)
                    
                    selected = self.planner.parse_judge_response(output_judging)
                    self.info_pool.plan = selected.get('plan', [self.info_pool.instruction])
                    self.info_pool.current_subgoal = selected.get('current_subgoal', self.info_pool.instruction)
                    thought = selected.get('thought', '')
                elif len(plan_candidates) == 1:
                    selected = plan_candidates[0]
                    self.info_pool.plan = selected.get('plan', [self.info_pool.instruction])
                    self.info_pool.current_subgoal = selected.get('current_subgoal', self.info_pool.instruction)
                    thought = selected.get('thought', '')
                else:
                    self.info_pool.plan = [self.info_pool.instruction]
                    self.info_pool.current_subgoal = self.info_pool.instruction
                    thought = "Fallback: could not parse plan candidates"
            else:
                # Continuing planning
                parsed = self.planner.parse_response(response)
                candidates = parsed.get("candidates", [])
                if candidates:
                    selected = candidates[0]
                    self.info_pool.plan = selected.get('plan', self.info_pool.plan)
                    self.info_pool.current_subgoal = selected.get('current_subgoal', self.info_pool.current_subgoal)
                    thought = selected.get('thought', '')
                else:
                    thought = "Plan unchanged"
            
            print(f"Thought: {thought}")
            print(f"Overall Plan: {self.info_pool.plan}")
            print(f"Current Subgoal: {self.info_pool.current_subgoal}")
        except Exception as e:
            logging.warning(f"Failed to parse plan: {e}")
            import traceback
            traceback.print_exc()
            if not self.info_pool.plan:
                self.info_pool.plan = [self.info_pool.instruction]
            if not self.info_pool.current_subgoal:
                self.info_pool.current_subgoal = self.info_pool.instruction
            print(f"Plan: {self.info_pool.plan} (fallback)")
    
    def _run_actor(self, screenshot_file: str) -> Dict:
        """Run Actor to decide on next action."""
        print("\n### Actor ... ###\n")
        
        prompt = self.actor.get_prompt(self.info_pool)
        chat = self.actor.init_chat()
        chat = add_response("user", prompt, chat, image=screenshot_file)
        
        response = get_model_api_response(chat, temperature=0.0)
        
        parsed = self.actor.parse_response(response)
        action_thought = parsed.get('thought', '')
        action_str = parsed.get('action', '')
        action_description = parsed.get('description', '')
        
        print(f"Action Thought: {action_thought}")
        print(f"Action: {action_str}")
        print(f"Action Description: {action_description}")
        
        self.info_pool.last_action_thought = action_thought
        
        return {
            'thought': action_thought,
            'action': action_str,
            'description': action_description,
            'raw_response': response
        }
    
    def _execute_action_adb(self, action_str: str) -> tuple:
        """Execute action using ADB via AndroidWorld's adb_utils."""
        print("\n### Action Execution ... ###\n")
        
        action_executed = None
        error_message = None
        # Access the underlying AndroidEnvInterface through controller
        android_env = self.env.controller.env
        
        try:
            # Parse the action JSON
            action_object = extract_json_object(action_str)
            
            if action_object is None:
                logging.warning(f"Could not parse action: {action_str}")
                return None, "Invalid action format"
            
            action_name = action_object.get("name", "").strip()
            arguments = action_object.get("arguments", {}) or {}
            
            print(f"Executing: {action_name} with args {arguments}")
            
            # Match original Agent-SAMA timing for reliable action execution
            if action_name.lower() == "tap":
                x = int(arguments.get("x", 0))
                y = int(arguments.get("y", 0))
                if x > 0 and y > 0:
                    adb_utils.issue_generic_request(
                        ['shell', 'input', 'tap', str(x), str(y)],
                        android_env
                    )
                    action_executed = {'name': 'Tap', 'x': x, 'y': y}
                    print(f"✓ Executed: Tap({x}, {y})")
                    time.sleep(5)  # Original: 5 seconds
            
            elif action_name.lower() == "swipe":
                x1 = int(arguments.get("x1", 0))
                y1 = int(arguments.get("y1", 0))
                x2 = int(arguments.get("x2", 0))
                y2 = int(arguments.get("y2", 0))
                adb_utils.issue_generic_request(
                    ['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), '500'],
                    android_env
                )
                action_executed = {'name': 'Swipe', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                print(f"✓ Executed: Swipe({x1}, {y1}) → ({x2}, {y2})")
                time.sleep(5)  # Original: 5 seconds
            
            elif action_name.lower() == "type":
                text = str(arguments.get("text", ""))
                if text:
                    adb_utils.type_text(text, android_env)
                    action_executed = {'name': 'Type', 'text': text}
                    print(f"✓ Executed: Type(\"{text}\")")
                    time.sleep(3)  # Original: 3 seconds
            
            elif action_name.lower() == "enter":
                adb_utils.issue_generic_request(
                    ['shell', 'input', 'keyevent', 'KEYCODE_ENTER'],
                    android_env
                )
                action_executed = {'name': 'Enter'}
                print("✓ Executed: Enter")
                time.sleep(10)  # Original: 10 seconds (for search/submit)
            
            elif action_name.lower() == "back":
                adb_utils.issue_generic_request(
                    ['shell', 'input', 'keyevent', '4'],
                    android_env
                )
                action_executed = {'name': 'Back'}
                print("✓ Executed: Back")
                time.sleep(3)  # Original: 3 seconds
            
            elif action_name.lower() == "home":
                adb_utils.issue_generic_request(
                    ['shell', 'input', 'keyevent', 'KEYCODE_HOME'],
                    android_env
                )
                action_executed = {'name': 'Home'}
                print("✓ Executed: Home")
                time.sleep(3)  # Original: 3 seconds
            
            elif action_name.lower() == "switch_app":
                adb_utils.issue_generic_request(
                    ['shell', 'input', 'keyevent', 'KEYCODE_APP_SWITCH'],
                    android_env
                )
                action_executed = {'name': 'Switch_App'}
                print("✓ Executed: Switch_App")
                time.sleep(3)  # Original: 3 seconds
            
            elif action_name.lower() == "wait":
                time.sleep(10)  # Original: 10 seconds
                action_executed = {'name': 'Wait'}
                print("✓ Executed: Wait")
            
            elif action_name.lower() == "open_app":
                app_name = str(arguments.get("app_name", ""))
                if app_name:
                    # ORIGINAL Agent-SAMA: Use OCR to find app name on screen, then tap on it
                    print(f"Opening app via OCR+Tap (original method): {app_name}")
                    
                    # Take fresh screenshot for OCR
                    screenshot_path = os.path.join(self.screenshot_dir, "screenshot.jpg")
                    
                    # Run OCR to find the app name
                    try:
                        from AgentsUtils.ocr_utils import run_ocr
                        text_list, coordinate_list = run_ocr(
                            screenshot_path, 
                            self.ocr_detection, 
                            self.ocr_recognition
                        )
                        
                        app_found = False
                        for ti in range(len(text_list)):
                            if app_name.lower() in text_list[ti].lower():
                                # Calculate center of the text bounding box
                                name_x = int((coordinate_list[ti][0] + coordinate_list[ti][2]) / 2)
                                # Tap slightly above center (like original: y - height offset)
                                text_height = coordinate_list[ti][3] - coordinate_list[ti][1]
                                name_y = int((coordinate_list[ti][1] + coordinate_list[ti][3]) / 2) - int(text_height)
                                
                                print(f"Found '{app_name}' at ({name_x}, {name_y}), tapping...")
                                adb_utils.issue_generic_request(
                                    ['shell', 'input', 'tap', str(name_x), str(name_y)],
                                    android_env
                                )
                                app_found = True
                                break
                        
                        if not app_found:
                            # Fallback: try package-based launch
                            print(f"App '{app_name}' not found on screen, trying package launch...")
                            launched = adb_utils.launch_app(app_name, android_env)
                            if not launched:
                                adb_utils.launch_app(app_name.lower(), android_env)
                        
                    except Exception as e:
                        logging.warning(f"OCR-based app launch failed: {e}, falling back to package launch")
                        adb_utils.launch_app(app_name, android_env)
                    
                    # Additional wait for slow apps (matches original)
                    if app_name in ['Fandango', 'Walmart', 'Best Buy']:
                        time.sleep(10)
                    time.sleep(10)  # Original: 10 seconds for app to fully load
                    action_executed = {'name': 'Open_App', 'app_name': app_name}
                    print(f"✓ Executed: Open_App('{app_name}')")
            
            elif action_name.lower() == "tap_type_and_enter":
                # Compound action: Tap → Type → Enter
                x = int(arguments.get("x", 0))
                y = int(arguments.get("y", 0))
                text = str(arguments.get("text", ""))
                
                if x > 0 and y > 0 and text:
                    # Step 1: Tap to focus the input field
                    adb_utils.issue_generic_request(
                        ['shell', 'input', 'tap', str(x), str(y)],
                        android_env
                    )
                    time.sleep(5)  # Wait for keyboard to appear
                    
                    # Step 2: Type the text
                    adb_utils.type_text(text, android_env)
                    time.sleep(3)  # Wait for typing to complete
                    
                    # Step 3: Press Enter
                    adb_utils.issue_generic_request(
                        ['shell', 'input', 'keyevent', 'KEYCODE_ENTER'],
                        android_env
                    )
                    
                    action_executed = {'name': 'Tap_Type_and_Enter', 'x': x, 'y': y, 'text': text}
                    print(f"✓ Executed: Tap_Type_and_Enter({x}, {y}, \"{text}\")")
                    time.sleep(10)  # Wait for search/submit result
                else:
                    print(f"⚠ Invalid Tap_Type_and_Enter args: x={x}, y={y}, text={text}")
            
            elif action_name.lower() == "tap_and_type":
                # Compound action: Tap → Type (no Enter)
                x = int(arguments.get("x", 0))
                y = int(arguments.get("y", 0))
                text = str(arguments.get("text", ""))
                
                if x > 0 and y > 0 and text:
                    # Step 1: Tap to focus
                    adb_utils.issue_generic_request(
                        ['shell', 'input', 'tap', str(x), str(y)],
                        android_env
                    )
                    time.sleep(5)  # Wait for keyboard to appear
                    
                    # Step 2: Type the text
                    adb_utils.type_text(text, android_env)
                    
                    action_executed = {'name': 'Tap_and_Type', 'x': x, 'y': y, 'text': text}
                    print(f"✓ Executed: Tap_and_Type({x}, {y}, \"{text}\")")
                    time.sleep(3)  # Wait for typing to complete
            
            elif action_name.lower() == "long_press":
                x = int(arguments.get("x", 0))
                y = int(arguments.get("y", 0))
                duration = int(arguments.get("duration", 1000))  # Default 1 second
                
                if x > 0 and y > 0:
                    # Long press is implemented as a swipe with same start/end
                    adb_utils.issue_generic_request(
                        ['shell', 'input', 'swipe', str(x), str(y), str(x), str(y), str(duration)],
                        android_env
                    )
                    action_executed = {'name': 'Long_Press', 'x': x, 'y': y, 'duration': duration}
                    print(f"✓ Executed: Long_Press({x}, {y}, {duration}ms)")
                    time.sleep(5)  # Wait for context menu
            
            elif action_name.lower() in ["finish", "done", "complete", "null", "none"]:
                action_executed = {'name': 'Finish'}
                print("✓ Task marked as complete")
            
            else:
                logging.warning(f"Unknown action: {action_name}")
                action_executed = {'name': 'Unknown', 'raw': action_str}
                print(f"⚠ Unknown action: {action_name}")
        
        except Exception as e:
            error_message = str(e)
            logging.error(f"Action execution failed: {e}")
            print(f"❌ Action failed: {e}")
        
        return action_executed, error_message
    
    def _run_reflection(self, before_screenshot: str, after_screenshot: str, action_str: str) -> Dict:
        """Run ReflectionAgent to evaluate action outcome."""
        print("\n### Reflector ... ###\n")
        
        prompt = self.reflection_agent.get_prompt(self.info_pool)
        chat = self.reflection_agent.init_chat()
        chat = add_response_with_images("user", prompt, chat, [before_screenshot, after_screenshot])
        
        response = get_reasoning_model_api_response(chat, temperature=0.0)
        
        try:
            parsed = self.reflection_agent.parse_response(response)
            outcome = parsed.get('outcome', 'A')
            error_description = parsed.get('error_description', '')
            progress_status = parsed.get('progress_status', '')
            
            if outcome == 'A':
                print(f"Outcome: {outcome} ✓ Action successful")
            elif outcome == 'B':
                print(f"Outcome: {outcome} ✗ Wrong page: {error_description}")
            elif outcome == 'C':
                print(f"Outcome: {outcome} ✗ No change: {error_description}")
            
            print(f"Progress: {progress_status}")
            
            return {
                'outcome': outcome,
                'error_description': error_description,
                'progress_status': progress_status
            }
        except Exception as e:
            logging.error(f"Failed to parse reflection: {e}")
            print(f"Error parsing reflection: {e}")
            return {
                'outcome': 'C',
                'error_description': str(e),
                'progress_status': ''
            }
    
    def _run_mentor(self) -> Dict:
        """Run Mentor for self-evolution reflection at end of successful task."""
        print("\n### Mentor Agent: Self-Evolution Reflection ... ###\n")
        
        if not self.info_pool:
            print("No info_pool available for Mentor")
            return {}
        
        try:
            # Get prompt and run LLM
            prompt = self.mentor.get_prompt(self.info_pool)
            chat = self.mentor.init_chat()
            chat = add_response("user", prompt, chat, image=None)
            
            response = get_reasoning_model_api_response(
                chat,
                model=KNOWLEDGE_REFLECTION_MODEL,
                temperature=0.0
            )
            
            # Process reflection results (updates shortcuts and tips in info_pool)
            reflection_results = self.mentor.reflect(self.info_pool, response)
            
            # Parse for logging
            parsed = self.mentor.parse_response(response)
            new_shortcut = parsed.get('new_shortcut', 'None')
            updated_tips = parsed.get('updated_tips', self.info_pool.tips)
            
            print(f"[Mentor] Shortcuts added: {reflection_results.get('shortcuts_added', 0)}")
            print(f"[Mentor] Tips updated: {reflection_results.get('tips_updated', False)}")
            
            if new_shortcut and new_shortcut.lower() != 'none':
                print(f"New Shortcut: {new_shortcut[:200]}..." if len(str(new_shortcut)) > 200 else f"New Shortcut: {new_shortcut}")
            
            tips_preview = updated_tips[:300] + "..." if len(str(updated_tips)) > 300 else updated_tips
            print(f"Updated Tips: {tips_preview}")
            
            # === FSM Pattern Extraction (Cross-Task Knowledge) ===
            print("\n### FSM Pattern Extraction ... ###\n")
            try:
                # Extract navigation patterns from FSM graphs
                nav_patterns = self.mentor.extract_fsm_patterns(self.info_pool)
                print(f"[FSM] Extracted {len(nav_patterns)} navigation patterns")
                
                # Extract common flows across apps
                common_flows = self.mentor.extract_common_flows(self.info_pool)
                print(f"[FSM] Extracted {len(common_flows)} common flows")
                
                # Print FSM summary
                print(f"\n[FSM Graph Statistics]")
                print(f"  Apps tracked: {len(self.info_pool.state_graphs)}")
                total_states = sum(g.number_of_nodes() for g in self.info_pool.state_graphs.values())
                total_transitions = sum(g.number_of_edges() for g in self.info_pool.state_graphs.values())
                print(f"  Total states: {total_states}")
                print(f"  Total transitions: {total_transitions}")
                for app_name, graph in self.info_pool.state_graphs.items():
                    beacons = [graph.nodes[n].get('beacon', '?') for n in graph.nodes]
                    print(f"  [{app_name}] {graph.number_of_nodes()} states: {', '.join(beacons[:4])}{'...' if len(beacons) > 4 else ''}")
                
                print(f"\n[FSM Patterns (Cross-Task Knowledge)]")
                print(f"  Navigation patterns: {len(self.info_pool.learned_navigation_patterns)}")
                print(f"  Common flows: {len(self.info_pool.common_flows)}")
                
            except Exception as e:
                logging.warning(f"FSM pattern extraction failed: {e}")
                print(f"[FSM] Pattern extraction error: {e}")
            
            # Log to steps
            self.steps_log.append({
                "step": self.step_count,
                "operation": "mentor_self_evolution",
                "new_shortcut": new_shortcut,
                "updated_tips": updated_tips,
                "shortcuts_added": reflection_results.get('shortcuts_added', 0),
                "tips_updated": reflection_results.get('tips_updated', False),
                "fsm_apps": len(self.info_pool.state_graphs),
                "fsm_patterns": len(self.info_pool.learned_navigation_patterns),
                "fsm_flows": len(self.info_pool.common_flows)
            })
            
            return {
                'new_shortcut': new_shortcut,
                'updated_tips': updated_tips,
                'shortcuts_added': reflection_results.get('shortcuts_added', 0),
                'tips_updated': reflection_results.get('tips_updated', False),
                'fsm_apps': len(self.info_pool.state_graphs),
                'fsm_patterns': len(self.info_pool.learned_navigation_patterns)
            }
            
        except Exception as e:
            logging.error(f"Mentor failed: {e}")
            print(f"[Mentor] Error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """
        Perform a single step using the FULL Agent-SAMA pipeline.
        
        Flow (matches infer.py):
        - First step: ScreenParser → StateAgent (first look) → Planner → Actor → Execute → ScreenParser → StateAgent → Reflector
        - Subsequent steps: (use cached perception) → Planner → Actor → Execute → ScreenParser → StateAgent → Reflector
        """
        step_data = {
            "step": self.step_count,
            "goal": goal,
            "action": None,
            "outcome": None,
            "summary": None,
        }
        
        self.step_count += 1
        is_first_step = self.step_count == 1
        
        print(f"\n{'='*60}")
        print(f"  Agent-SAMA Step {self.step_count}")
        print(f"{'='*60}")
        print(f"Goal: {goal}")
        
        # Get current state from AndroidWorld
        state = self.get_post_transition_state()
        screen_size = self.env.logical_screen_size
        
        # Save screenshot
        screenshot_file = self._save_screenshot_from_pixels(
            state.pixels,
            f"step_{self.step_count}_before.jpg"
        )
        
        # Initialize info pool on first step
        if self.info_pool is None:
            self._initialize_info_pool(goal, screen_size)
        
        # === PERCEPTION (only on first step) ===
        if is_first_step:
            # First step: run Screen Parser "first look"
            print("\n### Screen Parser (OCR + GroundingDINO) ... ###\n")
            perception_infos = self._get_perception_infos(screenshot_file)
            self.info_pool.perception_infos_pre = copy.deepcopy(perception_infos)
            self.info_pool.state_perception_infos = copy.deepcopy(perception_infos)
            self.info_pool.keyboard_pre = self._detect_keyboard(perception_infos)
            self._log_perception_infos(perception_infos)
            
            # First step: run State Agent "first look"
            print("\n### State Agent (First Look) ... ###\n")
            self._run_state_agent(screenshot_file, is_first=True)
        else:
            # Subsequent steps: perception already cached from previous step's "after action"
            # Just update the current screenshot path
            self.info_pool.current_screenshot_path = screenshot_file
            print(f"\nUsing cached perception from previous step ({len(self.info_pool.perception_infos_pre)} elements)")
        
        # === PLANNER ===
        if not self.info_pool.state_agent_recovery_mode:
            self._run_planner(screenshot_file)
            
            # Check for task completion
            if is_task_finished(self.info_pool.current_subgoal):
                print("\n" + "="*60)
                print("TASK COMPLETED!")
                print("="*60 + "\n")
                
                # Run Mentor for self-evolution
                self.task_succeeded = True
                self._run_mentor()
                
                # Save persistent knowledge for future tasks
                self._save_persistent_knowledge()
                
                step_data["summary"] = "Task completed"
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(done=True, data=step_data)
        
        # === ACTOR ===
        action_result = self._run_actor(screenshot_file)
        action_str = action_result['action']
        step_data["action"] = action_str
        
        # Check for finish action
        action_obj = extract_json_object(action_str) if AGENT_SAMA_AVAILABLE else None
        if action_obj and action_obj.get("name", "").lower() in ["finish", "done", "complete", "null", "none"]:
            print("\n" + "="*60)
            print("AGENT INDICATES TASK COMPLETE")
            print("="*60 + "\n")
            
            # Run Mentor for self-evolution
            self.task_succeeded = True
            self._run_mentor()
            
            # Save persistent knowledge for future tasks
            self._save_persistent_knowledge()
            
            step_data["summary"] = "Agent marked task as complete"
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(done=True, data=step_data)
        
        # Store pre-action state info for trajectory
        self.info_pool.pre_state_description = getattr(self.info_pool, '_current_state_description', '')
        self.info_pool.pre_state_pre_condition = getattr(self.info_pool, 'pre_condition', '')
        self.info_pool.pre_state_post_condition = getattr(self.info_pool, 'post_condition', '')
        
        # === EXECUTE ACTION (using original Actor.execute method) ===
        print("\n### Action Execution (Original Actor.execute) ... ###\n")
        
        # Set up screenshot log directory
        screenshot_log_dir = None
        if self.log_dir:
            screenshot_log_dir = os.path.join(self.log_dir, "screenshots")
            os.makedirs(screenshot_log_dir, exist_ok=True)
        
        # Use original Actor.execute() which handles shortcuts and atomic actions properly
        action_executed, num_atomic_actions, error = self.actor.execute(
            action_str,
            self.info_pool,
            screenshot_file=screenshot_file,  # Current screenshot
            ocr_detection=self.ocr_detection if self.perception_available else None,
            ocr_recognition=self.ocr_recognition if self.perception_available else None,
            thought=action_result.get('thought', ''),
            screenshot_log_dir=screenshot_log_dir,
            iter=str(self.step_count)
        )
        
        if error:
            step_data["summary"] = f"Action failed: {error}"
            self.info_pool.error_descriptions.append(error)
            self.info_pool.action_outcomes.append("C")
        elif action_executed is None:
            # Actor returned None - indicates finish or invalid action
            step_data["summary"] = "Action indicates completion or was invalid"
        
        # Update action history (matches original infer.py)
        self.info_pool.last_action = action_executed
        self.info_pool.last_summary = action_result.get('description', '')
        self.info_pool.action_history.append(action_executed if action_executed else action_str)
        self.info_pool.summary_history.append(action_result.get('description', ''))
        
        print(f"Action executed: {action_executed}, Atomic actions: {num_atomic_actions}")
        
        # Wait for screen to stabilize
        print(f"Waiting {self.wait_after_action_seconds}s for screen to stabilize...")
        time.sleep(self.wait_after_action_seconds)
        
        # === PERCEPTION (after action) ===
        print("\n### Screen Parser (after action) ... ###\n")
        after_state = self.env.get_state(wait_to_stabilize=False)
        after_screenshot = self._save_screenshot_from_pixels(
            after_state.pixels,
            f"step_{self.step_count}_after.jpg"
        )
        
        perception_infos_after = self._get_perception_infos(after_screenshot)
        self.info_pool.perception_infos_post = copy.deepcopy(perception_infos_after)
        self.info_pool.keyboard_post = self._detect_keyboard(perception_infos_after)
        print(f"Detected {len(perception_infos_after)} UI elements (after action)")
        
        # Cache for next step: "after action" becomes "pre" for next step
        self.info_pool.perception_infos_pre = copy.deepcopy(perception_infos_after)
        self.info_pool.state_perception_infos = copy.deepcopy(perception_infos_after)
        self.info_pool.keyboard_pre = self.info_pool.keyboard_post
        self.info_pool.current_screenshot_path = after_screenshot
        
        # === STATE AGENT (after action) ===
        self._run_state_agent(after_screenshot, is_first=False)
        
        # === REFLECTOR ===
        reflection = self._run_reflection(screenshot_file, after_screenshot, action_str)
        step_data["outcome"] = reflection["outcome"]
        step_data["summary"] = f"Action: {action_str}, Outcome: {reflection['outcome']}"
        
        # Update info pool with reflection results
        if not error:
            self.info_pool.action_outcomes.append(reflection["outcome"])
            if reflection["outcome"] != "A":
                self.info_pool.error_descriptions.append(reflection.get("error_description", ""))
            else:
                self.info_pool.error_descriptions.append("")
        
        if reflection.get("progress_status"):
            self.info_pool.progress_status = reflection["progress_status"]
        
        self.history.append(step_data)
        
        # === LOG STEP TO JSON (like original Agent-SAMA) ===
        step_log = {
            "step": self.step_count,
            "goal": goal,
            "operations": []
        }
        
        # Log perception
        step_log["operations"].append({
            "type": "perception",
            "screenshot": screenshot_file,
            "perception_count": len(self.info_pool.perception_infos_pre) if self.info_pool else 0,
        })
        
        # Log planner
        step_log["operations"].append({
            "type": "planning",
            "plan": self.info_pool.plan if self.info_pool else [],
            "current_subgoal": self.info_pool.current_subgoal if self.info_pool else "",
        })
        
        # Log action
        step_log["operations"].append({
            "type": "action",
            "action": action_str,
            "action_thought": action_result.get("thought", ""),
            "action_description": action_result.get("description", ""),
        })
        
        # Log reflection
        step_log["operations"].append({
            "type": "reflection",
            "outcome": reflection.get("outcome", ""),
            "error_description": reflection.get("error_description", ""),
            "progress_status": reflection.get("progress_status", ""),
        })
        
        self.steps_log.append(step_log)
        
        # Save steps.json after each step (incremental save)
        self._save_steps_json()
        
        # === TRAJECTORY COLLECTION ===
        if self.trajectory_collector and self.collect_trajectory:
            try:
                # Start task if first step
                if self.step_count == 1:
                    self.trajectory_collector.start_task(goal)
                
                # Create transition data using Agent-SAMA's helper
                transition_data = create_transition_from_info_pool(
                    self.info_pool,
                    reflection["outcome"],
                    reflection.get("error_description", "")
                )
                
                # Since screenshots are already saved directly to trajectory images_dir,
                # we record without copying (files are already in place)
                self.trajectory_collector.step_count += 1
                self.trajectory_collector.current_subgoal = self.info_pool.current_subgoal
                
                # Get relative paths (files already in images_dir)
                before_img_name = os.path.basename(screenshot_file)
                after_img_name = os.path.basename(after_screenshot)
                
                # Add validity to to_state
                # outcome is like "A: Successful..." or "B: ..." or "C: ..."
                is_valid = reflection["outcome"].strip().startswith("A")
                to_state_with_valid = {**transition_data["to_state"], "valid": is_valid}
                
                # Create transition record
                transition = {
                    "step": self.step_count,
                    "from_state": transition_data["from_state"],
                    "action": transition_data["action"],
                    "to_state": to_state_with_valid,
                    "before_image": f"images/{before_img_name}",
                    "after_image": f"images/{after_img_name}",
                    "outcome": reflection["outcome"],
                    "reason": reflection.get("error_description", "") or reflection.get("progress_status", "")
                }
                self.trajectory_collector.transitions.append(transition)
                
                # Generate message format for training
                message = self.trajectory_collector._create_message(
                    transition=transition,
                    previous_steps=self.trajectory_collector.get_previous_steps(k=2),
                    is_valid=is_valid,
                    reason=transition["reason"]
                )
                self.trajectory_collector.messages.append(message)
                
                # Autosave
                if self.trajectory_collector.autosave:
                    self.trajectory_collector._incremental_save()
                
                print(f"[Trajectory] Recorded transition for step {self.step_count}")
            except Exception as e:
                logging.warning(f"Failed to record trajectory: {e}")
                import traceback
                traceback.print_exc()
        
        # Step completion summary
        print(f"\n{'─'*40}")
        print(f"Step {self.step_count} completed. Outcome: {reflection['outcome']}")
        print(f"Progress: {self.info_pool.progress_status}")
        print(f"{'─'*40}\n")
        
        return base_agent.AgentInteractionResult(done=False, data=step_data)
    
    def _detect_keyboard(self, perception_infos: List[Dict]) -> bool:
        """Detect if keyboard is visible."""
        keyboard_height_limit = 0.9 * self.info_pool.height if self.info_pool else 2000
        for info in perception_infos:
            if info.get('coordinates', (0, 0))[1] < keyboard_height_limit:
                continue
            if 'ADB Keyboard' in info.get('text', ''):
                return True
        return False
    
    def finalize(self):
        """Called when task is complete."""
        # Save final steps.json
        self._save_steps_json()
        
        # Save trajectory data if collecting
        if self.trajectory_collector:
            try:
                self.trajectory_collector.save()
                stats = self.trajectory_collector.get_stats()
                print(f"[Trajectory] Final stats: {stats}")
            except Exception as e:
                logging.warning(f"Failed to save trajectory: {e}")
        
        if self.info_pool:
            # Print FSM summary
            try:
                self.state_agent.print_fsm_summary(self.info_pool)
            except Exception as e:
                logging.warning(f"Could not print FSM summary: {e}")
        
        # Save persistent knowledge for future tasks (if task succeeded)
        if self.task_succeeded:
            print("\n### Saving Persistent Knowledge for Future Tasks ... ###\n")
            self._save_persistent_knowledge()
        
        # Clean up temp screenshot directory (only if not collecting trajectories)
        # If collecting trajectories, screenshots are already copied to trajectory dir
        if not self.collect_trajectory:
            try:
                import shutil
                if os.path.exists(self.screenshot_dir):
                    shutil.rmtree(self.screenshot_dir)
            except Exception:
                pass
