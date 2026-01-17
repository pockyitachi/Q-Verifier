"""
Comprehensive Test Suite for Agent-SAMA Agents

This module contains unit tests for all agents in the Agent-SAMA framework:
- InfoPool
- BaseAgent (interface)
- StateAgent (FSM management)
- Planner
- Actor  
- ReflectionAgent
- Mentor
- ScreenParser (mocked)

Run with: python -m pytest tests/test_agents.py -v
Or simply: python tests/test_agents.py
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import json
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy dependencies before importing agents
sys.modules['modelscope'] = MagicMock()
sys.modules['modelscope.pipelines'] = MagicMock()
sys.modules['modelscope.utils'] = MagicMock()
sys.modules['modelscope.utils.constant'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()

# Mock tool_utils functions that require ADB
mock_tool_utils = MagicMock()
mock_tool_utils.get_current_app = MagicMock(return_value={"app_name": "Unknown", "package": None})
mock_tool_utils.normalize_app_name = lambda x: {
    "Maps": "Google Maps",
    "Chrome": "Google Chrome",
    "Calendar": "Google Calendar",
}.get(x, x)
sys.modules['AgentsUtils.tool_utils'] = mock_tool_utils

# Mock OCR utils
sys.modules['AgentsUtils.ocr_utils'] = MagicMock()

from Agents.info_pool import InfoPool
from Agents.state_agent import StateAgent
from Agents.planner import Planner
from Agents.actor import Actor
from Agents.reflection_agent import ReflectionAgent
from Agents.mentor import Mentor


class TestInfoPool(unittest.TestCase):
    """Test cases for InfoPool data structure."""
    
    def test_default_initialization(self):
        """Test that InfoPool initializes with correct default values."""
        pool = InfoPool()
        
        # Check defaults
        self.assertEqual(pool.instruction, "")
        self.assertEqual(pool.tips, "")
        self.assertEqual(pool.shortcuts, {})
        self.assertEqual(pool.width, 1080)
        self.assertEqual(pool.height, 2340)
        self.assertEqual(pool.perception_infos_pre, [])
        self.assertEqual(pool.perception_infos_post, [])
        self.assertEqual(pool.action_history, [])
        self.assertEqual(pool.state_graphs, {})
        self.assertEqual(pool.recovery_attempts, 0)
        self.assertEqual(pool.recovery_successes, 0)
        
    def test_custom_initialization(self):
        """Test InfoPool with custom values."""
        pool = InfoPool(
            instruction="Find a restaurant",
            width=1440,
            height=2960
        )
        
        self.assertEqual(pool.instruction, "Find a restaurant")
        self.assertEqual(pool.width, 1440)
        self.assertEqual(pool.height, 2960)
        
    def test_mutable_defaults(self):
        """Test that mutable defaults are not shared between instances."""
        pool1 = InfoPool()
        pool2 = InfoPool()
        
        pool1.action_history.append("action1")
        pool1.shortcuts["shortcut1"] = {"name": "test"}
        
        # pool2 should not be affected
        self.assertEqual(pool2.action_history, [])
        self.assertEqual(pool2.shortcuts, {})
        
    def test_fsm_related_fields(self):
        """Test FSM-related fields initialization."""
        pool = InfoPool()
        
        self.assertEqual(pool.state_graphs, {})
        self.assertEqual(pool.state_id_lists, {})
        self.assertEqual(pool.beacon_hist, set())
        self.assertEqual(pool.last_state_id, "")
        self.assertEqual(pool.active_app_name, "")


class TestStateAgent(unittest.TestCase):
    """Test cases for StateAgent (FSM management)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_agent = StateAgent(adb_path=None)
        self.info_pool = InfoPool()
        self.info_pool.instruction = "Find a restaurant on Google Maps"
        self.info_pool.current_screenshot_path = "/tmp/test.png"
        
    def test_init_chat(self):
        """Test init_chat returns proper system prompt."""
        chat_history = self.state_agent.init_chat()
        
        self.assertEqual(len(chat_history), 1)
        self.assertEqual(chat_history[0][0], "system")
        self.assertIn("state analyzer", chat_history[0][1][0]["text"])
        
    def test_init_chat_for_recovery(self):
        """Test recovery mode initialization."""
        chat_history = self.state_agent.init_chat_for_recovery()
        
        self.assertEqual(len(chat_history), 1)
        self.assertEqual(chat_history[0][0], "system")
        self.assertIn("recovery planner", chat_history[0][1][0]["text"])
        
    def test_generate_state_id(self):
        """Test state ID generation from beacon."""
        beacon = "Homepage of Google Maps"
        state_id = self.state_agent._generate_state_id(beacon)
        
        # Should be MD5 hash
        self.assertEqual(len(state_id), 32)
        # Should be consistent
        self.assertEqual(state_id, self.state_agent._generate_state_id(beacon))
        
    def test_normalize_beacon(self):
        """Test beacon normalization."""
        # Test various normalizations
        test_cases = [
            ("Home Page of Maps", "Homepage of Google Maps"),
            ("search page of Maps", "Search Page of Google Maps"),
            ("Homepage of Chrome", "Homepage of Google Chrome"),
            ("Detail page of Calendar", "Detail Page of Google Calendar"),
        ]
        
        for input_beacon, expected in test_cases:
            result = StateAgent._normalize_beacon(input_beacon)
            self.assertEqual(result, expected, f"Failed for input: {input_beacon}")
            
    def test_get_or_create_app_graph(self):
        """Test graph creation for new apps."""
        graph = self.state_agent.get_or_create_app_graph(self.info_pool, "Google Maps")
        
        self.assertIn("Google Maps", self.info_pool.state_graphs)
        self.assertEqual(graph.number_of_nodes(), 0)
        self.assertEqual(graph.number_of_edges(), 0)
        
    def test_update_state_graph_new_state(self):
        """Test adding a new state to the FSM graph."""
        self.state_agent.update_state_graph(
            info_pool=self.info_pool,
            state_description="Google Maps homepage showing search bar",
            current_app_name="Maps",
            current_state_beacon="Homepage of Maps",
            precondition="App is open",
            postcondition="Search bar is visible",
            predicted_next_state="Search results page",
            use_adb_app_detection=False
        )
        
        # Check app graph was created
        self.assertIn("Google Maps", self.info_pool.state_graphs)
        
        # Check state was added
        graph = self.info_pool.state_graphs["Google Maps"]
        self.assertEqual(graph.number_of_nodes(), 1)
        
        # Check active app was updated
        self.assertEqual(self.info_pool.active_app_name, "Google Maps")
        
        # Check beacon was normalized and added
        self.assertIn("Homepage of Google Maps", self.info_pool.beacon_hist)
        
    def test_update_state_graph_transition(self):
        """Test adding state transitions (edges) to FSM."""
        # Add first state
        self.state_agent.update_state_graph(
            info_pool=self.info_pool,
            state_description="Homepage",
            current_app_name="Maps",
            current_state_beacon="Homepage of Maps",
            precondition="",
            postcondition="",
            predicted_next_state="",
            use_adb_app_detection=False
        )
        
        # Simulate an action
        self.info_pool.last_action = {"name": "Tap", "arguments": {"x": 100, "y": 200}}
        
        # Add second state
        self.state_agent.update_state_graph(
            info_pool=self.info_pool,
            state_description="Search results",
            current_app_name="Maps",
            current_state_beacon="Search Page of Maps",
            precondition="",
            postcondition="",
            predicted_next_state="",
            use_adb_app_detection=False
        )
        
        # Check transition was created
        graph = self.info_pool.state_graphs["Google Maps"]
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        
    def test_no_duplicate_state_ids(self):
        """Test that duplicate state IDs are not added to lists."""
        # Add same state twice
        for _ in range(2):
            self.state_agent.update_state_graph(
                info_pool=self.info_pool,
                state_description="Homepage",
                current_app_name="Maps",
                current_state_beacon="Homepage of Maps",
                precondition="",
                postcondition="",
                predicted_next_state="",
                use_adb_app_detection=False
            )
        
        # Should only have one entry
        self.assertEqual(len(self.info_pool.state_id_lists["Google Maps"]), 1)
        
    def test_parse_response(self):
        """Test parsing StateAgent LLM response."""
        response = """### State Description ###
The current screen shows the Google Maps homepage with a search bar at the top.

### Predicted Next State ###
After searching, the results page will show matching locations.

### App Inference ###
Google Maps

### State Beacon ###
Homepage of Google Maps

### Post-condition of Current State ###
Search bar is visible and ready for input.

### Pre-condition of Next State ###
User must enter a search query.
"""
        
        parsed = self.state_agent.parse_response(response)
        
        self.assertIn("state_description", parsed)
        self.assertIn("predicted_next_state", parsed)
        self.assertIn("current_app_name", parsed)
        self.assertIn("state_beacon", parsed)
        self.assertEqual(parsed["current_app_name"], "Google Maps")
        self.assertEqual(parsed["state_beacon"], "Homepage of Google Maps")
        
    def test_get_prompt(self):
        """Test prompt generation includes necessary sections."""
        self.info_pool.current_subgoal = "Search for restaurants"
        self.info_pool.state_perception_infos = [
            {"coordinates": [100, 200], "text": "Search"}
        ]
        
        prompt = self.state_agent.get_prompt(self.info_pool)
        
        self.assertIn("### Current Screen State Analysis ###", prompt)
        self.assertIn("### Instruction ###", prompt)
        self.assertIn("### State Beacon ###", prompt)
        self.assertIn("### Your Output Format ###", prompt)
        
    def test_get_fsm_summary(self):
        """Test FSM summary generation."""
        # Add some states
        self.state_agent.update_state_graph(
            info_pool=self.info_pool,
            state_description="Homepage",
            current_app_name="Maps",
            current_state_beacon="Homepage of Maps",
            precondition="",
            postcondition="",
            predicted_next_state="",
            use_adb_app_detection=False
        )
        
        summary = self.state_agent.get_fsm_summary(self.info_pool)
        
        self.assertEqual(summary["total_apps"], 1)
        self.assertEqual(summary["total_states"], 1)
        self.assertIn("Google Maps", summary["apps"])
        
    def test_find_path_to_state(self):
        """Test pathfinding between FSM states."""
        # Create a simple graph: A -> B -> C
        import networkx as nx
        
        self.info_pool.state_graphs["TestApp"] = nx.DiGraph()
        graph = self.info_pool.state_graphs["TestApp"]
        graph.add_edge("state_a", "state_b")
        graph.add_edge("state_b", "state_c")
        
        path = self.state_agent.find_path_to_state(
            self.info_pool, "TestApp", "state_a", "state_c"
        )
        
        self.assertEqual(path, ["state_a", "state_b", "state_c"])
        
    def test_find_path_no_path(self):
        """Test pathfinding when no path exists."""
        import networkx as nx
        
        self.info_pool.state_graphs["TestApp"] = nx.DiGraph()
        graph = self.info_pool.state_graphs["TestApp"]
        graph.add_node("state_a")
        graph.add_node("state_b")
        # No edges - disconnected
        
        path = self.state_agent.find_path_to_state(
            self.info_pool, "TestApp", "state_a", "state_b"
        )
        
        self.assertIsNone(path)
        
    def test_resolve_target_state(self):
        """Test resolving target state from beacon or ID."""
        # Setup a state
        self.state_agent.update_state_graph(
            info_pool=self.info_pool,
            state_description="Homepage",
            current_app_name="Maps",
            current_state_beacon="Homepage of Maps",
            precondition="",
            postcondition="",
            predicted_next_state="",
            use_adb_app_detection=False
        )
        
        # Resolve by beacon
        beacon = "Homepage of Google Maps"
        resolved = self.state_agent.resolve_target_state(self.info_pool, beacon)
        
        self.assertIsNotNone(resolved)
        self.assertEqual(len(resolved), 32)  # MD5 hash


class TestPlanner(unittest.TestCase):
    """Test cases for Planner agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.planner = Planner()
        self.info_pool = InfoPool()
        self.info_pool.instruction = "Find a Korean restaurant in Chicago"
        
    def test_init_chat(self):
        """Test planner initialization."""
        chat_history = self.planner.init_chat()
        
        self.assertEqual(len(chat_history), 1)
        self.assertEqual(chat_history[0][0], "system")
        self.assertIn("high-level plans", chat_history[0][1][0]["text"])
        
    def test_init_chat_for_judging(self):
        """Test LLM-as-Judge initialization."""
        chat_history = self.planner.init_chat_for_judging()
        
        self.assertEqual(len(chat_history), 1)
        self.assertIn("LLM judge", chat_history[0][1][0]["text"])
        
    def test_get_prompt_initial_planning(self):
        """Test initial planning prompt generation."""
        self.info_pool.plan = ""  # No plan yet
        
        prompt = self.planner.get_prompt(self.info_pool)
        
        self.assertIn("### User Instruction ###", prompt)
        self.assertIn("Find a Korean restaurant", prompt)
        self.assertIn("5 high-level plans", prompt)  # LLM_Judge = True
        
    def test_get_prompt_continue_planning(self):
        """Test continue planning prompt."""
        self.info_pool.plan = "1. Open Maps\n2. Search for restaurants"
        self.info_pool.current_subgoal = "Open Maps"
        self.info_pool.progress_status = "Starting task"
        
        prompt = self.planner.get_prompt(self.info_pool)
        
        self.assertIn("### Current Plan ###", prompt)
        self.assertIn("### Previous Subgoal ###", prompt)
        self.assertIn("### Progress Status ###", prompt)
        
    def test_get_prompt_with_shortcuts(self):
        """Test prompt includes shortcuts when available."""
        self.info_pool.shortcuts = {
            "SearchAndTap": {
                "description": "Search and tap result",
                "precondition": "Search bar visible"
            }
        }
        
        prompt = self.planner.get_prompt(self.info_pool)
        
        self.assertIn("### Available Shortcuts", prompt)
        self.assertIn("SearchAndTap", prompt)
        
    def test_parse_response_single_plan(self):
        """Test parsing a single plan response."""
        response = """```json
{
    "thought": "This plan opens Maps first",
    "plan": ["Open Google Maps", "Search for Korean restaurant", "Check reviews"],
    "current_subgoal": "Open Google Maps"
}
```"""
        
        parsed = self.planner.parse_response(response)
        
        self.assertIn("candidates", parsed)
        self.assertEqual(len(parsed["candidates"]), 1)
        self.assertEqual(parsed["candidates"][0]["current_subgoal"], "Open Google Maps")
        
    def test_parse_response_multiple_plans(self):
        """Test parsing multiple plan candidates."""
        response = """```json
[
    {
        "thought": "Plan 1",
        "plan": ["Step A", "Step B"],
        "current_subgoal": "Step A"
    },
    {
        "thought": "Plan 2",
        "plan": ["Step X", "Step Y"],
        "current_subgoal": "Step X"
    }
]
```"""
        
        parsed = self.planner.parse_response(response)
        
        self.assertIn("candidates", parsed)
        self.assertEqual(len(parsed["candidates"]), 2)
        
    def test_parse_judge_response(self):
        """Test parsing judge response."""
        response = """### Thought ###
Plan 2 is best because it is more efficient.

### Plan ###
Step X, Step Y

### Current Subgoal ###
Step X
"""
        
        parsed = self.planner.parse_judge_response(response)
        
        self.assertIn("thought", parsed)
        self.assertIn("plan", parsed)
        self.assertIn("current_subgoal", parsed)
        self.assertEqual(parsed["current_subgoal"], "Step X")
        
    def test_extract_app_from_instruction(self):
        """Test app extraction from instruction."""
        apps = Planner.extract_app_from_instruction(
            "Find a restaurant on Google Maps and save it to Notes"
        )
        
        self.assertIn("Google Maps", apps)
        self.assertIn("Notes", apps)
        
    def test_build_fsm_patterns_section(self):
        """Test FSM patterns section building."""
        import networkx as nx
        
        # Add some state graph data
        self.info_pool.state_graphs["Google Maps"] = nx.DiGraph()
        graph = self.info_pool.state_graphs["Google Maps"]
        graph.add_node("state1", beacon="Homepage of Google Maps", visit_count=3)
        graph.add_node("state2", beacon="Search Page", visit_count=1)
        graph.add_edge("state1", "state2", action={"name": "Tap"})
        
        section = self.planner._build_fsm_patterns_section(self.info_pool)
        
        self.assertIn("Known App Navigation Patterns", section)
        self.assertIn("Google Maps", section)


class TestActor(unittest.TestCase):
    """Test cases for Actor agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.actor = Actor(adb_path=None)
        self.info_pool = InfoPool()
        self.info_pool.instruction = "Search for coffee shops"
        self.info_pool.plan = "Open Maps, Search for coffee"
        self.info_pool.current_subgoal = "Tap on search bar"
        self.info_pool.perception_infos_pre = [
            {"text": "text: Search", "coordinates": [540, 100]},
            {"text": "icon: magnifying glass", "coordinates": [50, 100]}
        ]
        
    def test_init_chat(self):
        """Test actor chat initialization."""
        chat_history = self.actor.init_chat()
        
        self.assertEqual(len(chat_history), 1)
        self.assertEqual(chat_history[0][0], "system")
        self.assertIn("operating mobile phones", chat_history[0][1][0]["text"])
        
    def test_init_chat_recovery(self):
        """Test recovery mode initialization."""
        chat_history = self.actor.init_chat_recovery()
        
        self.assertIn("recovery plan", chat_history[0][1][0]["text"])
        
    def test_get_prompt(self):
        """Test prompt generation includes all sections."""
        prompt = self.actor.get_prompt(self.info_pool)
        
        self.assertIn("### User Instruction ###", prompt)
        self.assertIn("### Overall Plan ###", prompt)
        self.assertIn("### Current Subtask ###", prompt)
        self.assertIn("### Screen Information ###", prompt)
        self.assertIn("#### Atomic Actions ####", prompt)
        self.assertIn("### Thought ###", prompt)
        self.assertIn("### Action ###", prompt)
        
    def test_get_prompt_keyboard_active(self):
        """Test prompt when keyboard is active."""
        self.info_pool.keyboard_pre = True
        
        prompt = self.actor.get_prompt(self.info_pool)
        
        self.assertIn("keyboard has been activated", prompt)
        self.assertIn("Type", prompt)  # Type action should be available
        
    def test_get_prompt_keyboard_inactive(self):
        """Test prompt when keyboard is inactive."""
        self.info_pool.keyboard_pre = False
        
        prompt = self.actor.get_prompt(self.info_pool)
        
        self.assertIn("keyboard has not been activated", prompt)
        self.assertIn("Unable to type", prompt)
        
    def test_get_prompt_with_tips(self):
        """Test prompt includes tips."""
        self.info_pool.tips = "Tip 1: Use search shortcuts"
        
        prompt = self.actor.get_prompt(self.info_pool)
        
        self.assertIn("### Tips ###", prompt)
        self.assertIn("Use search shortcuts", prompt)
        
    def test_get_prompt_with_action_history(self):
        """Test prompt includes action history."""
        self.info_pool.action_history = [{"name": "Tap", "arguments": {"x": 100, "y": 200}}]
        self.info_pool.summary_history = ["Tapped search bar"]
        self.info_pool.action_outcomes = ["A"]
        self.info_pool.error_descriptions = [""]
        
        prompt = self.actor.get_prompt(self.info_pool)
        
        self.assertIn("### Latest Action History ###", prompt)
        self.assertIn("Successful", prompt)
        
    def test_parse_response(self):
        """Test parsing actor response."""
        response = """### Thought ###
I need to tap on the search bar to enter my query.

### Action ###
{"name": "Tap", "arguments": {"x": 540, "y": 100}}

### Description ###
Tap on the search bar to activate text input.
"""
        
        parsed = self.actor.parse_response(response)
        
        self.assertIn("thought", parsed)
        self.assertIn("action", parsed)
        self.assertIn("description", parsed)
        self.assertIn("Tap", parsed["action"])
        
    def test_get_prompt_recovery(self):
        """Test recovery mode prompt generation."""
        self.info_pool.recovery_target_state_id = "abc123"
        self.info_pool.recovery_plan = {
            "abc123": {
                "recovery_goal": "Return to search page",
                "current_subgoal": "Press back button"
            }
        }
        
        prompt = self.actor.get_prompt_recovery(self.info_pool)
        
        self.assertIn("### Recovery Goal ###", prompt)
        self.assertIn("Return to search page", prompt)


class TestReflectionAgent(unittest.TestCase):
    """Test cases for ReflectionAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reflection = ReflectionAgent()
        self.info_pool = InfoPool()
        self.info_pool.instruction = "Find a restaurant"
        self.info_pool.current_subgoal = "Search for restaurants"
        self.info_pool.last_action = {"name": "Tap", "arguments": {"x": 100, "y": 200}}
        self.info_pool.last_summary = "Tap on search bar"
        self.info_pool.perception_infos_pre = [
            {"text": "text: Search", "coordinates": [100, 200]}
        ]
        self.info_pool.perception_infos_post = [
            {"text": "text: Enter query", "coordinates": [100, 200]}
        ]
        
    def test_init_chat(self):
        """Test reflection agent initialization."""
        chat_history = self.reflection.init_chat()
        
        self.assertEqual(len(chat_history), 1)
        self.assertIn("verify", chat_history[0][1][0]["text"])
        
    def test_get_prompt(self):
        """Test prompt generation."""
        prompt = self.reflection.get_prompt(self.info_pool)
        
        self.assertIn("### User Instruction ###", prompt)
        self.assertIn("### Current Subgoal ###", prompt)
        self.assertIn("### Screen Information Before the Action ###", prompt)
        self.assertIn("### Screen Information After the Action ###", prompt)
        self.assertIn("### Latest Action ###", prompt)
        self.assertIn("### Outcome ###", prompt)
        
    def test_get_prompt_with_predictions(self):
        """Test prompt includes state predictions."""
        self.info_pool.pre_state_pre_condition = "Search bar visible"
        self.info_pool.pre_state_post_condition = "Keyboard active"
        self.info_pool.pre_state_state_prediction = "Text input screen"
        
        prompt = self.reflection.get_prompt(self.info_pool)
        
        self.assertIn("### Precondition ###", prompt)
        self.assertIn("### Postcondition ###", prompt)
        self.assertIn("### Predicted Next Screen Description ###", prompt)
        
    def test_parse_response_success(self):
        """Test parsing successful action response."""
        response = """### Outcome ###
A

### Error Description ###
None

### Progress Status ###
Successfully tapped the search bar. Ready to enter query.
"""
        
        parsed = self.reflection.parse_response(response)
        
        self.assertEqual(parsed["outcome"], "A")
        self.assertEqual(parsed["error_description"], "None")
        self.assertIn("Successfully", parsed["progress_status"])
        
    def test_parse_response_failure(self):
        """Test parsing failed action response."""
        response = """### Outcome ###
B

### Error Description ###
The tap landed on wrong element, navigated to settings instead.

### Progress Status ###
Still trying to access search functionality.
"""
        
        parsed = self.reflection.parse_response(response)
        
        self.assertEqual(parsed["outcome"], "B")
        self.assertIn("wrong element", parsed["error_description"])
        
    def test_get_prompt_recovery_mode(self):
        """Test recovery mode prompt."""
        self.info_pool.recovery_target_state_id = "state123"
        self.info_pool.recovery_plan = {
            "state123": {
                "recovery_goal": "Return to homepage",
                "current_subgoal": "Press back button"
            }
        }
        self.info_pool.current_recovery_action = "Back"
        self.info_pool.current_recovery_thought = "Need to go back to previous screen"
        
        prompt = self.reflection.get_prompt_recovery_mode(self.info_pool)
        
        self.assertIn("### Recovery Mode", prompt)
        self.assertIn("### Recovery Goal", prompt)
        self.assertIn("### Target Recovery State ID ###", prompt)
        
    def test_get_prompt_for_note(self):
        """Test note-taking prompt."""
        self.info_pool.important_notes = "Found 3 restaurants nearby"
        
        prompt = self.reflection.get_prompt_for_note(self.info_pool)
        
        self.assertIn("### User Instruction", prompt)
        self.assertIn("### Existing Important Notes", prompt)
        self.assertIn("### Important Notes ###", prompt)
        
    def test_parse_response_note(self):
        """Test parsing note response."""
        response = """### Important Notes ###
Found 3 Korean restaurants: Kim's, Seoul Garden, and Korean BBQ. 
All have 4+ stars rating.
"""
        
        parsed = self.reflection.parse_response_note(response)
        
        self.assertIn("important_notes", parsed)
        self.assertIn("Korean restaurants", parsed["important_notes"])


class TestMentor(unittest.TestCase):
    """Test cases for Mentor agent (self-evolution)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mentor = Mentor(mode="unified")
        self.info_pool = InfoPool()
        self.info_pool.instruction = "Search for restaurants on Maps"
        self.info_pool.plan = "1. Open Maps\n2. Search for restaurants"
        self.info_pool.progress_status = "Task completed"
        self.info_pool.action_history = [
            {"name": "Tap", "arguments": {"x": 100, "y": 200}},
            {"name": "Type", "arguments": {"text": "restaurants"}},
            {"name": "Enter", "arguments": {}},
        ]
        self.info_pool.summary_history = [
            "Tapped search bar",
            "Typed search query",
            "Submitted search"
        ]
        self.info_pool.action_outcomes = ["A", "A", "A"]
        self.info_pool.error_descriptions = ["", "", ""]
        self.info_pool.progress_status_history = ["Starting", "Searching", "Done"]
        
    def test_init_unified_mode(self):
        """Test unified mode initialization."""
        mentor = Mentor(mode="unified")
        chat_history = mentor.init_chat()
        
        self.assertIn("Shortcuts", chat_history[0][1][0]["text"])
        self.assertIn("Tips", chat_history[0][1][0]["text"])
        
    def test_init_shortcut_mode(self):
        """Test shortcut-only mode initialization."""
        mentor = Mentor(mode="shortcut")
        chat_history = mentor.init_chat()
        
        self.assertIn("reusable action patterns", chat_history[0][1][0]["text"])
        
    def test_init_tips_mode(self):
        """Test tips-only mode initialization."""
        mentor = Mentor(mode="tips")
        chat_history = mentor.init_chat()
        
        self.assertIn("learning from mistakes", chat_history[0][1][0]["text"])
        
    def test_get_prompt(self):
        """Test prompt generation."""
        prompt = self.mentor.get_prompt(self.info_pool)
        
        self.assertIn("### Completed Task ###", prompt)
        self.assertIn("### Full Action History ###", prompt)
        self.assertIn("### Error & Recovery Analysis ###", prompt)
        self.assertIn("### New Shortcut ###", prompt)
        self.assertIn("### Updated Tips ###", prompt)
        
    def test_get_prompt_with_errors(self):
        """Test prompt includes error analysis."""
        self.info_pool.action_outcomes = ["A", "B", "A"]
        self.info_pool.error_descriptions = ["", "Wrong page", ""]
        self.info_pool.recovery_attempts = 1
        self.info_pool.recovery_successes = 1
        
        prompt = self.mentor.get_prompt(self.info_pool)
        
        self.assertIn("Total failures: 1", prompt)
        self.assertIn("Recovery attempts: 1", prompt)
        
    def test_parse_response_unified(self):
        """Test parsing unified mode response."""
        response = """### New Shortcut ###
```json
{
    "name": "SearchOnMaps",
    "arguments": ["query"],
    "description": "Search for a place on Maps",
    "precondition": "Maps is open",
    "atomic_action_sequence": [
        {"name": "Tap", "arguments_map": {"x": "search_x", "y": "search_y"}},
        {"name": "Type", "arguments_map": {"text": "query"}}
    ]
}
```

### Updated Tips ###
1. Always wait for search results to load
2. Check reviews before making decisions
"""
        
        parsed = self.mentor.parse_response(response)
        
        self.assertIn("new_shortcut", parsed)
        self.assertIn("updated_tips", parsed)
        self.assertIn("SearchOnMaps", parsed["new_shortcut"])
        
    def test_add_new_shortcut_valid(self):
        """Test adding a valid shortcut."""
        shortcut_json = """
{
    "name": "QuickSearch",
    "arguments": ["query"],
    "description": "Quick search shortcut",
    "precondition": "Search bar visible",
    "atomic_action_sequence": [
        {"name": "Tap", "arguments_map": {"x": 100, "y": 200}},
        {"name": "Type", "arguments_map": {"text": "query"}}
    ]
}
"""
        
        result = self.mentor.add_new_shortcut(shortcut_json, self.info_pool)
        
        self.assertTrue(result)
        self.assertIn("QuickSearch", self.info_pool.shortcuts)
        
    def test_add_new_shortcut_none(self):
        """Test adding None shortcut."""
        result = self.mentor.add_new_shortcut("None", self.info_pool)
        
        self.assertFalse(result)
        
    def test_add_new_shortcut_invalid_action(self):
        """Test adding shortcut with invalid atomic action."""
        shortcut_json = """
{
    "name": "InvalidShortcut",
    "arguments": [],
    "description": "Invalid",
    "precondition": "None",
    "atomic_action_sequence": [
        {"name": "InvalidAction", "arguments_map": {}}
    ]
}
"""
        
        result = self.mentor.add_new_shortcut(shortcut_json, self.info_pool)
        
        self.assertFalse(result)
        
    def test_add_new_shortcut_duplicate(self):
        """Test duplicate shortcut rejection."""
        shortcut_json = """
{
    "name": "TestShortcut",
    "arguments": [],
    "description": "Test",
    "precondition": "None",
    "atomic_action_sequence": [
        {"name": "Tap", "arguments_map": {"x": 0, "y": 0}}
    ]
}
"""
        
        # Add once
        self.mentor.add_new_shortcut(shortcut_json, self.info_pool)
        
        # Try to add again
        result = self.mentor.add_new_shortcut(shortcut_json, self.info_pool)
        
        self.assertFalse(result)
        
    def test_update_tips(self):
        """Test tips update."""
        new_tips = "1. Tip one\n2. Tip two"
        
        self.mentor.update_tips(new_tips, self.info_pool)
        
        self.assertEqual(self.info_pool.tips, new_tips)
        
    def test_extract_fsm_patterns(self):
        """Test FSM pattern extraction."""
        import networkx as nx
        
        # Create a state graph
        self.info_pool.state_graphs["Google Maps"] = nx.DiGraph()
        graph = self.info_pool.state_graphs["Google Maps"]
        graph.add_node("state1", beacon="Homepage of Google Maps", visit_count=5)
        graph.add_node("state2", beacon="Search Page of Google Maps", visit_count=3)
        graph.add_edge("state1", "state2")
        
        patterns = self.mentor.extract_fsm_patterns(self.info_pool)
        
        self.assertGreater(len(patterns), 0)
        self.assertIn("Google Maps_main_flow", patterns[0]["pattern_name"])
        
    def test_extract_common_flows(self):
        """Test common flow extraction."""
        # Add some patterns
        self.info_pool.learned_navigation_patterns = [
            {"app": "Google Maps", "type": "search_flow", "pattern_name": "Maps_search"},
            {"app": "Chrome", "type": "search_flow", "pattern_name": "Chrome_search"}
        ]
        
        flows = self.mentor.extract_common_flows(self.info_pool)
        
        self.assertIn("search_flow", flows)
        self.assertIn("Google Maps", flows["search_flow"]["apps"])
        
    def test_reflect(self):
        """Test full reflection process."""
        import networkx as nx
        
        # Setup state graph
        self.info_pool.state_graphs["Google Maps"] = nx.DiGraph()
        self.info_pool.state_graphs["Google Maps"].add_node(
            "s1", beacon="Homepage", visit_count=2
        )
        
        response = """### New Shortcut ###
None

### Updated Tips ###
1. Use shortcuts when possible
"""
        
        results = self.mentor.reflect(self.info_pool, response)
        
        self.assertIn("shortcuts_added", results)
        self.assertIn("tips_updated", results)
        self.assertIn("patterns_extracted", results)


class TestIntegration(unittest.TestCase):
    """Integration tests for agent interactions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.info_pool = InfoPool()
        self.info_pool.instruction = "Find a coffee shop"
        
    def test_state_agent_planner_integration(self):
        """Test StateAgent and Planner work together."""
        state_agent = StateAgent(adb_path=None)
        planner = Planner()
        
        # State agent updates state graph
        state_agent.update_state_graph(
            info_pool=self.info_pool,
            state_description="Maps homepage",
            current_app_name="Maps",
            current_state_beacon="Homepage of Maps",
            precondition="",
            postcondition="",
            predicted_next_state="",
            use_adb_app_detection=False
        )
        
        # Planner should see the FSM patterns
        prompt = planner.get_prompt(self.info_pool)
        
        # Note: Planner builds FSM patterns section
        self.assertIn("### User Instruction ###", prompt)
        
    def test_mentor_updates_planner_context(self):
        """Test Mentor updates are visible to Planner."""
        mentor = Mentor()
        planner = Planner()
        
        # Mentor adds tip
        self.info_pool.tips = "Use search shortcuts for faster results"
        
        # Planner should see tips
        prompt = planner.get_prompt(self.info_pool)
        
        self.assertIn("### Tips from Past Experience ###", prompt)
        self.assertIn("search shortcuts", prompt)
        
    def test_full_action_cycle(self):
        """Test a complete action-reflection cycle."""
        actor = Actor(adb_path=None)
        reflection = ReflectionAgent()
        
        # Setup perception data
        self.info_pool.perception_infos_pre = [
            {"text": "text: Search", "coordinates": [100, 200]}
        ]
        self.info_pool.perception_infos_post = [
            {"text": "text: Enter query", "coordinates": [100, 200]}
        ]
        self.info_pool.current_subgoal = "Search for coffee shops"
        self.info_pool.plan = "Search and review"
        
        # Actor generates action prompt
        actor_prompt = actor.get_prompt(self.info_pool)
        self.assertIn("### Action ###", actor_prompt)
        
        # Simulate action execution
        self.info_pool.last_action = {"name": "Tap", "arguments": {"x": 100, "y": 200}}
        self.info_pool.last_summary = "Tapped search bar"
        
        # Reflection evaluates
        reflection_prompt = reflection.get_prompt(self.info_pool)
        self.assertIn("### Outcome ###", reflection_prompt)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_info_pool(self):
        """Test agents handle empty InfoPool gracefully."""
        info_pool = InfoPool()
        
        # All agents should handle empty pool
        planner = Planner()
        prompt = planner.get_prompt(info_pool)
        self.assertIn("### User Instruction ###", prompt)
        
    def test_state_agent_parse_malformed_response(self):
        """Test StateAgent handles malformed response without crashing."""
        state_agent = StateAgent()
        
        malformed_response = "This is not a properly formatted response"
        
        parsed = state_agent.parse_response(malformed_response)
        
        # Should not crash, returns keys (may have partial content)
        self.assertIn("state_description", parsed)
        self.assertIn("current_app_name", parsed)
        self.assertIn("state_beacon", parsed)
        
    def test_mentor_parse_empty_response(self):
        """Test Mentor handles empty response."""
        mentor = Mentor()
        
        parsed = mentor.parse_response("")
        
        self.assertEqual(parsed["new_shortcut"], "")
        
    def test_actor_parse_no_json(self):
        """Test Actor handles response without valid JSON."""
        actor = Actor(adb_path=None)
        
        response = """### Thought ###
I will tap here.

### Action ###
Tap at 100, 200

### Description ###
Tapping the button.
"""
        
        parsed = actor.parse_response(response)
        
        # Should extract text even if JSON is invalid
        self.assertIn("thought", parsed)
        self.assertIn("action", parsed)


def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInfoPool))
    suite.addTests(loader.loadTestsFromTestCase(TestStateAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestActor))
    suite.addTests(loader.loadTestsFromTestCase(TestReflectionAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMentor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

