"""
State Agent for Agent-SAMA: FSM-based State Modeling

This module implements the State Agent, which models app execution as a Finite State Machine (FSM).
In this FSM representation:
- **States**: Each UI screen corresponds to a distinct FSM state, identified by a unique beacon
- **Transitions**: User actions (tap, swipe, type, etc.) trigger state transitions
- **Pre/Post-conditions**: Each state has conditions that must hold before/after transitions

The FSM is implemented using NetworkX DiGraph where:
- Nodes represent UI states with attributes: beacon, description, pre/post-conditions
- Edges represent transitions with the action that triggered them

Reference: Agent-SAMA paper (AAAI 2026) - "State-Aware Mobile Assistant"
"""

from Agents.base_agent import BaseAgent
from Agents.info_pool import InfoPool
from AgentsUtils.tool_utils import get_current_app, normalize_app_name
import json
import hashlib
import networkx as nx
import re
import time
from typing import Dict, List, Optional, Tuple


class StateAgent(BaseAgent):
    """
    State Agent for FSM-based mobile app state modeling.
    
    The State Agent is responsible for:
    1. Analyzing current UI screens and generating state descriptions
    2. Maintaining per-app FSM graphs (state_graphs) that model navigation flow
    3. Assigning unique state beacons for state identification
    4. Defining pre/post-conditions for verification
    5. Planning recovery paths when errors occur using the FSM structure
    
    Attributes:
        The agent uses InfoPool.state_graphs which maps app_name -> NetworkX DiGraph
        Each graph node (state) contains:
            - beacon: Unique identifier string (e.g., "Homepage of Maps")
            - description: Natural language description of the screen
            - precondition: What must be true before entering this state
            - postcondition: What should be true after completing actions in this state
            - screenshot_path: Path to the screenshot for this state
            - perception_info: Extracted UI elements from the screen
            - visit_count: Number of times this state has been visited
            - first_visited: Timestamp of first visit
            - last_visited: Timestamp of most recent visit
        
        Each graph edge (transition) contains:
            - action: The action that triggered the transition
            - transition_count: Number of times this transition occurred
            - last_outcome: Most recent outcome (A=success, B/C=failure)
    """
    
    def __init__(self, adb_path: str = None):
        """
        Initialize StateAgent with optional ADB path for reliable app detection.
        
        Args:
            adb_path: Path to ADB executable. If provided, enables reliable
                     app detection via ADB instead of relying solely on LLM inference.
        """
        self.adb_path = adb_path
    
    def get_current_app_info(self, llm_inferred_app: str = None) -> Dict:
        """
        Get current app information, preferring ADB detection over LLM inference.
        
        Args:
            llm_inferred_app: App name inferred by LLM (fallback if ADB fails)
            
        Returns:
            Dict with:
                - app_name: Normalized app name
                - source: "adb" or "llm" indicating detection method
                - package: Package name (if from ADB)
        """
        result = {
            "app_name": "Unknown",
            "source": "unknown",
            "package": None
        }
        
        # Try ADB detection first (most reliable)
        if self.adb_path:
            try:
                adb_info = get_current_app(self.adb_path)
                if adb_info.get("app_name") and adb_info["app_name"] != "Unknown":
                    result["app_name"] = adb_info["app_name"]
                    result["source"] = "adb"
                    result["package"] = adb_info.get("package")
                    return result
            except Exception as e:
                print(f"[StateAgent] ADB app detection failed: {e}")
        
        # Fallback to LLM inference with normalization
        if llm_inferred_app:
            result["app_name"] = normalize_app_name(llm_inferred_app)
            result["source"] = "llm"
        
        return result
    
    def get_or_create_app_graph(self, info_pool: InfoPool, app_name: str) -> nx.DiGraph:
        """
        Get the FSM graph for an app, creating it if it doesn't exist.
        
        Args:
            info_pool: Central information repository
            app_name: Name of the app
            
        Returns:
            NetworkX DiGraph for the app
        """
        # Normalize app name first
        normalized_name = normalize_app_name(app_name)
        
        if normalized_name not in info_pool.state_graphs:
            info_pool.state_graphs[normalized_name] = nx.DiGraph()
            print(f"[FSM] Created new graph for app: {normalized_name}")
        
        return info_pool.state_graphs[normalized_name]
    
    def init_chat(self):
        operation_history = []
        system_prompt = ("You are a state analyzer for a mobile automation agent. "
                         "Your goal is to describe the current screen state and predict the next state based on the current subgoal and the provided perception data.")
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def init_chat_for_recovery(self):
        operation_history = []
        system_prompt = (
            "You are a recovery planner for a mobile assistant.\n"
            "Decide whether the previous recovery target state is still valid for completing the current subgoal.\n"
            "If valid, give a new plan to reach it. If not, select a better state and generate a plan.\n"
            "Use action history, error info, and the state graph.\n"
            "Output the target state and recovery plan in the required format."
        )
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history
    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### Current Screen State Analysis ###\n"
        prompt += "Using the provided perception data, describe the current screen state.\n\n"

        prompt += "### Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        if info_pool.plan:
            prompt += "### Global Plan ###\n"
            prompt += f"{info_pool.plan}\n\n"

        if info_pool.current_subgoal:
            prompt += "### Current Subgoal ###\n"
            prompt += f"{info_pool.current_subgoal}\n\n"

        # Incorporate perception info if available
        prompt += "### Screen Information (Perception Data) ###\n"
        if info_pool.state_perception_infos and len(info_pool.state_perception_infos) > 0:
            for item in info_pool.state_perception_infos:
                prompt += f"{item['coordinates']}: {item['text']}\n"
        else:
            prompt += "No perception data available.\n"
        prompt += "\n"

        beacon_hist_set = info_pool.beacon_hist

        prompt += "### Your Output Format ###\n"
        prompt += "Respond in the exact format below:\n\n"
        prompt += "### State Description ###\n"
        prompt += "<Brief natural language summary of the screen>\n\n"
        prompt += "### Predicted Next State ###\n"
        prompt += "Based on the user instruction, the global plan, and the current subgoal, predict what the screen will display once the next subgoal is successfully achieved.\n\n"
        prompt += "### App Inference ###\n"
        prompt += "Guess the name of the app currently being used based on the screen content. If the screen shows the Home screen, App Drawer, or any system UI with no app actively open, respond with 'System Operation'.\n\n"
        prompt += "Note that: ONLY RETURN THE APP NAME here, no more other words, check the previous beacon history to avoid have different variants of the same APP\n"
        prompt += "### State Beacon ###\n"
        prompt += "Use a very strict format such as:\n"
        prompt += "- Homepage of <AppName>\n"
        prompt += "- Search Page of <AppName>\n"
        prompt += "- Item Detail Page in <AppName>\n"
        prompt += "- Notes List in <AppName>\n"
        prompt += "- Settings Page of <AppName>\n"
        prompt += "No other variation is allowed. Use consistent naming.\n"
        prompt += "If a similar beacon has appeared before, reuse it exactly.\n"
        prompt += "\n"
        prompt += "Previously seen previous beacon history:\n"

        for beacon in beacon_hist_set:
            prompt += f"- {beacon}\n"

        prompt += "\n"
        prompt += "### Post-condition of Current State ###\n"
        prompt += "After completing the current subgoal, what should be true in the app or screen? State it clearly.\n\n"

        prompt += "### Pre-condition of Next State ###\n"
        prompt += "What needs to be true before the next subgoal can begin execution?\n\n"

        # prompt += "### Expected Beacon of Next State ###\n"
        # prompt += "Predict the beacon label of the next state you expect to enter. It should match the format rules of State Beacon above.\n"

        return prompt

    def parse_response(self, response: str) -> dict:
        try:
            state_description = response.split("### State Description ###")[-1].split("### Predicted Next State ###")[
                0].strip()
            predicted_next_state = response.split("### Predicted Next State ###")[-1].split("### App Inference ###")[
                0].strip()
            current_app_name = response.split("### App Inference ###")[-1].split("### State Beacon ###")[0].strip()
            state_beacon = response.split("### State Beacon ###")[-1].split("### Post-condition of Current State ###")[
                0].strip()
            postcondition = \
                response.split("### Post-condition of Current State ###")[-1].split(
                    "### Pre-condition of Next State ###")[
                    0].strip()
            precondition = response.split("### Pre-condition of Next State ###")[-1].strip()

            return {
                "state_description": state_description,
                "predicted_next_state": predicted_next_state,
                "current_app_name": current_app_name,
                "state_beacon": state_beacon,
                "postcondition": postcondition,
                "precondition": precondition,
                "expected_next_beacon": ""  # not included in current response format
            }

        except Exception as e:
            print(f"[StateAgent] Failed to parse response: {e}")
            return {
                "state_description": "",
                "predicted_next_state": "",
                "current_app_name": "",
                "state_beacon": "",
                "postcondition": "",
                "precondition": "",
            }

    def update_state_graph(self, info_pool: InfoPool, state_description: str, current_app_name: str,
                           current_state_beacon: str, precondition: str, postcondition: str, predicted_next_state: str,
                           use_adb_app_detection: bool = True):
        """
        Update the FSM graph with a new state (node) and transition (edge).
        
        This method maintains the FSM structure by:
        1. Creating a new state node if the beacon hasn't been seen before
        2. Updating visit statistics for existing states
        3. Adding transitions (edges) from the previous state with the triggering action
        4. Tracking cross-app transitions separately
        
        Args:
            info_pool: Central information repository
            state_description: Natural language description of current UI screen
            current_app_name: Name of the app currently in focus (from LLM)
            current_state_beacon: Unique identifier for this state (e.g., "Search Page of Maps")
            precondition: Condition that must hold before this state
            postcondition: Condition that should hold after actions in this state
            predicted_next_state: Expected next state after current subgoal completion
            use_adb_app_detection: If True and ADB path is set, use ADB for reliable app detection
        
        FSM Node Structure:
            - beacon: State identifier string
            - description: NL description of the screen
            - precondition/postcondition: For verification
            - visit_count: Times visited (for learning frequent paths)
            - first_visited/last_visited: Timestamps
            
        FSM Edge Structure:
            - action: The action dict that triggered this transition
            - transition_count: Times this transition occurred
            - last_outcome: Most recent action outcome
        """
        screenshot_path = info_pool.current_screenshot_path
        current_time = time.time()
        
        # Get verified app name (prefer ADB over LLM inference)
        if use_adb_app_detection and self.adb_path:
            app_info = self.get_current_app_info(current_app_name)
            verified_app_name = app_info["app_name"]
            app_source = app_info["source"]
            if app_source == "adb" and verified_app_name != normalize_app_name(current_app_name):
                print(f"[FSM] App name corrected: LLM='{current_app_name}' → ADB='{verified_app_name}'")
        else:
            # Normalize LLM-provided app name
            verified_app_name = normalize_app_name(current_app_name)
            app_source = "llm"
        
        # Normalize beacon to avoid state explosion from minor LLM variations
        normalized_beacon = self._normalize_beacon(current_state_beacon)
        
        # Get or create app-specific FSM graph
        state_graph = self.get_or_create_app_graph(info_pool, verified_app_name)

        # Generate unique state ID from normalized beacon (FSM state identifier)
        state_id = self._generate_state_id(normalized_beacon)
        
        # Track visited states for this app (only unique states)
        if verified_app_name not in info_pool.state_id_lists:
            info_pool.state_id_lists[verified_app_name] = []
        if state_id not in info_pool.state_id_lists[verified_app_name]:
            info_pool.state_id_lists[verified_app_name].append(state_id)
        
        # Add normalized beacon to history for consistent naming
        info_pool.beacon_hist.add(normalized_beacon)
        
        # Handle cross-app transitions (app switching) - BEFORE updating graphs
        if info_pool.active_app_name and verified_app_name != info_pool.active_app_name:
            previous_app = info_pool.active_app_name
            info_pool.app_transition_history.append({
                "from": previous_app,
                "to": verified_app_name,
                "action": info_pool.last_action,
                "action_thought": info_pool.last_action_thought,
                "timestamp": current_time,
                "from_state": info_pool.last_state_id  # Track which state we left
            })
            # Reset state tracking for new app context
            info_pool.last_state_id = ""
            print(f"[FSM] App transition: {previous_app} → {verified_app_name}")

        # Add or update FSM node (state)
        if state_id not in state_graph.nodes:
            # New state - add node with full attributes
            state_graph.add_node(
                state_id,
                beacon=normalized_beacon,  # Use normalized beacon
                beacon_raw=current_state_beacon,  # Keep original for debugging
                description=state_description,
                screenshot_path=screenshot_path,
                perception_info=info_pool.state_perception_infos,
                precondition=precondition,
                postcondition=postcondition,
                predicted_next_state=predicted_next_state,
                visit_count=1,
                first_visited=current_time,
                last_visited=current_time,
                app_name=verified_app_name,  # Store verified app name
                app_source=app_source  # Track how app was detected (adb/llm)
            )
            print(f"[FSM] New state added: {normalized_beacon} (app: {verified_app_name})")
        else:
            # Existing state - update visit statistics
            state_graph.nodes[state_id]['visit_count'] = state_graph.nodes[state_id].get('visit_count', 0) + 1
            state_graph.nodes[state_id]['last_visited'] = current_time
            # Update description if we have a better one
            if state_description and len(state_description) > len(state_graph.nodes[state_id].get('description', '')):
                state_graph.nodes[state_id]['description'] = state_description
            print(f"[FSM] Revisiting state: {normalized_beacon} (visit #{state_graph.nodes[state_id]['visit_count']})")
        
        # Add FSM transition (edge) from previous state
        if (hasattr(info_pool, "last_state_id") and info_pool.last_state_id
                and info_pool.active_app_name == verified_app_name):
            if info_pool.last_action:
                prev_state_id = info_pool.last_state_id
                
                if state_graph.has_edge(prev_state_id, state_id):
                    # Update existing transition
                    edge_data = state_graph.edges[prev_state_id, state_id]
                    edge_data['transition_count'] = edge_data.get('transition_count', 0) + 1
                    edge_data['last_action'] = info_pool.last_action
                else:
                    # New transition
                    state_graph.add_edge(
                        prev_state_id, 
                        state_id, 
                        action=info_pool.last_action,
                        transition_count=1
                    )
        
        # Update current state reference
        info_pool.last_state_id = state_id
        
        # CRITICAL: Update active_app_name to the VERIFIED app name
        # This ensures consistency for edge creation in subsequent calls
        info_pool.active_app_name = verified_app_name
    
    @staticmethod
    def _normalize_beacon(beacon: str) -> str:
        """
        Normalize beacon string to avoid state explosion from minor variations.
        
        Examples of normalization:
        - "Home Page of Maps" → "Homepage of Google Maps"
        - "search page of Maps" → "Search Page of Google Maps"
        - "Homepage of Chrome" → "Homepage of Google Chrome"
        """
        # Strip and normalize whitespace
        beacon = ' '.join(beacon.split())
        
        # Common state type normalizations
        beacon = beacon.replace("Home Page", "Homepage")
        beacon = beacon.replace("home page", "Homepage")
        beacon = beacon.replace("Detail page", "Detail Page")
        beacon = beacon.replace("detail page", "Detail Page")
        beacon = beacon.replace("Search page", "Search Page")
        beacon = beacon.replace("search page", "Search Page")
        beacon = beacon.replace("Results page", "Results Page")
        beacon = beacon.replace("results page", "Results Page")
        beacon = beacon.replace("List page", "List Page")
        beacon = beacon.replace("list page", "List Page")
        
        # App name normalizations in beacons
        app_normalizations = {
            " of Maps": " of Google Maps",
            " in Maps": " in Google Maps",
            " of Chrome": " of Google Chrome",
            " in Chrome": " in Google Chrome",
            " of Calendar": " of Google Calendar",
            " in Calendar": " in Google Calendar",
            " of Photos": " of Google Photos",
            " in Photos": " in Google Photos",
            " of Drive": " of Google Drive",
            " in Drive": " in Google Drive",
            " of Keep": " of Google Keep",
            " in Keep": " in Google Keep",
        }
        
        for old, new in app_normalizations.items():
            if old in beacon and new not in beacon:
                beacon = beacon.replace(old, new)
        
        # Capitalize first letter of each word for consistency
        # But preserve app names as-is
        parts = beacon.split(" of ")
        if len(parts) == 2:
            # Format: "Something of AppName"
            state_part = parts[0].strip()
            app_part = parts[1].strip()
            beacon = f"{state_part} of {app_part}"
        
        parts = beacon.split(" in ")
        if len(parts) == 2:
            # Format: "Something in AppName"
            state_part = parts[0].strip()
            app_part = parts[1].strip()
            beacon = f"{state_part} in {app_part}"
        
        return beacon
    
    @staticmethod
    def _generate_state_id(beacon: str) -> str:
        """Generate a unique state ID from the normalized beacon string using MD5 hash."""
        normalized = StateAgent._normalize_beacon(beacon)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_state_by_beacon(self, info_pool: InfoPool, app_name: str, beacon: str) -> Optional[Dict]:
        """
        Retrieve state information by beacon string.
        
        Args:
            info_pool: Central information repository
            app_name: Name of the app
            beacon: State beacon string
            
        Returns:
            Dict with state attributes or None if not found
        """
        if app_name not in info_pool.state_graphs:
            return None
        
        state_id = self._generate_state_id(beacon)
        graph = info_pool.state_graphs[app_name]
        
        if state_id in graph.nodes:
            return dict(graph.nodes[state_id])
        return None
    
    def get_state_id_by_beacon(self, beacon: str) -> str:
        """
        Get the state ID for a given beacon string.
        Useful for recovery when LLM provides beacon instead of ID.
        """
        return self._generate_state_id(beacon)
    
    def beacon_to_id_mapping(self, info_pool: InfoPool, app_name: str) -> Dict[str, str]:
        """
        Create a mapping from beacon strings to state IDs for an app.
        
        Returns:
            Dict mapping beacon -> state_id
        """
        mapping = {}
        if app_name not in info_pool.state_graphs:
            return mapping
        
        graph = info_pool.state_graphs[app_name]
        for state_id in graph.nodes:
            beacon = graph.nodes[state_id].get('beacon', '')
            if beacon:
                mapping[beacon] = state_id
        
        return mapping
    
    def resolve_target_state(self, info_pool: InfoPool, target_input: str) -> Optional[str]:
        """
        Resolve a target state from either beacon or state ID.
        
        The LLM might provide either:
        - A full state ID (MD5 hash)
        - A beacon string ("Search Page of Google Maps")
        - A partial ID
        
        Args:
            info_pool: InfoPool with state graphs
            target_input: The target state identifier from LLM
            
        Returns:
            Valid state_id or None if not found
        """
        target_input = target_input.strip()
        app_name = info_pool.active_app_name
        
        if app_name not in info_pool.state_graphs:
            return None
        
        graph = info_pool.state_graphs[app_name]
        
        # Case 1: Direct state ID match
        if target_input in graph.nodes:
            return target_input
        
        # Case 2: Partial ID match (first N characters)
        for state_id in graph.nodes:
            if state_id.startswith(target_input) or target_input.startswith(state_id[:12]):
                return state_id
        
        # Case 3: Beacon string match
        beacon_mapping = self.beacon_to_id_mapping(info_pool, app_name)
        
        # Exact beacon match
        if target_input in beacon_mapping:
            return beacon_mapping[target_input]
        
        # Normalized beacon match
        normalized_target = self._normalize_beacon(target_input)
        if normalized_target in beacon_mapping:
            return beacon_mapping[normalized_target]
        
        # Generate ID from beacon and check
        generated_id = self._generate_state_id(target_input)
        if generated_id in graph.nodes:
            return generated_id
        
        # Case 4: Fuzzy beacon match (contains)
        target_lower = target_input.lower()
        for beacon, state_id in beacon_mapping.items():
            if target_lower in beacon.lower() or beacon.lower() in target_lower:
                print(f"[Recovery] Fuzzy matched '{target_input}' to beacon '{beacon}'")
                return state_id
        
        return None
    
    def verify_recovery_reached(self, info_pool: InfoPool, target_state_id: str, current_beacon: str) -> bool:
        """
        Verify if we've reached the target recovery state.
        
        Args:
            info_pool: InfoPool with state graphs
            target_state_id: The target state we're trying to reach
            current_beacon: The beacon of the current screen
            
        Returns:
            True if current state matches target
        """
        app_name = info_pool.active_app_name
        if app_name not in info_pool.state_graphs:
            return False
        
        graph = info_pool.state_graphs[app_name]
        
        # Get target beacon
        if target_state_id not in graph.nodes:
            return False
        
        target_beacon = graph.nodes[target_state_id].get('beacon', '')
        
        # Compare beacons (normalized)
        current_normalized = self._normalize_beacon(current_beacon)
        target_normalized = self._normalize_beacon(target_beacon)
        
        if current_normalized == target_normalized:
            return True
        
        # Also check if current state ID matches
        current_state_id = self._generate_state_id(current_beacon)
        return current_state_id == target_state_id
    
    def get_transitions_from_state(self, info_pool: InfoPool, app_name: str, state_id: str) -> List[Tuple[str, Dict]]:
        """
        Get all possible transitions (outgoing edges) from a given state.
        
        Args:
            info_pool: Central information repository
            app_name: Name of the app
            state_id: Source state ID
            
        Returns:
            List of (target_state_id, edge_attributes) tuples
        """
        if app_name not in info_pool.state_graphs:
            return []
        
        graph = info_pool.state_graphs[app_name]
        if state_id not in graph.nodes:
            return []
        
        return [(target, dict(graph.edges[state_id, target])) 
                for target in graph.successors(state_id)]
    
    def find_path_to_state(self, info_pool: InfoPool, app_name: str, 
                           source_id: str, target_id: str) -> Optional[List[str]]:
        """
        Find the shortest path between two states in the FSM.
        
        Useful for recovery planning - finding how to navigate back to a known state.
        
        Args:
            info_pool: Central information repository
            app_name: Name of the app
            source_id: Starting state ID
            target_id: Destination state ID
            
        Returns:
            List of state IDs forming the path, or None if no path exists
        """
        if app_name not in info_pool.state_graphs:
            return None
        
        graph = info_pool.state_graphs[app_name]
        try:
            return nx.shortest_path(graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
    
    def find_similar_beacon(self, beacon: str, info_pool: InfoPool) -> Optional[str]:
        """
        Find a similar beacon in history to avoid state explosion.
        
        Uses simple heuristics to match near-duplicate beacons:
        - Same app name
        - Similar state type (Homepage, Search, Detail, etc.)
        
        Args:
            beacon: The new beacon to check
            info_pool: InfoPool containing beacon_hist
            
        Returns:
            Matching existing beacon or None
        """
        normalized = self._normalize_beacon(beacon)
        
        # First check exact match after normalization
        if normalized in info_pool.beacon_hist:
            return normalized
        
        # Extract app name and state type for fuzzy matching
        app_name = None
        state_type = None
        
        for separator in [" of ", " in "]:
            if separator in normalized:
                parts = normalized.split(separator)
                state_type = parts[0].lower()
                app_name = parts[1]
                break
        
        if not app_name:
            return None
        
        # Look for similar beacons in history
        for existing in info_pool.beacon_hist:
            existing_lower = existing.lower()
            if app_name.lower() in existing_lower:
                # Same app - check state type similarity
                if state_type and state_type in existing_lower:
                    print(f"[FSM] Found similar beacon: '{beacon}' ≈ '{existing}'")
                    return existing
        
        return None
    
    def get_fsm_summary(self, info_pool: InfoPool) -> Dict:
        """
        Get a summary of all FSM graphs for debugging/logging.
        
        Returns:
            Dict with summary statistics for each app
        """
        summary = {
            "total_apps": len(info_pool.state_graphs),
            "total_states": 0,
            "total_transitions": 0,
            "apps": {}
        }
        
        for app_name, graph in info_pool.state_graphs.items():
            app_summary = {
                "states": graph.number_of_nodes(),
                "transitions": graph.number_of_edges(),
                "beacons": [],
                "most_visited": None
            }
            
            # Get beacons and find most visited
            max_visits = 0
            for node in graph.nodes:
                beacon = graph.nodes[node].get('beacon', 'Unknown')
                visit_count = graph.nodes[node].get('visit_count', 1)
                app_summary["beacons"].append(beacon)
                if visit_count > max_visits:
                    max_visits = visit_count
                    app_summary["most_visited"] = (beacon, visit_count)
            
            summary["apps"][app_name] = app_summary
            summary["total_states"] += app_summary["states"]
            summary["total_transitions"] += app_summary["transitions"]
        
        # Add cross-app transitions
        summary["cross_app_transitions"] = len(info_pool.app_transition_history)
        
        return summary
    
    def print_fsm_summary(self, info_pool: InfoPool) -> None:
        """Print a human-readable FSM summary."""
        summary = self.get_fsm_summary(info_pool)
        
        print("\n" + "="*60)
        print("FSM STATE GRAPH SUMMARY")
        print("="*60)
        print(f"Total Apps: {summary['total_apps']}")
        print(f"Total States: {summary['total_states']}")
        print(f"Total Transitions: {summary['total_transitions']}")
        print(f"Cross-App Transitions: {summary['cross_app_transitions']}")
        
        for app_name, app_info in summary["apps"].items():
            print(f"\n[{app_name}]")
            print(f"  States: {app_info['states']}, Transitions: {app_info['transitions']}")
            print(f"  Beacons: {', '.join(app_info['beacons'][:5])}")
            if app_info['most_visited']:
                print(f"  Most visited: {app_info['most_visited'][0]} ({app_info['most_visited'][1]}x)")
        
        print("="*60 + "\n")

    def get_prompt_for_recovery(self, info_pool: InfoPool) -> str:
        prompt = "### Recovery Mode Activated ###\n"
        prompt += "The agent has encountered an error during execution. Your goal is to identify a previously visited state that would allow the agent to recover and resume meaningful execution. You do NOT need to resume the original task after recovery — your goal is only to return to the selected state.\n\n"

        # Error message
        latest_error = info_pool.error_descriptions[-1] if info_pool.error_descriptions else "No error message"
        prompt += "### Latest Error Message ###\n"
        prompt += f"{latest_error}\n\n"

        # Last action
        prompt += "### Last Action ###\n"
        prompt += f"{info_pool.last_action}\n\n"

        # Recent action history
        prompt += "### Recent Action History ###\n"
        k = 3
        for i in range(1, k + 1):
            if len(info_pool.action_history) >= i:
                action = info_pool.action_history[-i]
                outcome = info_pool.action_outcomes[-i]
                summary = info_pool.summary_history[-i] if len(info_pool.summary_history) >= i else "No summary"
                prompt += f"- {action} | Outcome: {outcome} | {summary}\n"
        prompt += "\n"

        # Visited states (current app) - show both beacon AND full ID for LLM
        app_name = info_pool.active_app_name
        graph = info_pool.state_graphs.get(app_name, nx.DiGraph())
        visited = info_pool.state_id_lists.get(app_name, [])
        
        prompt += f"### Visited States in {app_name} ###\n"
        prompt += "**IMPORTANT**: You can specify the target state using EITHER the BEACON or STATE_ID below.\n"
        prompt += "Using the BEACON (e.g., 'Search Page of Google Maps') is recommended for readability.\n\n"
        
        valid_states_shown = 0
        for idx, sid in enumerate(visited):
            if sid in graph.nodes:
                bcn = graph.nodes[sid].get("beacon", "")
                desc = graph.nodes[sid].get("description", "")[:80]
                visits = graph.nodes[sid].get("visit_count", 1)
                prompt += f"[{idx + 1}] BEACON: \"{bcn}\"\n"
                prompt += f"    STATE_ID: {sid}\n"
                prompt += f"    Description: {desc} | Visits: {visits}\n\n"
                valid_states_shown += 1
        if valid_states_shown == 0:
            prompt += "No valid visited states found.\n"
        prompt += "\n"
        
        # Also show states from other apps (for cross-app recovery)
        other_apps = [a for a in info_pool.state_graphs.keys() if a != app_name]
        if other_apps:
            prompt += "### States in Other Apps (for reference) ###\n"
            for other_app in other_apps[:2]:  # Limit to 2 other apps
                other_graph = info_pool.state_graphs[other_app]
                prompt += f"**{other_app}**: "
                beacons = [other_graph.nodes[n].get('beacon', '?') for n in list(other_graph.nodes)[:3]]
                prompt += ", ".join(beacons) + "\n"
            prompt += "\n"

        # Current screen info
        prompt += "### Current Active APP ###\n"
        prompt += info_pool.active_app_name
        prompt += "\n### Current Screen Elements ###\n"
        for item in info_pool.state_perception_infos:
            if item["text"] and item["coordinates"] != (0, 0) and item["text"] != "icon: None":
                prompt += f"{item['coordinates']}; {item['text']}\n"
        prompt += "\n"

        # Output instructions
        prompt += "---\n"
        prompt += "**Instructions:**\n"
        prompt += "1. Examine the visited states and select a good recovery target.\n"
        prompt += "2. You can specify the target using EITHER the BEACON (recommended) or STATE_ID.\n"
        prompt += "3. Generate a short high-level plan to reach that state.\n"
        prompt += "4. Do NOT include mental/observational steps like 'verify', 'wait', 'check'.\n"
        prompt += "5. Only include executable actions: 'tap on X', 'type Y', 'swipe up', 'press back'.\n\n"
        
        prompt += "### Your Output Format ###\n"
        prompt += "Respond using the format below:\n\n"
        prompt += "### Recovery Goal ###\n"
        prompt += "A short instruction about the recovery goal\n\n"
        prompt += "### Recovery Target ###\n"
        prompt += "Provide EITHER the beacon string OR state_id from the visited states list above.\n"
        prompt += "Example: \"Search Page of Google Maps\" or \"a1b2c3d4e5f6...\"\n\n"
        prompt += "### Recovery Plan ###\n"
        prompt += "```json\n"
        prompt += "{\n"
        prompt += '  "target_beacon": "The beacon of the target state (copy exactly from list)",\n'
        prompt += '  "target_state_id": "The state_id (copy exactly from list)",\n'
        prompt += '  "thought": "Explain why this state was selected and how to reach it.",\n'
        prompt += '  "plan": ["Step 1: Press back button", "Step 2: Tap on search bar", "..."],\n'
        prompt += '  "current_subgoal": "Step 1: Press back button"\n'
        prompt += "}\n"
        prompt += "```\n"

        return prompt

    def parse_recovery_response(self, response: str, info_pool: InfoPool = None) -> dict:
        """
        Parse the recovery response from LLM.
        
        Handles both old format (### Recovery Target State ID ###) and 
        new format (### Recovery Target ### with beacon/state_id in JSON).
        """
        try:
            # Extract Recovery Goal (handle both "### Recovery Goal###" and "### Recovery Goal ###")
            goal_section = ""
            if "### Recovery Goal" in response:
                goal_parts = response.split("### Recovery Goal")
                if len(goal_parts) > 1:
                    goal_section = goal_parts[-1].split("###")[0].replace("#", "").strip()

            # Extract and parse the JSON plan block
            json_block = re.search(r"```json(.*?)```", response, re.DOTALL)
            if not json_block:
                # Try without code fence
                json_block = re.search(r'\{[^{}]*"plan"[^{}]*\}', response, re.DOTALL)
                if json_block:
                    recovery_plan_json = json.loads(json_block.group(0))
                else:
                    raise ValueError("Could not find recovery plan JSON block")
            else:
                recovery_plan_json = json.loads(json_block.group(1).strip())

            # Get target from JSON (preferred) or from old format section
            target_beacon = recovery_plan_json.get("target_beacon", "")
            target_state_id = recovery_plan_json.get("target_state_id", "")
            
            # Fallback: try old format extraction
            if not target_state_id and not target_beacon:
                if "### Recovery Target State ID ###" in response:
                    target_state_id = response.split("### Recovery Target State ID ###")[-1].split("###")[0].strip()
                elif "### Recovery Target ###" in response:
                    target_raw = response.split("### Recovery Target ###")[-1].split("###")[0].strip()
                    # Could be beacon or state_id
                    if len(target_raw) == 32 and all(c in '0123456789abcdef' for c in target_raw):
                        target_state_id = target_raw
                    else:
                        target_beacon = target_raw

            # Resolve target to state_id if we have info_pool
            resolved_state_id = target_state_id
            if info_pool and (target_beacon or target_state_id):
                # Try to resolve using our helper
                target_input = target_beacon if target_beacon else target_state_id
                resolved = self.resolve_target_state(info_pool, target_input)
                if resolved:
                    resolved_state_id = resolved
                    print(f"[Recovery] Resolved target '{target_input[:30]}...' to state_id: {resolved[:12]}...")
                else:
                    print(f"[Recovery] Warning: Could not resolve target '{target_input[:30]}...'")
                    # Keep original if resolution failed
                    resolved_state_id = target_state_id if target_state_id else self._generate_state_id(target_beacon)

            return {
                "recovery_goal": goal_section,
                "target_state_id": resolved_state_id,
                "target_beacon": target_beacon,
                "thought": recovery_plan_json.get("thought", ""),
                "plan": recovery_plan_json.get("plan", []),
                "current_subgoal": recovery_plan_json.get("current_subgoal", "")
            }

        except Exception as e:
            print(f"[parse_recovery_response] Failed to parse recovery response: {e}")
            import traceback
            traceback.print_exc()
            return {
                "recovery_goal": "",
                "target_state_id": "",
                "target_beacon": "",
                "thought": "",
                "plan": [],
                "current_subgoal": ""
            }

    def get_prompt_revise_recovery_plan(self, info_pool: InfoPool) -> str:
        prompt = "### Recovery Plan Revision Needed ###\n"
        prompt += "The current recovery plan failed. Your goal is to decide whether the previous recovery target state is still suitable. If not, pick a better one.\n"
        prompt += "Then, generate a new high-level plan (subgoals only) to reach the selected state. You do NOT need to resume the main task afterward.\n\n"

        # Current subgoal
        prompt += "### Current Subgoal ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        # Error
        latest_error = info_pool.error_descriptions[-1] if info_pool.error_descriptions else "No error available"
        prompt += "### Latest Error Message ###\n"
        prompt += f"{latest_error}\n\n"

        # Previous target state
        previous_target = info_pool.recovery_target_state_id
        prompt += "### Previous Recovery Target ###\n"
        prompt += f"{previous_target}\n\n"

        # State graph
        app = info_pool.active_app_name
        graph = info_pool.state_graphs.get(app, nx.DiGraph())
        visited = info_pool.state_id_lists.get(app, [])

        prompt += f"### Visited States in {app} ###\n"
        prompt += "**You can specify the target using EITHER the BEACON or STATE_ID.**\n\n"
        for idx, sid in enumerate(visited):
            if sid in graph.nodes:
                bcn = graph.nodes[sid].get("beacon", "")
                desc = graph.nodes[sid].get("description", "")[:60]
                is_prev_target = " ← (previous target)" if sid == previous_target else ""
                prompt += f"[{idx + 1}] BEACON: \"{bcn}\"{is_prev_target}\n"
                prompt += f"    STATE_ID: {sid}\n"
                prompt += f"    Description: {desc}\n\n"
        prompt += "\n"

        prompt += "### State Transitions ###\n"
        for u, v, data in graph.edges(data=True):
            action = data.get("action", "")
            prompt += f"{u[:6]} --[{action}]--> {v[:6]}\n"
        prompt += "\n"

        # Perception
        prompt += "### Current Active APP ###\n"
        prompt += info_pool.active_app_name
        prompt += "\n### Current Screen Elements ###\n"
        for elem in info_pool.state_perception_infos:
            if elem["text"] and elem["coordinates"] != (0, 0) and elem["text"] != "icon: None":
                prompt += f"{elem['coordinates']}; {elem['text']}\n"
        prompt += "\n"

        # Notes
        if info_pool.important_notes:
            prompt += "### Important Notes ###\n"
            prompt += f"{info_pool.important_notes}\n\n"

        # Tips
        if info_pool.tips:
            prompt += "### Tips ###\n"
            prompt += f"{info_pool.tips}\n\n"

        prompt += "---\n"
        prompt += "🧠 **Recovery Guidelines:**\n"
        prompt += "- If the original target is still good, reuse it. Otherwise, choose another.\n"
        prompt += "- Your goal is to return to that state only — not to resume the task.\n"
        prompt += "- The recovery plan should be a list of high-level subgoals.\n"
        prompt += "- Only include executable actions: 'tap on X', 'type Y', 'swipe up', 'press back'.\n\n"

        prompt += "### Output Format ###\n"
        prompt += "### Recovery Goal ###\n"
        prompt += "A short instruction about the recovery goal\n\n"
        prompt += "### Recovery Target ###\n"
        prompt += "Provide EITHER the beacon string OR state_id.\n\n"
        prompt += "### Recovery Plan ###\n"
        prompt += "```json\n"
        prompt += "{\n"
        prompt += '  "target_beacon": "The beacon of the target state",\n'
        prompt += '  "target_state_id": "The state_id",\n'
        prompt += '  "thought": "Rationale for choosing the target and overall strategy",\n'
        prompt += '  "plan": ["Step 1: Press back button", "Step 2: Tap on X"],\n'
        prompt += '  "current_subgoal": "Step 1: Press back button"\n'
        prompt += "}\n"
        prompt += "```\n"

        return prompt