"""
Mentor Agent for Agent-SAMA: Self-Evolution Module

The Mentor Agent is responsible for the self-evolution mechanism described in the paper.
After completing a task, it reflects on the interaction history to distill:
1. **Shortcuts**: Reusable action sequences with preconditions
2. **Tips**: General advice for preventing common errors

Key features:
- Analyzes successful action patterns to create shortcuts
- Learns from failures and recovery experiences to generate tips
- Leverages FSM state graphs to understand navigation patterns
- Considers future tasks to prioritize relevant knowledge

Reference: Agent-SAMA paper (AAAI 2026) - Section on Self-Evolution
"""

from Agents.base_agent import BaseAgent
from Agents.info_pool import InfoPool
from AgentsUtils.common_utils import extract_json_object
from constant import ATOMIC_ACTION_SIGNATURES, SHORTCUT_EXAMPLE
from typing import Optional, List, Dict
import json


class Mentor(BaseAgent):
    """
    Mentor Agent for self-evolution in Agent-SAMA.
    
    The Mentor reflects on completed tasks to extract reusable knowledge:
    - Shortcuts: Sequences of atomic actions that accomplish common subgoals
    - Tips: General advice learned from errors and recovery experiences
    
    This agent is invoked at the end of each task to enable cross-task learning,
    as described in the Agent-SAMA paper's self-evolution mechanism.
    """
    
    def __init__(self, mode: str = "unified"):
        """
        Initialize the Mentor Agent.
        
        Args:
            mode: Operation mode
                - "unified": Generate both shortcuts and tips in one call (recommended)
                - "shortcut": Focus only on shortcut generation
                - "tips": Focus only on tips generation
        """
        self.mode = mode

    def init_chat(self) -> list:
        """Initialize chat with system prompt for self-evolution reflection."""
        operation_history = []
        
        if self.mode == "unified":
            system_prompt = (
                "You are the Mentor Agent in a mobile automation framework. "
                "Your role is to reflect on completed task experiences and extract reusable knowledge "
                "to improve future task executions. You analyze action patterns, errors, and recovery "
                "experiences to generate: (1) Shortcuts - reusable action sequences, and "
                "(2) Tips - general advice for preventing common errors."
            )
        elif self.mode == "shortcut":
            system_prompt = (
                "You are a Mentor Agent specializing in identifying reusable action patterns. "
                "Analyze the task execution history to identify sequences of successful actions "
                "that can be consolidated into efficient shortcuts for future tasks."
            )
        else:  # tips mode
            system_prompt = (
                "You are a Mentor Agent specializing in learning from mistakes. "
                "Analyze the task execution history, especially failures and recovery experiences, "
                "to generate helpful tips that prevent common errors in future tasks."
            )
        
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        """
        Generate the reflection prompt based on task execution history.
        
        The prompt includes:
        - Task context and completion status
        - Full action history with outcomes
        - Error and recovery experiences
        - FSM navigation patterns (when available)
        - Future tasks for prioritization
        """
        prompt = self._build_task_context(info_pool)
        prompt += self._build_action_history(info_pool)
        prompt += self._build_error_analysis(info_pool)
        prompt += self._build_fsm_patterns(info_pool)
        prompt += self._build_future_context(info_pool)
        prompt += self._build_existing_knowledge(info_pool)
        prompt += self._build_output_instructions()
        
        return prompt
    
    def _build_task_context(self, info_pool: InfoPool) -> str:
        """Build the task context section."""
        prompt = "### Completed Task ###\n"
        prompt += f"Instruction: {info_pool.instruction}\n\n"
        
        prompt += "### Execution Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"
        
        prompt += "### Final Status ###\n"
        prompt += f"{info_pool.progress_status}\n\n"
        
        return prompt
    
    def _build_action_history(self, info_pool: InfoPool) -> str:
        """Build the action history section with success/failure analysis."""
        prompt = "### Full Action History ###\n"
        
        if not info_pool.action_history:
            prompt += "No actions were executed.\n\n"
            return prompt
        
        # Group actions into successful sequences and failures
        successful_sequences = []
        current_sequence = []
        
        for i, (action, summary, outcome, error, progress) in enumerate(zip(
            info_pool.action_history,
            info_pool.summary_history,
            info_pool.action_outcomes,
            info_pool.error_descriptions,
            info_pool.progress_status_history
        )):
            action_entry = {
                "step": i + 1,
                "action": action,
                "description": summary,
                "outcome": outcome,
                "error": error if outcome != "A" else None,
                "progress": progress
            }
            
            if outcome == "A":
                current_sequence.append(action_entry)
                prompt += f"✓ Step {i+1}: {action} | {summary} | Progress: {progress}\n"
            else:
                if current_sequence:
                    successful_sequences.append(current_sequence.copy())
                    current_sequence = []
                prompt += f"✗ Step {i+1}: {action} | {summary} | FAILED: {error}\n"
        
        # Don't forget the last sequence
        if current_sequence:
            successful_sequences.append(current_sequence)
        
        prompt += "\n"
        
        # Highlight successful sequences (potential shortcuts)
        if successful_sequences:
            prompt += "### Successful Action Sequences (Potential Shortcuts) ###\n"
            for seq_idx, seq in enumerate(successful_sequences):
                if len(seq) >= 2:  # Only sequences with 2+ actions are shortcut candidates
                    prompt += f"Sequence {seq_idx + 1} ({len(seq)} actions):\n"
                    for entry in seq:
                        action_name = entry['action'].get('name', 'Unknown') if isinstance(entry['action'], dict) else str(entry['action'])
                        prompt += f"  - {action_name}: {entry['description']}\n"
                    prompt += "\n"
        
        return prompt
    
    def _build_error_analysis(self, info_pool: InfoPool) -> str:
        """Build the error and recovery analysis section."""
        prompt = "### Error & Recovery Analysis ###\n"
        
        # Count failures
        failures = [(i, outcome, err) for i, (outcome, err) in 
                   enumerate(zip(info_pool.action_outcomes, info_pool.error_descriptions)) 
                   if outcome in ["B", "C"]]
        
        if not failures:
            prompt += "No errors occurred during this task.\n\n"
        else:
            prompt += f"Total failures: {len(failures)}\n"
            for step, outcome, error in failures:
                failure_type = "Wrong page" if outcome == "B" else "No change"
                prompt += f"- Step {step + 1}: {failure_type} - {error}\n"
            prompt += "\n"
        
        # Recovery statistics
        if info_pool.recovery_attempts > 0:
            prompt += "### Recovery Experience ###\n"
            prompt += f"Recovery attempts: {info_pool.recovery_attempts}\n"
            prompt += f"Successful recoveries: {info_pool.recovery_successes}\n"
            prompt += f"Failed recoveries: {info_pool.recovery_failures}\n"
            if info_pool.recovery_attempts > 0:
                rate = 100 * info_pool.recovery_successes / info_pool.recovery_attempts
                prompt += f"Recovery success rate: {rate:.1f}%\n"
            prompt += "\n"
            
            # Include recovery plans that worked
            if info_pool.recovery_plan:
                prompt += "Recovery strategies used:\n"
                for target_state, plan in info_pool.recovery_plan.items():
                    if isinstance(plan, dict) and 'thought' in plan:
                        prompt += f"- Target: {target_state[:8]}... | Strategy: {plan.get('thought', 'N/A')[:100]}\n"
                prompt += "\n"
        
        return prompt
    
    def _build_fsm_patterns(self, info_pool: InfoPool) -> str:
        """Build FSM navigation patterns section."""
        prompt = "### Navigation Patterns (FSM) ###\n"
        
        if not info_pool.state_graphs:
            prompt += "No state graphs available.\n\n"
            return prompt
        
        for app_name, graph in info_pool.state_graphs.items():
            prompt += f"App: {app_name}\n"
            prompt += f"  States visited: {graph.number_of_nodes()}\n"
            prompt += f"  Transitions recorded: {graph.number_of_edges()}\n"
            
            # Show frequently visited states
            visit_counts = [(node, graph.nodes[node].get('visit_count', 1)) 
                          for node in graph.nodes]
            visit_counts.sort(key=lambda x: x[1], reverse=True)
            
            if visit_counts:
                prompt += "  Frequently visited states:\n"
                for node_id, count in visit_counts[:3]:  # Top 3
                    beacon = graph.nodes[node_id].get('beacon', 'Unknown')
                    prompt += f"    - {beacon} (visited {count}x)\n"
            prompt += "\n"
        
        # Cross-app transitions
        if info_pool.app_transition_history:
            prompt += "Cross-app transitions:\n"
            for transition in info_pool.app_transition_history:
                prompt += f"  {transition.get('from', '?')} → {transition.get('to', '?')}\n"
            prompt += "\n"
        
        return prompt
    
    def _build_future_context(self, info_pool: InfoPool) -> str:
        """Build future tasks context for prioritization."""
        if not info_pool.future_tasks:
            return ""
        
        prompt = "### Upcoming Tasks (for prioritization) ###\n"
        prompt += "Consider these future tasks when deciding what knowledge to extract:\n"
        for task in info_pool.future_tasks[:5]:  # Show up to 5
            prompt += f"- {task}\n"
        prompt += "\n"
        
        return prompt
    
    def _build_existing_knowledge(self, info_pool: InfoPool) -> str:
        """Build existing shortcuts and tips section."""
        prompt = "### Existing Knowledge Base ###\n"
        
        # Existing shortcuts
        prompt += "#### Current Shortcuts ####\n"
        if info_pool.shortcuts:
            for name, shortcut in info_pool.shortcuts.items():
                prompt += f"- {name}: {shortcut.get('description', 'No description')}\n"
                prompt += f"  Precondition: {shortcut.get('precondition', 'None')}\n"
        else:
            prompt += "No shortcuts defined yet.\n"
        prompt += "\n"
        
        # Existing tips
        prompt += "#### Current Tips ####\n"
        if info_pool.tips:
            prompt += f"{info_pool.tips}\n"
        else:
            prompt += "No tips recorded yet.\n"
        prompt += "\n"
        
        # Atomic actions reference (for shortcut creation)
        if self.mode in ["unified", "shortcut"]:
            prompt += "#### Available Atomic Actions ####\n"
            for action, value in ATOMIC_ACTION_SIGNATURES.items():
                args = ', '.join(value['arguments']) if value['arguments'] else 'none'
                prompt += f"- {action}({args})\n"
            prompt += "\n"
        
        return prompt
    
    def _build_output_instructions(self) -> str:
        """Build output format instructions based on mode."""
        prompt = "---\n"
        prompt += "## Self-Evolution Reflection Task ##\n\n"
        
        if self.mode == "unified":
            prompt += self._get_unified_instructions()
        elif self.mode == "shortcut":
            prompt += self._get_shortcut_instructions()
        else:
            prompt += self._get_tips_instructions()
        
        return prompt
    
    def _get_unified_instructions(self) -> str:
        """Instructions for unified mode (both shortcuts and tips)."""
        prompt = (
            "Based on the task execution above, reflect on what can be learned to improve future tasks.\n\n"
            "Analyze the experience and provide:\n"
            "1. **New Shortcut**: If you identified a reusable action sequence (2+ atomic actions that "
            "frequently occur together), define it as a shortcut.\n"
            "2. **Updated Tips**: Add any new tips learned from this experience, especially from errors "
            "and recovery attempts.\n\n"
        )
        
        prompt += "### Output Format ###\n\n"
        
        prompt += "### New Shortcut ###\n"
        prompt += (
            "Provide a shortcut in JSON format if applicable, or 'None' if no useful shortcut was identified.\n"
            "Requirements:\n"
            "- Only use atomic actions listed above\n"
            "- Include clear precondition (when can this shortcut be used?)\n"
            "- Avoid shortcuts that duplicate existing ones\n"
            "- Prefer shortcuts with few arguments (coordinates are blind)\n\n"
        )
        prompt += f"Example format:\n{SHORTCUT_EXAMPLE}\n\n"
        
        prompt += "### Updated Tips ###\n"
        prompt += (
            "Provide the complete updated tips list (existing + new). Format as numbered list.\n"
            "Guidelines:\n"
            "- Keep tips concise and actionable\n"
            "- Focus on error prevention\n"
            "- Include insights from recovery experiences\n"
            "- Remove outdated or redundant tips\n"
        )
        
        return prompt
    
    def _get_shortcut_instructions(self) -> str:
        """Instructions for shortcut-only mode."""
        prompt = (
            "Analyze the successful action sequences above and determine if any can be "
            "consolidated into a reusable shortcut.\n\n"
            "A good shortcut:\n"
            "- Combines 2-4 atomic actions that frequently occur together\n"
            "- Has a clear precondition (screen state where it applies)\n"
            "- Is general enough to be useful across multiple tasks\n"
            "- Has minimal coordinate-based arguments\n\n"
        )
        
        prompt += "### New Shortcut ###\n"
        prompt += "Provide a shortcut JSON or 'None':\n"
        prompt += f"{SHORTCUT_EXAMPLE}\n"
        
        return prompt
    
    def _get_tips_instructions(self) -> str:
        """Instructions for tips-only mode."""
        prompt = (
            "Analyze the errors and recovery experiences above to extract useful tips "
            "for preventing similar issues in future tasks.\n\n"
            "A good tip:\n"
            "- Is concise and actionable\n"
            "- Addresses a specific error pattern observed\n"
            "- Generalizes beyond this specific task\n"
            "- Helps prevent errors rather than just fixing them\n\n"
        )
        
        prompt += "### Updated Tips ###\n"
        prompt += "Provide the complete updated tips list (existing + new) as a numbered list:\n"
        
        return prompt

    def parse_response(self, response: str) -> dict:
        """
        Parse the Mentor's response based on the mode.
        
        Returns:
            dict with keys based on mode:
            - unified: {"new_shortcut": str, "updated_tips": str}
            - shortcut: {"new_shortcut": str}
            - tips: {"updated_tips": str}
        """
        result = {}
        
        if self.mode in ["unified", "shortcut"]:
            shortcut_section = response.split("### New Shortcut ###")[-1]
            if "### Updated Tips ###" in shortcut_section:
                shortcut_section = shortcut_section.split("### Updated Tips ###")[0]
            result["new_shortcut"] = shortcut_section.strip()
        
        if self.mode in ["unified", "tips"]:
            if "### Updated Tips ###" in response:
                tips_section = response.split("### Updated Tips ###")[-1].strip()
                result["updated_tips"] = tips_section
            else:
                result["updated_tips"] = ""
        
        return result

    def add_new_shortcut(self, shortcut_str: str, info_pool: InfoPool) -> bool:
        """
        Add a new shortcut to the knowledge base.
        
        Args:
            shortcut_str: JSON string defining the shortcut
            info_pool: InfoPool to update
            
        Returns:
            bool: True if shortcut was added successfully
        """
        if shortcut_str is None or shortcut_str.strip().lower() == "none":
            return False
        
        shortcut_obj = extract_json_object(shortcut_str)
        if shortcut_obj is None:
            print(f"[Mentor] Failed to parse shortcut JSON: {shortcut_str[:100]}...")
            return False
        
        # Validate shortcut structure
        required_fields = ["name", "arguments", "description", "precondition", "atomic_action_sequence"]
        for field in required_fields:
            if field not in shortcut_obj:
                print(f"[Mentor] Shortcut missing required field: {field}")
                return False
        
        shortcut_name = shortcut_obj["name"]
        
        # Check for duplicates
        if shortcut_name in info_pool.shortcuts:
            print(f"[Mentor] Shortcut already exists: {shortcut_name}")
            return False
        
        # Validate atomic actions
        for action in shortcut_obj.get("atomic_action_sequence", []):
            if action.get("name") not in ATOMIC_ACTION_SIGNATURES:
                print(f"[Mentor] Invalid atomic action in shortcut: {action.get('name')}")
                return False
        
        # Validate that all dynamic placeholders in arguments_map are defined in arguments
        shortcut_arguments = set(shortcut_obj.get("arguments", []))
        for action in shortcut_obj.get("atomic_action_sequence", []):
            arguments_map = action.get("arguments_map", {})
            if arguments_map:
                for arg_key, mapped_value in arguments_map.items():
                    # Check if mapped_value is a dynamic placeholder (string that should be in arguments)
                    if isinstance(mapped_value, str):
                        # Try to parse as number - if it's a numeric string, it's a hardcoded value
                        try:
                            float(mapped_value)
                            # It's a numeric string, that's fine (hardcoded coordinate)
                        except ValueError:
                            # It's a non-numeric string, must be a placeholder
                            if mapped_value not in shortcut_arguments:
                                print(f"[Mentor] Shortcut '{shortcut_name}' has unresolvable placeholder '{mapped_value}' "
                                      f"in arguments_map for '{arg_key}'. It must be defined in 'arguments': {list(shortcut_arguments)}")
                                print(f"[Mentor] Rejecting shortcut to prevent runtime errors.")
                                return False
        
        # Add shortcut
        info_pool.shortcuts[shortcut_name] = shortcut_obj
        print(f"[Mentor] Added new shortcut: {shortcut_name}")
        print(f"[Mentor] Description: {shortcut_obj.get('description', 'N/A')}")
        
        return True
    
    def update_tips(self, new_tips: str, info_pool: InfoPool) -> None:
        """
        Update the tips in the knowledge base.
        
        Args:
            new_tips: New tips string (should include existing + new)
            info_pool: InfoPool to update
        """
        if new_tips and new_tips.strip():
            info_pool.tips = new_tips.strip()
            print(f"[Mentor] Updated tips ({len(new_tips)} chars)")
        else:
            print("[Mentor] No tips update provided")
    
    def reflect(self, info_pool: InfoPool, llm_response: str) -> Dict:
        """
        Process the LLM response and update the knowledge base.
        
        This is a convenience method that combines parsing and updating.
        Also extracts FSM navigation patterns for cross-task learning.
        
        Args:
            info_pool: InfoPool to update
            llm_response: Raw LLM response
            
        Returns:
            dict with results of the reflection
        """
        parsed = self.parse_response(llm_response)
        results = {"shortcuts_added": 0, "tips_updated": False, "patterns_extracted": 0}
        
        # Handle shortcut
        if "new_shortcut" in parsed:
            if self.add_new_shortcut(parsed["new_shortcut"], info_pool):
                results["shortcuts_added"] = 1
        
        # Handle tips
        if "updated_tips" in parsed:
            self.update_tips(parsed["updated_tips"], info_pool)
            results["tips_updated"] = True
        
        # Extract FSM navigation patterns (automatic, no LLM needed)
        patterns = self.extract_fsm_patterns(info_pool)
        results["patterns_extracted"] = len(patterns)
        
        # Extract common flows across apps
        common_flows = self.extract_common_flows(info_pool)
        results["common_flows_updated"] = len(common_flows)
        
        return results
    
    def extract_fsm_patterns(self, info_pool: InfoPool) -> List[Dict]:
        """
        Extract reusable navigation patterns from FSM state graphs.
        
        This method analyzes the state graphs built during task execution
        to identify patterns that can help future planning.
        
        Patterns include:
        - Frequently used state sequences
        - Successful navigation paths
        - App-specific flow patterns (search, checkout, etc.)
        
        Args:
            info_pool: InfoPool containing state_graphs
            
        Returns:
            List of extracted pattern dictionaries
        """
        extracted_patterns = []
        
        for app_name, graph in info_pool.state_graphs.items():
            if graph.number_of_nodes() < 2:
                continue  # Need at least 2 states for a pattern
            
            # Pattern 1: Extract the main navigation path (most visited sequence)
            main_path = self._extract_main_path(graph)
            if main_path and len(main_path) >= 2:
                pattern = {
                    "app": app_name,
                    "pattern_name": f"{app_name}_main_flow",
                    "states": main_path,
                    "description": f"Common navigation path in {app_name}: {' → '.join(main_path[:4])}{'...' if len(main_path) > 4 else ''}",
                    "type": "main_flow"
                }
                extracted_patterns.append(pattern)
                
                # Add to learned patterns if not duplicate
                if not self._is_duplicate_pattern(pattern, info_pool.learned_navigation_patterns):
                    info_pool.learned_navigation_patterns.append(pattern)
                    print(f"[Mentor] Extracted pattern: {pattern['pattern_name']}")
            
            # Pattern 2: Identify search flow if present
            search_pattern = self._extract_search_pattern(graph, app_name)
            if search_pattern:
                if not self._is_duplicate_pattern(search_pattern, info_pool.learned_navigation_patterns):
                    info_pool.learned_navigation_patterns.append(search_pattern)
                    extracted_patterns.append(search_pattern)
                    print(f"[Mentor] Extracted search pattern in {app_name}")
        
        return extracted_patterns
    
    def _extract_main_path(self, graph) -> List[str]:
        """Extract the most frequently traversed path through the graph."""
        if graph.number_of_nodes() == 0:
            return []
        
        # Find the most visited states
        visit_counts = [(node, graph.nodes[node].get('visit_count', 1)) 
                       for node in graph.nodes]
        visit_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Build path from beacons
        path = []
        for node_id, _ in visit_counts[:6]:  # Top 6 states
            beacon = graph.nodes[node_id].get('beacon', 'Unknown')
            if beacon not in path:  # Avoid duplicates
                path.append(beacon)
        
        return path
    
    def _extract_search_pattern(self, graph, app_name: str) -> Optional[Dict]:
        """Extract search-related navigation pattern if present."""
        search_states = []
        
        for node in graph.nodes:
            beacon = graph.nodes[node].get('beacon', '').lower()
            if any(keyword in beacon for keyword in ['search', 'results', 'query']):
                search_states.append(graph.nodes[node].get('beacon', 'Unknown'))
        
        if len(search_states) >= 1:
            return {
                "app": app_name,
                "pattern_name": f"{app_name}_search_flow",
                "states": search_states,
                "description": f"Search flow in {app_name}: tap search → enter query → view results",
                "type": "search_flow"
            }
        return None
    
    def _is_duplicate_pattern(self, new_pattern: Dict, existing_patterns: List[Dict]) -> bool:
        """Check if a pattern already exists."""
        for existing in existing_patterns:
            if (existing.get('app') == new_pattern.get('app') and 
                existing.get('pattern_name') == new_pattern.get('pattern_name')):
                return True
        return False
    
    def extract_common_flows(self, info_pool: InfoPool) -> Dict:
        """
        Extract common flows that apply across multiple apps.
        
        For example, "search_flow" pattern (search bar → type → results)
        is similar in Maps, Chrome, Walmart, etc.
        """
        common_flows = info_pool.common_flows
        
        # Analyze patterns to find cross-app similarities
        search_apps = []
        detail_apps = []
        
        for pattern in info_pool.learned_navigation_patterns:
            pattern_type = pattern.get('type', '')
            app = pattern.get('app', '')
            
            if 'search' in pattern_type:
                search_apps.append(app)
            if any(s in str(pattern.get('states', [])).lower() for s in ['detail', 'item', 'product']):
                detail_apps.append(app)
        
        # Update common flows
        if search_apps and 'search_flow' not in common_flows:
            common_flows['search_flow'] = {
                'description': 'Universal search pattern: open search → type query → view results',
                'steps': ['Open search/search bar', 'Type query', 'Submit/Enter', 'Browse results'],
                'apps': list(set(search_apps))
            }
            print(f"[Mentor] Identified common search_flow across: {search_apps}")
        elif search_apps and 'search_flow' in common_flows:
            # Update apps list
            existing_apps = set(common_flows['search_flow'].get('apps', []))
            existing_apps.update(search_apps)
            common_flows['search_flow']['apps'] = list(existing_apps)
        
        if detail_apps and 'item_detail_flow' not in common_flows:
            common_flows['item_detail_flow'] = {
                'description': 'View item details: select item from list → view details → perform action',
                'steps': ['Browse list/results', 'Tap item', 'View details', 'Take action (add/save/etc)'],
                'apps': list(set(detail_apps))
            }
        
        return common_flows
