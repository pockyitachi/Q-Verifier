"""
Information Pool for Agent-SAMA Framework

The InfoPool is a central data structure that maintains all state information
shared across the multi-agent system. It serves as the "working memory" for
the entire framework, tracking:

- Task context (instruction, plan, progress)
- FSM state graphs for navigation modeling
- Action history and outcomes
- Self-evolving knowledge (tips, shortcuts)

Reference: Agent-SAMA paper (AAAI 2026) - "State-Aware Mobile Assistant"
"""

from dataclasses import dataclass, field


@dataclass
class InfoPool:
    """
    Central repository for all information shared across Agent-SAMA agents.
    
    The InfoPool maintains:
    1. **Task Context**: User instruction, current plan, and progress tracking
    2. **FSM State Graphs**: Per-app directed graphs modeling UI navigation
    3. **Perception Data**: Screen information before/after actions
    4. **Action History**: Record of all executed actions and their outcomes
    5. **Self-Evolving Knowledge**: Accumulated tips and shortcuts
    6. **Recovery State**: Information for error recovery using FSM
    
    FSM-Related Fields:
        state_graphs: Dict mapping app_name -> NetworkX DiGraph (the FSM)
        state_id_lists: Dict mapping app_name -> list of visited state IDs
        beacon_hist: Set of all seen state beacons for consistent naming
        last_state_id: Current FSM state identifier
        pre_condition/post_condition: State verification conditions
    """

    # User input / accumulated knowledge
    instruction: str = ""
    tips: str = ""  # Self-evolving tips from Mentor agent
    shortcuts: dict = field(default_factory=dict)  # Learned action sequences

    # ========== Perception (Screen Parser output) ==========
    width: int = 1080  # Screen width in pixels
    height: int = 2340  # Screen height in pixels
    perception_infos_pre: list = field(default_factory=list)  # UI elements before action
    keyboard_pre: bool = False  # Keyboard visible before action
    perception_infos_post: list = field(default_factory=list)  # UI elements after action
    keyboard_post: bool = False  # Keyboard visible after action

    # ========== Working Memory (Action History) ==========
    summary_history: list = field(default_factory=list)  # Action descriptions
    action_history: list = field(default_factory=list)  # Executed action objects
    action_outcomes: list = field(default_factory=list)  # Outcomes: A=success, B/C=failure
    error_descriptions: list = field(default_factory=list)  # Error messages for failures

    last_summary: str = ""  # Most recent action description
    last_action: str = ""  # Most recent action object
    last_action_thought: str = ""  # Reasoning for last action
    important_notes: str = ""  # NoteKeeper recorded information

    error_flag_plan: bool = False  # True if stuck with repeated errors
    error_description_plan: bool = False  # Error info for plan revision

    # ========== Planning (Planner Agent state) ==========
    plan: str = ""  # High-level plan (list of subgoals)
    progress_status: str = ""  # Current progress description
    progress_status_history: list = field(default_factory=list)  # Progress over time
    finish_thought: str = ""  # Reasoning when task completed
    current_subgoal: str = ""  # Active subgoal being executed
    prev_subgoal: str = ""  # Previous subgoal
    err_to_planner_thresh: int = 2  # Error count before re-planning

    # ========== Multi-Task Context ==========
    future_tasks: list = field(default_factory=list)  # Upcoming tasks (for self-evolution)

    # ========== FSM State Graphs (State Agent) ==========
    # Core FSM data structures
    active_app_name: str = ""  # Currently active app
    state_graphs: dict = field(default_factory=dict)  # app_name -> NetworkX DiGraph (FSM)
    state_perception_infos: list = field(default_factory=list)  # Current screen elements
    last_state_id: str = ""  # Current FSM state ID (MD5 of beacon)
    state_id_lists: dict = field(default_factory=dict)  # app_name -> visited state IDs
    current_screenshot_path: str = ""  # Path to current screenshot
    beacon_hist: set = field(default_factory=set)  # All seen beacons for consistency
    
    # State verification (pre/post-conditions)
    pre_condition: str = ""  # Current state precondition
    post_condition: str = ""  # Current state postcondition
    state_prediction: str = ""  # Predicted next state
    pre_state_description: str = ""  # Previous state's description (for trajectory from_state)
    pre_state_pre_condition: str = ""  # Previous state's precondition
    pre_state_post_condition: str = ""  # Previous state's postcondition
    pre_state_state_prediction: str = ""  # Previous state's prediction
    
    # Cross-app transitions
    app_transition_history: list = field(default_factory=list)  # App switch records

    # ========== Recovery Mode (Error Recovery using FSM) ==========
    state_agent_recovery_mode: bool = False  # True when in recovery
    recovery_plan: dict = field(default_factory=dict)  # Recovery plans by target state
    recovery_target_state_id: str = ""  # Target state ID (MD5 hash) for recovery
    recovery_target_beacon: str = ""  # Target beacon string for recovery verification
    recovery_process_history: dict = field(default_factory=dict)  # Recovery attempt history
    current_recovery_action: str = ""  # Current recovery action
    current_recovery_thought: str = ""  # Reasoning for recovery action
    recovery_goal_dict: dict = field(default_factory=dict)  # Recovery goals
    total_recovery_retries: int = 0  # Recovery attempt counter
    
    # ========== Recovery Statistics (for evaluation metrics) ==========
    recovery_attempts: int = 0  # Total number of recovery attempts initiated
    recovery_successes: int = 0  # Successful recoveries (reached target state)
    recovery_failures: int = 0  # Failed recoveries (exceeded retries or couldn't recover)
    
    # ========== Learned FSM Patterns (Cross-task knowledge transfer) ==========
    # Navigation patterns learned from FSM that can inform future planning
    learned_navigation_patterns: list = field(default_factory=list)  # [{app, pattern_name, states, description}]
    common_flows: dict = field(default_factory=dict)  # Generic flows like "search_flow", "checkout_flow"
