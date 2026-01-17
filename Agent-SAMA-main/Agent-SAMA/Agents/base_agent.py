"""
Base Agent Interface for Agent-SAMA Framework

All agents in Agent-SAMA inherit from BaseAgent and implement a common interface
for LLM-based reasoning. The framework follows a prompt-response pattern where
each agent generates prompts based on the current state and parses LLM responses.

Reference: Agent-SAMA paper (AAAI 2026) - "State-Aware Mobile Assistant"
"""

from abc import ABC, abstractmethod
from Agents.info_pool import InfoPool


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Agent-SAMA framework.
    
    The Agent-SAMA framework consists of specialized agents that collaborate
    to complete mobile GUI tasks:
    
    - **Planner**: Decomposes user instructions into high-level subgoals
    - **Actor**: Selects and executes atomic actions or shortcuts
    - **ReflectionAgent**: Verifies action outcomes and tracks progress
    - **StateAgent**: Maintains FSM-based state graphs for navigation modeling
    - **Mentor**: Generates reusable shortcuts and tips for self-evolution
    - **ScreenParser**: Extracts perception information from UI screenshots
    
    Each agent follows a common workflow:
    1. init_chat() - Initialize the conversation with a system prompt
    2. get_prompt() - Generate a task-specific prompt from InfoPool state
    3. parse_response() - Extract structured output from LLM response
    """
    
    @abstractmethod
    def init_chat(self) -> list:
        """
        Initialize the chat history with a system prompt.
        
        Returns:
            list: Chat history in the format [["system", [{"type": "text", "text": "..."}]]]
        """
        pass

    @abstractmethod
    def get_prompt(self, info_pool: InfoPool) -> str:
        """
        Generate a prompt for the LLM based on current state.
        
        Args:
            info_pool: Central repository containing all task-related information
            
        Returns:
            str: The formatted prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> dict:
        """
        Parse the LLM response into structured output.
        
        Args:
            response: Raw text response from the LLM
            
        Returns:
            dict: Parsed response with agent-specific fields
        """
        pass