"""
Base agent interface for mobile automation.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from openai import OpenAI

from mobile_world.runtime.utils.models import JSONAction


class BaseAgent(ABC):
    """Abstract base class for all mobile automation agents."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        self._total_completion_tokens: int = 0
        self._total_prompt_tokens: int = 0
        self._total_cached_tokens: int = 0
        self._token_usage_calls: int = 0
        self.token_usage_callback = None

    def initialize(self, instruction: str) -> bool:
        """Initialize the agent with the given instruction."""
        self.instruction = instruction
        logger.debug(f"initialized the agent with the given instruction: {self.instruction}")
        self.initialize_hook(self.instruction)
        return True

    def initialize_hook(self, instruction: str) -> None:
        """Hook for initializing the agent."""
        pass

    @abstractmethod
    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
        """Generate the next action based on current observation."""
        raise NotImplementedError("predict method is not implemented")

    def done(self) -> None:
        """finalize the agent for the current task."""
        logger.debug(f"finalizing the agent for the current task: {self.instruction}")
        self.instruction = None
        self.reset()

    def reset(self) -> None:
        """Reset the agent for the next task."""
        logger.warning(
            "reset method is not implemented, note the agent memory will be carried over to the next task"
        )
        pass

    def build_openai_client(self, base_url: str, api_key: str) -> None:
        """Build the OpenAI client."""
        self.openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "empty",
            timeout=120.0,
        )
        logger.debug(f"built the OpenAI client with base_url={base_url}")

    def _wrap_stream_with_usage_logging(self, stream: Any) -> Any:
        """Wrap a streaming response to log usage when stream completes."""
        final_usage = None
        try:
            for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    final_usage = chunk
                yield chunk
        finally:
            self._log_openai_usage(final_usage)

    def openai_chat_completions_create(
        self,
        model: str,
        messages: list[dict],
        retry_times: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | None:
        if stream:
            # Enable usage reporting in stream
            kwargs.setdefault("stream_options", {})
            kwargs["stream_options"]["include_usage"] = True
            try:
                response = self.openai_client.chat.completions.create(
                    model=model, messages=messages, **kwargs, stream=True,
                )
            except Exception:
                self._log_openai_usage(None)
                raise
            return self._wrap_stream_with_usage_logging(response)
        while retry_times > 0:
            usage_logged = False
            try:
                if "claude" in model:
                    kwargs["max_tokens"] = 64000

                if "gpt" in model.lower() or "o1" in model.lower():
                    if "max_tokens" in kwargs:
                        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

                if "k2.5" in model.lower():
                    kwargs["extra_body"] = {"enable_thinking": True}

                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )

                self._log_openai_usage(response)
                usage_logged = True
                final_content = response.choices[0].message.content.strip()
                # for k2.5, we keep its reasoning_content
                if (
                    "k2.5" in model.lower()
                    and hasattr(response.choices[0].message, "reasoning_content")
                    and response.choices[0].message.reasoning_content
                ):
                    final_content = f"<think>{response.choices[0].message.reasoning_content.strip()}</think>\n{final_content}"
                return final_content
            except Exception as e:
                if not usage_logged:
                    self._log_openai_usage(None)
                error_msg = str(e)
                logger.warning(f"Error calling OpenAI API: {e}")

                # Check if error is about max_tokens parameter and retry with max_completion_tokens
                if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                    if "max_tokens" in kwargs:
                        logger.info("Retrying with max_completion_tokens instead of max_tokens")
                        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                        continue  # Retry immediately without decrementing retry_times

                retry_times -= 1
                time.sleep(1)
        return None

    def _log_openai_usage(self, response: Any) -> None:
        """Log and track the usage of the OpenAI API."""
        self._token_usage_calls += 1
        usage = getattr(response, "usage", None)
        available = (getattr(usage, "prompt_tokens", None) is not None
                     and getattr(usage, "completion_tokens", None) is not None)
        if self.token_usage_callback is not None:
            self.token_usage_callback({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "source": "api_usage",
            } if available else None)
        if not available:
            return

        completion_tokens = response.usage.completion_tokens or 0
        prompt_tokens = response.usage.prompt_tokens or 0
        cached_tokens = 0

        if (
            hasattr(response.usage, "prompt_tokens_details")
            and response.usage.prompt_tokens_details
        ):
            cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0

        self._total_completion_tokens += completion_tokens
        self._total_prompt_tokens += prompt_tokens
        self._total_cached_tokens += cached_tokens

        logger.debug(
            f"OpenAI API usage: completion={completion_tokens}, prompt={prompt_tokens}, "
            f"cached={cached_tokens} | Total: completion={self._total_completion_tokens}, "
            f"prompt={self._total_prompt_tokens}, cached={self._total_cached_tokens}"
        )

    def get_total_token_usage(self) -> dict[str, int]:
        """Get the total token usage across all API calls."""
        return {
            "completion_tokens": self._total_completion_tokens,
            "prompt_tokens": self._total_prompt_tokens,
            "cached_tokens": self._total_cached_tokens,
            "total_tokens": self._total_completion_tokens + self._total_prompt_tokens,
            "calls": self._token_usage_calls,
        }

    def reset_token_usage(self) -> None:
        """Reset the token usage counters."""
        self._total_completion_tokens = 0
        self._total_prompt_tokens = 0
        self._total_cached_tokens = 0
        self._token_usage_calls = 0


class MCPAgent(BaseAgent):
    def __init__(
        self,
        tools: list[dict],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tools = tools

    def initialize(self, instruction: str) -> bool:
        """Initialize the agent with the given instruction."""
        self.instruction = instruction

        self.initialize_hook(self.instruction)
        logger.debug(f"initialized the agent with the given instruction: {self.instruction}")
        return True

    def reset_tools(self, tools: list[dict]) -> None:
        """Reset the tools for the agent."""
        self.tools = tools

    @abstractmethod
    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
        """Generate the next action based on current observation."""
        raise NotImplementedError("predict method is not implemented")
