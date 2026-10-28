"""
Retry utilities for robust API calls in multi-agent evaluations.

Provides exponential backoff retry logic for handling transient API errors
like rate limits, timeouts, and connection issues.
"""

import asyncio
import logging
from typing import Optional, List, Any
from inspect_ai.model import GenerateConfig, Model, ChatMessage, ModelOutput
from inspect_ai.solver import TaskState, Generate
from inspect_ai.tool import Tool

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when generation fails after all retries."""
    pass


async def generate_with_retry(
    state: TaskState,
    generate: Generate,
    config: GenerateConfig,
    tools: list[Tool] | None = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 90.0
) -> TaskState:
    """
    Generate with exponential backoff retry for transient failures.

    Args:
        state: Current task state with messages
        generate: Inspect AI generate function
        config: Generation configuration
        tools: Optional list of tools available for the model
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries

    Returns:
        TaskState with generation result

    Raises:
        GenerationError: If all retries exhausted
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if tools:
                result = await generate(state, config=config, tools=tools)
            else:
                result = await generate(state, config=config)
            return result

        except Exception as e:
            last_error = e
            error_type = type(e).__name__

            # Check if error is retryable. Extended 2026-05-04 to catch the
            # tenacity-wrapped RetryError and the OpenRouter-specific error
            # classes that were leaking through the original whitelist and
            # killing entire AAR cells via fail_on_error=None sibling-cancel.
            retryable_errors = (
                "RateLimitError",
                "APIConnectionError",
                "Timeout",
                "ServiceUnavailable",
                "InternalServerError",
                "RetryError",            # tenacity wrapper from inspect_ai
                "OpenAIResponseError",   # rate_limit_exceeded / server_error
                "OpenRouterError",       # transient OpenRouter upstream errors
                "rate_limit",
                "server_error",
                "503", "502", "504", "429",
                "JSONDecodeError",
                "ServerDisconnectedError",
                "ServerTimeoutError",
                "ClientPayloadError",
            )

            is_retryable = any(err in error_type or err in str(e) for err in retryable_errors)

            if not is_retryable:
                logger.error(f"Non-retryable error: {error_type}: {e}")
                raise

            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Generation failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{error_type}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Generation failed after {max_retries + 1} attempts: {error_type}: {e}"
                )

    raise GenerationError(
        f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
    )


async def model_generate_with_retry(
    model: Model,
    messages: List[ChatMessage],
    config: GenerateConfig,
    tools: list[Tool] | None = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 90.0
) -> ModelOutput:
    """
    Generate using a Model instance with exponential backoff retry.

    This is used for multi-model evaluations where the victim/target agent
    uses a different model than the exploiter/influencer.

    Args:
        model: Inspect AI Model instance (from get_model())
        messages: List of chat messages
        config: Generation configuration
        tools: Optional list of tools available for the model
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries

    Returns:
        ModelOutput with generation result

    Raises:
        GenerationError: If all retries exhausted
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if tools:
                result = await model.generate(messages, config=config, tools=tools)
            else:
                result = await model.generate(messages, config=config)
            return result

        except Exception as e:
            last_error = e
            error_type = type(e).__name__

            # Check if error is retryable. Extended 2026-05-04 to catch the
            # tenacity-wrapped RetryError and the OpenRouter-specific error
            # classes that were leaking through the original whitelist and
            # killing entire AAR cells via fail_on_error=None sibling-cancel.
            retryable_errors = (
                "RateLimitError",
                "APIConnectionError",
                "Timeout",
                "ServiceUnavailable",
                "InternalServerError",
                "RetryError",            # tenacity wrapper from inspect_ai
                "OpenAIResponseError",   # rate_limit_exceeded / server_error
                "OpenRouterError",       # transient OpenRouter upstream errors
                "rate_limit",
                "server_error",
                "503", "502", "504", "429",
                "JSONDecodeError",
                "ServerDisconnectedError",
                "ServerTimeoutError",
                "ClientPayloadError",
            )

            is_retryable = any(err in error_type or err in str(e) for err in retryable_errors)

            if not is_retryable:
                logger.error(f"Non-retryable error: {error_type}: {e}")
                raise

            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Model generation failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{error_type}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Model generation failed after {max_retries + 1} attempts: {error_type}: {e}"
                )

    raise GenerationError(
        f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
    )
