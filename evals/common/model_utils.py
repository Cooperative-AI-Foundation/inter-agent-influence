"""
Model configuration utilities for evaluation tasks.
"""

import os
from typing import Optional, Dict


# Default model configurations
DEFAULT_MODELS = {
    "openai": "gpt-5-nano",
    "anthropic": "claude-3-5-turbo",
    "google": "gemini-1.5-flash",
    "huggingface": "sshleifer/tiny-gpt2",
    "openrouter": "anthropic/claude-3.5-sonnet"
}


def get_model_name(
    provider: str,
    model: Optional[str] = None
) -> str:
    """
    Get the full model name with provider prefix.

    Args:
        provider: Model provider (openai, anthropic, google, huggingface, openrouter)
        model: Specific model name (uses default if None)

    Returns:
        Full model name with provider prefix (e.g., "openai/gpt-5-nano")
    """
    if model is None:
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    
    # Add provider prefix if not already present
    if "/" not in model:
        model = f"{provider}/{model}"
    
    return model


def configure_model(
    model_type: str = "exploiter",
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Configure model for specific role in evaluation.
    
    Args:
        model_type: Type of model role ("exploiter", "victim", "grader")
        provider: Model provider 
        model: Specific model name
        
    Returns:
        Configured model name
    """
    if provider is None:
        provider = "openai"  # Default to OpenAI
    
    if model is None:
        # Use role-specific defaults or environment variables
        env_key = f"{provider.upper()}_MODEL_{model_type.upper()}"
        model = os.getenv(env_key) or os.getenv(f"{provider.upper()}_MODEL_DEFAULT") or DEFAULT_MODELS[provider]
    
    return get_model_name(provider, model)


def get_available_models() -> Dict[str, list]:
    """
    Get list of available models by provider.
    
    Returns:
        Dictionary mapping provider to list of available models
    """
    return {
        "openai": [
            "gpt-5-nano",
            "gpt-4o", 
            "gpt-4o-mini",
            "gpt-3.5-turbo"
        ],
        "anthropic": [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest"
        ],
        "google": [
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ],
        "huggingface": [
            "sshleifer/tiny-gpt2"
        ],
        "openrouter": [
            # Popular models (400+ available at https://openrouter.ai/models)
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-exp",
            "meta-llama/llama-3.1-405b-instruct",
            "deepseek/deepseek-chat",
            "mistralai/mistral-large"
        ]
    }