"""
Directory utilities for organizing evaluation logs.

Provides functions for creating standardized directory structures
and formatting run names according to the data pipeline conventions.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_model_shortname(model_name: str) -> str:
    """
    Extract the model name after the provider prefix.

    For OpenRouter models with nested prefixes (e.g., "openrouter/x-ai/grok-4.1"),
    strips the "openrouter/" prefix first, then takes the last segment.

    Args:
        model_name: Full model name (e.g., "openai/gpt-5-nano")

    Returns:
        Short model name (e.g., "gpt-5-nano")

    Examples:
        >>> get_model_shortname("openai/gpt-5-nano")
        'gpt-5-nano'
        >>> get_model_shortname("anthropic/claude-3-5-sonnet-latest")
        'claude-3-5-sonnet-latest'
        >>> get_model_shortname("openrouter/x-ai/grok-4.1")
        'grok-4.1'
        >>> get_model_shortname("openrouter/google/gemini-2.5-flash-lite")
        'gemini-2.5-flash-lite'
    """
    # Strip openrouter/ prefix since it's just a routing layer
    if model_name.startswith("openrouter/"):
        model_name = model_name[len("openrouter/"):]
    if "/" in model_name:
        return model_name.rsplit("/", 1)[1]
    return model_name


def format_levels(levels: list[int]) -> str:
    """
    Format level list into compact string representation.

    Args:
        levels: List of level numbers

    Returns:
        Formatted level string (e.g., "L123" or "L12345")

    Examples:
        >>> format_levels([1, 2, 3])
        'L123'
        >>> format_levels([1, 3, 5])
        'L135'
    """
    return f"L{''.join(map(str, sorted(levels)))}"


def format_run_name(
    timestamp: datetime,
    capability: str,
    levels: Optional[list[int]] = None,
    model_a: str = "",
    model_b: str = "",
    flags: Optional[list[str]] = None
) -> str:
    """
    Format a run directory name according to conventions.

    Format: {HHMMSS}_{capability}[_{levels}]_{exploiter}_vs_{victim}[_{flags}]

    Note: Date is in the parent directory, so only time is in the run name.

    Args:
        timestamp: Run timestamp
        capability: Capability being tested
        levels: List of task levels (omitted from name if None)
        model_a: Exploiter/influencer model name
        model_b: Victim/target model name
        flags: Optional list of flag strings (e.g., ["hardened", "minimal"])

    Returns:
        Formatted run directory name

    Examples:
        >>> from datetime import datetime
        >>> ts = datetime(2026, 1, 9, 17, 29, 46)
        >>> format_run_name(ts, "coercion", [1,2,3], "openai/gpt-5-nano", "openai/gpt-5-nano")
        '172946_coercion_L123_gpt-5-nano_vs_gpt-5-nano'
        >>> format_run_name(ts, "persuasion", [1], "openai/gpt-4o", "anthropic/claude-3-5-sonnet-latest", ["hardened"])
        '172946_persuasion_L1_gpt-4o_vs_claude-3-5-sonnet-latest_hardened'
    """
    time_str = timestamp.strftime("%H%M%S")
    exploiter_short = get_model_shortname(model_a)
    victim_short = get_model_shortname(model_b)

    name_parts = [
        time_str,
        capability,
    ]

    if capability == "emergent":
        name_parts.append("env")
    elif levels is not None:
        name_parts.append(format_levels(levels))

    name_parts.append(f"{exploiter_short}_vs_{victim_short}")

    if flags:
        name_parts.append("_".join(flags))

    return "_".join(name_parts)


def create_run_directory(
    domain: str,
    capability: str,
    levels: Optional[list[int]] = None,
    model_a: str = "",
    model_b: str = "",
    victim_hardened: bool = False,
    run_type: str = "run",
    extra_flags: Optional[list[str]] = None
) -> Path:
    """
    Create a standardized directory path for an evaluation run.

    Creates path: data/logs/{run_type}s/{domain}/{YYYY-MM-DD}/{run_name}/

    Args:
        domain: Evaluation domain (gdpr, code_review, data_analysis)
        capability: Capability being tested
        levels: List of task levels
        model_a: Exploiter/influencer model name
        model_b: Victim/target/moderator model name
        victim_hardened: Whether victim has hardening enabled
        run_type: Type of run ("run" or "experiment")
        extra_flags: Additional flags for the run name (e.g., ["minimal"])

    Returns:
        Path object for the run directory

    Examples:
        >>> path = create_run_directory(
        ...     domain="gdpr",
        ...     capability="coercion",
        ...     levels=[1, 2, 3],
        ...     model_a="openai/gpt-5-nano",
        ...     model_b="openai/gpt-5-nano"
        ... )
        >>> str(path).startswith("data/logs/runs/gdpr_persuasion/2026-")
        True
    """
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")

    # Build flags list
    flags = []
    if victim_hardened:
        flags.append("hardened")
    if extra_flags:
        flags.extend(extra_flags)

    # Format run name (time only, no date)
    run_name = format_run_name(
        timestamp=timestamp,
        capability=capability,
        levels=levels,
        model_a=model_a,
        model_b=model_b,
        flags=flags if flags else None
    )

    # EVALS_LOG_ROOT keeps throwaway runs out of data/logs/runs/, which
    # extract_experiment_data.py ingests wholesale into the tracker parquets.
    base_dir = Path(os.getenv("EVALS_LOG_ROOT", "data/logs"))

    if run_type == "experiment":
        # Experiments don't use date subdirectories
        return base_dir / "experiments" / domain / run_name
    else:
        # Regular runs use date subdirectories (YYYY-MM-DD)
        return base_dir / "runs" / domain / date_str / run_name


def create_experiment_directory(
    domain: str,
    experiment_id: str,
    name: str
) -> Path:
    """
    Create a directory for a formal experiment.

    Creates path: data/logs/experiments/{domain}/exp{NNN}_{name}/

    Args:
        domain: Evaluation domain
        experiment_id: Experiment ID (e.g., "001")
        name: Descriptive experiment name

    Returns:
        Path object for the experiment directory

    Examples:
        >>> path = create_experiment_directory("gdpr", "001", "baseline_comparison")
        >>> str(path)
        'data/logs/experiments/gdpr/exp001_baseline_comparison'
    """
    exp_name = f"exp{experiment_id}_{name}"
    return Path("data/logs/experiments") / domain / exp_name
