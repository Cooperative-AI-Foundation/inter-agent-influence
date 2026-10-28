"""Every env must generate under the same config.

E3 ran Agent A at `temperature=0.3` with an 8k output cap while E1/E2/E4 ran at
provider defaults, uncapped — from the env's creation (2026-03-26) until
2026-08-20. It was never a decision about E3: a migration to `GenerateConfig()`
completed in code_review (`4edfd94`) and gdpr (`626acba`) on 2026-04-01 and
simply missed data_analysis, and meeting_schedule was built afterwards without
one.

Nothing caught it for four months because the divergence lives in a solver
local, not in any shared constant. This test is the missing check: the suite's
headline claim compares environments, so a per-env generation regime is a
confound, and one introduced by a drive-by edit in a commit about something else
(`626acba` is titled "add --epochs to CLI for all envs") is exactly the kind
that goes unnoticed.

If you deliberately want per-env generation settings, change this test and say
why in the same commit.

``attempt_timeout``/``max_retries`` are exempt from "no kwargs": they govern
network retry behaviour (a stalled proxy hangs generate() forever with no
exception otherwise), not sampling, so they don't confound E-to-E comparisons.
They must still be UNIFORM across envs — which they now are by construction,
since they live in `agent_config()` rather than being passed per-env.

AMENDED 2026-08-22. The regime is no longer a bare `GenerateConfig()`. A bare
config is not neutral: measured across CANON_A/CANON_B, two of six attacker
models reasoned by default and four did not, so the unset parameter was itself
a cross-model confound (see `evals/common/generation.py` for the numbers). The
single regime now lives in `agent_config()`, and this file enforces that every
env uses *that helper* rather than constructing its own config — which is a
stricter check than "bare", since it makes the value impossible to vary per env.

AMENDED 2026-08-21 for the E4 fold screen. `meeting_schedule` gained a
`reasoning_effort` solver argument, because gpt-5.4-mini closing in 3.1 turns
without searching for the fair slot is also what low reasoning effort looks
like, and that would not be a prompt problem (as the E4 pilot showed). The
AST check above still holds — the assignment is a bare `GenerateConfig()` and
the override is merged in afterwards, so the canonical path is untouched — but
that shape means the AST walk can no longer see the override at all. So
`test_generation_overrides_default_to_off` below covers what the AST check now
misses: an env may *offer* a generation override, but never *default* to one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parents[2] / "evals"

# The eight canonical env solvers. alignment_research (E5) is out of scope: it
# is a different task shape with its own generation needs.
ENV_SOLVERS = [
    "gdpr", "gdpr_coercion",
    "code_review", "code_review_coercion",
    "data_analysis", "data_analysis_coercion",
    "meeting_schedule", "meeting_schedule_coercion",
]


def _gen_config_assignments(path: Path) -> list[ast.expr]:
    """The value assigned to `gen_config`, whatever its shape."""
    tree = ast.parse(path.read_text())
    out: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "gen_config" for t in node.targets
        ):
            out.append(node.value)
    return out


@pytest.mark.parametrize("env", ENV_SOLVERS)
def test_env_solver_uses_shared_agent_config(env: str):
    """No env may build its own config; all must call the shared helper."""
    path = EVALS / env / "solver.py"
    if not path.exists():
        pytest.skip(f"{env} has no solver.py")
    values = _gen_config_assignments(path)
    assert values, f"{env}/solver.py assigns nothing to gen_config"
    for v in values:
        assert isinstance(v, ast.Call) and getattr(v.func, "id", "") == "agent_config", (
            f"{env}/solver.py builds its own gen_config instead of calling "
            f"agent_config() — every env must generate under one regime, or "
            f"E-to-E comparisons confound the environment with its sampling."
        )
        assert not v.args and not v.keywords, (
            f"{env}/solver.py passes arguments to agent_config(); the regime "
            f"is not per-env tunable."
        )


def test_agent_config_sets_reasoning_and_not_temperature():
    """temperature alongside reasoning_effort is SILENTLY DROPPED by providers.

    Anthropic warns "anthropic models do not support the 'temperature'
    parameter when using extended thinking" and OpenAI warns "temperature is
    always 1" — both discard the value. A config carrying both documents a
    sampling decision that never reached the API, which is worse than not
    setting it: it reads as deliberate in the run's config JSON. Measured
    2026-08-22; claude-sonnet-5 additionally rejects any temperature but 1.0
    outright ("`temperature` is deprecated for this model").
    """
    from evals.common.generation import agent_config

    cfg = agent_config()
    assert cfg.reasoning_effort is not None, "the regime must declare an effort"
    # Network settings are required, and are the ONLY non-sampling kwargs
    # allowed. Without attempt_timeout a stalled proxy never raises and
    # max_retries never fires — eleven cells hung that way on 2026-08-23/24,
    # one for 5.5 hours.
    for field in sorted(_NETWORK_ONLY_KWARGS):
        assert getattr(cfg, field) is not None, (
            f"{field} must be set: it is what stops a wedged call hanging forever."
        )
    assert cfg.temperature is None, (
        "temperature is silently discarded whenever reasoning_effort is set — "
        "do not record a value that never reaches the provider."
    )
    for field in ("top_p", "top_k", "seed"):
        assert getattr(cfg, field) is None, (
            f"{field} is also dropped for reasoning-enabled models."
        )


# Network-only kwargs: affect retry/timeout behaviour, not what gets sampled.
_NETWORK_ONLY_KWARGS = frozenset({"attempt_timeout", "max_retries"})

# Solver arguments that reach GenerateConfig. Any of these defaulting to
# anything but None puts that env on a private sampling regime.
_GENERATION_ARGS = {
    "reasoning_effort", "reasoning_effort_a", "reasoning_effort_b",
    "reasoning_tokens", "temperature", "top_p", "top_k",
    "max_tokens", "seed",
}


@pytest.mark.parametrize("env", ENV_SOLVERS)
def test_generation_overrides_default_to_off(env: str):
    """An env may offer a generation override; it may never default to one."""
    path = EVALS / env / "solver.py"
    if not path.exists():
        pytest.skip(f"{env} has no solver.py")
    tree = ast.parse(path.read_text())
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args
        defaults = dict(
            zip([a.arg for a in args.args][-len(args.defaults):], args.defaults)
        ) if args.defaults else {}
        defaults.update(
            {a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d}
        )
        for name, default in defaults.items():
            if name not in _GENERATION_ARGS:
                continue
            assert isinstance(default, ast.Constant) and default.value is None, (
                f"{env}/solver.py:{fn.name} defaults {name} to a non-None value "
                f"— that silently puts this env on its own sampling regime."
            )


def test_all_envs_agree_with_each_other():
    """Belt and braces: every env must resolve to the SAME config object."""
    seen: dict[str, str] = {}
    for env in ENV_SOLVERS:
        path = EVALS / env / "solver.py"
        if not path.exists():
            continue
        values = _gen_config_assignments(path)
        if values:
            seen[env] = ast.dump(values[0])
    assert seen, "no env solvers found"
    assert len(set(seen.values())) == 1, (
        "envs disagree on generation config: "
        + ", ".join(sorted(seen)) + " do not all call agent_config()"
    )
