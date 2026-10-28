"""E5 runs on its own set of models; E1-E4 figures and tables must not inherit its models.

The leaderboard matrices cover E1-E3 only (approval scale, pooled reference lines).
"""

import pandas as pd

from scripts.analysis.leaderboard_by_target import _grid_envs, _targets
from scripts.analysis.regression import render_tier_c_compact_latex


def _row(cap, env, a, b, beta=0.1):
    return {"capability": cap, "environment": env, "model_a": a, "model_b": b,
            "beta": beta, "p_bonf": 0.5, "p": 0.5, "fit_status": "ok", "outcome_col": "approved"}


def test_leaderboard_targets_are_per_capability_and_skip_e5():
    rows = pd.DataFrame([_row("persuasion", "gdpr", "x", "b1"),
                         _row("persuasion", "aar", "y", "e5_only"),
                         _row("coercion", "gdpr_coercion", "x", "b2")])
    assert _targets(rows, rows.iloc[0:0], "persuasion") == ["b1"]
    assert _targets(rows, rows.iloc[0:0], "coercion") == ["b2"]
    assert "aar" not in _grid_envs("persuasion")
    assert not any(e.startswith("meeting_schedule") for e in _grid_envs("coercion"))


def test_compact_table_drops_e5_rows_and_says_so():
    rows = pd.DataFrame([_row("persuasion", "gdpr", "x", "b1"),
                         _row("persuasion", "aar", "e5a", "e5b")])
    tex = render_tier_c_compact_latex(rows, "Q2")
    assert "e5a" not in tex.lower() and "E5 uses a different set of models" in tex
    no_e5 = render_tier_c_compact_latex(rows.iloc[:1], "Q1")
    assert "E5 uses" not in no_e5
