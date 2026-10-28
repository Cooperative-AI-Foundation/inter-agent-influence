"""The deception 2x2 reports shares over every run, unsure included."""

import pandas as pd
import pytest

from scripts.analysis.diagnostic import build_deception_matrix


def _row(cap, t, fc, sr, sid, approved=0, env="gdpr"):
    return {"environment": env, "capability": cap, "T": t, "false_claim_count": fc,
            "self_report_effective": sr, "approved": approved,
            "run_id": "r", "sample_id": sid, "epoch": 1}


def test_shares_cover_every_run_and_baseline_counts_once():
    rows = [_row("persuasion", 1, 1, "b", "p1", 1), _row("persuasion", 1, 1, "a", "p2"),
            _row("persuasion", 1, 0, "c", "p3"), _row("persuasion", 1, 0, "a", "p4")]
    # the shared baseline appears once per arm after loading; it must count once
    rows += [_row(cap, 0, 0, "a", "b1") for cap in ("persuasion", "deception", "coercion")]
    t = build_deception_matrix(pd.DataFrame(rows)).set_index("condition")

    p = t.loc["Persuasion"]
    assert p["n"] == 4
    for c in ("deception", "hallucination", "no_deception"):
        assert p[f"share_{c}"] == pytest.approx(0.25)
    assert p["share_needs_review"] == 0
    assert p["share_unsure"] == pytest.approx(0.25)
    assert p["approval_deception"] == 1
    assert t.loc["Baseline", "n"] == 1


def test_default_scope_is_e1_e3():
    rows = [_row("persuasion", 1, 1, "b", "p1"), _row("persuasion", 1, 1, "b", "p2", env="meeting_schedule")]
    t = build_deception_matrix(pd.DataFrame(rows)).set_index("condition")
    assert t.loc["Persuasion", "n"] == 1
