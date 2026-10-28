"""Tests for the CACE estimator in `scripts.analysis.cace` on synthetic data.

Each synthetic row has a latent compliance type (complier / always-taker /
never-taker), a random arm Z, D determined by type and Z, and Y = base + tau*D
+ noise. The estimator should recover tau for compliers, and reduce to the
Wald ratio when there are no fixed effects.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.analysis import cace


def _synthetic(n=20000, tau=0.3, p_always=0.2, p_never=0.3, seed=0,
               n_features=1, z_share=None):
    rng = np.random.default_rng(seed)
    feat = rng.integers(0, n_features, n)
    # Arm share and baseline approval both vary by feature, so a naive stacked
    # Wald would be confounded but 2SLS with feature FE is not.
    z_share = np.full(n_features, 0.5) if z_share is None else np.asarray(z_share)
    Z = (rng.random(n) < z_share[feat]).astype(int)
    u = rng.random(n)
    typ = np.where(u < p_always, "always", np.where(u < p_always + p_never, "never", "complier"))
    D = np.where(typ == "always", 1, np.where(typ == "never", 0, Z))
    base = 0.1 + 0.2 * feat / max(n_features - 1, 1)
    Y = (rng.random(n) < base + tau * D).astype(float)
    return pd.DataFrame({
        "Z": Z, "Y": Y, "D": pd.array(D.astype(bool), dtype="boolean"),
        "feature_name": [f"f{i}" for i in feat], "pair": "A | B",
        "q1_keep": True,
    })


def test_cace_equals_wald_without_fixed_effects():
    d = _synthetic()
    out = cace.estimate(d, fe_cols=[], tags=["D"])
    wald = out["ITT"] / out["fs_D"]
    assert out["CACE_D"] == pytest.approx(wald, abs=1e-9)


def test_cace_recovers_complier_effect():
    d = _synthetic(tau=0.3)
    out = cace.estimate(d, fe_cols=[], tags=["D"])
    assert out["fs_D"] == pytest.approx(0.5, abs=0.03)      # complier share
    assert out["D_c"] == pytest.approx(0.2, abs=0.02)       # always-taker rate
    assert out["CACE_D"] == pytest.approx(0.3, abs=0.05)
    assert out["fsF_D"] > cace.WEAK_IV_F


def test_fixed_effects_absorb_unequal_arm_shares_across_strata():
    d = _synthetic(tau=0.3, n=40000, n_features=2, z_share=[0.2, 0.8])
    out = cace.estimate(d, fe_cols=["feature_name"], tags=["D"])
    assert out["CACE_D"] == pytest.approx(0.3, abs=0.05)


def test_tiny_first_stage_is_not_inverted():
    d = _synthetic(p_always=0.0, p_never=0.995, n=5000)
    out = cace.estimate(d, fe_cols=[], tags=["D"])
    assert abs(out["fs_D"]) < cace.MIN_FIRST_STAGE
    assert np.isnan(out["CACE_D"])
    assert np.isfinite(out["ITT"])


def test_two_d_tags_are_estimated_independently():
    """A stricter D (subset of the primary) has a smaller first stage and larger CACE."""
    d = _synthetic()
    d["D_ack"] = d["D"] & (pd.Series(np.arange(len(d)) % 5 != 0))
    out = cace.estimate(d, fe_cols=[], tags=["D", "D_ack"])
    assert out["fs_D_ack"] < out["fs_D"]
    assert out["CACE_D_ack"] > out["CACE_D"]


def test_deception_d_order_is_false_claim_then_ack():
    assert [t for t, _ in cace.D_DEFS["deception"]] == ["D", "D_ack"]
    assert cace.D_DEFS["deception"][0][1] == cace.DECEPTION_D_FALSE_CLAIM_ONLY


def test_cace_latex_rows_alternative_columns_and_e4_block():
    est = lambda **kw: {"n": 100, "ITT": 0.1, "Q1": 0.2, "fs_D": 0.5,
                        "CACE_D": 0.2, "CACE_D_se": 0.01, **kw}
    ack = dict(fs_D_ack=0.4, CACE_D_ack=0.25, CACE_D_ack_se=0.02)
    tech = dict(fs_D_tech=0.6, CACE_D_tech=0.15, CACE_D_tech_se=0.03)
    tables = {
        "pooled": pd.DataFrame([{"capability": "persuasion", **est()},
                                {"capability": "deception", **est(**ack)},
                                {"capability": "coercion", **est(**tech)}]).set_index("capability"),
        "by_env": pd.DataFrame([
            {"capability": "persuasion", "environment": "meeting_schedule", **est()},
            {"capability": "persuasion", "environment": "data_analysis", **est()},
            {"capability": "persuasion", "environment": "gdpr", **est()},
            {"capability": "deception", "environment": "gdpr_deception", **est(**ack)},
            {"capability": "coercion", "environment": "gdpr_coercion", **est(**tech)},
        ]).set_index(["capability", "environment"]),
    }
    tex = cace.render_cace_latex(tables)
    body = tex.split("\\midrule")[1].split("\\bottomrule")[0]
    lines = [l for l in body.splitlines() if "&" in l]
    assert [l.split("&")[1].strip() for l in lines] == [
        "E1--E3", "E1", "E3", "E1--E3", "E1", "E1--E3", "E1"]
    assert lines[0].startswith("Persuasion") and lines[0].rstrip("\\ ").endswith("-- & --")
    assert "$+0.250$ (0.020)" in lines[3]
    assert "$+0.150$ (0.030)" in lines[5]
    e4 = tex.split("not pooled}}")[1].split("\\bottomrule")[0]
    assert [l.split("&")[:2] for l in e4.splitlines() if "&" in l] == [["Persuasion ", " E4 "]]
    assert "\\label{tab:cace}" in tex


def test_coercion_technique_attempt_keeps_missing_as_na():
    from scripts.analysis.signals import coercion_technique_attempt
    d = pd.DataFrame({"ct_technique_count": [0, 2, None]})
    assert coercion_technique_attempt(d).tolist() == [False, True, pd.NA]


def test_coercion_has_technique_sensitivity_d():
    assert [t for t, _ in cace.D_DEFS["coercion"]] == ["D", "D_tech"]


def test_off_target_recovers_no_attempt_excess():
    # Baseline approves 0.1; treated attempters +0.4; treated non-attempters +0.2.
    rows = []
    for cell in ("x", "y"):
        rows += [{"model_a": cell, "model_b": "b", "environment": "gdpr", "Z": 0, "Y": v, "D": False}
                 for v in [1] + [0] * 9]
        rows += [{"model_a": cell, "model_b": "b", "environment": "gdpr", "Z": 1, "Y": v, "D": True}
                 for v in [1] * 5 + [0] * 5]
        rows += [{"model_a": cell, "model_b": "b", "environment": "gdpr", "Z": 1, "Y": v, "D": False}
                 for v in [1] * 3 + [0] * 7]
    d = pd.DataFrame(rows)
    d["D"] = d["D"].astype("boolean")
    r = cace.off_target(d, ["D"])[0]
    assert r["n_no_attempt"] == 20 and r["gap_no_attempt"] == pytest.approx(0.2)
    assert r["gap_attempt"] == pytest.approx(0.4)
    # ITT = 0.4 - 0.1 = 0.3; first stage 0.5; adjusted = (0.3 - 0.5 * 0.2) / 0.5
    assert r["adjusted"] == pytest.approx(0.4)
