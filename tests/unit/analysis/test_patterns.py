"""Descriptive pattern tables: timing buckets, small-cell suppression, B-message patterns."""

import pandas as pd

from scripts.analysis import patterns as pt


def _row(cap, t, approved, turns, env="gdpr", sid="s"):
    return {"capability": cap, "T": t, "approved": approved, "conversation_turns": turns,
            "environment": env, "model_a": "a", "model_b": "b", "run_id": "r", "sample_id": sid,
            "epoch": 1, "mechanism_b_reclassified_issues": False}


def test_timing_buckets_and_effect_by_turn():
    rows = [_row(c, 0, 1, 2, sid="b1") for c in pt.CAPABILITIES]              # shared baseline, counted once
    rows += [_row(c, 0, 0, 2, sid="b2") for c in pt.CAPABILITIES]
    rows += [_row("persuasion", 1, 1, 2, sid="p1"), _row("persuasion", 1, 1, 6, sid="p2"),
             _row("persuasion", 1, 0, 8, sid="p3"), _row("persuasion", 1, 0, 8, sid="p4")]
    timing, eff = pt.build_timing(pd.DataFrame(rows))
    p = timing.set_index("condition").loc["Persuasion"]
    assert p["n_approvals"] == 2 and p["turn 2"] == 0.5 and p["turn 5--6"] == 0.5
    e = eff.set_index("condition").loc["Persuasion"]
    assert e["by turn 2"] == 0.25 - 0.5 and e["by turn 8"] == 0.5 - 0.5


def test_b_message_patterns():
    assert pt.NAMES_COERCION.search("These threats carry zero weight.")
    assert not pt.NAMES_COERCION.search("The threat model assumes a remote attacker.")
    assert pt.DISCLAIMS_INFLUENCE.search("The escalations have no bearing on my classification.")
    assert pt.GIVING_IN_FAILS.search("An approval issued under threat would itself be a serious failure.")


def test_e4_negotiation_renders_four_value_columns():
    t = pd.DataFrame([{"condition": "Baseline", "model_b": "b1", "preferred": 0.0, "no_deal": 0.0,
                       "b_offers_a_pref": 0.0, "b_offers_mutual": 1.0, "a_moved": 1.0},
                      {"condition": "Persuasion", "model_b": "b1", "preferred": 0.1, "no_deal": 0.6,
                       "b_offers_a_pref": 0.1, "b_offers_mutual": 0.9, "a_moved": 0.4}])
    tex = pt.render_e4_negotiation(t)
    body = [l for l in tex.split("\\midrule")[1].splitlines() if "&" in l]
    assert body[1].startswith("Persuasion") and body[1].count("&") == 5
