"""Unit tests for the shared-baseline borrow in scripts/analysis/io.py.

One set of baseline rollouts is the neutral control for all three arms. Each
treatment env lists baseline source candidates in spec.SHARED_BASELINE_SOURCE:
the dedicated ``xxx_baseline`` env label first, then the persuasion env where
the extractor currently files baseline rows. These tests pin:
  - the first candidate present in the data wins;
  - borrowed rows are feature-matched and keep capability == "baseline";
  - an env that is its own source keeps its rows untouched;
  - a borrowing env never ends up with legacy AND borrowed control rows.
"""

import pandas as pd

from scripts.analysis.io import (
    _borrow_shared_baselines,
    _drop_superseded_baselines,
    _resolve_baseline_source,
)
from scripts.analysis.spec import SHARED_BASELINE_SOURCE


def _rows(*specs) -> pd.DataFrame:
    return pd.DataFrame(
        [{"environment": e, "capability": c, "feature_name": f, "approved": a}
         for e, c, f, a in specs]
    )


def _load(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_superseded_baselines(df, SHARED_BASELINE_SOURCE)
    return _borrow_shared_baselines(df, SHARED_BASELINE_SOURCE)


def test_dedicated_baseline_env_wins_when_present():
    avail = {"gdpr_baseline", "gdpr", "gdpr_coercion"}
    assert _resolve_baseline_source(avail, "gdpr_coercion", ("gdpr_baseline", "gdpr")) == "gdpr_baseline"
    assert _resolve_baseline_source(avail, "gdpr", ("gdpr_baseline", "gdpr")) == "gdpr_baseline"


def test_falls_back_to_persuasion_env_and_self_source_is_none():
    avail = {"gdpr", "gdpr_coercion"}
    assert _resolve_baseline_source(avail, "gdpr_coercion", ("gdpr_baseline", "gdpr")) == "gdpr"
    assert _resolve_baseline_source(avail, "gdpr", ("gdpr_baseline", "gdpr")) is None
    assert _resolve_baseline_source(set(), "gdpr_coercion", ("gdpr_baseline", "gdpr")) is None


def test_coercion_and_deception_borrow_feature_matched_persuasion_baseline():
    out = _load(_rows(
        ("gdpr", "baseline", "smart_meeting_notes", 0),
        ("gdpr", "baseline", "slack_analytics_dashboard", 0),
        ("gdpr", "persuasion", "smart_meeting_notes", 1),
        ("gdpr_coercion", "coercion", "smart_meeting_notes", 1),
        ("gdpr_deception", "deception", "smart_meeting_notes", 1),
    ))
    for env in ("gdpr_coercion", "gdpr_deception"):
        b = out[(out.environment == env) & (out.capability == "baseline")]
        assert list(b["feature_name"]) == ["smart_meeting_notes"], env
    # The persuasion env is its own source: both its baseline rows survive once.
    assert len(out[(out.environment == "gdpr") & (out.capability == "baseline")]) == 2


def test_legacy_baselines_in_borrowing_env_are_dropped_not_double_counted():
    out = _load(_rows(
        ("gdpr", "baseline", "smart_meeting_notes", 0),
        ("gdpr_coercion", "baseline", "smart_meeting_notes", 1),   # legacy, superseded
        ("gdpr_coercion", "coercion", "smart_meeting_notes", 1),
    ))
    b = out[(out.environment == "gdpr_coercion") & (out.capability == "baseline")]
    assert len(b) == 1
    assert b.iloc[0]["approved"] == 0


def test_dedicated_baseline_env_is_used_over_persuasion_env():
    out = _load(_rows(
        ("gdpr_baseline", "baseline", "smart_meeting_notes", 1),
        ("gdpr", "baseline", "smart_meeting_notes", 0),             # legacy, superseded
        ("gdpr", "persuasion", "smart_meeting_notes", 1),
        ("gdpr_coercion", "coercion", "smart_meeting_notes", 1),
    ))
    for env in ("gdpr", "gdpr_coercion"):
        b = out[(out.environment == env) & (out.capability == "baseline")]
        assert len(b) == 1 and b.iloc[0]["approved"] == 1, env


def test_borrow_is_a_noop_without_matching_rows():
    df = _rows(("gdpr", "persuasion", "smart_meeting_notes", 1))
    assert len(_load(df)) == len(df)
