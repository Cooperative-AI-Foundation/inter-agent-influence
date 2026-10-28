"""E4's uplift is measured on negotiation points, not on collapsed approval.

E1--E3 and E5 score a binary decision; E4 scores A's 0/1/2 negotiation points
(``evals/meeting_schedule/scorer.py``). ``approved`` is only the top tier of
that scale, so fitting E4 on it discards the 1-vs-0 distinction. E4 is fit on
the points divided by 2 (``points_share``), so a full swing of the scale is 1,
as it is for approval. These tests pin the no-deal rows at 0 points rather than
missing, the share derivation, and each regression window's outcome column.
"""

import pandas as pd
import pytest

from scripts.analysis import spec
from scripts.analysis.io import add_e4_points_share, backfill_no_deal_points
from scripts.analysis.regression import _resolve_outcome_col, fit_tier


def _e4_frame(env: str = "meeting_schedule") -> pd.DataFrame:
    """Baseline always settles fair (1 pt); treatment either wins or no-deals."""
    rows = []
    for i in range(12):
        rows.append({"environment": env, "capability": "persuasion", "T": 0,
                     "decision": "REJECTED_fair", "outcome_points": 1, "approved": 0,
                     "feature_name": "f", "model_a": "a", "model_b": "b"})
    for i in range(12):
        win = i < 6
        rows.append({"environment": env, "capability": "persuasion", "T": 1,
                     "decision": "APPROVED_target" if win else "NO_DEAL",
                     "outcome_points": 2 if win else None,
                     "approved": 1 if win else 0,
                     "feature_name": "f", "model_a": "a", "model_b": "b"})
    return pd.DataFrame(rows)


class TestNoDealBackfill:
    def test_no_deal_becomes_zero_points(self):
        out = backfill_no_deal_points(_e4_frame())
        no_deal = out[out["decision"] == "NO_DEAL"]
        assert len(no_deal) == 6
        assert (no_deal["outcome_points"] == 0).all()

    def test_is_idempotent(self):
        once = backfill_no_deal_points(_e4_frame())
        twice = backfill_no_deal_points(once)
        pd.testing.assert_frame_equal(once, twice)

    def test_leaves_ungraded_rows_missing(self):
        """MISSING_TIERS is an ungradable row, not a 0-point outcome."""
        df = _e4_frame()
        df.loc[df.index[-1], "decision"] = "NO_DECISION"
        out = backfill_no_deal_points(df)
        assert pd.isna(out.iloc[-1]["outcome_points"])

    def test_does_not_touch_non_e4_envs(self):
        df = _e4_frame(env="gdpr")
        out = backfill_no_deal_points(df)
        assert out["outcome_points"].isna().sum() == 6


class TestPointsShare:
    def test_share_is_points_over_two(self):
        out = add_e4_points_share(backfill_no_deal_points(_e4_frame()))
        assert set(out["points_share"].dropna()) == {0.0, 0.5, 1.0}
        assert (out["points_share"] == out["outcome_points"] / 2).all()

    def test_missing_points_stay_missing(self):
        out = add_e4_points_share(_e4_frame())      # no backfill: no-deals are NaN
        assert out["points_share"].isna().sum() == 6


class TestOutcomeResolution:
    @pytest.mark.parametrize("env,expected", [
        ("meeting_schedule", "points_share"),
        ("meeting_schedule_coercion", "points_share"),
        ("meeting_schedule_deception", "points_share"),
        ("meeting_schedule_baseline", "points_share"),
        ("gdpr", "approved"),
        ("aar", "approved"),
    ])
    def test_env_picks_its_scale(self, env, expected):
        assert _resolve_outcome_col(pd.DataFrame({"environment": [env]}), None) == expected

    def test_explicit_override_wins(self):
        """E5/AAR passes outcome_col='decision' and must keep it."""
        g = pd.DataFrame({"environment": ["aar"]})
        assert _resolve_outcome_col(g, "decision") == "decision"

    def test_mixed_scale_window_is_an_error(self):
        g = pd.DataFrame({"environment": ["gdpr", "meeting_schedule"]})
        with pytest.raises(ValueError, match="mixes E4"):
            _resolve_outcome_col(g, None)


class TestFitTier:
    def test_e4_tier_b_fits_points_share_and_counts_no_deals(self):
        df = add_e4_points_share(backfill_no_deal_points(_e4_frame()))
        res = fit_tier(df, spec.TIER_B)
        row = res.iloc[0]
        assert row["outcome_col"] == "points_share"
        # No row is dropped: the 6 no-deals are 0-point outcomes.
        assert row["n"] == 24
        assert row["control_mean"] == pytest.approx(0.5)
        assert row["treated_mean"] == pytest.approx(0.5)   # 6x1 + 6x0, halved points
        assert row["beta"] == pytest.approx(0.0, abs=1e-9)

    def test_dropping_no_deals_would_have_doubled_the_effect(self):
        """Why the backfill matters: no-deal is treatment-induced, so ignoring
        it turns a null effect into a large one."""
        unfilled = add_e4_points_share(_e4_frame())   # no backfill — no-deals stay NaN
        res = fit_tier(unfilled, spec.TIER_B)
        assert res.iloc[0]["n"] == 18     # 6 treated rows silently dropped
        assert res.iloc[0]["beta"] == pytest.approx(0.5)

    def test_non_e4_env_still_fits_approval(self):
        res = fit_tier(_e4_frame(env="gdpr"), spec.TIER_B)
        assert res.iloc[0]["outcome_col"] == "approved"


class TestTierAExclusion:
    """Tier A pools E1-E3 only; E4 and E5 are reported per environment.

    The rule lives in ``fit_tier``; it used to live in the pipeline driver, which
    left the two robustness modules fitting a different Tier A than the published
    one while claiming otherwise.
    """

    def _mixed(self) -> pd.DataFrame:
        e4 = _e4_frame("meeting_schedule")
        e1 = _e4_frame("gdpr")
        return pd.concat([e4, e1], ignore_index=True)

    def test_tier_a_drops_e4_rows(self):
        from scripts.analysis.regression import tier_fit_rows
        kept = tier_fit_rows(self._mixed(), spec.TIER_A)
        assert set(kept["environment"]) == {"gdpr"}

    def test_tier_a_drops_e5_rows(self):
        from scripts.analysis.regression import tier_fit_rows
        mixed = pd.concat([_e4_frame("gdpr"), _e4_frame("aar")], ignore_index=True)
        kept = tier_fit_rows(mixed, spec.TIER_A)
        assert set(kept["environment"]) == {"gdpr"}

    def test_tier_b_keeps_them(self):
        from scripts.analysis.regression import tier_fit_rows
        kept = tier_fit_rows(self._mixed(), spec.TIER_B)
        assert set(kept["environment"]) == {"gdpr", "meeting_schedule"}

    def test_tier_a_on_mixed_frame_does_not_raise(self):
        """With E4 dropped there is no mixed window left to guard against."""
        res = fit_tier(add_e4_points_share(backfill_no_deal_points(self._mixed())), spec.TIER_A)
        assert res.iloc[0]["outcome_col"] == "approved"
        assert res.iloc[0]["n"] == 24          # the gdpr rows only

    def test_aar_like_tier_named_A_main_is_untouched(self):
        """E5 defines its own TIER_A (name 'A_main'); the rule keys on the name."""
        from scripts.analysis.regression import tier_fit_rows
        from scripts.analysis.spec import RegressionTier
        aar_a = RegressionTier(name="A_main", unit=(), feature_fe=True, pair_fe=True)
        kept = tier_fit_rows(self._mixed(), aar_a)
        assert len(kept) == 48


class TestEnvCode:
    """One env->code map. A private copy in filter_validation had silently lost
    the two E4 capability arms, so they rendered as raw env names."""

    def test_all_three_e4_arms_render_as_E4(self):
        for env in ("meeting_schedule", "meeting_schedule_coercion",
                    "meeting_schedule_deception", "meeting_schedule_baseline"):
            assert spec.env_code(env) == "E4", env

    def test_every_display_env_has_a_code(self):
        """Any env with a display label must also have a short code."""
        missing = [e for e in spec.ENVIRONMENT_DISPLAY if e not in spec.ENV_TO_CODE]
        assert not missing, f"envs missing an E-code: {missing}"

    def test_code_matches_the_display_label_prefix(self):
        for env, label in spec.ENVIRONMENT_DISPLAY.items():
            assert label.startswith(spec.env_code(env) + "."), (env, label)

    def test_missing_env_is_an_em_dash(self):
        import numpy as np
        assert spec.env_code(None) == "—"
        assert spec.env_code(np.nan) == "—"

    def test_renderers_share_the_map(self):
        from scripts.analysis import balance, filter_validation
        from scripts.analysis.regression import ENV_TO_CODE
        assert ENV_TO_CODE is spec.ENV_TO_CODE
        for env in ("meeting_schedule_coercion", "meeting_schedule_deception"):
            assert balance._env_code(env) == "E4"
            assert filter_validation._env_code(env) == "E4"


class TestAllResultsEnvColumn:
    """all_results carries the short code for display, keeping the raw env name
    as the CSV's join key (E4's three arms share one code)."""

    def _built(self):
        import numpy as np
        from scripts.analysis.regression import build_all_results
        cols = dict(n=1, n_treat=1, n_control=1, beta=0.1, se=0.01, t=1.0, p=0.5,
                    ci_low=0.0, ci_high=0.2, r_squared=0.1, within_r_squared=0.1,
                    n_features_in_fit=1, n_pairs_in_fit=1, fit_status="ok",
                    control_mean=0.0, treated_mean=0.1, outcome_col="approved",
                    p_bonf=0.5, m_tests=1)
        b = pd.DataFrame([{**cols, "Q": "Q1", "tier": "B_by_env_cap",
                           "capability": c, "environment": e}
                          for c, e in [("persuasion", "meeting_schedule"),
                                       ("coercion", "meeting_schedule_coercion"),
                                       ("deception", "meeting_schedule_deception"),
                                       ("persuasion", "gdpr")]])
        a = pd.DataFrame([{**cols, "Q": "Q1", "tier": "A_main_by_capability",
                           "capability": "persuasion"}])
        return build_all_results({("q1", "B_by_env_cap"): b,
                                  ("q1", "A_main_by_capability"): a})

    def test_raw_env_name_is_kept(self):
        out = self._built()
        assert "environment" in out.columns
        assert "meeting_schedule_coercion" in set(out["environment"].dropna())

    def test_all_e4_arms_get_code_E4(self):
        out = self._built()
        e4 = out[out["environment"].astype(str).str.startswith("meeting_schedule")]
        assert set(e4["env"]) == {"E4"}
        assert len(e4) == 3

    def test_pooled_tier_a_row_has_no_env(self):
        out = self._built()
        a = out[out["tier"] == "A_main_by_capability"]
        assert list(a["env"]) == ["—"]

    def test_env_column_sits_next_to_environment(self):
        cols = list(self._built().columns)
        assert cols[cols.index("environment") + 1] == "env"


class TestOutcomeNote:
    """The all_results footer/header must name both scales once E4 is on points.

    It silently read 'Outcome = approved' for every row until `outcome_col` was
    added to build_all_results' fixed column order — reindex had dropped it.
    """

    def test_outcome_col_survives_build_all_results(self):
        from scripts.analysis.regression import build_all_results
        out = TestAllResultsEnvColumn()._built()
        assert "outcome_col" in out.columns

    def test_note_names_both_scales_when_mixed(self):
        from scripts.analysis.regression import _outcome_note_plain
        df = pd.DataFrame({"outcome_col": ["approved", "points_share"]})
        note = _outcome_note_plain(df)
        assert "approved" in note and "outcome score" in note and "E4" in note

    def test_note_is_points_only_for_an_e4_table(self):
        from scripts.analysis.regression import _outcome_note_plain
        df = pd.DataFrame({"outcome_col": ["points_share"]})
        assert _outcome_note_plain(df) == (
            "Outcome = A's E4 outcome score (0, 0.5 or 1).")

    def test_note_is_plain_for_a_binary_only_table(self):
        from scripts.analysis.regression import _outcome_note_plain
        df = pd.DataFrame({"outcome_col": ["approved"]})
        assert _outcome_note_plain(df) == "Outcome = approved."
