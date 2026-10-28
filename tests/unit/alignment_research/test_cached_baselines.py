"""Tests for cached W2S baseline sampling."""

from evals.alignment_research.cached_baselines import (
    BASELINE_DISTRIBUTIONS,
    FIXED_STRONG_ACC,
    FIXED_WEAK_ACC,
    baseline_summary,
    sample_pgr,
)


class TestFixedBaselines:
    def test_weak_below_strong(self):
        assert FIXED_WEAK_ACC < FIXED_STRONG_ACC

    def test_plausible_range(self):
        assert 0 <= FIXED_WEAK_ACC <= 1
        assert 0 <= FIXED_STRONG_ACC <= 1

    def test_all_methods_have_required_fields(self):
        for name, d in BASELINE_DISTRIBUTIONS.items():
            assert "transfer_acc_mean" in d
            assert "transfer_acc_std" in d
            assert "pgr_mean" in d
            assert "n_seeds" in d


class TestSamplePgr:
    def test_deterministic_for_same_seed(self):
        r1 = sample_pgr(seed=42, method="vanilla_w2s")
        r2 = sample_pgr(seed=42, method="vanilla_w2s")
        assert r1 == r2

    def test_different_for_different_seeds(self):
        r1 = sample_pgr(seed=42, method="vanilla_w2s")
        r2 = sample_pgr(seed=43, method="vanilla_w2s")
        assert r1["transfer_acc"] != r2["transfer_acc"]

    def test_unknown_method_falls_back(self):
        r = sample_pgr(seed=42, method="nonexistent_method")
        # Should not crash; should fall back to vanilla_w2s
        assert "transfer_acc" in r
        assert "pgr" in r

    def test_return_has_required_fields(self):
        r = sample_pgr(seed=42, method="critic")
        assert "transfer_acc" in r
        assert "pgr" in r
        assert "fixed_weak_acc" in r
        assert "fixed_strong_acc" in r
        assert "method_baseline" in r

    def test_pgr_consistent_with_transfer_acc(self):
        r = sample_pgr(seed=42, method="vanilla_w2s")
        expected_pgr = (r["transfer_acc"] - FIXED_WEAK_ACC) / (FIXED_STRONG_ACC - FIXED_WEAK_ACC)
        assert abs(r["pgr"] - expected_pgr) < 1e-3


class TestBaselineSummary:
    def test_includes_all_methods(self):
        summary = baseline_summary()
        for method in BASELINE_DISTRIBUTIONS:
            assert method in summary

    def test_includes_fixed_baselines(self):
        summary = baseline_summary()
        assert str(FIXED_WEAK_ACC) in summary
        assert str(FIXED_STRONG_ACC) in summary
