"""E5 appendix tables rebuild from the released per-sample table and the ablation pilot file."""

import pandas as pd

from scripts.analysis import aar_tables


def test_wilson_ci_matches_the_published_row():
    lo, hi = aar_tables.wilson_ci(3, 15)
    assert (round(lo, 3), round(hi, 3)) == (0.070, 0.452)


def test_samples_are_labelled_by_arm_and_level():
    df = aar_tables.load_samples()
    assert set(df["cond"].unique()) == {"baseline", "L1", "L3"}
    assert (df.loc[df.condition == "baseline", "cond"] == "baseline").all()
    assert (df.loc[(df.condition == "attack") & (df.instruction_level == 1), "cond"] == "L1").all()


def test_pair_tables_use_the_paper_arm_names_and_targets():
    df = aar_tables.load_samples()
    rate = aar_tables.pair_rate_table(df)
    counts = aar_tables.pair_n_table(df)
    for tex in (rate, counts):
        assert "\\textbf{Goal-only} & \\textbf{Scaffolded}" in tex
        assert "L1" not in tex and "L3" not in tex
    assert counts.count("\\texttt{") == 2 * len(aar_tables.A_ORDER) * len(aar_tables.B_ORDER) - 2 * len(aar_tables.A_ORDER)


def test_ablation_table_pools_t16_and_adds_the_reference_row():
    raw = pd.read_csv(aar_tables.ABLATION_RAW)
    tex = aar_tables.turns_ablation_table(raw, aar_tables.load_samples())
    assert f"$16$ & {int((raw.max_turns == 16).sum())} &" in tex
    assert "$16^{\\dagger}$ & 102 &" in tex
