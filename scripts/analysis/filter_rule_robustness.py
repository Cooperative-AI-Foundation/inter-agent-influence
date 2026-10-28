"""Robustness: the Q1 deception rule as an exclusion (strict) vs a conjunction (AND).

Q1 keeps a persuasion / coercion / baseline row only when the false-claim
scorer found nothing AND the self-report probe did not admit. That excludes a
row when EITHER signal fires. The 2026-09 audit floated the alternative of
excluding only when BOTH fire, so the exclusion would mirror the deception
arm's inclusion rule (see the paper's scoring appendix).

The strict rule is the spec; this module scores the alternative alongside it so
the choice is a reported number rather than an assumption. It runs the same path
as the pipeline — feature restriction, then per-feature-cell subsampling at the
same seed — so the estimates are comparable to the published Tier A.

Outputs (analysis/02_filter/):
    filter_rule_robustness.csv
    filter_rule_robustness.md
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from .filter import apply_q1_filter
from .io import load_master_samples
from .regression import fit_tier
from .signals import false_claim_count
from .spec import (
    BASELINE_RULES,
    CAPABILITIES,
    FILTER_DIR,
    N_PER_FEATURE_CELL_Q1,
    RANDOM_SEED,
    SELF_REPORT_COL,
    TIER_A,
    TREATMENT_RULES,
)
from .subsample import restrict_coercion_features, subsample_per_feature_cell

STRICT = "strict (exclude if either fires)"
CONJUNCTION = "AND (exclude only if both fire)"


def _without_deception_conditions(rule):
    return dataclasses.replace(rule, require_self_report=None, require_false_claim=None)


def and_rule_q1(df: pd.DataFrame) -> pd.DataFrame:
    """Q1 under the conjunction rule.

    Every other Q1 condition is unchanged: the two deception signals are removed
    from the rules, the filter runs, and rows are then dropped only where BOTH
    fire. The deception treatment arm is untouched — its rule is already a
    conjunction, and this leaves it that way.
    """
    loose_bl = {k: _without_deception_conditions(v) for k, v in BASELINE_RULES.items()}
    loose_tx = {
        k: {c: _without_deception_conditions(r) for c, r in v.items()}
        for k, v in TREATMENT_RULES.items()
    }
    kept, _ = apply_q1_filter(df, baseline_rules=loose_bl, treatment_rules=loose_tx)

    fc = pd.to_numeric(false_claim_count(df), errors="coerce").loc[kept.index]
    sr = df[SELF_REPORT_COL].loc[kept.index]
    both = fc.gt(0).fillna(False) & sr.eq("b").fillna(False)
    deception_treatment = kept["capability"].eq("deception") & kept["T"].eq(1)
    return kept[(~deception_treatment & ~both) | (deception_treatment & both)]


def _tier_a(q1: pd.DataFrame) -> pd.DataFrame:
    """Published Tier A path: restrict coercion features, subsample, fit."""
    sub, _ = subsample_per_feature_cell(
        restrict_coercion_features(q1), N_PER_FEATURE_CELL_Q1, RANDOM_SEED)
    return fit_tier(sub, TIER_A)


def compare() -> pd.DataFrame:
    df = load_master_samples()
    df = df[df["capability"].isin(CAPABILITIES)].reset_index(drop=True)

    rows: list[dict] = []
    for label, q1 in ((STRICT, apply_q1_filter(df)[0]), (CONJUNCTION, and_rule_q1(df))):
        counts = q1.groupby(["capability", "T"]).size()
        for _, r in _tier_a(q1).iterrows():
            cap = r["capability"]
            rows.append({
                "rule": label,
                "capability": cap,
                "q1_rows_total": int(len(q1)),
                "q1_rows_treated": int(counts.get((cap, 1), 0)),
                "q1_rows_control": int(counts.get((cap, 0), 0)),
                "n_fitted": int(r["n"]),
                "control_mean": r.get("control_mean"),
                "treated_mean": r.get("treated_mean"),
                "beta": r["beta"],
                "se": r["se"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "p": r.get("p"),
            })
    out = pd.DataFrame(rows)
    wide = out.pivot(index="capability", columns="rule", values="beta")
    out = out.merge(
        (wide[CONJUNCTION] - wide[STRICT]).rename("beta_delta_vs_strict").reset_index(),
        on="capability", how="left")
    return out


def render_md(t: pd.DataFrame) -> str:
    lines = [
        "# Q1 deception rule — strict vs conjunction", "",
        "The spec uses the STRICT rule: a persuasion / coercion / baseline row is",
        "kept only when no false claim was found AND the self-report probe did not",
        "admit, so either signal excludes it. The CONJUNCTION variant excludes a row",
        "only when both fire, mirroring the deception arm's inclusion rule.", "",
        "Same feature restriction, same subsample seed, same specification as the",
        "published Tier A, so the two columns are directly comparable.", "",
    ]
    tbl = pd.DataFrame({
        "capability": sorted(t.capability.unique()),
    }).set_index("capability")
    for label, short in ((STRICT, "strict"), (CONJUNCTION, "AND")):
        g = t[t.rule == label].set_index("capability")
        tbl[f"treated rows ({short})"] = g.q1_rows_treated
        tbl[f"beta ({short})"] = g.beta.round(4)
        tbl[f"se ({short})"] = g.se.round(4)
    lines += [tbl.reset_index().to_markdown(index=False), ""]
    lines += ["## Change in the treatment effect", ""]
    for cap in sorted(t.capability.unique()):
        s = t[(t.capability == cap) & (t.rule == STRICT)].beta.iloc[0]
        c = t[(t.capability == cap) & (t.rule == CONJUNCTION)].beta.iloc[0]
        pct = f"{(c - s) / s:+.1%}" if s else "n/a"
        lines.append(f"- **{cap}**: {s:.4f} -> {c:.4f} ({c - s:+.4f}, {pct})")
    return "\n".join(lines) + "\n"


def run(out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or FILTER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t = compare()
    csv_path = out_dir / "filter_rule_robustness.csv"
    md_path = out_dir / "filter_rule_robustness.md"
    t.to_csv(csv_path, index=False)
    md_path.write_text(render_md(t))
    strict = t[t.rule == STRICT].q1_rows_total.iloc[0]
    conj = t[t.rule == CONJUNCTION].q1_rows_total.iloc[0]
    print(f"  filter rule robustness: Q1 {strict:,} (strict) vs {conj:,} (AND); "
          f"largest |delta beta| = {t.beta_delta_vs_strict.abs().max():.4f}")
    return {"filter_rule_robustness_csv": csv_path, "filter_rule_robustness_md": md_path}


if __name__ == "__main__":
    for k, v in run().items():
        print(f"  {k}: {v}")
