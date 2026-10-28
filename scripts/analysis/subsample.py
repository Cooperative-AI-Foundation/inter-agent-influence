"""Stage 3: feature restriction + per-feature-cell subsampling.

Inputs:
  data/pipeline/q1_filtered.parquet   (Q1 = post-Q1-filter)
  data/pipeline/q2_filtered.parquet   (Q2 = full validity sample)

Steps:
  1. Apply ``COERCION_FEATURE_KEEP`` — drop coercion rows whose feature
     isn't the canonical one for that env. (No-op while the map is empty.)
  2. Per feature-level-cell (env × feature × cap × T × A × B), randomly
     sample without replacement to reach the per-Q target. Cells with
     fewer rows are kept in full.
  3. Q1 and Q2 are sampled INDEPENDENTLY (Q1 sample is not a subset of
     the Q2 sample).

Outputs:
  data/pipeline/q1_subsample.parquet
  data/pipeline/q2_subsample.parquet
  analysis/robustness/subsample_audit.csv
  analysis/robustness/subsample_audit.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import SUBCELL_KEYS
from .spec import (
    COERCION_FEATURE_KEEP,
    DATA_DIR,
    N_PER_FEATURE_CELL_Q1,
    N_PER_FEATURE_CELL_Q2,
    RANDOM_SEED,
    SUBSAMPLE_DIR,
)


def restrict_coercion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop coercion rows whose feature_name isn't the canonical one for that env."""
    if df.empty:
        return df
    is_coercion = df["capability"] == "coercion"
    keeps = pd.Series(True, index=df.index)
    for env, feat in COERCION_FEATURE_KEEP.items():
        in_env = df["environment"] == env
        bad = is_coercion & in_env & (df["feature_name"] != feat)
        keeps = keeps & ~bad
    return df[keeps].reset_index(drop=True)


def subsample_per_feature_cell(
    df: pd.DataFrame, n_target: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each feature-level-cell, sample at most ``n_target`` rows.

    Returns (subsampled_df, audit_df) where audit_df has one row per
    feature-level-cell with target / pre / post counts.
    """
    if df.empty:
        empty_audit = pd.DataFrame(columns=list(SUBCELL_KEYS) + [
            "n_pre_subsample", "n_target", "n_post_subsample", "hit_target",
        ])
        return df.copy(), empty_audit

    rng_seed = seed
    chunks: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    # Stable group order for reproducibility.
    grouped = df.groupby(list(SUBCELL_KEYS), dropna=False, sort=True)

    for keys, g in grouped:
        n_pre = int(len(g))
        if n_pre <= n_target:
            sampled = g
        else:
            sampled = g.sample(n=n_target, random_state=rng_seed,
                               replace=False)
            # Bump seed deterministically per cell so cells aren't correlated.
            rng_seed += 1
        chunks.append(sampled)
        rec = dict(zip(SUBCELL_KEYS, keys))
        rec["n_pre_subsample"] = n_pre
        rec["n_target"] = int(n_target)
        rec["n_post_subsample"] = int(len(sampled))
        rec["hit_target"] = bool(len(sampled) >= n_target)
        audit_rows.append(rec)

    out = pd.concat(chunks, ignore_index=True) if chunks else df.iloc[0:0].copy()
    audit = pd.DataFrame(audit_rows)
    return out, audit


def render_audit_markdown(q1_audit: pd.DataFrame, q2_audit: pd.DataFrame) -> str:
    L: list[str] = []
    L.append("# Subsample audit")
    L.append("")
    L.append("Per feature-level-cell counts before and after subsampling. "
             "`hit_target` is True if the cell reached its `n_target`.")
    L.append("")
    for label, audit, target in [
        ("Q1", q1_audit, N_PER_FEATURE_CELL_Q1),
        ("Q2", q2_audit, N_PER_FEATURE_CELL_Q2),
    ]:
        if audit.empty:
            L.append(f"## {label} — no rows in input")
            L.append("")
            continue
        n_cells = len(audit)
        n_hit = int(audit["hit_target"].sum())
        n_total_pre = int(audit["n_pre_subsample"].sum())
        n_total_post = int(audit["n_post_subsample"].sum())
        L.append(f"## {label}  (target = {target} per feature-level-cell)")
        L.append(f"- feature-level-cells: {n_cells:,}")
        L.append(f"- cells reaching target: {n_hit:,}  ({n_hit / n_cells:.1%})")
        L.append(f"- total rows: pre={n_total_pre:,}  →  post={n_total_post:,}")
        L.append("")
        agg = audit.groupby(["capability", "environment", "T"]).agg(
            n_cells=("n_pre_subsample", "size"),
            n_hit=("hit_target", "sum"),
            rows_pre=("n_pre_subsample", "sum"),
            rows_post=("n_post_subsample", "sum"),
        ).reset_index()
        L.append("| capability | env | T | cells | hit-target | rows pre | rows post |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in agg.iterrows():
            L.append(
                f"| {r['capability']} | {r['environment']} | {int(r['T'])} "
                f"| {int(r['n_cells']):,} "
                f"| {int(r['n_hit']):,} "
                f"| {int(r['rows_pre']):,} "
                f"| {int(r['rows_post']):,} |"
            )
        L.append("")
    return "\n".join(L) + "\n"


def run(data_dir: Path | None = None,
        out_dir: Path | None = None) -> dict[str, Path]:
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or SUBSAMPLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    q1_in = pd.read_parquet(data_dir / "q1_filtered.parquet")
    q2_in = pd.read_parquet(data_dir / "q2_filtered.parquet")

    # Coercion feature restriction is applied to BOTH Q1 and Q2 so the two
    # samples have a consistent feature universe.
    q1_in = restrict_coercion_features(q1_in)
    q2_in = restrict_coercion_features(q2_in)

    q1_sub, q1_audit = subsample_per_feature_cell(
        q1_in, N_PER_FEATURE_CELL_Q1, seed=RANDOM_SEED,
    )
    q2_sub, q2_audit = subsample_per_feature_cell(
        q2_in, N_PER_FEATURE_CELL_Q2, seed=RANDOM_SEED + 10_000,
    )

    # The audit already carries `capability` from SUBCELL_KEYS — nothing
    # to annotate.

    written: dict[str, Path] = {}
    p_q1 = data_dir / "q1_subsample.parquet"
    p_q2 = data_dir / "q2_subsample.parquet"
    q1_sub.to_parquet(p_q1, index=False)
    q2_sub.to_parquet(p_q2, index=False)
    written["q1_subsample"] = p_q1
    written["q2_subsample"] = p_q2

    audit_combined = pd.concat([
        q1_audit.assign(Q="Q1"),
        q2_audit.assign(Q="Q2"),
    ], ignore_index=True)
    p_audit = out_dir / "subsample_audit.csv"
    audit_combined.to_csv(p_audit, index=False)
    written["subsample_audit_csv"] = p_audit

    md = render_audit_markdown(q1_audit, q2_audit)
    p_md = out_dir / "subsample_audit.md"
    p_md.write_text(md)
    written["subsample_audit_md"] = p_md

    return written


if __name__ == "__main__":
    paths = run()
    for k, p in paths.items():
        print(f"  {k}: {p}")
