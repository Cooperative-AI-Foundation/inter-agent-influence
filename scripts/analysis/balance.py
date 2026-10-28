"""Stage 4: balance audits.

Two audits land in ``analysis/robustness/``:

  q1_vs_q2_approvals.csv   per (env, cap, T, A, B): approval rate and
                            sample size for Q1 (post-filter) vs Q2.
                            PNG: paired bar chart per capability × T.

  balance_q1.csv           per (env, cap, T, A, B): approval rate and N
  balance_q2.csv           comparing the SUBSAMPLE to the corresponding
                            FULL pre-subsample population. PNG: scatter
                            (full-rate vs subsample-rate, one dot per cell).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import CELL_KEYS
from .spec import (
    significance_stars,
    BALANCE_DIR,
    DATA_DIR,
    ENV_TO_CODE,
    env_code as _env_code,
    Q1_COLOR,
    Q2_COLOR,
)


# Capability ordering: persuasion → deception → coercion (paper convention).
_CAP_ORDER: dict[str, int] = {"persuasion": 0, "deception": 1, "coercion": 2}
_CAP_DISPLAY: dict[str, str] = {
    "persuasion": "Persuasion", "deception": "Deception", "coercion": "Coercion",
}
# This module restricts analysis to E1-E3.
_E1_E3_ENVS: frozenset[str] = frozenset(
    e for e, c in ENV_TO_CODE.items() if c in {"E1", "E2", "E3"}
)


def _cap_sort(cap: str) -> int:
    return _CAP_ORDER.get(str(cap), 99)


def _cap_disp(cap: str) -> str:
    return _CAP_DISPLAY.get(str(cap), str(cap).title())





def _restrict_e1_e3(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["environment"].isin(_E1_E3_ENVS)].reset_index(drop=True)


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return significance_stars(p)


# ---------------------------------------------------------------------------
# Q1 (post-filter) vs Q2 (full) approval comparison
# ---------------------------------------------------------------------------


def q1_vs_q2_approvals(q1_filt: pd.DataFrame, q2_full: pd.DataFrame) -> pd.DataFrame:
    keys = list(CELL_KEYS)
    g_q1 = q1_filt.groupby(keys, dropna=False).agg(
        n_q1=("approved", "size"),
        approval_q1=("approved", "mean"),
    ).reset_index()
    g_q2 = q2_full.groupby(keys, dropna=False).agg(
        n_q2=("approved", "size"),
        approval_q2=("approved", "mean"),
    ).reset_index()
    out = g_q2.merge(g_q1, on=keys, how="left")
    out["delta_q1_minus_q2"] = out["approval_q1"] - out["approval_q2"]
    return out


def plot_q1_vs_q2(comparison: pd.DataFrame, out_path: Path) -> None:
    """Per (capability, T) scatter of Q1 approval vs Q2 approval; one dot per cell."""
    from . import style
    from .spec import BASELINE_COLOR, CAPABILITIES, CAPABILITY_COLORS, CAPABILITY_DISPLAY
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(style.TEXT_WIDTH_IN * 0.9, 2.85), sharey=True)
    capabilities = sorted(comparison["capability"].unique()) if "capability" in comparison.columns else []
    if not capabilities:
        from .spec import env_to_capability
        comparison = comparison.copy()
        comparison["capability"] = comparison["environment"].apply(env_to_capability)
        capabilities = sorted(comparison["capability"].unique())
    capabilities = [c for c in CAPABILITIES if c in capabilities]

    for ax, T in zip(axes, [0, 1]):
        ax.plot([0, 1], [0, 1], color="#3a3a3a", lw=style.LW_REF, linestyle=(0, (2, 2)), zorder=1)
        if T == 0:
            # The baseline is shared by all three arms, so each cell appears
            # once per capability with identical values; plot it once.
            sub = comparison[(comparison["T"] == 0)
                             & (comparison["capability"] == capabilities[0])]
            ax.scatter(
                sub["approval_q2"], sub["approval_q1"],
                s=12, alpha=0.85, label="Baseline (shared by all arms)",
                color=BASELINE_COLOR, lw=0, zorder=3,
            )
        else:
            for capability in capabilities:
                sub = comparison[(comparison["capability"] == capability) & (comparison["T"] == T)]
                ax.scatter(
                    sub["approval_q2"], sub["approval_q1"],
                    s=12, alpha=0.85,
                    label=CAPABILITY_DISPLAY.get(capability, capability.title()),
                    color=CAPABILITY_COLORS.get(capability, BASELINE_COLOR), lw=0, zorder=3,
                )
        ax.set_xlabel("Q2 Approval Rate (Full Sample)")
        if T == 0:
            ax.set_ylabel("Q1 Approval Rate (Post-Filter)")
        ax.set_title("Baseline (T = 0)" if T == 0 else "Treatment (T = 1)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        style.tidy_axes(ax, grid_axis="both")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               handletextpad=0.3, markerscale=1.3, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.12, left=0.1, right=0.99, top=0.92, bottom=0.27)
    style.save(fig, out_path)


# ---------------------------------------------------------------------------
# Subsample vs full-population balance
# ---------------------------------------------------------------------------


def subsample_balance(full: pd.DataFrame, sub: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per cell, compare approval rate in full population vs subsample."""
    keys = list(CELL_KEYS)
    g_full = full.groupby(keys, dropna=False).agg(
        n_full=("approved", "size"),
        approval_full=("approved", "mean"),
    ).reset_index()
    g_sub = sub.groupby(keys, dropna=False).agg(
        n_sub=("approved", "size"),
        approval_sub=("approved", "mean"),
    ).reset_index()
    out = g_full.merge(g_sub, on=keys, how="left")
    out["abs_diff"] = (out["approval_sub"] - out["approval_full"]).abs()
    out["sample"] = label
    from .spec import env_to_capability
    out["capability"] = out["environment"].apply(env_to_capability)
    return out


def plot_balance(bal: pd.DataFrame, out_path: Path, label: str,
                  q_color: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    sub = bal.dropna(subset=["approval_full", "approval_sub"])
    ax.scatter(
        sub["approval_full"], sub["approval_sub"],
        s=22, alpha=0.7, color=q_color, edgecolor="black", linewidth=0.3,
    )
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle="--",
            label="y = x (perfect balance)")
    mae = sub["abs_diff"].mean()
    ax.set_xlabel("approval rate — full pre-subsample population")
    ax.set_ylabel(f"approval rate — {label} subsample")
    ax.set_title(f"{label} subsample balance — MAE = {mae:.3f}", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_balance_latex(q1_bal: pd.DataFrame, q2_bal: pd.DataFrame) -> str:
    """Paper-ready LaTeX: subsample-vs-population per-(capability, env, T) abs_diff.

    Capability rows are ordered Persuasion → Deception → Coercion; env in
    E1-E5 codes; treatment in $\\{0, 1\\}$.
    """
    L: list[str] = []
    L.append("% Requires preamble: \\usepackage{booktabs}")
    L.append("\\begin{table}[!ht]")
    L.append("\\caption{\\textbf{Subsample vs.\\ full-population balance.} Per-cell "
             "absolute difference in approval rate, summarised by capability, "
             "environment, and treatment. $|\\Delta|$ is the absolute "
             "difference between the cell's approval rate in the subsample "
             "and in the corresponding full pre-subsample population. Q1 "
             "reference population $=$ post-Q1-filter sample; Q2 reference "
             "population $=$ full validity-passing sample.}")
    L.append("\\label{tab:subsample_balance}")
    L.append("\\centering")
    L.append("\\footnotesize")
    L.append("\\begin{tabular}{lllcrrr}")
    L.append("\\toprule")
    L.append("\\textbf{Sample} & \\textbf{Capability} & \\textbf{Env} & $\\bm{T}$ & \\textbf{Cells} & "
             "\\textbf{Mean} $\\bm{|\\Delta|}$ & \\textbf{Max} $\\bm{|\\Delta|}$ \\\\")
    L.append("\\midrule")
    for label, bal in [("Q1", q1_bal), ("Q2", q2_bal)]:
        if bal is None or bal.empty:
            continue
        sub = bal.dropna(subset=["approval_full", "approval_sub"]).copy()
        if sub.empty:
            continue
        sub["_env"] = sub["environment"].map(_env_code)
        agg = sub.groupby(["capability", "_env", "T"]).agg(
            cells=("abs_diff", "size"),
            mean_abs_diff=("abs_diff", "mean"),
            max_abs_diff=("abs_diff", "max"),
        ).reset_index()
        agg["_cap_order"] = agg["capability"].map(_cap_sort)
        agg = agg.sort_values(["_cap_order", "_env", "T"]).drop(columns="_cap_order")
        first = True
        for _, r in agg.iterrows():
            sample_cell = label if first else ""
            first = False
            L.append(
                f"{sample_cell} & {_cap_disp(r['capability'])} & {r['_env']} & "
                f"{int(r['T'])} & {int(r['cells']):,} & "
                f"{r['mean_abs_diff']:.4f} & {r['max_abs_diff']:.4f} \\\\"
            )
        L.append("\\midrule")
    if L[-1] == "\\midrule":
        L.pop()
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


def render_balance_markdown(q1_bal: pd.DataFrame, q2_bal: pd.DataFrame) -> str:
    L: list[str] = []
    L.append("# Subsample balance audit")
    L.append("")
    L.append("Per cell: approval rate in the full pre-subsample population vs. "
             "in the subsample. `abs_diff` is the absolute difference; "
             "small numbers mean the subsample is representative.")
    L.append("")
    for label, bal in [("Q1", q1_bal), ("Q2", q2_bal)]:
        if bal.empty:
            continue
        sub = bal.dropna(subset=["approval_full", "approval_sub"])
        L.append(f"## {label}")
        L.append(f"- cells with both samples: {len(sub):,}")
        L.append(f"- mean abs_diff: {sub['abs_diff'].mean():.4f}")
        L.append(f"- median abs_diff: {sub['abs_diff'].median():.4f}")
        L.append(f"- 95th-percentile abs_diff: {sub['abs_diff'].quantile(0.95):.4f}")
        L.append("")
        # Per-capability summary.
        agg = sub.groupby(["capability", "T"]).agg(
            cells=("abs_diff", "size"),
            mean_abs_diff=("abs_diff", "mean"),
            max_abs_diff=("abs_diff", "max"),
        ).reset_index()
        agg["_cap_order"] = agg["capability"].map(_cap_sort)
        agg = agg.sort_values(["_cap_order", "T"]).drop(columns="_cap_order")
        L.append("| capability | T | cells | mean abs_diff | max abs_diff |")
        L.append("|---|---:|---:|---:|---:|")
        for _, r in agg.iterrows():
            L.append(f"| {_cap_disp(r['capability'])} | {int(r['T'])} | {int(r['cells']):,} "
                     f"| {r['mean_abs_diff']:.4f} | {r['max_abs_diff']:.4f} |")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(data_dir: Path | None = None,
        out_dir: Path | None = None) -> dict[str, Path]:
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or BALANCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    q1_filt = pd.read_parquet(data_dir / "q1_filtered.parquet")
    q2_full = pd.read_parquet(data_dir / "q2_filtered.parquet")
    q1_sub = pd.read_parquet(data_dir / "q1_subsample.parquet")
    q2_sub = pd.read_parquet(data_dir / "q2_subsample.parquet")

    # All four parquets already carry the (derived) `capability` column
    # from io.load_master_samples; just compute the comparison.
    # The Q1-vs-Q2 comparison is restricted to E1-E3 (the envs shared across
    # all three capability arms); Q2-only persuasion envs (E4, E5) have no Q1
    # counterpart, so they're not part of this comparison.
    cmp = q1_vs_q2_approvals(
        _restrict_e1_e3(q1_filt), _restrict_e1_e3(q2_full),
    )
    from .spec import env_to_capability
    cmp["capability"] = cmp["environment"].apply(env_to_capability)
    written: dict[str, Path] = {}
    p = out_dir / "q1_vs_q2_approvals.csv"
    cmp.to_csv(p, index=False)
    written["q1_vs_q2_csv"] = p
    plot_q1_vs_q2(cmp, out_dir / "q1_vs_q2_approvals")
    written["q1_vs_q2_png"] = out_dir / "q1_vs_q2_approvals.png"

    # Stage 4b: subsample balance vs the corresponding full population
    # (For Q1: full = q1_filt (pre-subsample, post-filter). For Q2: full = q2_full.)
    q1_bal = subsample_balance(q1_filt, q1_sub, label="Q1")
    q2_bal = subsample_balance(q2_full, q2_sub, label="Q2")
    p1 = out_dir / "balance_q1.csv"
    p2 = out_dir / "balance_q2.csv"
    q1_bal.to_csv(p1, index=False)
    q2_bal.to_csv(p2, index=False)
    written["balance_q1_csv"] = p1
    written["balance_q2_csv"] = p2

    plot_balance(q1_bal, out_dir / "balance_q1", "Q1", Q1_COLOR)
    plot_balance(q2_bal, out_dir / "balance_q2", "Q2", Q2_COLOR)
    written["balance_q1_png"] = out_dir / "balance_q1.png"
    written["balance_q2_png"] = out_dir / "balance_q2.png"

    md = render_balance_markdown(q1_bal, q2_bal)
    p_md = out_dir / "balance.md"
    p_md.write_text(md)
    written["balance_md"] = p_md

    p_tex = out_dir / "balance.tex"
    p_tex.write_text(render_balance_latex(q1_bal, q2_bal))
    written["balance_tex"] = p_tex

    return written


if __name__ == "__main__":
    paths = run()
    for k, p in paths.items():
        print(f"  {k}: {p}")
