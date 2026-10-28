"""Unified data-quality summary.

A single entry point that synthesizes the per-stage audits from stages 01-04
into one table and one figure. Lets a fresh reader answer "what happened
to my data?" without jumping between four folders.

Inputs:
  data/pipeline/q1_filtered.parquet
  data/pipeline/q2_filtered.parquet     (= validity-passing master sample)
  data/pipeline/q1_subsample.parquet
  data/pipeline/q2_subsample.parquet

Outputs (top of analysis/):
  data_quality.csv         tidy long table, one row per (capability, T, stage)
  data_quality.md          human-readable wide table + balance summary
  data_quality.png/pdf     three rows (one per capability), three small panels
                            each: row-count waterfall · approval by stage · balance histogram
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import CELL_KEYS, load_master_samples
from .spec import (
    ANALYSIS_DIR,
    SUMMARY_DIR,
    CAPABILITIES,
    CAPABILITY_DISPLAY,
    DATA_DIR,
    N_PER_FEATURE_CELL_Q1,
    N_PER_FEATURE_CELL_Q2,
    Q1_COLOR,
    Q2_COLOR,
)


# ---------------------------------------------------------------------------
# Per-stage stats: rows + cell-level approval (mean ± SD across cells)
# ---------------------------------------------------------------------------


def _cell_mean_sd(df: pd.DataFrame) -> tuple[int, float, float]:
    """Compute (n_cells, mean of cell means, SD of cell means).

    Cell = (environment, capability, T, model_a, model_b).
    """
    if df.empty:
        return 0, float("nan"), float("nan")
    cell_means = (
        df.groupby(list(CELL_KEYS), dropna=False)["approved"].mean().to_numpy()
    )
    if cell_means.size == 0:
        return 0, float("nan"), float("nan")
    return int(cell_means.size), float(np.mean(cell_means)), float(np.std(cell_means, ddof=1)) if cell_means.size > 1 else 0.0


def _stage_record(stage: str, df: pd.DataFrame, capability: str, T: int) -> dict:
    """Sub-row of the data-quality table for one (capability, T, stage)."""
    sub = df[(df["capability"] == capability) & (df["T"] == T)]
    n_cells, cell_mean, cell_sd = _cell_mean_sd(sub)
    return {
        "capability": capability,
        "T": int(T),
        "stage": stage,
        "n_rows": int(len(sub)),
        "n_cells": n_cells,
        "cell_mean_approval": cell_mean,
        "cell_sd_approval": cell_sd,
        "pooled_approval": float(sub["approved"].mean()) if not sub.empty else float("nan"),
    }


def build_table(stages: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for stage_name, df in stages.items():
        for cap in CAPABILITIES:
            for T in (0, 1):
                rows.append(_stage_record(stage_name, df, cap, T))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subsample balance: MAE of approval per cell, full vs subsample
# ---------------------------------------------------------------------------


def cell_abs_diff(full: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
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
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


STAGE_ORDER = ["Raw", "Q1 post-filter", "Q1 subsample", "Q2 subsample"]


def render_markdown(table: pd.DataFrame,
                     q1_balance: pd.DataFrame,
                     q2_balance: pd.DataFrame) -> str:
    L: list[str] = []
    L.append("# Data quality at a glance")
    L.append("")
    L.append("**One table to answer**: how many rows, how variable across cells, "
             "and is the subsample representative — at every stage, by capability × T.")
    L.append("")
    L.append("- `cell mean ± SD` reports approval rate per cell "
             "(env × capability × T × A × B), then takes mean and SD across cells. "
             "Captures both the typical cell and how heterogeneous cells are.")
    L.append("- `pooled` is mean(approved) across all rows in that stage (weights big cells more).")
    L.append("")

    pivot_n = table.pivot_table(
        index=["capability", "T"], columns="stage",
        values="n_rows", aggfunc="first",
    ).reindex(columns=STAGE_ORDER)
    pivot_cells = table.pivot_table(
        index=["capability", "T"], columns="stage",
        values="n_cells", aggfunc="first",
    ).reindex(columns=STAGE_ORDER)
    pivot_mean = table.pivot_table(
        index=["capability", "T"], columns="stage",
        values="cell_mean_approval", aggfunc="first",
    ).reindex(columns=STAGE_ORDER)
    pivot_sd = table.pivot_table(
        index=["capability", "T"], columns="stage",
        values="cell_sd_approval", aggfunc="first",
    ).reindex(columns=STAGE_ORDER)
    pivot_pool = table.pivot_table(
        index=["capability", "T"], columns="stage",
        values="pooled_approval", aggfunc="first",
    ).reindex(columns=STAGE_ORDER)

    L.append("## Row counts and cell-level approval, by stage")
    L.append("")
    L.append("| capability | T | metric | " + " | ".join(STAGE_ORDER) + " |")
    L.append("|---|---|---|" + "---:|" * len(STAGE_ORDER))
    for (cap, T) in pivot_n.index:
        T_label = "baseline" if T == 0 else "treatment"
        # n_rows
        cells = [f"{int(pivot_n.loc[(cap, T), s]):,}" if pd.notna(pivot_n.loc[(cap, T), s]) else "—"
                 for s in STAGE_ORDER]
        L.append(f"| {cap} | {T_label} | n_rows | " + " | ".join(cells) + " |")
        # n_cells
        cells = [f"{int(pivot_cells.loc[(cap, T), s])}" if pd.notna(pivot_cells.loc[(cap, T), s]) else "—"
                 for s in STAGE_ORDER]
        L.append(f"| | | n_cells | " + " | ".join(cells) + " |")
        # cell mean ± SD
        cells = []
        for s in STAGE_ORDER:
            m = pivot_mean.loc[(cap, T), s]
            sd = pivot_sd.loc[(cap, T), s]
            cells.append(f"{m:.3f} ± {sd:.3f}" if pd.notna(m) else "—")
        L.append(f"| | | cell mean ± SD | " + " | ".join(cells) + " |")
        # pooled
        cells = [f"{pivot_pool.loc[(cap, T), s]:.3f}" if pd.notna(pivot_pool.loc[(cap, T), s]) else "—"
                 for s in STAGE_ORDER]
        L.append(f"| | | pooled | " + " | ".join(cells) + " |")
    L.append("")

    # Balance summary.
    L.append("## Subsample balance — `abs_diff` of approval rate (subsample vs full pre-subsample)")
    L.append("")
    L.append("Lower is better. MAE is the mean absolute deviation across cells; the 95th-pct "
             "is the deviation in the worst-balanced 5% of cells.")
    L.append("")
    L.append("| capability | T | Q1 cells | Q1 MAE | Q1 95pct | Q2 cells | Q2 MAE | Q2 95pct |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for cap in CAPABILITIES:
        for T in (0, 1):
            T_label = "baseline" if T == 0 else "treatment"
            q1 = q1_balance[(q1_balance["capability"] == cap) & (q1_balance["T"] == T)]
            q2 = q2_balance[(q2_balance["capability"] == cap) & (q2_balance["T"] == T)]
            q1c = q1.dropna(subset=["abs_diff"])
            q2c = q2.dropna(subset=["abs_diff"])
            row = (
                f"| {cap} | {T_label} "
                f"| {len(q1c)} "
                f"| {q1c['abs_diff'].mean():.4f} " if len(q1c) else f"| {cap} | {T_label} | 0 | — "
            )
            row += f"| {q1c['abs_diff'].quantile(0.95):.4f} " if len(q1c) else "| — "
            row += f"| {len(q2c)} "
            row += f"| {q2c['abs_diff'].mean():.4f} " if len(q2c) else "| — "
            row += f"| {q2c['abs_diff'].quantile(0.95):.4f} |" if len(q2c) else "| — |"
            L.append(row)
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Figure: per-capability dashboard
# ---------------------------------------------------------------------------


def plot_dashboard(table: pd.DataFrame,
                    q1_balance: pd.DataFrame,
                    q2_balance: pd.DataFrame,
                    out_path: Path) -> None:
    n_caps = len(CAPABILITIES)
    fig, axes = plt.subplots(
        n_caps, 3,
        figsize=(13.5, 3.2 * n_caps),
        gridspec_kw={"width_ratios": [2.0, 1.4, 1.4]},
        sharey="col",   # same y-axis range within each column
    )
    if n_caps == 1:
        axes = axes.reshape(1, -1)

    for row_i, cap in enumerate(CAPABILITIES):
        ax_water = axes[row_i, 0]
        ax_app = axes[row_i, 1]
        ax_bal = axes[row_i, 2]

        # ---- Panel 1: row-count waterfall (baseline + treatment side by side)
        cap_table = table[table["capability"] == cap].copy()
        for T, color, label in [(0, "#888888", "baseline"),
                                 (1, "#444444", "treatment")]:
            sub = cap_table[cap_table["T"] == T].set_index("stage").reindex(STAGE_ORDER)
            x = np.arange(len(STAGE_ORDER))
            offset = -0.18 if T == 0 else +0.18
            ax_water.bar(x + offset, sub["n_rows"].fillna(0).to_numpy(),
                          width=0.34, color=color, label=label)
        ax_water.set_xticks(np.arange(len(STAGE_ORDER)))
        ax_water.set_xticklabels(STAGE_ORDER, fontsize=8, rotation=10)
        ax_water.set_ylabel("rows")
        ax_water.set_title(f"{CAPABILITY_DISPLAY.get(cap, cap.title())} — row counts by stage",
                            fontsize=10)
        ax_water.legend(loc="upper right", fontsize=7, frameon=False)
        ax_water.grid(axis="y", linestyle=":", color="#bbb", lw=0.4)

        # ---- Panel 2: cell-mean approval (± SD across cells) by stage
        # Baseline and treatment offset side-by-side so error bars don't overlap.
        for T, color, label, marker, offset in [
            (0, "#666666", "baseline",  "o", -0.18),
            (1, "#d62728", "treatment", "s", +0.18),
        ]:
            sub = cap_table[cap_table["T"] == T].set_index("stage").reindex(STAGE_ORDER)
            mean = sub["cell_mean_approval"].to_numpy()
            sd = sub["cell_sd_approval"].to_numpy()
            x = np.arange(len(STAGE_ORDER)) + offset
            ax_app.errorbar(x, mean, yerr=sd, fmt=marker, color=color,
                             markersize=6, capsize=3, label=label, lw=1.0)
        ax_app.set_xticks(np.arange(len(STAGE_ORDER)))
        ax_app.set_xticklabels(STAGE_ORDER, fontsize=8, rotation=10)
        ax_app.set_ylabel("cell-level approval (mean ± SD)")
        ax_app.set_title("approval by stage", fontsize=10)
        ax_app.legend(loc="upper left", fontsize=7, frameon=False)
        ax_app.grid(axis="y", linestyle=":", color="#bbb", lw=0.4)
        ax_app.set_ylim(-0.05, 1.05)

        # ---- Panel 3: balance histogram (cells' abs_diff)
        q1c = q1_balance[q1_balance["capability"] == cap]["abs_diff"].dropna()
        q2c = q2_balance[q2_balance["capability"] == cap]["abs_diff"].dropna()
        bins = np.linspace(0, max(q1c.max() if len(q1c) else 0,
                                    q2c.max() if len(q2c) else 0,
                                    0.05) + 1e-6, 25)
        if len(q1c):
            ax_bal.hist(q1c, bins=bins, color=Q1_COLOR, alpha=0.6, label=f"Q1  (n={len(q1c)})")
            ax_bal.axvline(q1c.median(), color=Q1_COLOR, lw=1.2, linestyle=":")
        if len(q2c):
            ax_bal.hist(q2c, bins=bins, color=Q2_COLOR, alpha=0.6, label=f"Q2  (n={len(q2c)})")
            ax_bal.axvline(q2c.median(), color=Q2_COLOR, lw=1.2, linestyle=":")
        ax_bal.set_xlabel("|approval_subsample − approval_full|")
        ax_bal.set_ylabel("cells")
        ax_bal.set_title("subsample balance", fontsize=10)
        ax_bal.legend(loc="upper right", fontsize=7, frameon=False)
        ax_bal.grid(axis="y", linestyle=":", color="#bbb", lw=0.4)

    fig.suptitle("Data quality — per capability",
                  fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(data_dir: Path | None = None,
        out_dir: Path | None = None) -> dict[str, Path]:
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or SUMMARY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_master_samples()
    raw = raw[raw["capability"].isin(CAPABILITIES)].reset_index(drop=True)
    q1_filt = pd.read_parquet(data_dir / "q1_filtered.parquet")
    q1_sub = pd.read_parquet(data_dir / "q1_subsample.parquet")
    q2_sub = pd.read_parquet(data_dir / "q2_subsample.parquet")

    stages = {
        "Raw": raw,
        "Q1 post-filter": q1_filt,
        "Q1 subsample": q1_sub,
        "Q2 subsample": q2_sub,
    }

    table = build_table(stages)

    # Balance: subsample vs the corresponding full pre-subsample population.
    q1_balance = cell_abs_diff(q1_filt, q1_sub)
    q2_balance = cell_abs_diff(raw, q2_sub)
    # Annotate with capability + T for grouping.
    for b in (q1_balance, q2_balance):
        # Already has capability + T from CELL_KEYS.
        pass

    written: dict[str, Path] = {}
    p_csv = out_dir / "data_quality.csv"
    table.to_csv(p_csv, index=False)
    written["data_quality_csv"] = p_csv

    md = render_markdown(table, q1_balance, q2_balance)
    p_md = out_dir / "data_quality.md"
    p_md.write_text(md)
    written["data_quality_md"] = p_md

    plot_dashboard(table, q1_balance, q2_balance, out_dir / "data_quality")
    written["data_quality_png"] = out_dir / "data_quality.png"

    return written


if __name__ == "__main__":
    paths = run()
    for k, p in paths.items():
        print(f"  {k}: {p}")
