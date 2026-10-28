"""Technique frequency and difference-from-baseline figures — one per capability.

Each figure has two subplots sharing the x-axis (techniques):

  Top — Frequency by A model
    Grouped bar chart: proportion of treatment rows where each technique
    was used, broken down by model_a.

  Bottom — Difference from baseline by A model (grouped bar chart)
    For each (model_a, technique): mean outcome when technique is used minus
    the capability-wide baseline (T=0) mean outcome.
    Bars with n < MIN_N are shown at reduced alpha.
    Reference line at y=0 (= no difference from baseline).

Main figures cover E1-E3 on approval; the ``_e4`` figures cover E4 on the
negotiation points share.

Outputs: analysis/technique_analysis/technique_{capability}[_e4].png/pdf
Run:     python -m scripts.analysis.technique_plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import add_e4_points_share, backfill_no_deal_points, technique_columns_for_capability
from . import style
from .leaderboard_by_target import _model_label
from .spec import (
    CAPABILITIES,
    DATA_DIR,
    E4_ENVS,
    E4_OUTCOME_COL,
    MODEL_COLORS,
    PALETTE,
    TIER_A_EXCLUDE_ENVS,
    TECHNIQUE_ANALYSIS_DIR,
    TECHNIQUES_BY_ARM,
)

MIN_N = 10
LOW_ALPHA = 0.3
FULL_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _technique_display(capability: str) -> dict[str, str]:
    """Map tech_<slug> column name → canonical display name for a capability."""
    def _slug(name: str) -> str:
        return (
            name.lower()
                .replace("(", "").replace(")", "")
                .replace("-", "_").replace(" ", "_")
        )
    return {f"tech_{_slug(t)}": t for t in TECHNIQUES_BY_ARM[capability]}


_FALLBACK = [PALETTE[k] for k in ("blue", "orange", "green", "yellow", "magenta",
                                  "violet", "red", "dkgreen")]


def _model_colours(all_models: list[str]) -> dict[str, str]:
    """Fixed colour per attacker from spec.MODEL_COLORS; unlisted models take the
    next free palette slot so a hue is never reused within one figure."""
    out: dict[str, str] = {}
    used = set()
    for m in sorted(all_models):
        c = MODEL_COLORS.get(m)
        if c is None or c in used:
            c = next(x for x in _FALLBACK if x not in used)
        out[m] = c
        used.add(c)
    return out


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------


def plot_technique_figure(
    capability: str,
    df: pd.DataFrame,
    out_path: Path,
    *,
    outcome_col: str = "approved",
    scope: str = "E1-E3",
) -> None:
    style.apply()
    tech_cols = technique_columns_for_capability(capability)
    display = _technique_display(capability)
    tech_labels = [display[c] for c in tech_cols]
    n_techs = len(tech_cols)

    cap_df = df[df["capability"] == capability]
    treat = cap_df[cap_df["T"] == 1]
    baseline_rate = cap_df[cap_df["T"] == 0][outcome_col].mean()

    all_models = sorted(treat["model_a"].unique())
    colours = _model_colours(all_models)
    n_models = len(all_models)

    # Drawn at the printed width. The two panels always get the same height;
    # the figure grows below them by however much the rotated technique
    # labels (which differ in length between capabilities) and the legend need.
    import math
    longest_in = max(len(t) for t in tech_labels) * 0.42 * style.FS_BASE / 72
    labels_in = longest_in * math.sin(math.radians(30)) + 0.04
    legend_in = 0.42
    axes_in, top_in = 2.95, 0.08
    bottom_in = labels_in + 0.16 + legend_in
    fig_h = axes_in + top_in + bottom_in
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(style.TEXT_WIDTH_IN, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.22},
    )

    x = np.arange(n_techs)
    group_w = 0.78
    bar_w = group_w / n_models

    # ---- top: frequency ----
    for i, model_a in enumerate(all_models):
        sub = treat[treat["model_a"] == model_a]
        freqs = [sub[c].mean() if len(sub) > 0 else 0.0 for c in tech_cols]
        offset = (i - n_models / 2 + 0.5) * bar_w
        ax_top.bar(x + offset, freqs, bar_w * 0.92, color=colours[model_a],
                   alpha=FULL_ALPHA, label=_model_label(model_a), lw=0, zorder=3)

    ax_top.set_ylabel("Frequency\n(Share of Treatment Transcripts)")
    ax_top.set_ylim(0, 1)
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    style.tidy_axes(ax_top, grid_axis="y")

    # ---- bottom: difference from baseline ----
    ax_bot.axhline(0, color="#3a3a3a", lw=style.LW_REF, zorder=2)
    for i, model_a in enumerate(all_models):
        sub = treat[treat["model_a"] == model_a]
        offset = (i - n_models / 2 + 0.5) * bar_w
        for j, col in enumerate(tech_cols):
            used = sub[sub[col] == True]
            n = len(used)
            if n == 0:
                continue
            uplift = used[outcome_col].mean() - baseline_rate
            alpha = FULL_ALPHA if n >= MIN_N else LOW_ALPHA
            ax_bot.bar(x[j] + offset, uplift, bar_w * 0.92, color=colours[model_a],
                       alpha=alpha, lw=0, zorder=3)

    unit = ("E4 Outcome Score" if outcome_col == E4_OUTCOME_COL else "Approval Rate")
    ax_bot.set_ylabel(f"Difference from Baseline\n({unit})")
    ax_bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    style.tidy_axes(ax_bot, grid_axis="y")

    # ---- shared x-axis ----
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(tech_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax_bot.set_xlim(-0.6, n_techs - 0.4)
    ax_bot.tick_params(axis="x", length=0)
    ax_top.tick_params(axis="x", length=0)

    # ---- legend (below the x tick labels) ----
    fig.subplots_adjust(left=0.6 / style.TEXT_WIDTH_IN, right=0.995,
                        top=1 - top_in / fig_h, bottom=bottom_in / fig_h)
    handles, labels = ax_top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(n_models, 6), title="Attacker Model (Agent A)",
               handlelength=1.0, handleheight=1.0, columnspacing=1.2, handletextpad=0.5)

    style.save(fig, out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    out_dir: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or TECHNIQUE_ANALYSIS_DIR
    data_dir = data_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_e4_points_share(backfill_no_deal_points(
        pd.read_parquet(data_dir / "q2_filtered.parquet")))
    e1_e3 = df[~df["environment"].isin(TIER_A_EXCLUDE_ENVS)]
    e4 = df[df["environment"].isin(E4_ENVS)]

    written: dict[str, Path] = {}
    for capability in CAPABILITIES:
        for suffix, sub, kw in (
            ("", e1_e3, {}),
            ("_e4", e4, {"outcome_col": E4_OUTCOME_COL, "scope": "E4"}),
        ):
            if sub[sub["capability"] == capability].empty:
                continue
            out = out_dir / f"technique_{capability}{suffix}"
            plot_technique_figure(capability, sub, out, **kw)
            written[f"technique_{capability}{suffix}_png"] = out.with_suffix(".png")
            print(f"  wrote {out.with_suffix('.png').relative_to(out.parents[2])}")

    return written


if __name__ == "__main__":
    paths = run()
    print(f"\n{len(paths)} figures written.")
