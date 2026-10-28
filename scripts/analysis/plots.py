"""Stage 6: forest plots — one figure per capability (paper Figs 4 and 5).

Each figure has one panel per environment. Within a panel there is one row
per (model_a, model_b) pair, grouped by attacker (model_a): the attacker is
named once, to the left of its block, and the three rows of a block are the
three target models (model_b) in a fixed alphabetical order that the figure
caption states. Each row carries two estimates with 95% HC1 CIs: Q1 (orange)
and Q2 (blue). A filled marker is significant at Bonferroni-corrected
p < SIG_ALPHA, an open one is not, and a faded row has fewer observations than
the cell's quota. The pooled (Tier B) estimate for the environment sits in the
top row, and dotted vertical lines repeat it so each pair can be read against
it.

``render_forest`` is the single renderer; ``standalone_env`` (Fig 2) and
``leaderboard_by_target`` (Figs 11-13) share its conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .spec import (
    CAPABILITIES,
    ENVIRONMENT_DISPLAY,
    ENV_BY_CAP,
    FOREST_PAIR_ENVS_EXCLUDE,
    N_PER_FEATURE_CELL_Q1,
    N_PER_FEATURE_CELL_Q2,
    PLOTS_DIR,
    Q1_COLOR,
    Q2_COLOR,
    REGRESSIONS_DIR,
    ROW_SORT,
    SIG_ALPHA,
    effect_axis_label,
)


# ---------------------------------------------------------------------------
# Per-cell expected target — for the fade decision
# ---------------------------------------------------------------------------


def cell_target_n(env: str, capability: str, q_label: str, full_q2: pd.DataFrame) -> int:
    """Expected N for a Tier-C regression in this (env, capability), under the spec.

    target = (#baseline features + #treatment features) × per_cell, where
    per_cell = 40 (Q1) or 100 (Q2). Baseline and treatment features are
    counted separately so the target reflects the cells that actually exist.
    """
    sub = full_q2[(full_q2["environment"] == env) & (full_q2["capability"] == capability)]
    if "feature_name" in sub.columns and "T" in sub.columns:
        n_base = int(sub.loc[sub["T"] == 0, "feature_name"].nunique())
        n_treat = int(sub.loc[sub["T"] == 1, "feature_name"].nunique())
    else:
        n_feat = int(sub["feature_name"].nunique())
        n_base = n_treat = n_feat
    base = N_PER_FEATURE_CELL_Q1 if q_label == "Q1" else N_PER_FEATURE_CELL_Q2
    return int((n_base + n_treat) * base)


def _short_model(m: str) -> str:
    """Display label for a model: its bare id, as the paper's prose writes it."""
    return m.rsplit("/", 1)[-1] if "/" in m else m


def _pair_label(model_a: str, model_b: str) -> str:
    return f"{_short_model(model_a)} → {_short_model(model_b)}"


def is_significant(row) -> bool:
    """Bonferroni-corrected p < SIG_ALPHA (falls back to raw p if no p_bonf)."""
    p = row["p_bonf"] if ("p_bonf" in row and np.isfinite(row.get("p_bonf", np.nan))) else row["p"]
    return bool(np.isfinite(p) and p < SIG_ALPHA)


def is_faded(row, target: int) -> bool:
    """A cell is faded when its regression has fewer rows than its quota."""
    if target <= 0:
        return False
    return bool(row["n"] < target)


# ---------------------------------------------------------------------------
# Shared forest renderer
# ---------------------------------------------------------------------------


@dataclass
class ForestPanel:
    """Everything one environment panel needs."""
    title: str
    xlabel: str
    q2: pd.DataFrame                      # Tier C rows, indexed by (model_a, model_b)
    q2_pooled: pd.Series | None = None    # Tier B row (or None)
    q2_target: int = 0                    # fade threshold on regression n
    q1: pd.DataFrame | None = None        # Q1 overlay (None if the env has no Q1)
    q1_pooled: pd.Series | None = None
    q1_target: int = 0
    xlim: tuple[float, float] | None = None  # explicit limits (None = auto)
    extra: dict = field(default_factory=dict)


ROW_PITCH_IN = 0.13      # vertical space per pair row
GROUP_GAP = 0.45          # extra gap (in row units) between attacker blocks
POOLED_GAP = 0.6          # gap between the pooled row and the first block
Y_OFFSET = 0.21           # Q1 above / Q2 below the row centre


def _row_positions(pairs: pd.DataFrame) -> tuple[dict, list[tuple[str, float, float, float]]]:
    """Map each pair to a y position; return (positions, blocks).

    ``blocks`` is a list of (model_a, y_top, y_bottom, y_centre) for the
    attacker groups, in display order (top to bottom).
    """
    positions: dict[tuple[str, str], float] = {}
    blocks: list[tuple[str, float, float, float]] = []
    y = 0.0 + POOLED_GAP          # row 0 is the pooled row
    prev_a = None
    block_rows: list[float] = []
    for a, b in zip(pairs["model_a"], pairs["model_b"]):
        if prev_a is not None and a != prev_a:
            blocks.append((prev_a, block_rows[0] - 0.5, block_rows[-1] + 0.5,
                           float(np.mean(block_rows))))
            block_rows = []
            y += GROUP_GAP
        y += 1.0
        positions[(a, b)] = y
        block_rows.append(y)
        prev_a = a
    if block_rows:
        blocks.append((prev_a, block_rows[0] - 0.5, block_rows[-1] + 0.5,
                       float(np.mean(block_rows))))
    return positions, blocks


def render_forest(pairs: pd.DataFrame, panels: list[ForestPanel], out_path: Path,
                  *, width_in: float = style.TEXT_WIDTH_IN,
                  label_col_in: float = 1.05, legend: bool = True) -> None:
    """Draw the multi-panel forest figure and save it as PDF + PNG."""
    style.apply()
    pairs = pairs.sort_values(list(ROW_SORT)).reset_index(drop=True)
    positions, blocks = _row_positions(pairs)
    y_max = max(positions.values()) + 0.5
    has_q1 = any(p.q1 is not None for p in panels)

    n_panels = len(panels)
    n_rows_eq = y_max + 0.5
    plot_h = n_rows_eq * ROW_PITCH_IN
    top_in, bottom_in = 0.34, 0.42 + (0.5 if legend else 0.0)
    fig_h = plot_h + top_in + bottom_in
    fig, axes = plt.subplots(1, n_panels, figsize=(width_in, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        # Background bands: pooled row and every other attacker block.
        ax.axhspan(-0.5, 0.5, color=style.POOLED_BAND_COLOR, zorder=0, lw=0)
        for i, (_, y0, y1, _) in enumerate(blocks):
            if i % 2 == 0:
                ax.axhspan(y0, y1, color=style.BAND_COLOR, zorder=0, lw=0)

        xs: list[float] = []
        series = [(panel.q2, panel.q2_pooled, panel.q2_target, Q2_COLOR,
                   +Y_OFFSET if panel.q1 is not None else 0.0)]
        if panel.q1 is not None:
            series.insert(0, (panel.q1, panel.q1_pooled, panel.q1_target, Q1_COLOR, -Y_OFFSET))
        for rows, pooled, target, color, dy in series:
            for key, y in positions.items():
                if key not in rows.index:
                    continue
                r = rows.loc[key]
                if not np.isfinite(r["beta"]):
                    continue
                style.plot_estimate(ax, y + dy, r["beta"], r["ci_low"], r["ci_high"],
                                    color=color, significant=is_significant(r),
                                    faded=is_faded(r, target))
                xs += [r["ci_low"], r["ci_high"]]
            if pooled is not None and np.isfinite(pooled["beta"]):
                style.plot_estimate(ax, 0.0 + dy, pooled["beta"], pooled["ci_low"],
                                    pooled["ci_high"], color=color,
                                    significant=is_significant(pooled), faded=False)
                style.reference_line(ax, pooled["beta"], color)
                xs += [pooled["ci_low"], pooled["ci_high"]]

        style.zero_line(ax)
        ax.axhline(0.5 + POOLED_GAP / 2, color=style.SEPARATOR_COLOR, lw=0.5, zorder=1)

        style.panel_title(ax, panel.title)
        ax.set_xlabel(panel.xlabel.replace(" (", "\n("), labelpad=3)
        if panel.xlim is not None:
            ax.set_xlim(*panel.xlim)
        else:
            xs = [x for x in xs if np.isfinite(x)] + [0.0]
            lo, hi = min(xs), max(xs)
            pad = 0.06 * (hi - lo) if hi > lo else 0.1
            ax.set_xlim(lo - pad, hi + pad)
        style.tidy_axes(ax, grid_axis=None)
        style.effect_ticks(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    # Row labels: "Pooled" plus one attacker name per block, on the left axis.
    ax0 = axes[0]
    ax0.set_ylim(y_max, -0.5)        # inverted: pooled row on top
    ax0.set_yticks([0.0] + [c for (_, _, _, c) in blocks])
    labels = ax0.set_yticklabels(["Pooled"] + [a for (a, _, _, _) in blocks])
    labels[0].set_fontweight("bold")

    fig.subplots_adjust(left=label_col_in / width_in, right=1 - 0.06 / width_in,
                        top=1 - top_in / fig_h, bottom=bottom_in / fig_h, wspace=0.22)

    extras = []
    if legend:
        handles = style.forest_legend_handles(q1=has_q1, q2=True)
        leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                         ncol=3, handletextpad=0.4, columnspacing=1.4,
                         borderaxespad=0.0, handlelength=1.6)
        extras.append(leg)
    style.save(fig, out_path, extra=extras)


# ---------------------------------------------------------------------------
# Per-capability figure (Figs 4, 5; persuasion's twin of Fig 2)
# ---------------------------------------------------------------------------


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(["model_a", "model_b"])


# The paper prints the key once, under the persuasion figure (Fig 2); the
# deception and coercion figures (Figs 4, 5) say in their captions that the key
# is the same, so they are drawn without one to save height.
FOREST_LEGEND_CAPABILITIES: frozenset[str] = frozenset({"persuasion"})


def plot_capability_forest(
    capability: str,
    tier_c_q1: pd.DataFrame,
    tier_c_q2: pd.DataFrame,
    tier_b_q1: pd.DataFrame,
    tier_b_q2: pd.DataFrame,
    full_q2_for_targets: pd.DataFrame,
    out_path: Path,
) -> None:
    sub_q1 = tier_c_q1[tier_c_q1["capability"] == capability]
    sub_q2 = tier_c_q2[tier_c_q2["capability"] == capability]
    present = set(sub_q1["environment"]) | set(sub_q2["environment"])
    envs = [e for e in ENV_BY_CAP.get(capability, ())
            if e in present and e not in FOREST_PAIR_ENVS_EXCLUDE]
    if not envs:
        return

    pq1 = sub_q1[sub_q1["environment"].isin(envs)]
    pq2 = sub_q2[sub_q2["environment"].isin(envs)]
    pairs = (pd.concat([pq1[["model_a", "model_b"]], pq2[["model_a", "model_b"]]])
             .drop_duplicates().sort_values(list(ROW_SORT)).reset_index(drop=True))

    b_q1 = tier_b_q1[tier_b_q1["capability"] == capability].set_index("environment")
    b_q2 = tier_b_q2[tier_b_q2["capability"] == capability].set_index("environment")

    panels: list[ForestPanel] = []
    for env in envs:
        env_q1 = sub_q1[sub_q1["environment"] == env]
        env_q2 = sub_q2[sub_q2["environment"] == env]
        panels.append(ForestPanel(
            title=ENVIRONMENT_DISPLAY.get(env, env),
            xlabel=effect_axis_label(env),
            q2=_indexed(env_q2),
            q2_pooled=b_q2.loc[env] if env in b_q2.index else None,
            q2_target=cell_target_n(env, capability, "Q2", full_q2_for_targets),
            q1=_indexed(env_q1) if not env_q1.empty else None,
            q1_pooled=b_q1.loc[env] if env in b_q1.index else None,
            q1_target=cell_target_n(env, capability, "Q1", full_q2_for_targets),
        ))
    render_forest(pairs, panels, out_path,
                  legend=capability in FOREST_LEGEND_CAPABILITIES)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(plots_dir: Path | None = None,
        regressions_dir: Path | None = None,
        data_dir: Path | None = None) -> dict[str, Path]:
    from .spec import DATA_DIR
    plots_dir = plots_dir or PLOTS_DIR
    regressions_dir = regressions_dir or REGRESSIONS_DIR
    data_dir = data_dir or DATA_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    tier_b_q1 = pd.read_parquet(regressions_dir / "tier_B_q1.parquet")
    tier_b_q2 = pd.read_parquet(regressions_dir / "tier_B_q2.parquet")
    tier_c_q1 = pd.read_parquet(regressions_dir / "tier_C_q1.parquet")
    tier_c_q2 = pd.read_parquet(regressions_dir / "tier_C_q2.parquet")

    full_q2 = pd.read_parquet(data_dir / "q2_filtered.parquet")

    written: dict[str, Path] = {}
    for capability in CAPABILITIES:
        out = plots_dir / f"forest_{capability}"
        plot_capability_forest(
            capability=capability,
            tier_c_q1=tier_c_q1, tier_c_q2=tier_c_q2,
            tier_b_q1=tier_b_q1, tier_b_q2=tier_b_q2,
            full_q2_for_targets=full_q2,
            out_path=out,
        )
        written[f"forest_{capability}_png"] = out.with_suffix(".png")

    return written


if __name__ == "__main__":
    paths = run()
    for k, p in paths.items():
        print(f"  {k}: {p}")
