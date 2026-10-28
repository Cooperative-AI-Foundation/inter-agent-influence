"""Leaderboard plots — one figure per (capability, target model_b) + 3x3 matrix per capability.

Per-target figures:
  - 3 env-panels (one per environment in the capability)
  - Up to 6 rows per panel (one per model_a), filtered to the single target model_b
  - Each panel sorted independently by Q1 beta descending (highest exploiter at top)
  - Individual y-axis labels per panel (model_a name only, no shared axis)

Matrix figures (one per capability):
  - 3 rows (one per target model_b) × 3 columns (one per environment)
  - Same per-cell content and sorting as the per-target figures

Both use the same Q1/Q2 visual language as the forest plots.

Outputs: analysis/leaderboard_by_target/
  leaderboard_{capability}_{slug}.png/pdf      — per-target (9 total)
  leaderboard_{capability}_matrix.png/pdf      — combined 3x3 (3 total)

Run: python -m scripts.analysis.leaderboard_by_target
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .plots import cell_target_n, is_faded, is_significant
from .spec import (
    CAPABILITIES,
    DATA_DIR,
    ENVIRONMENT_DISPLAY,
    ENV_BY_CAP,
    TIER_A_EXCLUDE_ENVS,
    LEADERBOARD_DIR,
    Q1_COLOR,
    Q2_COLOR,
    REGRESSIONS_DIR,
    effect_axis_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_label(model: str) -> str:
    """Display name for a model: the bare model id, as the paper's prose uses it."""
    return model.rsplit("/", 1)[-1] if "/" in model else model


def _model_slug(model: str) -> str:
    """Filename-safe slug for a model name."""
    last = model.rsplit("/", 1)[-1]
    return (
        last.lower()
            .replace("-it", "")
            .replace("-", "_")
            .replace(".", "_")
    )


def _sorted_models(
    env_q1: pd.DataFrame,
    env_q2: pd.DataFrame,
) -> list[str]:
    """Return model_a list sorted by Q1 beta desc (NaN last), Q2-only models appended."""
    q1_sorted = (
        env_q1
        .assign(_sort=env_q1["beta"].where(env_q1["fit_status"] == "ok"))
        .sort_values("_sort", ascending=False, na_position="last")["model_a"]
        .tolist()
    )
    q2_only = [m for m in env_q2["model_a"].tolist() if m not in q1_sorted]
    return q1_sorted + q2_only


# ---------------------------------------------------------------------------
# Per-cell drawing (shared by single-target and matrix figures)
# ---------------------------------------------------------------------------

Y_OFFSET = 0.19


def _draw_cell(
    ax: plt.Axes,
    env: str,
    capability: str,
    model_b: str,
    cap_q1: pd.DataFrame,
    cap_q2: pd.DataFrame,
    main_q1: float | None,
    main_q2: float | None,
    full_q2_for_targets: pd.DataFrame,
    *,
    show_xlabel: bool = True,
) -> None:
    """Draw one (env, model_b) cell onto ax. Caller sets title/ylabel.

    Rows are attackers (model_a), sorted by the Q1 estimate (largest at the
    top). Dotted lines mark the capability's pooled Tier-A estimates.
    """
    target_q1 = cell_target_n(env, capability, "Q1", full_q2_for_targets)
    target_q2 = cell_target_n(env, capability, "Q2", full_q2_for_targets)

    env_q1 = cap_q1[cap_q1["environment"] == env].set_index("model_a")
    env_q2 = cap_q2[cap_q2["environment"] == env].set_index("model_a")

    models = _sorted_models(
        env_q1.reset_index()[["model_a", "beta", "fit_status"]],
        env_q2.reset_index()[["model_a"]],
    )
    model_index = {m: i for i, m in enumerate(models)}

    for q_color, df_env, target, dy in [
        (Q1_COLOR, env_q1, target_q1, -Y_OFFSET),
        (Q2_COLOR, env_q2, target_q2, +Y_OFFSET),
    ]:
        for model_a, i in model_index.items():
            if model_a not in df_env.index:
                continue
            row = df_env.loc[model_a]
            if not np.isfinite(row["beta"]):
                continue
            style.plot_estimate(ax, i + dy, row["beta"], row["ci_low"], row["ci_high"],
                                color=q_color, significant=is_significant(row),
                                faded=is_faded(row, target))

    style.zero_line(ax)
    if main_q1 is not None and np.isfinite(main_q1):
        style.reference_line(ax, main_q1, Q1_COLOR)
    if main_q2 is not None and np.isfinite(main_q2):
        style.reference_line(ax, main_q2, Q2_COLOR)

    for i in range(len(models)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=style.BAND_COLOR, zorder=0, lw=0)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([_model_label(m) for m in models], fontsize=style.FS_BASE - 0.7)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    if show_xlabel:
        ax.set_xlabel(effect_axis_label(env).replace(" (", "\n("))
    style.tidy_axes(ax, grid_axis=None)
    style.effect_ticks(ax)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# Per-target figure (1 row × 3 env columns)
# ---------------------------------------------------------------------------


def _grid_envs(capability: str) -> list[str]:
    """E1-E3: one outcome scale (approval) and one set of models; E4 and E5 are reported elsewhere."""
    return [e for e in ENV_BY_CAP.get(capability, ()) if e not in TIER_A_EXCLUDE_ENVS]


def _targets(tier_c_q1: pd.DataFrame, tier_c_q2: pd.DataFrame, capability: str) -> list[str]:
    """Target models that appear in this capability's grid environments."""
    envs = set(_grid_envs(capability))
    rows = pd.concat([tier_c_q1, tier_c_q2])
    rows = rows[(rows["capability"] == capability) & rows["environment"].isin(envs)]
    return sorted(rows["model_b"].unique())


# The paper prints the key once, under the persuasion leaderboard (Fig 12); the
# deception and coercion leaderboards refer to it in their captions.
LEADERBOARD_LEGEND_CAPABILITIES: frozenset[str] = frozenset({"persuasion"})


def _finish(fig, out_path: Path, *, legend: bool = True) -> None:
    if legend:
        fig.legend(handles=style.forest_legend_handles(), loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=3, handletextpad=0.4,
                   columnspacing=1.4, borderaxespad=0.0, handlelength=1.6)
    style.save(fig, out_path)


def plot_leaderboard(
    capability: str,
    model_b: str,
    tier_c_q1: pd.DataFrame,
    tier_c_q2: pd.DataFrame,
    main_q1: float | None,
    main_q2: float | None,
    full_q2_for_targets: pd.DataFrame,
    out_path: Path,
) -> None:
    style.apply()
    envs = _grid_envs(capability)
    if not envs:
        return

    cap_q1 = tier_c_q1[(tier_c_q1["capability"] == capability) & (tier_c_q1["model_b"] == model_b)]
    cap_q2 = tier_c_q2[(tier_c_q2["capability"] == capability) & (tier_c_q2["model_b"] == model_b)]

    fig, axes = plt.subplots(1, len(envs), figsize=(style.TEXT_WIDTH_IN, 2.4),
                             sharey=False, sharex=True)
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        _draw_cell(ax, env, capability, model_b, cap_q1, cap_q2,
                   main_q1, main_q2, full_q2_for_targets)
        style.panel_title(ax, ENVIRONMENT_DISPLAY.get(env, env))
    fig.subplots_adjust(left=1.0 / style.TEXT_WIDTH_IN, right=0.99, top=0.85,
                        bottom=0.42, wspace=0.9)
    _finish(fig, out_path)


# ---------------------------------------------------------------------------
# Matrix figure (n_targets rows × 3 env columns) — paper Figs 11-13
# ---------------------------------------------------------------------------


def plot_leaderboard_matrix(
    capability: str,
    all_model_bs: list[str],
    tier_c_q1: pd.DataFrame,
    tier_c_q2: pd.DataFrame,
    main_q1: float | None,
    main_q2: float | None,
    full_q2_for_targets: pd.DataFrame,
    out_path: Path,
) -> None:
    style.apply()
    envs = _grid_envs(capability)
    if not envs:
        return

    n_rows = len(all_model_bs)
    n_cols = len(envs)
    row_h = 1.25                       # plot height per target row
    row_title_in = 0.32                # space above each row for its title
    legend = capability in LEADERBOARD_LEGEND_CAPABILITIES
    top_in, bottom_in = 0.55, 0.55 + (0.5 if legend else 0.0)
    label_in = 1.15                    # room for the longest model id
    plot_in = (style.TEXT_WIDTH_IN - 0.05 - n_cols * label_in) / n_cols
    fig_h = (row_h + row_title_in) * n_rows + top_in + bottom_in - row_title_in
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(style.TEXT_WIDTH_IN, fig_h),
                             sharey=False, sharex="col")
    axes = np.atleast_2d(axes)
    if n_cols == 1 and n_rows > 1:
        axes = axes.T

    for row_idx, model_b in enumerate(all_model_bs):
        cap_q1 = tier_c_q1[(tier_c_q1["capability"] == capability) & (tier_c_q1["model_b"] == model_b)]
        cap_q2 = tier_c_q2[(tier_c_q2["capability"] == capability) & (tier_c_q2["model_b"] == model_b)]
        is_last_row = row_idx == n_rows - 1
        for col_idx, env in enumerate(envs):
            ax = axes[row_idx, col_idx]
            _draw_cell(ax, env, capability, model_b, cap_q1, cap_q2,
                       main_q1, main_q2, full_q2_for_targets, show_xlabel=is_last_row)
            if row_idx == 0:
                style.panel_title(ax, ENVIRONMENT_DISPLAY.get(env, env), pad_pt=18)
        # Row title: the target model, above the row's first cell.
        axes[row_idx, 0].annotate(
            f"Target: {_model_label(model_b)}", xy=(0, 1.0), xycoords="axes fraction",
            xytext=(-label_in * 72, 4), textcoords="offset points",
            ha="left", va="bottom", fontsize=style.FS_LABEL, fontweight="bold",
            annotation_clip=False)

    fig.subplots_adjust(left=label_in / style.TEXT_WIDTH_IN, right=1 - 0.05 / style.TEXT_WIDTH_IN,
                        top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
                        wspace=label_in / plot_in, hspace=row_title_in / row_h)
    _finish(fig, out_path, legend=legend)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    leaderboard_dir: Path | None = None,
    regressions_dir: Path | None = None,
    data_dir: Path | None = None,
) -> dict[str, Path]:
    leaderboard_dir = leaderboard_dir or LEADERBOARD_DIR
    regressions_dir = regressions_dir or REGRESSIONS_DIR
    data_dir = data_dir or DATA_DIR
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    tier_a_q1 = pd.read_parquet(regressions_dir / "tier_A_q1.parquet")
    tier_a_q2 = pd.read_parquet(regressions_dir / "tier_A_q2.parquet")
    tier_c_q1 = pd.read_parquet(regressions_dir / "tier_C_q1.parquet")
    tier_c_q2 = pd.read_parquet(regressions_dir / "tier_C_q2.parquet")
    full_q2 = pd.read_parquet(data_dir / "q2_filtered.parquet")

    written: dict[str, Path] = {}
    for capability in CAPABILITIES:
        all_model_bs = _targets(tier_c_q1, tier_c_q2, capability)
        a1 = tier_a_q1[tier_a_q1["capability"] == capability]
        a2 = tier_a_q2[tier_a_q2["capability"] == capability]
        main_q1 = float(a1["beta"].iloc[0]) if not a1.empty else None
        main_q2 = float(a2["beta"].iloc[0]) if not a2.empty else None

        for model_b in all_model_bs:
            slug = _model_slug(model_b)
            out = leaderboard_dir / f"leaderboard_{capability}_{slug}"
            plot_leaderboard(
                capability=capability,
                model_b=model_b,
                tier_c_q1=tier_c_q1,
                tier_c_q2=tier_c_q2,
                main_q1=main_q1,
                main_q2=main_q2,
                full_q2_for_targets=full_q2,
                out_path=out,
            )
            written[f"leaderboard_{capability}_{slug}_png"] = out.with_suffix(".png")
            print(f"  wrote {out.with_suffix('.png').relative_to(out.parents[2])}")

        # Matrix: all targets × all envs in one figure
        matrix_out = leaderboard_dir / f"leaderboard_{capability}_matrix"
        plot_leaderboard_matrix(
            capability=capability,
            all_model_bs=all_model_bs,
            tier_c_q1=tier_c_q1,
            tier_c_q2=tier_c_q2,
            main_q1=main_q1,
            main_q2=main_q2,
            full_q2_for_targets=full_q2,
            out_path=matrix_out,
        )
        written[f"leaderboard_{capability}_matrix_png"] = matrix_out.with_suffix(".png")
        print(f"  wrote {matrix_out.with_suffix('.png').relative_to(matrix_out.parents[2])}")

    return written


if __name__ == "__main__":
    paths = run()
    print(f"\n{len(paths)} figures written.")
