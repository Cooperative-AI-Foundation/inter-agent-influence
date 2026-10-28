"""Standalone tables + forest plot for one (or several) environments.

Designed for E4 (meeting_schedule) and E5 (aar), which the paper presents as
their own appendix sections rather than alongside E1-E3. Reads the canonical
Q2 outputs (tier_B_q2.parquet, tier_C_q2.parquet) produced by
``scripts.analysis.regression`` and renders:

  - <env>_tier_B.tex          Tier B (env x cap) — same format as tier_B_q2.tex
  - <env>_tier_C.tex          Tier C (per-pair) — same format as tier_C_q2.tex
  - forest_<env>.{pdf,png}    single-panel forest plot

In ``--envs`` mode (multiple envs), produces a single Tier B / Tier C
table covering all the named envs (one row per env in B, one row per
pair-in-env in C), with the prefix supplied via ``--out-prefix``.

Usage:
    python -m scripts.analysis.standalone_env --env meeting_schedule
    python -m scripts.analysis.standalone_env --env aar
    python -m scripts.analysis.standalone_env --envs meeting_schedule,aar --out-prefix e4_e5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .regression import render_tier_b_latex, render_tier_c_latex
from .spec import (
    ENVIRONMENT_DISPLAY,
    PLOTS_DIR,
    REGRESSIONS_DIR,
    ROW_SORT,
    SIG_ALPHA,
    effect_axis_label,
    outcome_label,
)


from .plots import ForestPanel, cell_target_n, render_forest as _render_forest_grid


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(["model_a", "model_b"])


def _load_full_q2() -> pd.DataFrame | None:
    """Q2 sample used to size each cell's quota (for the fade decision)."""
    from .spec import DATA_DIR
    p = DATA_DIR / "q2_filtered.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _target(env: str, capability: str, q_label: str, full_q2: pd.DataFrame | None) -> int:
    """Quota for one cell; 0 (never fade) if the Q2 sample is unavailable."""
    if full_q2 is None:
        return 0
    return cell_target_n(env, capability, q_label, full_q2)


def render_forest(rows: pd.DataFrame, pooled: pd.Series, env: str, out_pdf: Path) -> None:
    """Single-panel forest for one environment (Q2 only)."""
    from .spec import env_to_capability
    full_q2 = _load_full_q2()
    cap = env_to_capability(env)
    pairs = rows[["model_a", "model_b"]].drop_duplicates()
    panel = ForestPanel(
        title=ENVIRONMENT_DISPLAY.get(env, env),
        xlabel=effect_axis_label(env),
        q2=_indexed(rows), q2_pooled=pooled,
        q2_target=_target(env, cap, "Q2", full_q2),
    )
    _render_forest_grid(pairs, [panel], out_pdf, width_in=3.4)


def render_multi_env_forest(
    tier_c: pd.DataFrame,
    tier_b: pd.DataFrame,
    envs: list[str],
    out_pdf: Path,
    title: str | None = None,
    dropped_per_env: dict[str, int] | None = None,
    subtitle: str | None = None,
    tier_c_q1: pd.DataFrame | None = None,
    tier_b_q1: pd.DataFrame | None = None,
) -> None:
    """Multi-panel forest: one panel per env, shared y-axis (union of pairs).

    Each panel shows per-pair Q2 effects with HC1 95% CIs and a dotted line at
    the env's pooled (Tier B) effect. If ``tier_c_q1`` and ``tier_b_q1`` are
    given, Q1 estimates are overlaid wherever the env has them. Pairs absent
    from an env leave that row blank in that panel. ``title``, ``subtitle`` and
    ``dropped_per_env`` are accepted for compatibility but no longer drawn:
    everything a reader needs beyond the plot belongs in the LaTeX caption.
    """
    from .spec import env_to_capability
    panels_src = []
    for env in envs:
        b = tier_b[tier_b["environment"] == env]
        c = tier_c[tier_c["environment"] == env].copy()
        if b.empty or c.empty:
            continue
        panels_src.append((env, b.iloc[0], c))
    if not panels_src:
        raise SystemExit(f"no data for envs={envs!r}")

    union = (
        pd.concat([p[2][["model_a", "model_b"]] for p in panels_src])
        .drop_duplicates()
        .sort_values(list(ROW_SORT))
        .reset_index(drop=True)
    )
    full_q2 = _load_full_q2()
    panels: list[ForestPanel] = []
    for env, pooled, cell_rows in panels_src:
        cap = env_to_capability(env)
        q1 = q1_pooled = None
        if tier_c_q1 is not None and tier_b_q1 is not None:
            q1_rows = tier_c_q1[tier_c_q1["environment"] == env]
            if not q1_rows.empty:
                q1 = _indexed(q1_rows)
                q1_pool = tier_b_q1[tier_b_q1["environment"] == env]
                q1_pooled = q1_pool.iloc[0] if not q1_pool.empty else None
        panels.append(ForestPanel(
            title=ENVIRONMENT_DISPLAY.get(env, env),
            xlabel=effect_axis_label(env),
            q2=_indexed(cell_rows), q2_pooled=pooled,
            q2_target=_target(env, cap, "Q2", full_q2),
            q1=q1, q1_pooled=q1_pooled,
            q1_target=_target(env, cap, "Q1", full_q2),
        ))
    _render_forest_grid(union, panels, out_pdf)


def render_influence_matrix(
    tier_c: pd.DataFrame,
    env: str,
    out_pdf: Path,
    pooled_beta: float | None = None,
) -> None:
    """A×B heatmap of per-pair point uplift for one environment.

    Pivots Tier C β into a model_a (rows) × model_b (columns) matrix with a
    diverging colormap centered at 0. Cells with non-significant effects
    (Bonferroni p >= SIG_ALPHA) are hatched.
    """
    rows = tier_c[tier_c["environment"] == env].copy()
    if rows.empty:
        return

    pivot = rows.pivot_table(index="model_a", columns="model_b", values="beta")
    p_pivot = rows.pivot_table(index="model_a", columns="model_b", values="p_bonf")

    label = ENVIRONMENT_DISPLAY.get(env, env)
    olabel = outcome_label(env)

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.01)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pivot.columns) + 2),
                                     max(5, 0.7 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Agent B (target)", fontsize=11)
    ax.set_ylabel("Agent A (influencer)", fontsize=11)
    ax.set_title(f"{label} — influence matrix", fontsize=12,
                 fontweight="bold", pad=10)

    for (i, j), v in np.ndenumerate(pivot.values):
        if not np.isfinite(v):
            continue
        p_val = p_pivot.values[i, j] if np.isfinite(p_pivot.values[i, j]) else 1.0
        sig = p_val < SIG_ALPHA
        txt = f"{v:+.3f}"
        color = "white" if abs(v) > vmax * 0.6 else "black"
        weight = "bold" if sig else "normal"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                color=color, fontweight=weight)
        if not sig:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, hatch="//",
                                        edgecolor="grey", lw=0))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"point uplift ({olabel})", fontsize=10)

    if pooled_beta is not None and np.isfinite(pooled_beta):
        fig.text(0.5, 0.01, f"Pooled (Tier B) β = {pooled_beta:+.4f}",
                 ha="center", fontsize=10, style="italic", color="dimgrey")

    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def _write_tables(tier_b_sub: pd.DataFrame, tier_c_sub: pd.DataFrame,
                  out_prefix: str, q_label: str = "Q2") -> tuple[Path, Path]:
    b_tex = REGRESSIONS_DIR / f"{out_prefix}_tier_B.tex"
    c_tex = REGRESSIONS_DIR / f"{out_prefix}_tier_C.tex"
    b_tex.write_text(render_tier_b_latex(tier_b_sub, q_label))
    c_tex.write_text(render_tier_c_latex(tier_c_sub, q_label))
    return b_tex, c_tex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env",
        help="Single environment (e.g. meeting_schedule). Renders <env>_tier_B.tex, "
             "<env>_tier_C.tex, forest_<env>.{pdf,png}.",
    )
    ap.add_argument(
        "--envs",
        help="Comma-separated list of envs (e.g. 'meeting_schedule,aar'). Renders "
             "combined Tier B / Tier C tables under --out-prefix.",
    )
    ap.add_argument(
        "--out-prefix",
        default="e4_e5",
        help="Filename prefix for combined-mode outputs (default: e4_e5).",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional super-title for the multi-panel forest plot.",
    )
    ap.add_argument(
        "--shared-only",
        action="store_true",
        help="In --envs mode, restrict to (model_a, model_b) pairs present in ALL "
             "named envs (the cross-env shared subset).",
    )
    ap.add_argument(
        "--pairs-from",
        default=None,
        help="In --envs mode, define the y-axis pair universe as the UNION of "
             "pairs across the comma-separated env list given here (e.g. "
             "'gdpr,code_review_sabotage,data_analysis,meeting_schedule'). All "
             "panels then use that universe; pairs outside it are dropped, "
             "pairs missing from a given env are blank in that panel. "
             "Mutually exclusive with --shared-only.",
    )
    args = ap.parse_args()
    if args.shared_only and args.pairs_from:
        ap.error("--shared-only and --pairs-from are mutually exclusive")

    tier_b = pd.read_parquet(REGRESSIONS_DIR / "tier_B_q2.parquet")
    tier_c = pd.read_parquet(REGRESSIONS_DIR / "tier_C_q2.parquet")
    q1_b_path = REGRESSIONS_DIR / "tier_B_q1.parquet"
    q1_c_path = REGRESSIONS_DIR / "tier_C_q1.parquet"
    tier_b_q1 = pd.read_parquet(q1_b_path) if q1_b_path.exists() else None
    tier_c_q1 = pd.read_parquet(q1_c_path) if q1_c_path.exists() else None

    if args.envs:
        envs = [e.strip() for e in args.envs.split(",") if e.strip()]
        b_sub = tier_b[tier_b["environment"].isin(envs)].copy()
        c_sub = tier_c[tier_c["environment"].isin(envs)].copy()
        if b_sub.empty or c_sub.empty:
            raise SystemExit(f"no rows for envs={envs!r} in tier_B/tier_C Q2 outputs")

        dropped_per_env: dict[str, int] | None = None
        subtitle = None
        if args.shared_only:
            # Pair = (model_a, model_b). Shared = present in every requested env.
            per_env_pairs = {
                env: set(map(tuple, c_sub[c_sub["environment"] == env][["model_a", "model_b"]].drop_duplicates().values))
                for env in envs
            }
            shared = set.intersection(*per_env_pairs.values()) if per_env_pairs else set()
            if not shared:
                raise SystemExit("no pairs are shared across all requested envs")
            dropped_per_env = {env: len(per_env_pairs[env] - shared) for env in envs}
            mask = c_sub.apply(lambda r: (r["model_a"], r["model_b"]) in shared, axis=1)
            c_sub = c_sub[mask].copy()
            n_panels_with_drop = sum(1 for v in dropped_per_env.values() if v > 0)
            subtitle = (
                f"Restricted to the {len(shared)} (model_a, model_b) pairs present in all "
                f"{len(envs)} environments; non-shared pairs dropped from "
                f"{n_panels_with_drop} of {len(envs)} panels (counts in panel titles)."
            )
        elif args.pairs_from:
            anchor_envs = [e.strip() for e in args.pairs_from.split(",") if e.strip()]
            anchor_pairs: set[tuple[str, str]] = set()
            for env in anchor_envs:
                e_pairs = tier_c[tier_c["environment"] == env][["model_a", "model_b"]].drop_duplicates()
                anchor_pairs |= set(map(tuple, e_pairs.values))
            if not anchor_pairs:
                raise SystemExit(f"no pairs found in --pairs-from envs={anchor_envs!r}")
            per_env_pairs = {
                env: set(map(tuple, c_sub[c_sub["environment"] == env][["model_a", "model_b"]].drop_duplicates().values))
                for env in envs
            }
            dropped_per_env = {env: len(per_env_pairs[env] - anchor_pairs) for env in envs}
            mask = c_sub.apply(lambda r: (r["model_a"], r["model_b"]) in anchor_pairs, axis=1)
            c_sub = c_sub[mask].copy()
            anchor_label = ", ".join(
                ENVIRONMENT_DISPLAY.get(e, e).split(".")[0] for e in anchor_envs
            )
            subtitle = (
                f"Pairs shown: {len(anchor_pairs)} A→B combinations from "
                f"{anchor_label}. Blank rows indicate the panel's env did not run that pair."
            )

        b_tex, c_tex = _write_tables(b_sub, c_sub, args.out_prefix)
        print(f"wrote {b_tex.relative_to(REGRESSIONS_DIR.parent.parent)}")
        print(f"wrote {c_tex.relative_to(REGRESSIONS_DIR.parent.parent)}")

        # Multi-panel forest in the requested env order. No suptitle by
        # default — the figure title goes in the LaTeX caption, matching the
        # capability-pooled forest plots in plots.py. Pass --title to opt in.
        forest_pdf = PLOTS_DIR / f"forest_{args.out_prefix}.pdf"
        title = args.title if args.title else None
        render_multi_env_forest(
            c_sub, tier_b, envs, forest_pdf, title=title,
            dropped_per_env=dropped_per_env, subtitle=subtitle,
            tier_c_q1=tier_c_q1, tier_b_q1=tier_b_q1,
        )
        print(f"wrote {forest_pdf.relative_to(REGRESSIONS_DIR.parent.parent)} (+ .png)")
        return

    if not args.env:
        raise SystemExit("provide either --env or --envs")

    env = args.env
    b_sub = tier_b[tier_b["environment"] == env].copy()
    c_sub = tier_c[tier_c["environment"] == env].copy()
    if b_sub.empty:
        raise SystemExit(f"no tier_B Q2 row for environment={env!r}")
    if c_sub.empty:
        raise SystemExit(f"no tier_C Q2 rows for environment={env!r}")

    b_tex, c_tex = _write_tables(b_sub, c_sub, env)

    # Forest plot
    forest_pdf = PLOTS_DIR / f"forest_{env}.pdf"
    pooled = b_sub.iloc[0]
    render_forest(c_sub, pooled, env, forest_pdf)

    print(f"wrote {b_tex.relative_to(REGRESSIONS_DIR.parent.parent)}")
    print(f"wrote {c_tex.relative_to(REGRESSIONS_DIR.parent.parent)}")
    print(f"wrote {forest_pdf.relative_to(REGRESSIONS_DIR.parent.parent)} (+ .png)")

    # Influence matrix heatmap (E4 envs)
    from .spec import E4_ENVS
    if env in E4_ENVS:
        matrix_pdf = PLOTS_DIR / f"influence_matrix_{env}.pdf"
        render_influence_matrix(c_sub, env, matrix_pdf,
                                pooled_beta=float(pooled["beta"]))
        print(f"wrote {matrix_pdf.relative_to(REGRESSIONS_DIR.parent.parent)} (+ .png)")


if __name__ == "__main__":
    main()
