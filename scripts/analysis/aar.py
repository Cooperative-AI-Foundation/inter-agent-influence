"""E5 (AAR) analysis — Q2-only, single regression per tier.

The AAR env differs from E1-E3 in two structural ways that warrant a
dedicated module:

1. **Different schema** — `final_data/e5.parquet` (the raw pooled AAR file) has
   `decision` (already int 0/1), `score_aar_drift_scorer`, and an
   `instruction_level` column. No `mechanism_a_*` flags, so no Q1
   behavioral filter applies.

2. **Two persuasion sub-conditions in the data, pooled into one effect.**
   The data has baseline + goal-only (L1) persuasion + scaffolded (L3)
   persuasion. We pool both attack sub-conditions against the shared
   baseline (T = condition=='attack'). The baseline is treatment-neutral
   (empty techniques_str) so it serves as the no-persuasion comparator for
   both levels; its stored instruction_level=3 is a storage artifact, not a
   treatment contrast.

Spec mirrors the main pipeline:
   y = α + β·T + α_f + γ_a + δ_b + ε
where y = `decision`, α_f = scenario FE, γ_a / δ_b = (model_a × model_b)
FE. HC1 robust SEs.

Subsample target: n=100 per feature-level-cell (env=AAR collapses; cell =
scenario × condition × model_a × model_b). Cells below quota are flagged
in the audit and rendered with lighter color in plots.

Outputs (in the same stage folders as E1-E3, prefixed `aar_`):
    data/pipeline/aar_subsample.parquet
    analysis/03_subsample/aar_subsample_audit.{csv,md}
    analysis/05_regressions/aar_tier_{A,B,C}_main.{parquet,csv}
    analysis/05_regressions/aar_main_effects.{md,tex,png,pdf}
    analysis/06_plots/aar_summary.{png,pdf}           (two panels:
                                                       heatmap + bars)
    analysis/06_plots/aar_forest.{png,pdf}            (forest plot, standalone)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .plots import _pair_label, _short_model as _model_id
from .regression import (
    _controls_label,
    _padj,
    _short_model,
    _stars,
    add_bonferroni,
    fit_tier,
)
from .spec import (
    DATA_DIR,
    PLOTS_DIR,
    REGRESSIONS_DIR,
    ROOT,
    SUBSAMPLE_DIR,
    RegressionTier,
)


# ---------------------------------------------------------------------------
# Constants — all knobs in one place
# ---------------------------------------------------------------------------

# Pooled AAR samples (L1 + L3 attacks + shared baseline). T = (condition ==
# "attack") pools both attack levels against the baseline. Single source of
# truth, shared with the core pipeline (io._normalize_aar reads the same file).
INPUT_PARQUET = ROOT / "final_data" / "e5.parquet"

# AAR outputs land alongside the E1-E3 outputs in the existing stage folders,
# distinguished by the `aar_` filename prefix. Intermediate parquets land in
# data/pipeline/ alongside q{1,2}_subsample.parquet.
AAR_DATA_DIR    = DATA_DIR
AAR_SUBSAMPLE_DIR    = SUBSAMPLE_DIR
AAR_REGRESSIONS_DIR  = REGRESSIONS_DIR
AAR_PLOTS_DIR        = PLOTS_DIR

# Subsample target. Q2 only (no Q1 behavioral filter for E5 — AAR has no
# mechanism_a_* flags; the technique-count signal IS the outcome via drift).
N_PER_CELL = 100
RANDOM_SEED = 42

OUTCOME_COL = "decision"   # int 0/1 — research-pivot indicator (drift ≥ 0.5)

SIG_ALPHA = 0.05
NORMAL_ALPHA = 0.95
LIGHT_ALPHA = 0.35

# Three regression specifications, run on disjoint slices.
@dataclass(frozen=True)
class RegressionDef:
    name: str
    label: str        # display label
    description: str  # one-line description for the audit


REGRESSIONS: tuple[RegressionDef, ...] = (
    RegressionDef(
        name="main",
        label="Persuasion vs baseline",
        description=(
            "T=1 iff condition=='attack'. Pools both persuasion sub-conditions "
            "(goal-only L1 and scaffolded L3) against the shared baseline. "
            "Baseline is treatment-neutral so it serves as comparator for both."
        ),
    ),
)


# Three regression tiers (mirrors the main pipeline but with AAR-specific
# units — no env dimension since AAR is a single env, and a scenario-level B
# tier instead of env × capability).
TIER_A = RegressionTier(name="A_main",        unit=(),                                feature_fe=True,  pair_fe=True)
TIER_B = RegressionTier(name="B_by_scenario", unit=("feature_name",),                 feature_fe=False, pair_fe=True)
TIER_C = RegressionTier(name="C_by_cell",     unit=("feature_name","model_a","model_b"), feature_fe=False, pair_fe=False)
# (Tier C: pair is constant within cell, so pair FE collapses; same for feature FE.)
# Tier B: feature is constant within unit, so feature FE collapses.

TIERS: tuple[RegressionTier, ...] = (TIER_A, TIER_B, TIER_C)


# Cell unit for subsampling and quota checks (no env — single env in AAR data).
CELL_KEYS: tuple[str, ...] = ("feature_name", "instruction_level",
                                "model_a", "model_b")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_aar_samples(path: Path | None = None) -> pd.DataFrame:
    """Load AAR samples — baseline + pooled L1+L3 persuasion attacks.

    The raw parquet has two sub-conditions of persuasion: goal-only (L1, a
    light-weight prompt that just states the redirect objective) and
    scaffolded (L3, a 5-lever rhetorical playbook). Both are kept and pooled
    against the shared baseline:
        - baseline rows         (condition == "baseline", techniques_str empty)
        - L1 attack rows        (condition == "attack" AND instruction_level == 1)
        - L3 attack rows        (condition == "attack" AND instruction_level == 3)

    All baselines are stored at instruction_level=3, but baseline runs apply
    no persuasion technique (techniques_str is empty), so the level label on
    baseline rows is a storage artifact and the baseline is treatment-neutral
    with respect to attack level. T = (condition == 'attack') gives the
    pooled persuasion effect averaged over the L1/L3 mix.

    Adds:
        T_main   — 1 iff condition == "attack" (pooled L1+L3)
        approved — alias for `decision` (int 0/1) for spec consistency
    """
    p = path or INPUT_PARQUET
    df = pd.read_parquet(p)

    # Outcome alias.
    df["approved"] = df[OUTCOME_COL].astype(int)

    # T_main: 1 if attack (L1 or L3), 0 if baseline.
    df["T_main"] = (df["condition"] == "attack").astype(int)

    return df


# ---------------------------------------------------------------------------
# Subsample
# ---------------------------------------------------------------------------


def subsample_per_cell(df: pd.DataFrame, n_target: int = N_PER_CELL,
                        seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample at most ``n_target`` rows per cell.

    Cell here = (feature_name, instruction_level, model_a, model_b). Baseline
    rows have instruction_level=3 in the data (the underlying default), so
    we use the (condition, instruction_level) joint key implicitly via
    subsetting before sampling.

    Cells with fewer than n_target rows are kept in full and flagged
    `hit_target = False` in the audit.
    """
    if df.empty:
        return df.copy(), pd.DataFrame()

    # Subsample separately within (condition, instruction_level) so each
    # condition gets its full quota per scenario × pair.
    chunks: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    rng_seed = seed

    group_keys = ["feature_name", "condition", "instruction_level",
                  "model_a", "model_b"]
    for keys, g in df.groupby(group_keys, dropna=False, sort=True):
        n_pre = int(len(g))
        if n_pre <= n_target:
            sampled = g
        else:
            sampled = g.sample(n=n_target, random_state=rng_seed, replace=False)
            rng_seed += 1
        chunks.append(sampled)
        rec = dict(zip(group_keys, keys))
        rec["n_pre"] = n_pre
        rec["n_target"] = int(n_target)
        rec["n_post"] = int(len(sampled))
        rec["hit_target"] = bool(len(sampled) >= n_target)
        audit_rows.append(rec)

    out = pd.concat(chunks, ignore_index=True) if chunks else df.iloc[0:0].copy()
    audit = pd.DataFrame(audit_rows)
    return out, audit


# LPM fit core (FitResult, fit_one, fit_tier, _stars, _controls_label) lives
# in regression.py — see the imports at the top of this module. AAR-specific
# wiring: pass T_col="T_main" and outcome_col=OUTCOME_COL ("decision") to the
# shared fit functions.


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_main_effects_md(tier_a_results: dict[str, pd.DataFrame]) -> str:
    """Tier A summary."""
    L: list[str] = []
    L.append("# AAR (E5) — main effect")
    L.append("")
    L.append("Persuasion (pooled L1 + L3) vs baseline. Both attack sub-conditions "
             "(goal-only L1 and scaffolded L3) share the treatment-neutral baseline. "
             "Q2 only — no Q1 behavioral filter because AAR has no mechanism flags.")
    L.append("")
    L.append("Spec: `y = α + β·T + feature FE + (A×B) FE + ε`, HC1 SE. "
             "y = `decision` (research-pivot indicator).")
    L.append("")
    L.append("| Regression | N | β̂_T | HC1 SE | 95% CI | p | sig | Control mean | Treated mean | R² |")
    L.append("|---|---:|---:|---:|---|---:|---|---:|---:|---:|")
    for reg in REGRESSIONS:
        df = tier_a_results.get(reg.name)
        if df is None or df.empty:
            continue
        r = df.iloc[0]
        ci = (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
              if np.isfinite(r["ci_low"]) else "—")
        L.append(
            f"| {reg.label} "
            f"| {int(r['n']):,} "
            f"| {r['beta']:+.4f} "
            f"| ({r['se']:.4f}) "
            f"| {ci} "
            f"| {r['p']:.4g} "
            f"| {_stars(_padj(r))} "
            f"| {r['control_mean']:.3f} "
            f"| {r['treated_mean']:.3f} "
            f"| {r['r_squared']:.3f} |"
        )
    L.append("")
    return "\n".join(L) + "\n"


def render_main_effects_tex(tier_a_results: dict[str, pd.DataFrame]) -> str:
    """LaTeX: Tier A as a single-column table."""
    out: list[str] = []
    out.append("% Requires preamble: \\usepackage{booktabs,threeparttable,amsmath}")
    out.append("\\begin{table}[!htbp]")
    out.append("\\centering")
    out.append("\\caption{AAR (E5) main treatment effect — persuasion vs baseline.}")
    out.append("\\label{tab:aar_main_effect}")
    out.append("\\begin{threeparttable}")
    out.append("\\begin{tabular}{lc}")
    out.append("\\toprule")
    out.append(" & Persuasion vs baseline \\\\")
    out.append("\\midrule")

    def cell(reg_name: str, fn) -> str:
        df = tier_a_results.get(reg_name)
        if df is None or df.empty:
            return "--"
        return fn(df.iloc[0])

    fmt_beta = lambda r: f"${r['beta']:+.4f}^{{{_stars(_padj(r))}}}$" if np.isfinite(r["beta"]) else "--"
    fmt_se = lambda r: f"$({r['se']:.4f})$"
    fmt_ci = lambda r: f"$[{r['ci_low']:+.3f},\\;{r['ci_high']:+.3f}]$"
    fmt_3 = lambda field: (lambda r: f"{r[field]:.3f}")
    fmt_int = lambda field: (lambda r: f"{int(r[field]):,}")

    for label, fn in [
        (r"$\hat\beta_T$",   fmt_beta),
        (r"HC1 SE",          fmt_se),
        (r"95\% CI",         fmt_ci),
        (r"Control mean",    fmt_3("control_mean")),
        (r"Treated mean",    fmt_3("treated_mean")),
        (r"$N$",             fmt_int("n")),
        (r"$R^2$",           fmt_3("r_squared")),
    ]:
        cells = [cell(r.name, fn) for r in REGRESSIONS]
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\begin{tablenotes}[flushleft]\\footnotesize")
    out.append("\\item LPM, HC1 robust SE. Outcome: \\texttt{decision} (research-pivot indicator).")
    out.append("\\item All specifications include feature FE (3 scenarios) "
               "and (\\agentA{} $\\times$ \\agentB{}) FE.")
    out.append("\\item Significance (Bonferroni-corrected $p$, family = the tests in this table): "
               "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.")
    out.append("\\end{tablenotes}")
    out.append("\\end{threeparttable}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def render_main_effects_png(tier_a_results: dict[str, pd.DataFrame],
                              out_path: Path) -> None:
    """Same headline table as PNG (no LaTeX needed)."""
    headers = ["Persuasion vs baseline"]

    def fmt(reg_name: str, kind: str) -> str:
        df = tier_a_results.get(reg_name)
        if df is None or df.empty:
            return "—"
        r = df.iloc[0]
        if kind == "beta":
            return f"{r['beta']:+.4f}{_stars(_padj(r))}" if np.isfinite(r["beta"]) else "—"
        if kind == "se":
            return f"({r['se']:.4f})" if np.isfinite(r["se"]) else "—"
        if kind == "ci":
            return (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                    if np.isfinite(r["ci_low"]) else "—")
        if kind == "p":
            return f"{r['p']:.4g}" if np.isfinite(r["p"]) else "—"
        if kind == "control_mean":
            return f"{r['control_mean']:.3f}" if np.isfinite(r["control_mean"]) else "—"
        if kind == "treated_mean":
            return f"{r['treated_mean']:.3f}" if np.isfinite(r["treated_mean"]) else "—"
        if kind == "n":
            return f"{int(r['n']):,}"
        if kind == "r2":
            return f"{r['r_squared']:.3f}" if np.isfinite(r["r_squared"]) else "—"
        return "—"

    rows = [
        ("β̂_T",          "beta"),
        ("HC1 SE",        "se"),
        ("95% CI",        "ci"),
        ("p",             "p"),
        ("Control mean",  "control_mean"),
        ("Treated mean",  "treated_mean"),
        ("N",             "n"),
        ("R²",            "r2"),
    ]
    cell_text = [[""] + headers] + [
        [label] + [fmt(r.name, kind) for r in REGRESSIONS] for label, kind in rows
    ]

    fig, ax = plt.subplots(figsize=(6.0, 0.45 * (len(rows) + 1) + 1.2))
    ax.set_axis_off()
    table = ax.table(cellText=cell_text, loc="center", cellLoc="center",
                      colWidths=[0.30, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for j in range(len(headers) + 1):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#eeeeee")
    for i in range(1, len(rows) + 1):
        table[(i, 0)].set_facecolor("#fafafa")
        table[(i, 0)].set_text_props(ha="left")

    ax.set_title("AAR (E5) — main treatment effect (persuasion vs baseline)",
                  fontsize=12, fontweight="bold", pad=10)
    fig.text(0.5, 0.04,
              "LPM, HC1 SE. y = decision (research-pivot indicator). "
              "FE: feature (3 scenarios) + (a × b) pair. "
              "Stars use Bonferroni-corrected p: *** p<0.001, ** p<0.01, * p<0.05.",
              ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — heatmap (panel a) and bar chart (panel b)
# ---------------------------------------------------------------------------


def load_pipeline_cells() -> pd.DataFrame:
    """E5 rows of the pipeline's tier C Q2 table (one pooled fit per (A, B)
    cell with scenario FE, Bonferroni over the whole tier C family).

    The results text counts significant E5 cells from this table, so the
    heatmap stars the same cells. Empty if the pipeline has not been run.
    """
    path = REGRESSIONS_DIR / "tier_C_q2.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["environment"] == "aar"]
    cols = ["model_a", "model_b", "n", "beta", "se", "p_bonf", "fit_status"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def plot_heatmap_main(tier_c_main: pd.DataFrame,
                        out_path: Path | None = None,
                        *,
                        ax: "plt.Axes | None" = None,
                        panel_label: str | None = None,
                        cell_rows: pd.DataFrame | None = None) -> None:
    """Per (A, B) cell uplift heatmap for persuasion vs baseline.

    Visual conventions (match E1-E3):
      - Red-blue diverging colormap (RdBu_r). Higher β = more pivot = more
        successful research-direction redirection = worse from a safety
        perspective. So we map positive β to red, negative β to blue.
      - Significance threshold: single `*` for p < 0.05 (matches the
        filled-marker convention in E1-E3 forest plots).

    Args:
        tier_c_main: per-(scenario, A, B) regression rows from persuasion-vs-baseline.
        out_path: if set, save standalone PNG/PDF here. Ignored if ax is given.
        ax: if set, draw onto this Axes; caller manages figure save.
        panel_label: optional prefix for the title (e.g., "(a)").
        cell_rows: pooled per-(A, B) rows from the pipeline's tier C table
            (`load_pipeline_cells`). When given, the cell estimates and stars
            come from these rows, so the figure matches the results text;
            otherwise the scenario-level rows are pooled here.
    """
    df = tier_c_main.copy()
    if df.empty and (cell_rows is None or cell_rows.empty):
        return

    # Pool the 3 scenario-level estimates per (A, B) cell using inverse-
    # variance weighting. This gives β̂_(A,B) and its SE.
    #
    # Two failure modes are handled:
    #   (a) "se_zero_or_nan" — outcome is constant within both T=0 and T=1
    #       (e.g., the attacker produces zero pivots in either condition).
    #       This means β = 0 by construction; we treat it as a valid zero
    #       estimate (with no usable SE) rather than as missing data.
    #   (b) Genuinely no fits (all T_constant or n_too_small) — display "—".
    def _pool(group: pd.DataFrame) -> pd.Series:
        ok = group[(group["fit_status"] == "ok") & np.isfinite(group["se"]) & (group["se"] > 0)]
        deg = group[
            ((group["fit_status"] == "se_zero_or_nan") |
             ((group["fit_status"] == "ok") & (group["se"] == 0)))
            & (group["beta"].abs() < 1e-12)
        ]

        if not ok.empty:
            w = 1.0 / (ok["se"] ** 2)
            beta = float((ok["beta"] * w).sum() / w.sum())
            se = float(np.sqrt(1.0 / w.sum()))
            _pcol = "p_bonf" if "p_bonf" in ok.columns else "p"
            return pd.Series({
                "beta_pooled": beta,
                "se_pooled": se,
                "n_total": int(group["n"].sum()),
                "n_scenarios": int(len(ok)),
                "p_min": float(ok[_pcol].min()),  # Bonferroni-adjusted across scenarios
                "is_degenerate": False,
            })

        if not deg.empty:
            # All scenarios are degenerate — y is constant in both T arms,
            # so β=0 with infinite precision (in the limit). Display as +0.0
            # with no stars; the SE is shown as "(deg)" for transparency.
            return pd.Series({
                "beta_pooled": 0.0,
                "se_pooled": np.nan,
                "n_total": int(group["n"].sum()),
                "n_scenarios": 0,
                "p_min": np.nan,
                "is_degenerate": True,
            })

        return pd.Series({
            "beta_pooled": np.nan, "se_pooled": np.nan,
            "n_total": int(group["n"].sum()),
            "n_scenarios": 0, "p_min": np.nan,
            "is_degenerate": False,
        })

    if cell_rows is not None and not cell_rows.empty:
        c = cell_rows.copy()
        deg = (c["fit_status"] == "se_zero_or_nan") & (c["beta"].abs() < 1e-12)
        agg = pd.DataFrame({
            "model_a": c["model_a"], "model_b": c["model_b"],
            "beta_pooled": np.where(deg, 0.0, c["beta"]),
            "se_pooled": np.where(deg, np.nan, c["se"]),
            "n_total": c["n"].astype(int),
            "n_scenarios": np.where(deg, 0, 3),
            "p_min": np.where(deg, np.nan, c["p_bonf"]),
            "is_degenerate": deg,
        })
    else:
        agg = df.groupby(["model_a", "model_b"], dropna=False).apply(_pool, include_groups=False).reset_index()

    pivot_beta = agg.pivot_table(index="model_a", columns="model_b",
                                    values="beta_pooled")
    pivot_se = agg.pivot_table(index="model_a", columns="model_b",
                                  values="se_pooled")
    pivot_p = agg.pivot_table(index="model_a", columns="model_b",
                                 values="p_min")
    pivot_n = agg.pivot_table(index="model_a", columns="model_b",
                                 values="n_total")

    # Sort rows and columns alphabetically by raw model id (matches E1-E3).
    pivot_beta = pivot_beta.sort_index().sort_index(axis=1)
    pivot_se = pivot_se.reindex_like(pivot_beta)
    pivot_p = pivot_p.reindex_like(pivot_beta)
    pivot_n = pivot_n.reindex_like(pivot_beta)

    row_labels = [_model_id(m) for m in pivot_beta.index]
    col_labels = [_model_id(m) for m in pivot_beta.columns]

    style.apply()
    own_fig = ax is None
    if own_fig:
        # Drawn at half the text width so the two E5 panels sit side by side
        # as subfigures without being scaled down.
        n_r, n_c = pivot_beta.shape
        fig_w = style.TEXT_WIDTH_IN * 0.57
        fig_h = 0.25 * n_r + 0.95
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=1.05 / fig_w, right=1 - 0.5 / fig_w,
                            top=0.99, bottom=0.9 / fig_h)
    else:
        fig = ax.figure
    arr = pivot_beta.to_numpy() * 100  # convert to pp
    se_arr = pivot_se.to_numpy() * 100
    p_arr = pivot_p.to_numpy()
    n_arr = pivot_n.to_numpy()
    deg_pivot = (
        agg.pivot_table(index="model_a", columns="model_b",
                          values="is_degenerate", aggfunc="first")
            .reindex_like(pivot_beta)
    )
    deg_arr = deg_pivot.to_numpy()
    vmax = np.nanmax(np.abs(arr)) if arr.size and np.isfinite(arr).any() else 30

    # Diverging map from the shared palette: positive β (more pivot, i.e. a
    # more successful redirection) is warm, negative is cool, zero is neutral.
    from matplotlib.colors import LinearSegmentedColormap
    from .spec import DIVERGING_COLORS
    cmap = LinearSegmentedColormap.from_list("effect", list(DIVERGING_COLORS))
    im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(pivot_beta.shape[1]))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(pivot_beta.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Target Model")
    ax.set_ylabel("Attacker Model")
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if panel_label:
        ax.set_title(panel_label, loc="left")

    # Annotate cells with β + (single * if p<.05) + SE + n.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center",
                         fontsize=style.FS_SMALL, color="#666", style="italic")
                continue
            p_v = p_arr[i, j]
            star = "*" if (np.isfinite(p_v) and p_v < SIG_ALPHA) else ""
            se_v = se_arr[i, j]
            n_v = n_arr[i, j]
            is_deg = bool(deg_arr[i, j]) if np.isfinite(arr[i, j]) and not pd.isna(deg_arr[i, j]) else False
            color = "white" if abs(v) > vmax * 0.55 else style.PALETTE["ink"]
            txt = f"{round(v, 1) + 0.0:+.1f}{star}"  # +0.0 rounds away a negative zero
            if is_deg:
                txt += " (deg.)"
            elif np.isfinite(se_v):
                txt += f" ({se_v:.1f})"
            if np.isfinite(n_v):
                txt += f"\nn = {int(n_v)}"
            ax.text(j, i, txt, ha="center", va="center",
                     fontsize=style.FS_SMALL - 0.5, color=color, linespacing=1.25)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Effect on Pivot Rate (pp)")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, width=style.LW_AXIS)

    if own_fig and out_path is not None:
        style.save(fig, out_path)


def plot_forest_aar(tier_c_main: pd.DataFrame,
                     tier_a_main: pd.DataFrame,
                     out_path: Path | None = None,
                     *,
                     axes: "list[plt.Axes] | None" = None,
                     suptitle: bool = True) -> None:
    """Forest plot for E5 — analogous to the E1-E3 per-capability forest plots.

    Three panels (one per scenario), rows for each (model_a, model_b) pair
    sorted alphabetical, single marker per row showing β̂_T from the
    persuasion-vs-baseline regression at that cell. Filled if p<0.05; lighter
    colour if regression N below the per-cell target.

    The pooled main effect (Tier A) is drawn as a dotted vline shared
    across all panels.

    Args:
        out_path: standalone save path. Ignored if axes is given.
        axes: list of pre-created Axes (one per scenario) to draw onto.
              If None, creates its own figure.
        suptitle: whether to draw a figure-level title (only used when
                  creating the figure standalone).
    """
    if tier_c_main.empty:
        return

    df = tier_c_main.copy()
    scenarios = sorted(df["feature_name"].dropna().unique())
    if not scenarios:
        return

    # Universe of pairs across all scenarios — sorted alphabetical (matches E1-E3).
    pairs = (
        df[["model_a", "model_b"]]
        .drop_duplicates()
        .sort_values(["model_a", "model_b"])
        .reset_index(drop=True)
    )
    pair_labels = [_pair_label(a, b) for a, b in zip(pairs["model_a"], pairs["model_b"])]
    pair_index = {(a, b): i for i, (a, b) in enumerate(zip(pairs["model_a"], pairs["model_b"]))}

    n_panels = len(scenarios)
    n_rows = len(pairs)

    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(5.0 * n_panels, max(5.5, 0.42 * n_rows + 2.0)),
            sharey=True, sharex=False,
        )
        if n_panels == 1:
            axes = [axes]
    else:
        fig = axes[0].figure
        if len(axes) != n_panels:
            raise ValueError(
                f"plot_forest_aar: got {len(axes)} axes but expected {n_panels} "
                f"(one per scenario)"
            )

    # Per-cell target N for the lighter-color rule.
    # AAR Tier C cell = (scenario, A, B). Each scenario contributes
    # roughly 17 baseline + 68 attack rows per pair (post-subsample),
    # so target N ~ 85.
    target_n = 80

    # Pooled main β for the dotted vline.
    pooled_beta = (
        float(tier_a_main["beta"].iloc[0]) if not tier_a_main.empty
        and np.isfinite(tier_a_main["beta"].iloc[0]) else None
    )

    main_color = "#1f77b4"  # blue, same hue as Q2 in E1-E3 forest plots

    for ax, scenario in zip(axes, scenarios):
        sub = df[df["feature_name"] == scenario].set_index(["model_a", "model_b"])

        for (a, b), i in pair_index.items():
            if (a, b) not in sub.index:
                continue
            row = sub.loc[(a, b)]
            if not np.isfinite(row["beta"]):
                continue

            beta = row["beta"]
            ci_lo = row["ci_low"]
            ci_hi = row["ci_high"]
            p_val = row["p_bonf"] if ("p_bonf" in row and np.isfinite(row.get("p_bonf", np.nan))) else row["p"]
            n_val = row["n"]

            # Below-quota cells get lighter colour
            lighter = bool(np.isfinite(n_val) and n_val < target_n)
            alpha = 0.35 if lighter else 0.95
            lw = 1.0 if lighter else 1.2

            # Horizontal CI bar (only if CI is finite)
            if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                ax.hlines(i, ci_lo, ci_hi, color=main_color, lw=lw, alpha=alpha)

            # Marker: filled if p<0.05, open otherwise
            sig = bool(np.isfinite(p_val) and p_val < SIG_ALPHA)
            if sig:
                ax.scatter([beta], [i], s=46, marker="o",
                            color=main_color, edgecolor="black", linewidth=0.6,
                            alpha=alpha, zorder=3)
            else:
                ax.scatter([beta], [i], s=46, marker="o",
                            facecolor="white", edgecolor=main_color, linewidth=1.3,
                            alpha=alpha, zorder=3)

        # Vertical reference lines
        ax.axvline(0, color="grey", lw=0.6, linestyle="-", alpha=0.4)
        if pooled_beta is not None:
            ax.axvline(pooled_beta, color=main_color, lw=1.0, linestyle=":",
                        alpha=0.85)

        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(pair_labels, fontsize=7.5)
        ax.set_xlabel(r"$\hat\beta_T$ on $\mathrm{decision}$ (pp)", fontsize=9)
        # Pretty scenario title
        scenario_title = scenario.replace("_", " ").title()
        ax.set_title(scenario_title, fontsize=10)
        ax.invert_yaxis()  # alphabetical top-to-bottom

    if own_fig:
        # Legend at the bottom of the figure when standalone.
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="none",
                    markerfacecolor=main_color, markeredgecolor="black",
                    markersize=8, label="persuasion vs baseline: β̂_T ± 95% CI (filled if Bonferroni-corrected p<0.05; one star in tables)"),
            Line2D([0], [0], marker="o", color="none",
                    markerfacecolor="white", markeredgecolor=main_color,
                    markeredgewidth=1.3, markersize=8, label="open marker: p ≥ 0.05"),
            Line2D([0], [0], color=main_color, lw=1.5, alpha=0.95,
                    label="full sample"),
            Line2D([0], [0], color=main_color, lw=1.5, alpha=0.35,
                    label=f"below quota (n < {target_n})"),
            Line2D([0], [0], color=main_color, lw=1.0, linestyle=":",
                    label=r"pooled main β̂_T (Tier A)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0), ncol=3,
                    fontsize=8, frameon=False)
        if suptitle:
            fig.suptitle("AAR forest — per-pair persuasion vs baseline, by scenario",
                          fontsize=12, fontweight="bold", y=1.0)
        fig.tight_layout(rect=(0, 0.10, 1, 0.97))

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_bars_by_attacker(tier_c_results: dict[str, pd.DataFrame],
                            df_subsample: pd.DataFrame,
                            out_path: Path | None = None,
                            *,
                            ax: "plt.Axes | None" = None,
                            panel_label: str | None = None) -> None:
    """Bar chart: baseline rate vs persuasion rate per attacker, derived from regression.

    For each attacker model (averaging across targets and scenarios):
      - Baseline rate     mean control_mean over the attacker's cells
      - Persuasion rate   mean of (control_mean + β̂_T) over the same cells
    Error bars on the persuasion bars are 95% CIs for that mean, with
    SE = sqrt(sum(se_i^2)) / k over the k cells (independent cell fits).

    Args:
        out_path: standalone save path. Ignored if ax is given.
        ax: if set, draw onto this Axes; caller manages figure save.
        panel_label: optional title (e.g., "(b)").
    """
    from .spec import BASELINE_COLOR, Q2_COLOR
    style.apply()
    own_fig = ax is None
    if own_fig:
        fig_w, fig_h = style.TEXT_WIDTH_IN * 0.41, 0.25 * 8 + 0.80
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0.42 / fig_w, right=0.99, top=0.99, bottom=0.9 / fig_h)
    else:
        fig = ax.figure

    main = tier_c_results.get("main", pd.DataFrame())
    if main.empty:
        return

    d = main.copy()
    d["treated_rate"] = d["control_mean"] + d["beta"]
    d["se2"] = d["se"].fillna(0.0) ** 2
    main_agg = d.groupby("model_a").agg(
        control_rate=("control_mean", "mean"),
        treated_rate=("treated_rate", "mean"),
        se2=("se2", "sum"), k=("beta", "size"), n=("n", "sum"),
    ).reset_index()
    main_agg["se"] = np.sqrt(main_agg["se2"]) / main_agg["k"]

    attackers = sorted(main_agg["model_a"].tolist())
    if not attackers:
        return
    x = np.arange(len(attackers))
    width = 0.38
    look = main_agg.set_index("model_a")
    baseline_vals = [look.loc[a, "control_rate"] for a in attackers]
    treated_vals = [look.loc[a, "treated_rate"] for a in attackers]
    treated_errs = [1.96 * look.loc[a, "se"] for a in attackers]

    ax.bar(x - width / 2, baseline_vals, width * 0.94, color=BASELINE_COLOR,
           label="Baseline", lw=0, zorder=3)
    ax.bar(x + width / 2, treated_vals, width * 0.94, color=Q2_COLOR,
           label="Persuasion", lw=0, zorder=3)
    ax.errorbar(x + width / 2, treated_vals, yerr=treated_errs, fmt="none",
                ecolor="#3a3a3a", elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_id(a) for a in attackers], rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Attacker Model")
    ax.set_ylabel("Pivot Rate")
    if panel_label:
        ax.set_title(panel_label, loc="left")
    ax.legend(loc="upper right", handlelength=1.0, handleheight=1.0)
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    style.tidy_axes(ax, grid_axis="y")

    if own_fig and out_path is not None:
        style.save(fig, out_path)


def plot_aar_summary(tier_c_main: pd.DataFrame,
                        tier_c_results: dict[str, pd.DataFrame],
                        df_subsample: pd.DataFrame,
                        out_path: Path,
                        cell_rows: pd.DataFrame | None = None) -> dict[str, Path]:
    """The paper's E5 figure: (a) heatmap and (b) per-attacker bars.

    The two panels are written as SEPARATE files (``<out>_heatmap`` and
    ``<out>_bars``) so the paper can place them as LaTeX subfigures with
    their own subcaptions; ``<out>`` itself is the two-panel composite kept
    for quick inspection. Panel labels and titles live in the LaTeX captions.
    """
    out_path = Path(out_path)
    heat = out_path.with_name(out_path.stem + "_heatmap")
    bars = out_path.with_name(out_path.stem + "_bars")
    plot_heatmap_main(tier_c_main, heat, cell_rows=cell_rows)
    plot_bars_by_attacker(tier_c_results, df_subsample, bars)

    fig, (ax_heatmap, ax_bars) = plt.subplots(
        1, 2, figsize=(style.TEXT_WIDTH_IN * 1.4, 3.4),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.45},
    )
    plot_heatmap_main(tier_c_main, ax=ax_heatmap, panel_label="(a)",
                      cell_rows=cell_rows)
    plot_bars_by_attacker(tier_c_results, df_subsample, ax=ax_bars, panel_label="(b)")
    style.save(fig, out_path)
    return {"heatmap": heat.with_suffix(".pdf"), "bars": bars.with_suffix(".pdf")}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run() -> dict[str, Path]:
    """End-to-end AAR analysis. Outputs land in the existing stage folders
    alongside the E1-E3 outputs, prefixed with `aar_` in their filenames.
    """
    AAR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    AAR_SUBSAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    AAR_REGRESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    AAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_aar_samples()
    print(f"  loaded {len(df):,} AAR samples; "
          f"capabilities={df['capability'].value_counts().to_dict()}")

    # Subsample to n=100 per cell.
    sub, audit = subsample_per_cell(df)
    print(f"  subsampled: {len(sub):,} rows; "
          f"{audit['hit_target'].sum()}/{len(audit)} cells hit target")

    written: dict[str, Path] = {}

    # Intermediate parquet — sits with q1/q2 subsample parquets in data/pipeline/
    p_sub = AAR_DATA_DIR / "aar_subsample.parquet"
    sub.to_parquet(p_sub, index=False)
    written["aar_subsample_parquet"] = p_sub

    # Subsample audit lands in 03_subsample/ alongside the E1-E3 audit
    p_audit = AAR_SUBSAMPLE_DIR / "aar_subsample_audit.csv"
    audit.to_csv(p_audit, index=False)
    written["aar_subsample_audit_csv"] = p_audit

    L = ["# AAR subsample audit", "",
         f"Target: n={N_PER_CELL} per "
         f"(scenario, condition, instruction_level, model_a, model_b) cell.", ""]
    by_cond = audit.groupby(["condition", "instruction_level"]).agg(
        cells=("n_pre", "size"),
        hit=("hit_target", "sum"),
        rows_pre=("n_pre", "sum"),
        rows_post=("n_post", "sum"),
    ).reset_index()
    L.append("| condition | instruction_level | cells | hit-target | rows pre | rows post |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for _, r in by_cond.iterrows():
        L.append(f"| {r['condition']} | {int(r['instruction_level'])} | "
                  f"{int(r['cells'])} | {int(r['hit'])} | "
                  f"{int(r['rows_pre']):,} | {int(r['rows_post']):,} |")
    p_md = AAR_SUBSAMPLE_DIR / "aar_subsample_audit.md"
    p_md.write_text("\n".join(L) + "\n")
    written["aar_subsample_audit_md"] = p_md

    # Regressions: 3 specs × 3 tiers = 9 result tables → 05_regressions/
    tier_a_results: dict[str, pd.DataFrame] = {}
    tier_c_results: dict[str, pd.DataFrame] = {}

    print()
    for reg in REGRESSIONS:
        T_col = f"T_{reg.name}"
        for tier in TIERS:
            res = fit_tier(sub, tier, T_col=T_col, outcome_col=OUTCOME_COL)
            if res.empty:
                continue
            res["regression"] = reg.name
            res["controls"] = [
                _controls_label(nf, np_) for nf, np_ in
                zip(res["n_features_in_fit"], res["n_pairs_in_fit"])
            ]
            # Bonferroni within this (tier x regression-spec) family — adds
            # m_tests + p_bonf; stars / marker fills downstream key on p_bonf.
            res = add_bonferroni(res)
            res["sig"] = res["p_bonf"].apply(_stars)

            stem = f"aar_tier_{tier.name.split('_')[0]}_{reg.name}"
            res.to_parquet(AAR_REGRESSIONS_DIR / f"{stem}.parquet", index=False)
            res.to_csv(AAR_REGRESSIONS_DIR / f"{stem}.csv", index=False)
            written[f"{stem}_parquet"] = AAR_REGRESSIONS_DIR / f"{stem}.parquet"

            n_ok = int((res["fit_status"] == "ok").sum())
            n_sig = int(((res["p_bonf"] < SIG_ALPHA) & np.isfinite(res["p_bonf"])).sum())
            print(f"  [{reg.name:>6}/{tier.name}] "
                  f"{len(res):>3} regs · ok={n_ok} · m={int(res['m_tests'].iloc[0])} · "
                  f"sig(Bonf)={n_sig} · median |β|={float(np.nanmedian(res['beta'].abs())):.4f}")

            if tier.name == "A_main":
                tier_a_results[reg.name] = res
            elif tier.name == "C_by_cell":
                tier_c_results[reg.name] = res

    # Headline tables → 05_regressions/
    p_md = AAR_REGRESSIONS_DIR / "aar_main_effects.md"
    p_md.write_text(render_main_effects_md(tier_a_results))
    written["aar_main_effects_md"] = p_md

    p_tex = AAR_REGRESSIONS_DIR / "aar_main_effects.tex"
    p_tex.write_text(render_main_effects_tex(tier_a_results))
    written["aar_main_effects_tex"] = p_tex

    p_png = AAR_REGRESSIONS_DIR / "aar_main_effects"
    render_main_effects_png(tier_a_results, p_png)
    written["aar_main_effects_png"] = p_png.with_suffix(".png")

    # Plots → 06_plots/
    if "main" in tier_c_results:
        tier_a_main = tier_a_results.get("main", pd.DataFrame())

        # Two-panel summary: (a) heatmap + (b) per-attacker bars.
        parts = plot_aar_summary(tier_c_results["main"], tier_c_results, sub,
                                 AAR_PLOTS_DIR / "aar_summary",
                                 cell_rows=load_pipeline_cells())
        written["aar_summary"] = AAR_PLOTS_DIR / "aar_summary.png"
        written["aar_summary_heatmap"] = parts["heatmap"]
        written["aar_summary_bars"] = parts["bars"]

        # Forest plot — kept as a standalone file, not in the summary.
        plot_forest_aar(tier_c_results["main"], tier_a_main,
                          AAR_PLOTS_DIR / "aar_forest")
        written["aar_forest"] = AAR_PLOTS_DIR / "aar_forest.png"

    return written


if __name__ == "__main__":
    paths = run()
    print()
    for k, p in paths.items():
        print(f"  {k}: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
