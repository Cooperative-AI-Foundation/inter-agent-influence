"""Stage 5: regressions on Q1 / Q2 subsamples.

Three tiers per Q (see spec.py):
  Tier A — per (capability)
  Tier B — per (env, capability)
  Tier C — per (env, capability, model_a, model_b)

Spec: y = α + β·T + feature FE + (A×B) FE
LPM with HC1 robust standard errors. FE columns that collapse to a constant
within the fit window are dropped automatically (so feature FE collapses for
1-feature envs in tier C; pair FE always collapses in tier C).

Outputs (analysis/results/):
  tier_<X>_q<n>.parquet     tidy long table — one row per regression
  tier_<X>_q<n>.csv         same as parquet, csv form
  main_effects_wide.tex     four-column LaTeX (Q1/Q2 × persuasion/coercion)
                             of the tier-A regressions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .io import add_e4_points_share, backfill_no_deal_points
from .spec import (
    significance_stars,
    CONFIDENCE_LEVEL,
    DATA_DIR,
    E4_ENVS,
    ENV_TO_CODE,
    env_code,
    E4_OUTCOME_COL,
    OUTCOME_COL,
    outcome_col_for_env,
    REGRESSIONS_DIR,
    REGRESSION_TIERS,
    RegressionTier,
    SE_TYPE,
    TIER_A,
    TIER_A_EXCLUDE_ENVS,
    FOREST_PAIR_ENVS_EXCLUDE,
)


# ---------------------------------------------------------------------------
# Single regression
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    n: int
    n_treat: int
    n_control: int
    beta: float
    se: float
    t: float
    p: float
    ci_low: float
    ci_high: float
    r_squared: float
    within_r_squared: float
    n_features: int
    n_pairs: int
    fit_status: str
    control_mean: float
    treated_mean: float
    outcome_col: str = OUTCOME_COL


def fit_one(df: pd.DataFrame, *, feature_fe: bool, pair_fe: bool,
            T_col: str = "T", outcome_col: str | None = None,
            confidence: float = CONFIDENCE_LEVEL,
            se_type: str = SE_TYPE) -> FitResult:
    """Fit y ~ T + (feature FE)? + (A×B FE)? on ``df`` with HC1.

    Drops FE columns that have only one unique level in the fit window.
    Rows with NaN in ``T_col`` are dropped defensively before fitting.

    Args:
        T_col: column to use as the treatment indicator (default ``"T"``;
            E5/AAR passes ``"T_main"``).
        outcome_col: outcome column name; defaults to ``spec.OUTCOME_COL``
            (= ``"approved"`` in the main pipeline; E5/AAR overrides to
            ``"decision"`` via its own module-level constant).
    """
    out_col = outcome_col if outcome_col is not None else OUTCOME_COL
    df = df.dropna(subset=[T_col, out_col])
    n = int(len(df))
    n_treat = int((df[T_col] == 1).sum())
    n_control = int((df[T_col] == 0).sum())
    control_mean = float(df.loc[df[T_col] == 0, out_col].mean()) if n_control else float("nan")
    treated_mean = float(df.loc[df[T_col] == 1, out_col].mean()) if n_treat else float("nan")

    base = FitResult(
        n=n, n_treat=n_treat, n_control=n_control,
        beta=float("nan"), se=float("nan"), t=float("nan"), p=float("nan"),
        ci_low=float("nan"), ci_high=float("nan"),
        r_squared=float("nan"),
        within_r_squared=float("nan"),
        n_features=int(df["feature_name"].nunique()) if "feature_name" in df.columns else 0,
        n_pairs=int(df.groupby(["model_a", "model_b"]).ngroups) if "model_a" in df.columns else 0,
        fit_status="ok",
        control_mean=control_mean, treated_mean=treated_mean,
        outcome_col=out_col,
    )

    if n_treat == 0 or n_control == 0:
        base.fit_status = "T_constant"
        return base
    if n < 4:
        base.fit_status = "n_too_small"
        return base

    y = df[out_col].astype(float).to_numpy()
    X_parts = [
        pd.Series(1.0, index=df.index, name="const"),
        df[T_col].astype(float).rename("T"),
    ]

    # Feature FE: only when more than one feature is present.
    if feature_fe and "feature_name" in df.columns:
        if df["feature_name"].nunique(dropna=False) > 1:
            X_parts.append(
                pd.get_dummies(df["feature_name"], prefix="feat",
                                drop_first=True).astype(float)
            )

    # Pair FE: interaction of (model_a, model_b). Skip if only one pair.
    if pair_fe and {"model_a", "model_b"}.issubset(df.columns):
        pair = df["model_a"].astype(str) + "||" + df["model_b"].astype(str)
        if pair.nunique() > 1:
            X_parts.append(
                pd.get_dummies(pair, prefix="pair",
                                drop_first=True).astype(float)
            )

    X = pd.concat(X_parts, axis=1)

    try:
        result = sm.OLS(y, X.to_numpy()).fit(cov_type=se_type)
    except Exception as e:  # noqa: BLE001
        base.fit_status = f"fit_error:{type(e).__name__}"
        return base

    coefs = pd.Series(result.params, index=X.columns)
    ses = pd.Series(result.bse, index=X.columns)
    beta = float(coefs["T"])
    se = float(ses["T"])
    base.beta = beta
    base.se = se
    base.r_squared = float(result.rsquared)

    # Within-R² (partial R² of T after partialing out FE): fraction of
    # outcome variance explained by treatment, net of feature & pair FE.
    X_no_T = X.drop(columns=["T"])
    if X_no_T.shape[1] > 0:
        try:
            ssr_full = float(result.ssr)
            ssr_short = float(sm.OLS(y, X_no_T.to_numpy()).fit().ssr)
            if ssr_short > 0:
                base.within_r_squared = max(0.0, 1.0 - ssr_full / ssr_short)
        except Exception:  # noqa: BLE001
            pass

    if not (np.isfinite(se) and se > 0):
        base.fit_status = "se_zero_or_nan"
        return base

    tval = beta / se
    df_resid = max(int(result.df_resid), 1)
    pval = 2 * (1 - stats.t.cdf(abs(tval), df=df_resid))
    crit = stats.t.ppf(0.5 + confidence / 2, df=df_resid)
    base.t = float(tval)
    base.p = float(pval)
    base.ci_low = float(beta - crit * se)
    base.ci_high = float(beta + crit * se)
    return base


# ---------------------------------------------------------------------------
# Tier driver
# ---------------------------------------------------------------------------


def _fr_to_dict(fr: FitResult) -> dict:
    """Tidy-dict view of a FitResult, with the canonical output column names."""
    return {
        "n": fr.n, "n_treat": fr.n_treat, "n_control": fr.n_control,
        "beta": fr.beta, "se": fr.se, "t": fr.t, "p": fr.p,
        "ci_low": fr.ci_low, "ci_high": fr.ci_high,
        "r_squared": fr.r_squared,
        "within_r_squared": fr.within_r_squared,
        "n_features_in_fit": fr.n_features,
        "n_pairs_in_fit": fr.n_pairs,
        "control_mean": fr.control_mean, "treated_mean": fr.treated_mean,
        "outcome_col": fr.outcome_col,
        "fit_status": fr.fit_status,
    }


def _resolve_outcome_col(g: pd.DataFrame, explicit: str | None) -> str:
    """Outcome column for one fit window: points in E4, binary approval elsewhere.

    An explicit ``outcome_col`` from the caller always wins (E5/AAR passes
    ``"decision"``). Otherwise the window's environments decide. The two scales
    cannot be pooled in one fit — a percentage-point effect and a points effect
    are different units — so a window that mixes E4 with anything else is a
    specification error, not something to average over. Tier A already excludes
    E4 (``spec.TIER_A_EXCLUDE_ENVS``) and tiers B/C group by environment, so no
    mixed window is reachable through the pipeline; this guards a hand-built one.
    """
    if explicit is not None:
        return explicit
    if "environment" not in g.columns:
        return OUTCOME_COL
    envs = set(g["environment"].dropna().unique())
    e4 = envs & E4_ENVS
    if not e4:
        return OUTCOME_COL
    if envs - E4_ENVS:
        raise ValueError(
            "regression window mixes E4 (negotiation points share) with "
            f"non-E4 (binary approval) environments: {sorted(envs)}. "
            "Split the fit or pass outcome_col explicitly."
        )
    return outcome_col_for_env(next(iter(e4)))


def tier_fit_rows(df: pd.DataFrame, tier: RegressionTier) -> pd.DataFrame:
    """Rows ``tier`` is fit on, after dropping envs the tier cannot pool.

    Tier A pools a capability's E1-E3 environments into one estimate; E4 and E5
    are dropped (``TIER_A_EXCLUDE_ENVS``). Tiers B and C group by environment
    and are unaffected. Applied here so every
    caller of Tier A inherits one row set; keyed on the tier's name, so E5/AAR's
    own ``A_main`` is untouched.
    """
    if (tier.name == TIER_A.name and TIER_A_EXCLUDE_ENVS
            and "environment" in df.columns):
        return df[~df["environment"].isin(TIER_A_EXCLUDE_ENVS)]
    return df


def fit_tier(df: pd.DataFrame, tier: RegressionTier, *,
             T_col: str = "T", outcome_col: str | None = None) -> pd.DataFrame:
    """Fit one regression per group defined by ``tier.unit``.

    If ``tier.unit`` is empty, a single regression is fit on the whole frame
    (used by E5/AAR's Tier A). ``T_col`` lets callers override the treatment
    column (E5/AAR passes ``"T_main"``).

    The outcome column is resolved per group by ``_resolve_outcome_col`` unless
    the caller names one, so E4 groups are fit on negotiation points while
    E1-E3/E5 groups stay on binary approval. The column used is reported back
    in the ``outcome_col`` field of every output row.
    """
    rows: list[dict] = []
    df = tier_fit_rows(df, tier)
    if tier.unit:
        grouped = df.groupby(list(tier.unit), dropna=False, sort=True)
        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(tier.unit, keys))
            rec["tier"] = tier.name
            fr = fit_one(g, feature_fe=tier.feature_fe, pair_fe=tier.pair_fe,
                         T_col=T_col, outcome_col=_resolve_outcome_col(g, outcome_col))
            rec.update(_fr_to_dict(fr))
            rows.append(rec)
    else:
        rec = {"tier": tier.name}
        fr = fit_one(df, feature_fe=tier.feature_fe, pair_fe=tier.pair_fe,
                     T_col=T_col, outcome_col=_resolve_outcome_col(df, outcome_col))
        rec.update(_fr_to_dict(fr))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multiple-hypothesis correction (Bonferroni, per tier x Q family)
# ---------------------------------------------------------------------------


def add_bonferroni(res: pd.DataFrame) -> pd.DataFrame:
    """Add Bonferroni-adjusted p-values to one tier x Q result table.

    The correction family is the set of regressions in THIS table — all
    conducted tests within a single (tier, Q). ``m`` is the number of tests that
    produced a finite p-value (failed fits don't count). ``p_bonf = min(p*m, 1)``.

    Significance stars and forest-plot marker fills key on ``p_bonf``; the raw
    ``p`` is retained for transparency.
    """
    res = res.copy()
    valid = res["p"].notna() & np.isfinite(res["p"])
    m = int(valid.sum())
    res["m_tests"] = m
    res["p_bonf"] = np.where(valid, np.minimum(res["p"] * m, 1.0), np.nan)
    return res


# ---------------------------------------------------------------------------
# LaTeX rendering for the headline 4-col table
# ---------------------------------------------------------------------------


def _outcome_note_plain(res: pd.DataFrame) -> str:
    """Plain-text twin of ``_outcome_clause`` for figure footers and markdown."""
    cols = set(res["outcome_col"].dropna().unique()) if "outcome_col" in res.columns else set()
    if E4_OUTCOME_COL in cols and len(cols) > 1:
        return ("Outcome = approved, except E4 rows = A's outcome score (0, 0.5 or 1), "
                "so E4 effects are in outcome score and the rest are in approval probability.")
    if cols == {E4_OUTCOME_COL}:
        return "Outcome = A's E4 outcome score (0, 0.5 or 1)."
    return "Outcome = approved."


def _outcome_clause(res: pd.DataFrame) -> str:
    """Caption clause naming the outcome(s) a table's regressions were fit on.

    Tables that span E1--E5 mix two scales: binary approval everywhere except
    E4, whose outcome is A's outcome score (0, 0.5 or 1). Reads the ``outcome_col``
    the fits reported rather than re-deriving it from the env list.
    """
    cols = set(res["outcome_col"].dropna().unique()) if "outcome_col" in res.columns else set()
    if E4_OUTCOME_COL in cols and len(cols) > 1:
        return ("outcome \\texttt{approved} outside E4 and, in E4, \\agentA{}'s "
                "outcome score (0, 0.5, or 1), so E4 effects are not an approval probability")
    if cols == {E4_OUTCOME_COL}:
        return "outcome \\agentA{}'s E4 outcome score (0, 0.5, or 1)"
    return "outcome \\texttt{approved}"


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return significance_stars(p)


def _padj(r) -> float:
    """Adjusted p for a result row, falling back to raw p if uncorrected."""
    v = r.get("p_bonf", np.nan) if hasattr(r, "get") else np.nan
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return r["p"]
    return v


def _build_main_effects_grid(tier_a_q1: pd.DataFrame,
                              tier_a_q2: pd.DataFrame
                              ) -> tuple[list[str], list[tuple[str, str]],
                                          list[tuple[str, str]],
                                          dict[tuple[str, str], pd.Series]]:
    """Shared helper: build column keys + headers + lookup for table renderers."""
    from .spec import CAPABILITIES

    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, r in tier_a_q1.iterrows():
        lookup[("Q1", r["capability"])] = r
    for _, r in tier_a_q2.iterrows():
        lookup[("Q2", r["capability"])] = r

    col_keys: list[tuple[str, str]] = []
    headers: list[str] = []
    for cap in CAPABILITIES:
        for q in ("Q1", "Q2"):
            col_keys.append((q, cap))
            headers.append(f"{q} {cap.title()}")
    return headers, col_keys, [], lookup


def render_main_effects_latex_wide(tier_a_q1: pd.DataFrame,
                                     tier_a_q2: pd.DataFrame) -> str:
    """Compact main effects table: rows = (Q, capability), cols = stats.

    Designed to fit a single-column NeurIPS page without \\resizebox. Each row
    is a (Q, capability) combination; columns are headline statistics. Drops
    Control/Treated means and the redundant ``p`` column (significance is
    encoded in the stars on $\\hat{\\beta}_T$).
    """
    from .spec import CAPABILITIES

    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, r in tier_a_q1.iterrows():
        lookup[("Q1", r["capability"])] = r
    for _, r in tier_a_q2.iterrows():
        lookup[("Q2", r["capability"])] = r

    def fmt_beta(r: pd.Series) -> str:
        if not np.isfinite(r["beta"]):
            return "--"
        return f"${r['beta']:+.4f}^{{{_stars(_padj(r))}}}$"

    def fmt_se(r: pd.Series) -> str:
        return f"$({r['se']:.4f})$" if np.isfinite(r["se"]) else "--"

    def fmt_ci(r: pd.Series) -> str:
        if not np.isfinite(r["ci_low"]):
            return "--"
        return f"$[{r['ci_low']:+.3f},\\,{r['ci_high']:+.3f}]$"

    def fmt_int(r: pd.Series) -> str:
        return f"{int(r['n']):,}" if pd.notna(r["n"]) else "--"

    def fmt_r2(r: pd.Series) -> str:
        return f"{r['r_squared']:.3f}" if np.isfinite(r["r_squared"]) else "--"

    out: list[str] = []
    out.append("% Requires preamble: \\usepackage{booktabs,amsmath,amssymb}")
    out.append("\\begin{table}")
    out.append("\\caption{Per-capability effect $\\hat\\tau^c$ for Q1 and Q2, "
               "pooling over $(a, b, e)$ within each capability. All "
               "specifications include feature FE and (\\agentA{} $\\times$ "
               "\\agentB{}) pair FE with HC1-robust standard errors. Stars use "
               "Bonferroni-corrected $p$ (family = the tests in this table). "
        "Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}")
    out.append("\\label{tab:main_effects_wide}")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{llcccc}")
    out.append("\\toprule")
    out.append("Sample & Capability & $\\hat{\\beta}_T$ & HC1 SE & 95\\% CI & $N$ \\\\")
    out.append("\\midrule")
    for q in ("Q1", "Q2"):
        for cap in CAPABILITIES:
            r = lookup.get((q, cap))
            cap_disp = _cap_display(cap)
            if r is None:
                out.append(f"{q} & {cap_disp} & -- & -- & -- & -- \\\\")
                continue
            out.append(
                f"{q} & {cap_disp} & {fmt_beta(r)} & {fmt_se(r)} & {fmt_ci(r)} & {fmt_int(r)} \\\\"
            )
        if q == "Q1":
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def render_main_effects_png(tier_a_q1: pd.DataFrame,
                             tier_a_q2: pd.DataFrame,
                             out_path: Path) -> None:
    """Same headline table as the LaTeX, rendered as a matplotlib figure (PNG/PDF).

    Uses plain text (not LaTeX) so it renders without a TeX install.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers, col_keys, _, lookup = _build_main_effects_grid(tier_a_q1, tier_a_q2)

    def fmt(key, kind: str) -> str:
        if key not in lookup:
            return "—"
        r = lookup[key]
        if kind == "beta":
            if not np.isfinite(r["beta"]):
                return "—"
            return f"{r['beta']:+.4f}{_stars(r['p'])}"
        if kind == "se":
            return f"({r['se']:.4f})" if np.isfinite(r["se"]) else "—"
        if kind == "ci":
            if not np.isfinite(r["ci_low"]):
                return "—"
            return f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
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

    row_specs = [
        ("β̂_T",          "beta"),
        ("HC1 SE",        "se"),
        ("95% CI",        "ci"),
        ("p",             "p"),
        ("Control mean",  "control_mean"),
        ("Treated mean",  "treated_mean"),
        ("N",             "n"),
        ("R²",            "r2"),
    ]

    n_cols = len(col_keys) + 1   # +1 for the row-label column
    n_rows = len(row_specs) + 1  # +1 for header row

    fig_w = max(8.0, 1.6 + 1.7 * len(col_keys))
    fig_h = 0.45 * n_rows + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # Build cell text matrix.
    cell_text: list[list[str]] = []
    cell_text.append([""] + headers)
    for label, kind in row_specs:
        cell_text.append([label] + [fmt(k, kind) for k in col_keys])

    table = ax.table(
        cellText=cell_text,
        loc="center",
        cellLoc="center",
        colWidths=[0.16] + [(1.0 - 0.16) / len(col_keys)] * len(col_keys),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Header row styling.
    for j in range(n_cols):
        c = table[(0, j)]
        c.set_text_props(weight="bold")
        c.set_facecolor("#eeeeee")
    # Row label column styling.
    for i in range(1, n_rows):
        c = table[(i, 0)]
        c.set_text_props(ha="left")
        c.set_facecolor("#fafafa")
    # Section divider after the first 4 stat rows (β/SE/CI/p).
    for j in range(n_cols):
        c = table[(4, j)]   # last row of the coefficient block
        c.visible_edges = "BLR"  # extra-thick bottom

    ax.set_title("Main effects (Tier A): Q1 / Q2 by capability",
                  fontsize=12, fontweight="bold", pad=10)
    fig.text(0.5, 0.04,
              "Each cell shows the coefficient on T from a separate LPM with "
              "feature FE + (a × b) pair FE, HC1 robust SE.\n"
              "Stars: *** p<0.001, ** p<0.01, * p<0.05.",
              ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tier-B / Tier-C LaTeX renderers (one .tex per tier per Q)
# ---------------------------------------------------------------------------


def _fmt_beta_cell(r: pd.Series) -> str:
    if not np.isfinite(r["beta"]):
        return "--"
    s = _stars(_padj(r))
    sup = f"^{{{s}}}" if s else ""
    return f"${r['beta']:+.4f}{sup}$"


def _fmt_se_cell(r: pd.Series) -> str:
    return f"$({r['se']:.4f})$" if np.isfinite(r["se"]) else "--"


def _fmt_ci_cell(r: pd.Series) -> str:
    if not np.isfinite(r["ci_low"]):
        return "--"
    return f"$[{r['ci_low']:+.3f},\\;{r['ci_high']:+.3f}]$"


def _fmt_int_cell(v) -> str:
    return f"{int(v):,}" if pd.notna(v) else "--"


def _fmt_3_cell(v) -> str:
    return f"{v:.3f}" if (pd.notna(v) and np.isfinite(v)) else "--"


# Map env keys produced by the pipeline to short paper-register labels (E1-E5).
# Persuasion env keys: gdpr, code_review_sabotage, data_analysis, meeting_schedule, aar.

CAPABILITY_ORDER: dict[str, int] = {"persuasion": 0, "deception": 1, "coercion": 2}
CAPABILITY_DISPLAY: dict[str, str] = {
    "persuasion": "Persuasion", "coercion": "Coercion", "deception": "Deception",
}


def _env_code(env: str) -> str:
    return ENV_TO_CODE.get(str(env), str(env))


def _cap_sort_key(cap: str) -> int:
    return CAPABILITY_ORDER.get(str(cap), 99)


def _cap_display(cap: str) -> str:
    return CAPABILITY_DISPLAY.get(str(cap), str(cap).title())


def render_tier_a_latex(tier_a: pd.DataFrame, q_label: str) -> str:
    """Per-capability pooled main effect table for one Q.

    Same column structure as ``render_tier_b_latex`` minus the ``Env`` column:
    one row per capability, ordered Persuasion → Coercion → Deception.
    """
    L: list[str] = []
    L.append("% Requires preamble: \\usepackage{booktabs,amsmath,amssymb}")
    L.append("\\begin{table}")
    L.append(
        f"\\caption{{\\textbf{{{q_label} regressions: capability-pooled main effect.}} "
        "Effect $\\hat\\tau^c$. LPM fit per capability, pooling E1--E3 and their "
        "feature-level cells, outcome \\texttt{approved}, "
        "coefficient on $T$ with HC1-robust standard errors. Fixed effects: "
        "feature FE (dropped in single-feature environments); "
        "$(a \\times b)$ pair FE. $R^2_w$ is the "
        "within-$R^2$ (variance of $y$ explained by $T$ after partialling "
        "out the FE structure). Stars use Bonferroni-corrected $p$ (family = "
        "the tests in this table). "
        "The 95\\% CIs are not corrected, so a CI can exclude zero without a star. "
        "Significance: $^{*}p<0.05$, $^{**}p<0.01$, "
        "$^{***}p<0.001$.}"
    )
    L.append(f"\\label{{tab:tier_a_{q_label.lower()}}}")
    L.append("\\centering")
    L.append("\\footnotesize")
    L.append("\\begin{tabular}{lrcccc}")
    L.append("\\toprule")
    L.append(
        "\\textbf{Capability} & $\\bm{N}$ & $\\bm{\\hat{\\tau}^{c}}$ & \\textbf{HC1 SE} & "
        "\\textbf{95\\% CI} & $\\bm{R^2_w}$ \\\\"
    )
    L.append("\\midrule")
    sub = tier_a.copy()
    sub["__cap_order"] = sub["capability"].map(_cap_sort_key)
    sub = sub.sort_values("__cap_order")
    has_within = "within_r_squared" in sub.columns
    for _, r in sub.iterrows():
        cap = _cap_display(r.get("capability", "--"))
        r2 = r["within_r_squared"] if has_within else r["r_squared"]
        L.append(
            f"{cap} & {_fmt_int_cell(r['n'])} & {_fmt_beta_cell(r)} & "
            f"{_fmt_se_cell(r)} & {_fmt_ci_cell(r)} & {_fmt_3_cell(r2)} \\\\"
        )
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


def render_tier_b_latex(
    tier_b: pd.DataFrame,
    q_label: str,
    tier_a: pd.DataFrame | None = None,
) -> str:
    """Per (capability, environment) regression table for one Q.

    Uses E1-E5 environment labels and orders capabilities Persuasion → Coercion
    → Deception. Reports within-R² (variance in y explained by T after
    partialing out feature and pair fixed effects). When ``tier_a`` is
    supplied, a bolded ``\\textbf{<Cap>} & (all envs)`` row precedes each
    capability's per-env rows, showing the capability-pooled main effect
    (the same statistic plotted as the suptitle reference in the forest plots).
    """
    L: list[str] = []
    L.append("% Requires preamble: \\usepackage{booktabs,amsmath,amssymb}")
    L.append("\\begin{table}")
    if tier_a is not None:
        caption = (
            f"{q_label} regressions: capability-pooled main effect "
            "$\\hat\\tau^c$ (bold rows, ``all envs'') and per-(capability, "
            "environment) pooled treatment effect $\\hat\\tau^{ce}$. LPM "
            "fit on the corresponding sample, " + _outcome_clause(tier_b) + ", "
            "coefficient on $T$ with HC1-robust standard errors. Fixed "
            "effects: feature FE (dropped in single-feature environments); "
            "$(a \\times b)$ pair FE in every "
            "regression. The per-environment rows match the pooled "
            "$\\hat\\beta_T$ reference lines drawn in the forest plots; "
            "the capability rows match the headline effects in "
            "Tab.~\\ref{tab:main_effects_wide}. $R^2_w$ is the within-$R^2$. "
            "Stars use Bonferroni-corrected $p$ (family = the tests in this "
            "table). The 95\\% CIs are not corrected, so a CI can exclude zero without a star. Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
        )
    else:
        caption = (
            f"\\textbf{{{q_label} regressions: pooled treatment effect per (capability, "
            "environment).} Effect $\\hat\\tau^{ce}$. LPM fit per (capability, "
            "environment) on the corresponding sample $\\mathcal{S}_k^{ce}$; "
            + _outcome_clause(tier_b) +
            "; coefficient on $T$ with HC1-robust "
            "standard errors. Fixed effects: feature FE (dropped in single-feature environments); "
            "$(a \\times b)$ pair FE in every "
            "regression. $R^2_w$ is the "
            "within-$R^2$ (variance of $y$ explained by $T$ after partialling "
            "out the FE structure). Stars use Bonferroni-corrected $p$ (family = "
        "the tests in this table). "
        "The 95\\% CIs are not corrected, so a CI can exclude zero without a star. "
        "Significance: $^{*}p<0.05$, $^{**}p<0.01$, "
            "$^{***}p<0.001$."
        )
    L.append(f"\\caption{{{caption}}}")
    L.append(f"\\label{{tab:tier_b_{q_label.lower()}}}")
    L.append("\\centering")
    L.append("\\footnotesize")
    L.append("\\begin{tabular}{llrcccc}")
    L.append("\\toprule")
    beta_header = (
        "$\\bm{\\hat{\\tau}^{c}}$ / $\\bm{\\hat{\\tau}^{ce}}$"
        if tier_a is not None else "$\\bm{\\hat{\\tau}^{ce}}$"
    )
    L.append(
        f"\\textbf{{Capability}} & \\textbf{{Env}} & $\\bm{{N}}$ & {beta_header} & \\textbf{{HC1 SE}} & "
        "\\textbf{95\\% CI} & $\\bm{R^2_w}$ \\\\"
    )
    L.append("\\midrule")

    sub = tier_b.copy()
    sub["__cap_order"] = sub["capability"].map(_cap_sort_key)
    sub["__env_code"] = sub["environment"].map(_env_code)
    sub = sub.sort_values(["__cap_order", "__env_code", "environment"])
    has_within = "within_r_squared" in sub.columns

    tier_a_by_cap: dict[str, pd.Series] = {}
    if tier_a is not None:
        for _, r in tier_a.iterrows():
            tier_a_by_cap[str(r["capability"])] = r

    capabilities_in_order = [
        cap for cap in sorted(sub["capability"].unique(), key=_cap_sort_key)
    ]
    for i, cap in enumerate(capabilities_in_order):
        if i > 0 and tier_a is not None:
            L.append("\\midrule")
        if tier_a is not None and cap in tier_a_by_cap:
            ar = tier_a_by_cap[cap]
            r2 = ar["within_r_squared"] if has_within else ar["r_squared"]
            L.append(
                f"\\textbf{{{_cap_display(cap)}}} & \\textit{{(all envs)}} & "
                f"{_fmt_int_cell(ar['n'])} & {_fmt_beta_cell(ar)} & "
                f"{_fmt_se_cell(ar)} & {_fmt_ci_cell(ar)} & "
                f"{_fmt_3_cell(r2)} \\\\"
            )
        for _, r in sub[sub["capability"] == cap].iterrows():
            env = _env_code(r.get("environment", "--"))
            r2 = r["within_r_squared"] if has_within else r["r_squared"]
            L.append(
                f"{_cap_display(cap)} & {env} & {_fmt_int_cell(r['n'])} & "
                f"{_fmt_beta_cell(r)} & {_fmt_se_cell(r)} & "
                f"{_fmt_ci_cell(r)} & {_fmt_3_cell(r2)} \\\\"
            )

    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


# Short model labels for compact tables. Mirrors persuasion_regression.MODEL_DISPLAY.
MODEL_SHORT: dict[str, str] = {
    "claude-opus-4-6":          "Opus 4.6",
    "claude-opus-4-7":          "Opus 4.7",
    "claude-sonnet-4-6":        "Sonnet 4.6",
    "gemini-3.1-pro-preview":   "Gemini 3.1",
    "gemini-2.5-pro":           "Gemini 2.5",
    "gemini-2.5-flash":         "Gemini 2.5 F",
    "gemini-3-flash-preview":   "Gemini 3 F",
    "gpt-5":                    "GPT-5",
    "gpt-5.4":                  "GPT-5.4",
    "gpt-5.4-mini":             "GPT-5.4 Mini",
    "grok-4.20":                "Grok 4.20",
    "grok-4.1-fast":            "Grok 4.1",
    "kimi-k2.6":                "Kimi K2.6",
    "o4-mini":                  "o4-mini",
    "gemma-4-31b-it":           "Gemma 4-31B",
}


def _short_model(name: str) -> str:
    return MODEL_SHORT.get(str(name), str(name))


def _compact_cell(beta: float, p: float) -> str:
    """Single cell content for the compact Tier C table: signed β with stars."""
    if not (pd.notna(beta) and np.isfinite(beta)):
        return "--"
    s = _stars(p)
    sup = f"^{{{s}}}" if s else ""
    sign = "+" if beta >= 0 else "-"
    return f"${sign}{abs(beta):.2f}{sup}$"


def render_tier_c_compact_latex(tier_c: pd.DataFrame, q_label: str) -> str:
    """Compact per-pair Tier C table: rows = pairs, columns = (capability, env).

    One landscape-friendly table per question. Cells show $\\hat\\tau^{abe}$ with
    significance stars; HC1 SEs and 95% CIs available in the long-format
    companion table tier_C_<q>.tex and the parquet tier_C_<q>.parquet.
    """
    sub = tier_c.copy()
    off_grid = sub["environment"].isin(FOREST_PAIR_ENVS_EXCLUDE)
    has_off_grid = bool(off_grid.any())
    sub = sub[(sub["fit_status"] == "ok") & ~off_grid].copy()
    sub["pair"] = sub["model_a"].astype(str) + "||" + sub["model_b"].astype(str)
    sub["env_code"] = sub["environment"].map(_env_code)
    sub["cap_order"] = sub["capability"].map(_cap_sort_key)

    cap_envs = (
        sub[["capability", "cap_order", "env_code", "environment"]]
        .drop_duplicates()
        .sort_values(["cap_order", "env_code", "environment"])
    )
    column_keys = list(zip(cap_envs["capability"], cap_envs["environment"]))

    pairs = sorted(sub["pair"].unique())

    _pcol = "p_bonf" if "p_bonf" in sub.columns else "p"
    sub["cell"] = [_compact_cell(b, p) for b, p in zip(sub["beta"], sub[_pcol])]
    pivot = sub.pivot_table(
        index="pair",
        columns=["capability", "environment"],
        values="cell",
        aggfunc="first",
    )
    pivot = pivot.reindex(index=pairs, columns=column_keys)

    L: list[str] = []
    L.append("% Requires preamble: \\usepackage{booktabs,amsmath,rotating}")
    L.append("\\begin{sidewaystable}")
    L.append(
        f"\\caption{{\\textbf{{{q_label} regressions: per-pair treatment effect (compact).}} "
        "Effect $\\hat\\tau^{abe}$ for each $(a, b, e)$ cell. LPM with HC1-robust SE; "
        + _outcome_clause(tier_c) + "; coefficient on $T$. Each cell fits one "
        "regression on its own slice; feature FE (dropped in single-feature environments); pair FE "
        "absent (pair is constant within a cell). Cells show $\\hat\\tau$ with "
        "significance stars; HC1 SEs and 95\\% CIs are in the long-format "
        f"companion (\\cref{{tab:tier_c_{q_label.lower()}}}). Cells with "
        "fit failures (e.g.\\ \\texttt{T\\_constant}) are shown as `--'. "
        + ("E5 uses a different set of models and is reported in the long-format "
           "table only. " if has_off_grid else "")
        + "Stars use Bonferroni-corrected $p$ (family = the tests in the long-format "
        "table). Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}"
    )
    L.append(f"\\label{{tab:tier_c_compact_{q_label.lower()}}}")
    L.append("\\centering")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{4pt}")

    n_cols = len(column_keys)
    L.append("\\begin{tabular}{l" + "c" * n_cols + "}")
    L.append("\\toprule")

    cap_groups: list[tuple[str, int]] = []
    for cap, _env in column_keys:
        if cap_groups and cap_groups[-1][0] == cap:
            cap_groups[-1] = (cap, cap_groups[-1][1] + 1)
        else:
            cap_groups.append((cap, 1))
    cap_header = "\\textbf{Pair $\\bm{(A \\to B)}$}"
    for cap, span in cap_groups:
        cap_header += f" & \\multicolumn{{{span}}}{{c}}{{\\textbf{{{_cap_display(cap)}}}}}"
    cap_header += " \\\\"
    L.append(cap_header)

    cmidrule_parts: list[str] = []
    col_idx = 2
    for _, span in cap_groups:
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + span - 1}}}")
        col_idx += span
    L.append(" ".join(cmidrule_parts))

    env_header = ""
    for _cap, env in column_keys:
        env_header += f" & \\textbf{{{_env_code(env)}}}"
    L.append(env_header + " \\\\")
    L.append("\\midrule")

    for pair in pairs:
        a, _, b = pair.partition("||")
        a_disp = _short_model(a)
        b_disp = _short_model(b)
        pair_label = f"{a_disp} $\\to$ {b_disp}"
        row_cells: list[str] = []
        for col in column_keys:
            v = pivot.loc[pair, col] if col in pivot.columns else None
            row_cells.append(v if isinstance(v, str) and v else "--")
        L.append(pair_label + " & " + " & ".join(row_cells) + " \\\\")

    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{sidewaystable}")
    return "\n".join(L) + "\n"


def render_tier_c_latex(tier_c: pd.DataFrame, q_label: str) -> str:
    """Per-cell regression table for one Q (one row per (env, cap, A, B)).

    Uses ``longtable`` so the table paginates naturally — many rows would
    otherwise overflow a single page.
    """
    L: list[str] = []
    L.append("% Requires preamble: \\usepackage{booktabs,amsmath,longtable}")
    L.append("\\begin{center}")
    L.append("\\footnotesize")
    L.append("\\begin{longtable}{lllrccc}")
    L.append(
        f"\\caption{{\\textbf{{{q_label} regressions: per-pair treatment effect.}} Effect "
        "$\\hat\\tau^{abe}$ for each $(a, b, e)$ cell. LPM with HC1-robust SE; " + _outcome_clause(tier_c) + "; "
        "coefficient on $T$. Feature FE (dropped in single-feature environments); pair FE absent (pair is "
        "constant within a cell). Cells with no estimate (one arm empty, or outcome "
        "constant in both arms) are shown as `--'. "
        "Stars use Bonferroni-corrected $p$ (family = the tests in this table). "
        "The 95\\% CIs are not corrected, so a CI can exclude zero without a star. "
        "Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}"
    )
    L.append(f"\\label{{tab:tier_c_{q_label.lower()}}} \\\\")
    L.append("\\toprule")
    L.append(
        "\\textbf{Cap.}\\ & \\textbf{Env.}\\ & \\textbf{Pair $\\bm{(a, b)}$} & $\\bm{N}$ & "
        "$\\bm{\\hat{\\tau}^{abe}}$ & \\textbf{HC1 SE} & \\textbf{95\\% CI} \\\\"
    )
    L.append("\\midrule")
    L.append("\\endfirsthead")
    L.append(
        "\\multicolumn{7}{l}{\\footnotesize\\textit{(continued from previous page)}} \\\\"
    )
    L.append("\\toprule")
    L.append(
        "\\textbf{Cap.}\\ & \\textbf{Env.}\\ & \\textbf{Pair $\\bm{(a, b)}$} & $\\bm{N}$ & "
        "$\\bm{\\hat{\\tau}^{abe}}$ & \\textbf{HC1 SE} & \\textbf{95\\% CI} \\\\"
    )
    L.append("\\midrule")
    L.append("\\endhead")
    L.append("\\midrule")
    L.append(
        "\\multicolumn{7}{r}{\\footnotesize\\textit{(continued on next page)}} \\\\"
    )
    L.append("\\endfoot")
    L.append("\\bottomrule")
    L.append("\\endlastfoot")
    sub = tier_c.copy()
    sub["__cap_order"] = sub["capability"].map(_cap_sort_key)
    sub["__env_code"] = sub["environment"].map(_env_code)
    sub = sub.sort_values(["__cap_order", "__env_code", "model_a", "model_b"])
    for _, r in sub.iterrows():
        cap = _cap_display(r.get("capability", "--"))
        env = _env_code(r.get("environment", "--"))
        a_disp = _short_model(r.get("model_a", ""))
        b_disp = _short_model(r.get("model_b", ""))
        pair = f"{a_disp} $\\to$ {b_disp}"
        L.append(
            f"{cap} & {env} & {pair} & {_fmt_int_cell(r['n'])} & {_fmt_beta_cell(r)} & "
            f"{_fmt_se_cell(r)} & {_fmt_ci_cell(r)} \\\\"
        )
    L.append("\\end{longtable}")
    L.append("\\end{center}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Unified all-results table — every regression in one place
# ---------------------------------------------------------------------------


def _controls_label(n_features_in_fit: int, n_pairs_in_fit: int) -> str:
    """Describe the FE that ACTUALLY entered the fit for one regression.

    A FE column is dropped at fit time if it has only one unique value (i.e.
    is constant within the fit window). This means a regression in a
    1-feature env doesn't include feature FE; a regression on a single
    (model_a, model_b) cell doesn't include pair FE.
    """
    parts: list[str] = []
    if int(n_features_in_fit or 0) > 1:
        parts.append("feature FE")
    if int(n_pairs_in_fit or 0) > 1:
        parts.append("(model_A × model_B) FE")
    return "; ".join(parts) if parts else "(none — no FE applied)"


def build_all_results(tier_outputs: dict[tuple[str, str], pd.DataFrame]
                       ) -> pd.DataFrame:
    """Concatenate every (tier × Q) into one tidy long table with extras.

    Columns (in fixed order):
      Q · tier · capability · environment · env · model_a · model_b
      N · N_treat · N_control · control_mean · treated_mean · outcome_col
      beta · HC1_se · t · p · sig · ci_low · ci_high
      r_squared · controls · n_features_in_fit · n_pairs_in_fit · fit_status

    The `controls` column is derived per-row from `n_features_in_fit` and
    `n_pairs_in_fit` — only FE that survived constant-column drop are listed.
    """
    pieces: list[pd.DataFrame] = []
    for (q_label, tier_name), df in tier_outputs.items():
        d = df.copy()
        # Ensure the unit columns exist for tiers that don't define them.
        for c in ("environment", "model_a", "model_b"):
            if c not in d.columns:
                d[c] = pd.NA
        d["controls"] = [
            _controls_label(nf, np_) for nf, np_ in
            zip(d["n_features_in_fit"], d["n_pairs_in_fit"])
        ]
        if "p_bonf" not in d.columns:        # uncorrected fallback
            d["p_bonf"] = d["p"]
            d["m_tests"] = d["p"].notna().sum()
        d["sig"] = d["p_bonf"].apply(_stars)   # stars on the Bonferroni-adjusted p
        d = d.rename(columns={"se": "HC1_se"})
        pieces.append(d)
    out = pd.concat(pieces, ignore_index=True)
    # Short code beside the raw env name, not instead of it: the CSV is the
    # machine-readable artifact and `environment` is its join key, while E4's
    # three capability arms all share the code E4.
    out["env"] = out["environment"].map(env_code)

    col_order = [
        "Q", "tier", "capability", "environment", "env", "model_a", "model_b",
        "n", "n_treat", "n_control",
        "control_mean", "treated_mean", "outcome_col",
        "beta", "HC1_se", "t", "p", "m_tests", "p_bonf", "sig", "ci_low", "ci_high",
        "r_squared", "controls",
        "n_features_in_fit", "n_pairs_in_fit", "fit_status",
    ]
    return out.reindex(columns=col_order)


def render_all_results_markdown(df: pd.DataFrame) -> str:
    """Markdown view: full Tier A + B; Tier C summarized (count + range)."""
    L: list[str] = []
    L.append("# All regression results")
    L.append("")
    L.append("Every regression run by the pipeline, in one table. "
             "See `all_results.csv` for the machine-readable version "
             "(includes Tier C, 324 rows).")
    L.append("")
    L.append("Significance stars are on the **Bonferroni-adjusted** `p_bonf` "
             "(family = the tests in each tier×Q): `***` p<0.001, `**` p<0.01, "
             "`*` p<0.05. Raw `p` is shown alongside.")
    L.append("")
    L.append(_outcome_note_plain(df) + " `env` is the short E1-E5 code; "
             "`all_results.csv` keeps the full environment name alongside it.")
    L.append("")

    def fmt_row(r: pd.Series) -> str:
        beta = f"{r['beta']:+.4f}{r['sig']}" if np.isfinite(r["beta"]) else "—"
        se = f"({r['HC1_se']:.4f})" if np.isfinite(r["HC1_se"]) else "—"
        ci = (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
              if np.isfinite(r["ci_low"]) else "—")
        p = f"{r['p']:.4g}" if np.isfinite(r["p"]) else "—"
        pb = f"{r['p_bonf']:.4g}" if np.isfinite(r.get("p_bonf", np.nan)) else "—"
        return (
            f"| {r['Q']} | {r['capability']} | {r['env']} | "
            f"{int(r['n']):,} | {beta} | {se} | {p} | {pb} | {ci} | "
            f"{r['r_squared']:.3f} | {r['controls']} |"
        )

    # Tier A
    a = df[df["tier"] == "A_main_by_capability"]
    L.append("## Tier A — main effect, per capability")
    L.append("")
    L.append("| Q | capability | env | N | β̂_T | HC1 SE | p | p_bonf | 95% CI | R² | controls |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---|---:|---|")
    for _, r in a.sort_values(["Q", "capability"]).iterrows():
        L.append(fmt_row(r))
    L.append("")

    # Tier B
    b = df[df["tier"] == "B_by_env_cap"]
    L.append("## Tier B — by environment × capability")
    L.append("")
    L.append("| Q | capability | env | N | β̂_T | HC1 SE | p | p_bonf | 95% CI | R² | controls |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---|---:|---|")
    for _, r in b.sort_values(["Q", "capability", "env"]).iterrows():
        L.append(fmt_row(r))
    L.append("")

    # Tier C summary (full table is in CSV)
    c = df[df["tier"] == "C_by_cell"]
    L.append("## Tier C — by cell (env × capability × A × B)")
    L.append("")
    L.append("Full per-cell table is in `all_results.csv`. Summary by (Q, capability):")
    L.append("")
    L.append("| Q | capability | n_regressions | n_ok | n_sig (Bonf p<.05) | median β | median N |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for (q, cap), g in c.groupby(["Q", "capability"]):
        ok = (g["fit_status"] == "ok").sum()
        sig = ((g["p_bonf"] < 0.05) & np.isfinite(g["p_bonf"])).sum()
        med_beta = float(np.nanmedian(g["beta"]))
        med_n = int(np.nanmedian(g["n"]))
        L.append(
            f"| {q} | {cap} | {len(g)} | {int(ok)} | {int(sig)} "
            f"| {med_beta:+.4f} | {med_n:,} |"
        )
    L.append("")

    return "\n".join(L) + "\n"


def render_all_results_png(df: pd.DataFrame, out_path: Path) -> None:
    """Wide PNG showing Tier A + Tier B (Tier C is too big for a useful image)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = df[df["tier"].isin(["A_main_by_capability", "B_by_env_cap"])].copy()
    show = show.sort_values(["tier", "Q", "capability", "env"]).reset_index(drop=True)

    headers = [
        "Q", "tier", "capability", "env",
        "N", "β̂_T", "HC1 SE", "95% CI", "p", "sig",
        "Ctrl mean", "Treat mean", "R²", "Controls",
    ]

    def cells(r: pd.Series) -> list[str]:
        return [
            r["Q"],
            r["tier"].split("_", 1)[0],
            r["capability"],
            str(r["env"]),
            f"{int(r['n']):,}",
            (f"{r['beta']:+.4f}" if np.isfinite(r["beta"]) else "—"),
            (f"{r['HC1_se']:.4f}" if np.isfinite(r["HC1_se"]) else "—"),
            (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]" if np.isfinite(r["ci_low"]) else "—"),
            (f"{r['p']:.4g}" if np.isfinite(r["p"]) else "—"),
            r["sig"] or "",
            (f"{r['control_mean']:.3f}" if np.isfinite(r["control_mean"]) else "—"),
            (f"{r['treated_mean']:.3f}" if np.isfinite(r["treated_mean"]) else "—"),
            (f"{r['r_squared']:.3f}" if np.isfinite(r["r_squared"]) else "—"),
            r["controls"],
        ]

    table_data = [headers] + [cells(r) for _, r in show.iterrows()]
    n_rows = len(table_data)

    fig_w = 17.0
    fig_h = 0.38 * n_rows + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.04, 0.05, 0.09, 0.13, 0.06, 0.07, 0.06, 0.13, 0.06, 0.03,
                    0.06, 0.06, 0.05, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Header styling.
    for j in range(len(headers)):
        c = table[(0, j)]
        c.set_text_props(weight="bold")
        c.set_facecolor("#dddddd")
    # Tier separator: light shading per tier.
    for i in range(1, n_rows):
        tier_short = table_data[i][1]
        bg = "#fafafa" if tier_short == "A" else "#ffffff"
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(bg)

    ax.set_title("All Tier A + Tier B regressions (Tier C — 324 rows — in all_results.csv)",
                  fontsize=12, fontweight="bold", pad=10)
    fig.text(0.5, 0.02,
              "All specifications: LPM, HC1 robust SE, coefficient on T. "
              + _outcome_note_plain(show) +
              " Stars: *** p<0.001, ** p<0.01, * p<0.05.",
              ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(data_dir: Path | None = None,
        out_dir: Path | None = None) -> dict[str, Path]:
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or REGRESSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-apply the E4 no-deal backfill: these parquets are read straight off
    # disk, so a subsample written before the fix would otherwise drop every
    # no-deal row from the points fits. Idempotent on a fresh subsample.
    q1 = add_e4_points_share(backfill_no_deal_points(pd.read_parquet(data_dir / "q1_subsample.parquet")))
    q2 = add_e4_points_share(backfill_no_deal_points(pd.read_parquet(data_dir / "q2_subsample.parquet")))

    written: dict[str, Path] = {}
    tier_outputs: dict[tuple[str, str], pd.DataFrame] = {}

    for q_label, q_df in [("q1", q1), ("q2", q2)]:
        for tier in REGRESSION_TIERS:
            # Tier A's env exclusion is applied inside fit_tier, so every
            # caller of that tier gets the same row set.
            res = fit_tier(q_df, tier)
            res.insert(0, "Q", q_label.upper())
            # Bonferroni correction within this (tier, Q) family — adds m_tests
            # and p_bonf; stars / forest fill downstream key on p_bonf.
            res = add_bonferroni(res)
            tier_outputs[(q_label, tier.name)] = res
            stem = f"tier_{tier.name.split('_')[0]}_{q_label}"
            res.to_parquet(out_dir / f"{stem}.parquet", index=False)
            res.to_csv(out_dir / f"{stem}.csv", index=False)
            written[f"{stem}_parquet"] = out_dir / f"{stem}.parquet"
            written[f"{stem}_csv"] = out_dir / f"{stem}.csv"
            n_sig = int(((res['p_bonf'] < 0.05) & np.isfinite(res['p_bonf'])).sum())
            print(
                f"  [{q_label.upper()}/{tier.name}] {len(res)} regressions; "
                f"ok={int((res['fit_status']=='ok').sum())}; m={int(res['m_tests'].iloc[0])}; "
                f"sig(Bonf)={n_sig}; median |β|={float(np.nanmedian(res['beta'].abs())):.4f}"
            )

    tier_a_q1 = tier_outputs[("q1", "A_main_by_capability")]
    tier_a_q2 = tier_outputs[("q2", "A_main_by_capability")]

    tex = render_main_effects_latex_wide(tier_a_q1, tier_a_q2)
    p_tex = out_dir / "main_effects_wide.tex"
    p_tex.write_text(tex)
    written["main_effects_wide_tex"] = p_tex

    p_png = out_dir / "main_effects_wide"
    render_main_effects_png(tier_a_q1, tier_a_q2, p_png)
    written["main_effects_wide_png"] = p_png.with_suffix(".png")

    # Per-tier paper-ready LaTeX (one .tex per tier per Q). These are
    # \input'd into the hand-authored Overleaf appendix (appendix/*.tex).
    for q_label in ("q1", "q2"):
        ta = tier_outputs[(q_label, "A_main_by_capability")]
        tb = tier_outputs[(q_label, "B_by_env_cap")]
        tc = tier_outputs[(q_label, "C_by_cell")]
        p_a_tex = out_dir / f"tier_A_{q_label}.tex"
        p_b_tex = out_dir / f"tier_B_{q_label}.tex"
        p_c_tex = out_dir / f"tier_C_{q_label}.tex"
        p_c_compact_tex = out_dir / f"tier_C_compact_{q_label}.tex"
        p_a_tex.write_text(render_tier_a_latex(ta, q_label.upper()))
        p_b_tex.write_text(render_tier_b_latex(tb, q_label.upper()))
        p_c_tex.write_text(render_tier_c_latex(tc, q_label.upper()))
        p_c_compact_tex.write_text(render_tier_c_compact_latex(tc, q_label.upper()))
        written[f"tier_A_{q_label}_tex"] = p_a_tex
        written[f"tier_B_{q_label}_tex"] = p_b_tex
        written[f"tier_C_{q_label}_tex"] = p_c_tex
        written[f"tier_C_compact_{q_label}_tex"] = p_c_compact_tex

    # Unified all-results table: every regression across all tiers and Q's,
    # one row per regression, with controls / significance columns added.
    all_results = build_all_results(tier_outputs)
    p_all_csv = out_dir / "all_results.csv"
    p_all_md = out_dir / "all_results.md"
    p_all_png = out_dir / "all_results"   # only Tier A + B as a figure
    all_results.to_csv(p_all_csv, index=False)
    p_all_md.write_text(render_all_results_markdown(all_results))
    render_all_results_png(all_results, p_all_png)
    written["all_results_csv"] = p_all_csv
    written["all_results_md"] = p_all_md
    written["all_results_png"] = p_all_png.with_suffix(".png")

    return written


if __name__ == "__main__":
    paths = run()
    for k, p in paths.items():
        print(f"  {k}: {p}")
