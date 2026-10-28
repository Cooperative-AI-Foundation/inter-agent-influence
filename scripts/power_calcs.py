"""
Simulation-based power analysis for treatment uplift detection.

Design per regression (one for coercion, one for persuasion):
    - n_features: unique prompts (cluster unit for inference)
    - n_model_a, n_model_b: model factors, entered as additive fixed effects
    - binary treatment: control-prompt vs. treatment-prompt
    - K resamples per (feature, model_a, model_b, arm) cell
    - binary outcome y in {0, 1}

Balance:
    Fully balanced: K replicates per (feature, model_a, model_b, arm) cell.
    Total N per regression = n_features * n_model_a * n_model_b * 2 * K.
    Balance across arms within cell minimizes Var(beta_1_hat) at fixed compute;
    balance across (feature, model) cells ensures symmetric FE identification.

Regression:
    y = beta_0 + beta_1 * treatment + alpha_f + gamma_a + delta_b + eps

where alpha_f, gamma_a, delta_b are feature, model A, and model B fixed effects
(additive, not interacted). Estimated by OLS (linear probability model).

LPM over logit: the estimand is the average treatment effect on the probability
scale, which OLS estimates directly. Fixed-effects logit suffers the incidental
parameters problem; OLS does not. A logit robustness check is provided via
`fit_logit_marginal_effect`.

Standard errors:
    - Primary:   cluster-robust at feature level, t(G-1) critical values
    - Sensitivity: heteroskedasticity-robust (treats resamples as independent
                   draws from the super-population of model outputs)

Usage:
    python3 scripts/power_calcs.py --e1e4 --bonferroni   # the paper's MDE table (power appendix)
    python3 scripts/power_calcs.py                       # power curves over K

Output:
    Power curves over K for main and conservative scenarios, plus Type I error
    calibration check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis.pilot_variance import (
    compute_heterogeneity,
    find_eval_files,
    load_all_samples,
    print_heterogeneity_report,
)


def load_variance_from_data(path: str) -> dict | None:
    """Load heterogeneity estimates from .eval files or a saved JSON.

    Returns the heterogeneity dict from variance_analysis, or None on failure.
    """
    p = Path(path)

    if p.is_file() and p.suffix == ".json":
        with open(p) as f:
            return json.load(f)

    if p.is_dir():
        eval_paths = find_eval_files(str(p))
        if not eval_paths:
            print(f"No .eval files found under {p}")
            return None
        print(f"Found {len(eval_paths)} eval files, computing heterogeneity...")
        samples = load_all_samples(eval_paths)
        if not samples:
            print("No samples extracted.")
            return None
        print(f"Loaded {len(samples)} samples")
        het = compute_heterogeneity(samples)
        print_heterogeneity_report(het)
        return het

    print(f"Not a valid path: {path}")
    return None


# -----------------------------------------------------------------------------
# Core simulation
# -----------------------------------------------------------------------------

def simulate_one(
    K: int,
    baseline: float,
    treat_effect: float,
    feature_sd: float,
    model_a_sd: float,
    model_b_sd: float,
    effect_heterogeneity: float,
    n_features: int,
    n_model_a: int,
    n_model_b: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Simulate one dataset and return p-values under two SE specifications.

    Data-generating process:
        p_ijkm = baseline + feature_shift_j + model_a_shift_k + model_b_shift_m
                 + treat_ijkm * (treat_effect + effect_shift_j)
        y_ijkm ~ Bernoulli(clip(p_ijkm, 0.01, 0.99))

    Returns:
        (p_value_hc_robust, p_value_clustered, beta_hat)
    """
    n_arms = 2
    n_obs_per_feature = n_model_a * n_model_b * n_arms * K

    feat_ids = np.repeat(np.arange(n_features), n_obs_per_feature)
    model_a_ids = np.tile(
        np.repeat(np.arange(n_model_a), n_model_b * n_arms * K),
        n_features,
    )
    model_b_ids = np.tile(
        np.repeat(np.arange(n_model_b), n_arms * K),
        n_features * n_model_a,
    )
    arm_ids = np.tile(
        np.repeat(np.arange(n_arms), K),
        n_features * n_model_a * n_model_b,
    )

    feature_shifts = rng.normal(0, feature_sd, n_features)
    model_a_shifts = rng.normal(0, model_a_sd, n_model_a)
    model_b_shifts = rng.normal(0, model_b_sd, n_model_b)
    feature_specific_effects = treat_effect + rng.normal(
        0, effect_heterogeneity, n_features
    )

    p = (
        baseline
        + feature_shifts[feat_ids]
        + model_a_shifts[model_a_ids]
        + model_b_shifts[model_b_ids]
    )
    p = p + (arm_ids == 1) * feature_specific_effects[feat_ids]
    p = np.clip(p, 0.01, 0.99)
    y = rng.binomial(1, p).astype(float)
    treatment = (arm_ids == 1).astype(float)

    # --- Within transformation: absorb three-way additive FE by iterative demeaning
    df = pd.DataFrame(
        {
            "y": y,
            "T": treatment,
            "feature": feat_ids,
            "model_a": model_a_ids,
            "model_b": model_b_ids,
        }
    )
    for col in ("y", "T"):
        for _ in range(10):  # converges quickly for balanced panels
            df[col] -= df.groupby("feature")[col].transform("mean")
            df[col] -= df.groupby("model_a")[col].transform("mean")
            df[col] -= df.groupby("model_b")[col].transform("mean")

    Y = df["y"].values
    X = df["T"].values
    xtx_inv = 1.0 / float(X @ X)
    beta = xtx_inv * float(X @ Y)
    resid = Y - X * beta

    # --- HC-robust SE
    meat_hc = float(np.sum((X * resid) ** 2))
    se_hc = np.sqrt(xtx_inv * meat_hc * xtx_inv)
    p_hc = 2 * (1 - stats.norm.cdf(abs(beta / se_hc)))

    # --- Cluster-robust SE at feature level with finite-cluster correction.
    # Degenerate when n_features == 1 (single cluster); in that case the
    # cluster-robust SE is undefined, so we report NaN and let HC stand alone.
    feat_arr = df["feature"].values
    G = n_features
    if G < 2:
        p_clust = float("nan")
    else:
        meat_c = 0.0
        for f in np.unique(feat_arr):
            mask = feat_arr == f
            score = float(np.sum(X[mask] * resid[mask]))
            meat_c += score * score
        dof_adjust = G / (G - 1)
        se_c = np.sqrt(dof_adjust * xtx_inv * meat_c * xtx_inv)
        t_c = beta / se_c
        p_clust = 2 * (1 - stats.t.cdf(abs(t_c), df=G - 1))

    return p_hc, p_clust, beta


def power_at(
    K: int,
    n_sims: int = 500,
    baseline: float = 0.10,
    treat_effect: float = 0.03,
    feature_sd: float = 0.03,
    model_a_sd: float = 0.02,
    model_b_sd: float = 0.02,
    effect_heterogeneity: float = 0.0,
    n_features: int = 5,
    n_model_a: int = 6,
    n_model_b: int = 3,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Estimate rejection rate (power) at a given K over n_sims replications.

    ``alpha`` is the two-sided significance threshold; pass a Bonferroni-adjusted
    value (e.g. 0.05 / m) to size for multiple comparisons.
    """
    p_hc_list, p_clust_list, beta_list = [], [], []
    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        p_hc, p_clust, beta = simulate_one(
            K=K,
            baseline=baseline,
            treat_effect=treat_effect,
            feature_sd=feature_sd,
            model_a_sd=model_a_sd,
            model_b_sd=model_b_sd,
            effect_heterogeneity=effect_heterogeneity,
            n_features=n_features,
            n_model_a=n_model_a,
            n_model_b=n_model_b,
            rng=rng,
        )
        p_hc_list.append(p_hc)
        p_clust_list.append(p_clust)
        beta_list.append(beta)

    p_clust_arr = np.array(p_clust_list)
    if np.isnan(p_clust_arr).all():
        power_clust = float("nan")
    else:
        power_clust = float(np.mean(p_clust_arr[~np.isnan(p_clust_arr)] < alpha))

    return {
        "K": K,
        "N_total": n_features * n_model_a * n_model_b * 2 * K,
        "power_hc": float(np.mean(np.array(p_hc_list) < alpha)),
        "power_clustered": power_clust,
        "mean_beta": float(np.mean(beta_list)),
        "sd_beta": float(np.std(beta_list)),
    }


# -----------------------------------------------------------------------------
# Per-cell power (single (a, b, e) triplet, individual alpha = 0.05 test)
# -----------------------------------------------------------------------------

def power_at_per_cell(
    K: int,
    n_features: int,
    baseline: float,
    treat_effect: float,
    feature_sd: float,
    effect_heterogeneity: float = 0.0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Power for a single per-cell regression y = beta0 + tau * T + alpha_f + eps,
    fit on n_features * 2 * K observations (one cell = one (a, b, e) triplet,
    one model pair, one environment). Each replicate draws features and arms
    independently; pair-level shifts are absorbed by the cell intercept.

    Returns the empirical power for the individual two-sided test of tau = 0
    at level alpha (HC-robust SE), with no multiplicity correction.
    """
    n_arms = 2
    n = n_features * n_arms * K
    feat_ids = np.repeat(np.arange(n_features), n_arms * K)
    arm_ids = np.tile(np.repeat(np.arange(n_arms), K), n_features)
    T = arm_ids.astype(float)

    rng_master = np.random.default_rng(seed)
    rejects = 0
    n_used = 0
    for _ in range(n_sims):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        feature_shifts = rng.normal(0, feature_sd, n_features)
        feat_specific_eff = treat_effect + rng.normal(0, effect_heterogeneity, n_features)
        p = baseline + feature_shifts[feat_ids] + (arm_ids == 1) * feat_specific_eff[feat_ids]
        p = np.clip(p, 0.01, 0.99)
        y = rng.binomial(1, p).astype(float)

        y_d, T_d = y.copy(), T.copy()
        if n_features > 1:
            for f in range(n_features):
                m = feat_ids == f
                y_d[m] -= y[m].mean()
                T_d[m] -= T[m].mean()
        else:
            y_d -= y.mean()
            T_d -= T.mean()

        XtX = float(T_d @ T_d)
        if XtX == 0:
            continue
        beta = float(T_d @ y_d) / XtX
        resid = y_d - beta * T_d
        se_hc = float(np.sqrt(np.sum((T_d * resid) ** 2)) / XtX)
        if se_hc == 0:
            continue
        n_used += 1
        p_two = 2 * (1 - stats.norm.cdf(abs(beta / se_hc)))
        if p_two < alpha:
            rejects += 1

    return {
        "K": K,
        "n_features": n_features,
        "N_per_cell": n,
        "n_sims_used": n_used,
        "power": float(rejects / n_sims) if n_sims else float("nan"),
    }


# -----------------------------------------------------------------------------
# Subgroup power: per-(model_A, model_B) treatment effects with Romano-Wolf
# -----------------------------------------------------------------------------

def _fit_within_feature(y: np.ndarray, T: np.ndarray, feat: np.ndarray,
                        n_features: int) -> tuple[float, float, float, np.ndarray]:
    """LPM with feature FE via within-transform. Returns (beta, se_hc, XX, T_demean).

    Also returns the within-demeaned T vector and XX so callers doing wild
    bootstrap can reuse them without re-demeaning.
    """
    y_d = y.copy()
    T_d = T.copy()
    for f in range(n_features):
        mask = feat == f
        if mask.any():
            y_d[mask] -= y[mask].mean()
            T_d[mask] -= T[mask].mean()
    XX = float(T_d @ T_d)
    if XX <= 0:
        return 0.0, np.nan, XX, T_d
    beta = float(T_d @ y_d) / XX
    resid = y_d - beta * T_d
    meat = float(np.sum((T_d * resid) ** 2))
    se = np.sqrt(meat) / XX
    return beta, se, XX, T_d


def simulate_one_subgroup_interaction(
    K: int,
    baseline: float,
    treat_effect: float,
    feature_sd: float,
    model_a_sd: float,
    model_b_sd: float,
    effect_heterogeneity: float,
    n_features: int,
    n_model_a: int,
    n_model_b: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Saturated T x A x B interaction model with feature FE.

    Spec: y = alpha_f + gamma_{ab} * Z_{ab} + beta_{ab} * (T * Z_{ab}) + eps
    where alpha_f is feature FE, Z_{ab} indicates cell (a, b), and beta_{ab}
    is the cell-specific treatment effect (our estimand; 18 per capability).

    Under balanced design with additive feature FE, point estimates equal
    the per-cell regression approach, but HC-robust SEs pool residual
    information across cells via the full-model sandwich formula.

    Fits the full model once, then wild-bootstraps restricted residuals
    (under joint H0: all beta_{ab} = 0) for Romano-Wolf stepdown.
    """
    n_arms = 2
    n_cells = n_model_a * n_model_b
    n_obs_per_feature = n_cells * n_arms * K

    feat_ids = np.repeat(np.arange(n_features), n_obs_per_feature)
    model_a_ids = np.tile(
        np.repeat(np.arange(n_model_a), n_model_b * n_arms * K), n_features,
    )
    model_b_ids = np.tile(
        np.repeat(np.arange(n_model_b), n_arms * K), n_features * n_model_a,
    )
    arm_ids = np.tile(
        np.repeat(np.arange(n_arms), K), n_features * n_model_a * n_model_b,
    )
    cell_ids = model_a_ids * n_model_b + model_b_ids  # in [0, n_cells)

    feature_shifts = rng.normal(0, feature_sd, n_features)
    model_a_shifts = rng.normal(0, model_a_sd, n_model_a)
    model_b_shifts = rng.normal(0, model_b_sd, n_model_b)
    feature_specific_effects = treat_effect + rng.normal(
        0, effect_heterogeneity, n_features,
    )

    p = (
        baseline
        + feature_shifts[feat_ids]
        + model_a_shifts[model_a_ids]
        + model_b_shifts[model_b_ids]
    )
    p = p + (arm_ids == 1) * feature_specific_effects[feat_ids]
    p = np.clip(p, 0.01, 0.99)
    y = rng.binomial(1, p).astype(float)
    T = (arm_ids == 1).astype(float)

    n = len(y)

    # Design: feature dummies (drop feat 0) + cell indicators (all) + T x cell indicators (all)
    F = np.zeros((n, n_features - 1))
    for f in range(1, n_features):
        F[:, f - 1] = (feat_ids == f).astype(float)
    C = np.zeros((n, n_cells))
    for c in range(n_cells):
        C[:, c] = (cell_ids == c).astype(float)
    TC = C * T[:, np.newaxis]
    X = np.hstack([F, C, TC])
    tc_start = (n_features - 1) + n_cells  # index of first T x cell coefficient

    # Full OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    H = XtX_inv @ X.T  # (p, n) hat map
    beta = H @ y
    resid = y - X @ beta

    # HC-robust SE for the 18 T x cell coefficients:
    #   V_kk = sum_i H[k, i]^2 * resid[i]^2
    H_tc = H[tc_start:tc_start + n_cells, :]  # (n_cells, n)
    var_tc = (H_tc ** 2 * resid ** 2).sum(axis=1)
    se_tc = np.sqrt(var_tc)
    beta_tc = beta[tc_start:tc_start + n_cells]

    t_mat = np.where(se_tc > 0, beta_tc / se_tc, 0.0).reshape(n_model_a, n_model_b)
    beta_mat = beta_tc.reshape(n_model_a, n_model_b)
    se_mat = se_tc.reshape(n_model_a, n_model_b)

    # Restricted model for wild bootstrap: drop T x cell columns
    X_r = X[:, :tc_start]
    beta_r = np.linalg.solve(X_r.T @ X_r, X_r.T @ y)
    fit_r = X_r @ beta_r
    resid_r = y - fit_r  # restricted residuals for wild bootstrap

    # Vectorized wild bootstrap across n_boot draws (avoids Python loop overhead)
    U = rng.choice([-1.0, 1.0], size=(n_boot, n))
    Y_star = fit_r[None, :] + U * resid_r[None, :]           # (n_boot, n)
    Beta_b = Y_star @ H.T                                    # (n_boot, p)
    Resid_b = Y_star - Beta_b @ X.T                          # (n_boot, n)
    H_tc_sq = H_tc ** 2                                      # (n_cells, n)
    Var_b = (Resid_b ** 2) @ H_tc_sq.T                       # (n_boot, n_cells)
    SE_b = np.sqrt(Var_b)
    Beta_tc_b = Beta_b[:, tc_start:tc_start + n_cells]       # (n_boot, n_cells)
    T_b = np.where(SE_b > 0, Beta_tc_b / SE_b, 0.0)          # (n_boot, n_cells)
    t_boot_mat = T_b.T.reshape(n_model_a, n_model_b, n_boot)

    return {"t_obs": t_mat, "t_boot": t_boot_mat, "beta": beta_mat, "se": se_mat}


def simulate_one_subgroup(
    K: int,
    baseline: float,
    treat_effect: float,
    feature_sd: float,
    model_a_sd: float,
    model_b_sd: float,
    effect_heterogeneity: float,
    n_features: int,
    n_model_a: int,
    n_model_b: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Simulate one dataset; return per-(a, b) HC-robust t-stats and wild
    bootstrap null t-stats for Romano-Wolf stepdown.

    Same DGP as `simulate_one`, but fits a separate LPM per (model_a, model_b)
    cell and additionally wild-bootstraps restricted (β=0) residuals for the
    joint null distribution needed by RW.
    """
    n_arms = 2
    n_obs_per_feature = n_model_a * n_model_b * n_arms * K

    feat_ids = np.repeat(np.arange(n_features), n_obs_per_feature)
    model_a_ids = np.tile(
        np.repeat(np.arange(n_model_a), n_model_b * n_arms * K), n_features,
    )
    model_b_ids = np.tile(
        np.repeat(np.arange(n_model_b), n_arms * K), n_features * n_model_a,
    )
    arm_ids = np.tile(
        np.repeat(np.arange(n_arms), K), n_features * n_model_a * n_model_b,
    )

    feature_shifts = rng.normal(0, feature_sd, n_features)
    model_a_shifts = rng.normal(0, model_a_sd, n_model_a)
    model_b_shifts = rng.normal(0, model_b_sd, n_model_b)
    feature_specific_effects = treat_effect + rng.normal(
        0, effect_heterogeneity, n_features,
    )

    p = (
        baseline
        + feature_shifts[feat_ids]
        + model_a_shifts[model_a_ids]
        + model_b_shifts[model_b_ids]
    )
    p = p + (arm_ids == 1) * feature_specific_effects[feat_ids]
    p = np.clip(p, 0.01, 0.99)
    y = rng.binomial(1, p).astype(float)
    T = (arm_ids == 1).astype(float)

    t_obs = np.zeros((n_model_a, n_model_b))
    beta_arr = np.zeros((n_model_a, n_model_b))
    se_arr = np.zeros((n_model_a, n_model_b))
    t_boot = np.zeros((n_model_a, n_model_b, n_boot))

    for a in range(n_model_a):
        for b in range(n_model_b):
            mask = (model_a_ids == a) & (model_b_ids == b)
            y_c, T_c, f_c = y[mask], T[mask], feat_ids[mask]
            beta, se, XX, T_d = _fit_within_feature(y_c, T_c, f_c, n_features)
            beta_arr[a, b], se_arr[a, b] = beta, se
            if np.isnan(se) or se == 0:
                continue
            t_obs[a, b] = beta / se

            # Wild bootstrap under β=0. Restricted residual = within-feature-
            # demeaned y (since under H0 the model is just feature FE).
            y_d = y_c.copy()
            for f in range(n_features):
                m = f_c == f
                y_d[m] -= y_c[m].mean()
            n_c = y_d.shape[0]
            u = rng.choice([-1.0, 1.0], size=(n_boot, n_c))
            y_star = u * y_d[np.newaxis, :]
            beta_b = (y_star @ T_d) / XX
            resid_b = y_star - beta_b[:, np.newaxis] * T_d[np.newaxis, :]
            meat_b = np.sum((T_d[np.newaxis, :] * resid_b) ** 2, axis=1)
            se_b = np.sqrt(meat_b) / XX
            # Bootstrap t-stats under the null (centered by construction)
            t_boot[a, b, :] = np.where(se_b > 0, beta_b / se_b, 0.0)

    return {"t_obs": t_obs, "t_boot": t_boot, "beta": beta_arr, "se": se_arr}


def romano_wolf_pvalues(t_obs: np.ndarray, t_boot: np.ndarray) -> np.ndarray:
    """Romano-Wolf stepdown FWER-adjusted two-sided p-values.

    Args:
        t_obs:  (M,) observed t-statistics.
        t_boot: (M, B) bootstrap t-statistics drawn under the joint null.
    Returns:
        (M,) RW-adjusted p-values with monotonicity enforced.
    """
    abs_t = np.abs(t_obs)
    abs_tb = np.abs(t_boot)
    M, B = abs_tb.shape
    order = np.argsort(-abs_t)  # strongest first
    adj_p = np.empty(M)
    prev = 0.0
    for s in range(M):
        remaining = order[s:]
        max_tb = np.max(abs_tb[remaining, :], axis=0)
        p_raw = float(np.mean(max_tb >= abs_t[order[s]]))
        prev = max(p_raw, prev)
        adj_p[order[s]] = prev
    return adj_p


def power_subgroup(
    K: int,
    n_sims: int = 100,
    n_boot: int = 200,
    baseline: float = 0.10,
    treat_effect: float = 0.04,
    feature_sd: float = 0.03,
    model_a_sd: float = 0.02,
    model_b_sd: float = 0.02,
    effect_heterogeneity: float = 0.0,
    n_features: int = 5,
    n_model_a: int = 6,
    n_model_b: int = 3,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Per-(model_A, model_B) treatment-effect power under three corrections:
    uncorrected, Bonferroni, and Romano-Wolf stepdown (across M = n_a * n_b
    tests within a single capability).
    """
    M = n_model_a * n_model_b
    rej_unc = np.zeros((n_sims, n_model_a, n_model_b))
    rej_bonf = np.zeros_like(rej_unc)
    rej_rw = np.zeros_like(rej_unc)
    betas = np.zeros_like(rej_unc)

    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        out = simulate_one_subgroup(
            K=K, baseline=baseline, treat_effect=treat_effect,
            feature_sd=feature_sd, model_a_sd=model_a_sd, model_b_sd=model_b_sd,
            effect_heterogeneity=effect_heterogeneity,
            n_features=n_features, n_model_a=n_model_a, n_model_b=n_model_b,
            n_boot=n_boot, rng=rng,
        )
        t = out["t_obs"]
        tb = out["t_boot"]
        betas[i] = out["beta"]

        # Uncorrected two-sided
        p_unc = 2 * (1 - stats.norm.cdf(np.abs(t)))
        rej_unc[i] = p_unc < alpha

        # Bonferroni
        rej_bonf[i] = p_unc < (alpha / M)

        # Romano-Wolf
        adj = romano_wolf_pvalues(t.flatten(), tb.reshape(M, n_boot))
        rej_rw[i] = (adj < alpha).reshape(n_model_a, n_model_b)

    return {
        "K": K,
        "N_per_cell": n_features * 2 * K,
        "power_uncorrected_per_cell": rej_unc.mean(axis=0),
        "power_bonferroni_per_cell": rej_bonf.mean(axis=0),
        "power_rw_per_cell": rej_rw.mean(axis=0),
        "power_uncorrected_mean": float(rej_unc.mean()),
        "power_bonferroni_mean": float(rej_bonf.mean()),
        "power_rw_mean": float(rej_rw.mean()),
        "power_rw_min": float(rej_rw.mean(axis=0).min()),
        "power_rw_max": float(rej_rw.mean(axis=0).max()),
        "mean_beta": float(betas.mean()),
    }


def power_subgroup_joint(
    K: int,
    coercion_cfg: dict,
    persuasion_cfg: dict,
    n_sims: int = 100,
    n_boot: int = 200,
    baseline: float = 0.10,
    treat_effect: float = 0.04,
    n_features: int = 5,
    n_model_a: int = 6,
    n_model_b: int = 3,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Joint subgroup power across both capabilities (2 * n_a * n_b = 36 cells).

    Coercion and persuasion are run as separate 2-arm experiments in the real
    study (each capability has its own baseline arm plus its own treatment arm).
    Since they use completely disjoint data, the per-cell t-stats for the two
    capabilities are independent, so we can simulate them independently within
    each sim and concatenate the resulting t-stats into a single 36-element
    family for Romano-Wolf stepdown.

    `coercion_cfg` and `persuasion_cfg` are dicts with keys:
        feature_sd, model_a_sd, model_b_sd, effect_heterogeneity.

    Fitting spec: saturated T x A x B interaction model with feature FE
    (see `simulate_one_subgroup_interaction`).
    """
    n_cells = n_model_a * n_model_b
    M = 2 * n_cells

    rej_unc = np.zeros((n_sims, 2, n_model_a, n_model_b))
    rej_bonf = np.zeros_like(rej_unc)
    rej_rw = np.zeros_like(rej_unc)
    betas = np.zeros_like(rej_unc)

    for i in range(n_sims):
        rng_c = np.random.default_rng(seed + 2 * i)
        rng_p = np.random.default_rng(seed + 2 * i + 1)

        out_c = simulate_one_subgroup_interaction(
            K=K, baseline=baseline, treat_effect=treat_effect,
            feature_sd=coercion_cfg["feature_sd"],
            model_a_sd=coercion_cfg["model_a_sd"],
            model_b_sd=coercion_cfg["model_b_sd"],
            effect_heterogeneity=coercion_cfg["effect_heterogeneity"],
            n_features=n_features, n_model_a=n_model_a, n_model_b=n_model_b,
            n_boot=n_boot, rng=rng_c,
        )
        out_p = simulate_one_subgroup_interaction(
            K=K, baseline=baseline, treat_effect=treat_effect,
            feature_sd=persuasion_cfg["feature_sd"],
            model_a_sd=persuasion_cfg["model_a_sd"],
            model_b_sd=persuasion_cfg["model_b_sd"],
            effect_heterogeneity=persuasion_cfg["effect_heterogeneity"],
            n_features=n_features, n_model_a=n_model_a, n_model_b=n_model_b,
            n_boot=n_boot, rng=rng_p,
        )

        t_joint = np.concatenate([out_c["t_obs"].flatten(), out_p["t_obs"].flatten()])
        tb_joint = np.concatenate(
            [out_c["t_boot"].reshape(n_cells, n_boot),
             out_p["t_boot"].reshape(n_cells, n_boot)],
            axis=0,
        )

        p_unc = 2 * (1 - stats.norm.cdf(np.abs(t_joint)))
        rej_unc[i] = (p_unc < alpha).reshape(2, n_model_a, n_model_b)
        rej_bonf[i] = (p_unc < alpha / M).reshape(2, n_model_a, n_model_b)
        adj_p = romano_wolf_pvalues(t_joint, tb_joint)
        rej_rw[i] = (adj_p < alpha).reshape(2, n_model_a, n_model_b)

        betas[i, 0] = out_c["beta"]
        betas[i, 1] = out_p["beta"]

    per_cell_rw = rej_rw.mean(axis=0)
    per_cell_bonf = rej_bonf.mean(axis=0)
    per_cell_unc = rej_unc.mean(axis=0)

    return {
        "K": K,
        "N_per_cell": n_features * 2 * K,
        "M_total": M,
        "coercion_rw_per_cell": per_cell_rw[0],
        "persuasion_rw_per_cell": per_cell_rw[1],
        "coercion_rw_mean": float(per_cell_rw[0].mean()),
        "coercion_rw_min": float(per_cell_rw[0].min()),
        "coercion_rw_max": float(per_cell_rw[0].max()),
        "persuasion_rw_mean": float(per_cell_rw[1].mean()),
        "persuasion_rw_min": float(per_cell_rw[1].min()),
        "persuasion_rw_max": float(per_cell_rw[1].max()),
        "overall_rw_mean": float(per_cell_rw.mean()),
        "overall_rw_min": float(per_cell_rw.min()),
        "overall_bonf_mean": float(per_cell_bonf.mean()),
        "overall_unc_mean": float(per_cell_unc.mean()),
        "mean_beta_coercion": float(betas[:, 0].mean()),
        "mean_beta_persuasion": float(betas[:, 1].mean()),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_table(results: list[dict], title: str) -> None:
    print(f"\n--- {title} ---")
    print(
        f"{'K':>6} {'N':>9} {'Pwr(HC)':>9} {'Pwr(clust)':>11} {'SD(beta)':>10}"
    )
    for r in results:
        print(
            f"{r['K']:>6} {r['N_total']:>9,} {r['power_hc']:>8.1%} "
            f"{r['power_clustered']:>10.1%} {r['sd_beta']:>10.4f}"
        )


# -----------------------------------------------------------------------------
# Robustness utilities (to be run on real data, not in the power simulation)
# -----------------------------------------------------------------------------

def fit_logit_marginal_effect(df: pd.DataFrame) -> dict:
    """
    Fit a fixed-effects logit via dummy variables and compute the average
    marginal effect (AME) of the treatment. Use as a robustness check
    alongside the LPM specification.

    Expects df with columns: y, T, feature, model_a, model_b.
    Returns the AME, its delta-method SE, and the raw log-odds coefficient.

    Note: with a moderate number of FE levels and many observations per cell,
    the incidental parameters bias is small but not zero. Interpret divergence
    between LPM and logit-AME carefully.
    """
    import statsmodels.api as sm

    X_parts = [
        df[["T"]].astype(float),
        pd.get_dummies(df["feature"], prefix="f", drop_first=True).astype(float),
        pd.get_dummies(df["model_a"], prefix="a", drop_first=True).astype(float),
        pd.get_dummies(df["model_b"], prefix="b", drop_first=True).astype(float),
    ]
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X)
    y = df["y"].astype(float).values

    model = sm.Logit(y, X).fit(disp=False, maxiter=200)
    # Average marginal effect of T via manual delta method
    linpred = model.predict(X, linear=True)
    phi = np.exp(linpred) / (1 + np.exp(linpred)) ** 2
    ame = float(np.mean(phi) * model.params["T"])

    # Approximate SE via delta method on T coefficient scaled by mean phi
    se_ame = float(np.mean(phi) * model.bse["T"])
    return {
        "ame": ame,
        "se_ame": se_ame,
        "log_odds_coef": float(model.params["T"]),
        "log_odds_se": float(model.bse["T"]),
    }


def intra_environment_residual_correlation(
    df: pd.DataFrame, feature_to_env: dict[int, int]
) -> float:
    """
    After fitting the primary LPM, compute the average pairwise correlation of
    feature-level residual means for features sharing an environment.

    Use this as a pre-registered diagnostic: if the returned value exceeds 0.10,
    report environment-clustered SEs as an additional specification.

    Expects df with columns: y, T, feature, model_a, model_b, and a mapping
    from feature id to environment id.
    """
    # Fit LPM with dummies, extract residuals
    import statsmodels.api as sm

    X_parts = [
        df[["T"]].astype(float),
        pd.get_dummies(df["feature"], prefix="f", drop_first=True).astype(float),
        pd.get_dummies(df["model_a"], prefix="a", drop_first=True).astype(float),
        pd.get_dummies(df["model_b"], prefix="b", drop_first=True).astype(float),
    ]
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X)
    y = df["y"].astype(float).values

    fit = sm.OLS(y, X).fit()
    resid = fit.resid

    work = df[["feature"]].copy()
    work["resid"] = resid
    work["env"] = work["feature"].map(feature_to_env)

    # Mean residual per feature
    feat_means = work.groupby("feature")["resid"].mean().reset_index()
    feat_means["env"] = feat_means["feature"].map(feature_to_env)

    # Average pairwise correlation of feature residual means within environments.
    # With single-number residual means per feature we cannot compute a Pearson
    # correlation directly; instead compute average product normalized by variance.
    overall_var = float(feat_means["resid"].var(ddof=1))
    if overall_var == 0:
        return 0.0

    products = []
    for env, group in feat_means.groupby("env"):
        values = group["resid"].values
        if len(values) < 2:
            continue
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                products.append(values[i] * values[j])

    if not products:
        return 0.0
    return float(np.mean(products) / overall_var)


# -----------------------------------------------------------------------------
# E1-E4 minimum detectable effects (final design x pilot variance)
# -----------------------------------------------------------------------------
# Variance components are the pilot priors documented in the appendix; the
# design (feature counts per environment, 18 model pairs) is the final E1-E4
# structure. Inference uses the HC-robust SE, matching the pipeline (spec.SE_TYPE
# = "HC1"); cluster-robust is the canonical script's sensitivity variant.

# Pilot priors.
_BASELINE = 0.048
_FEATURE_SD = 0.05 * 1.5            # 1.5x inflation for E2-E4 vs E1
_PAIR_SD = {"persuasion": 0.07, "coercion": 0.09, "deception": 0.09}  # deception ~ coercion proxy
_SIGMA_BETA = {"persuasion": 0.039, "coercion": 0.024, "deception": 0.024}
# Final design: features per (capability, environment).
_FEATURES = {
    "persuasion": {"E1": 3, "E2": 2, "E3": 1, "E4": 1},
    "deception":  {"E1": 3, "E2": 2, "E3": 1, "E4": 1},
    "coercion":   {"E1": 3, "E2": 2, "E3": 1, "E4": 1},
}


def _mde_search(power_fn, grid=None) -> int | None:
    """Smallest treat_effect (pp) with HC power >= 0.80.

    Power is monotone increasing in the effect size, so we bisect the grid
    (~8 evaluations) instead of scanning it linearly (~70). Returns the grid
    point rounded to whole pp.
    """
    if grid is None:
        grid = np.arange(0.0050, 0.4001, 0.0025)
    lo, hi = 0, len(grid) - 1
    if power_fn(float(grid[hi])) < 0.80:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power_fn(float(grid[mid])) >= 0.80:
            hi = mid
        else:
            lo = mid + 1
    return round(float(grid[lo]) * 100)


# Bonferroni family sizes per Q (a priori design), matching the regression
# tiers: Tier C = 18 pairs x 12 (env x capability) per-cell tests; Tier B = 12
# env x capability cells; Tier A = 3 capabilities. Pass bonferroni=False for the
# uncorrected (raw alpha = 0.05) MDEs.
_M_CELL, _M_ENV, _M_CAP = 216, 12, 3
# The per-capability estimator pools E1-E3 only (E4 is on a different outcome scale).
_POOL_ENVS = ("E1", "E2", "E3")


def mde_e1_e4(K: int = 100, n_sims_cell: int = 2000, n_sims_pool: int = 800,
              bonferroni: bool = False) -> None:
    """Print E1-E4 MDEs (HC inference) for the three estimators at replicate K.

    With ``bonferroni=True`` each estimator class is sized at its Bonferroni-
    adjusted threshold: per-cell alpha = 0.05/216, per-env 0.05/12, per-cap
    0.05/3.
    """
    a_cell = 0.05 / _M_CELL if bonferroni else 0.05
    a_env = 0.05 / _M_ENV if bonferroni else 0.05
    a_cap = 0.05 / _M_CAP if bonferroni else 0.05
    tag = "Bonferroni" if bonferroni else "raw alpha=0.05"
    print(f"\n===== E1-E4 minimum detectable effects, K = {K} (HC-robust, {tag}) =====")
    for cap, envs in _FEATURES.items():
        pair_sd = _PAIR_SD[cap]
        ma_sd = mb_sd = pair_sd / np.sqrt(2.0)   # split combined pair SD across A,B FE
        sbeta = _SIGMA_BETA[cap]
        print(f"\n{cap}:")
        for env, nf in envs.items():
            cell = _mde_search(lambda t, nf=nf: power_at_per_cell(
                K=K, n_features=nf, baseline=_BASELINE, treat_effect=t,
                feature_sd=_FEATURE_SD, effect_heterogeneity=sbeta,
                n_sims=n_sims_cell, alpha=a_cell)["power"])
            roll = _mde_search(lambda t, nf=nf: power_at(
                K=K, n_sims=n_sims_pool, baseline=_BASELINE, treat_effect=t,
                feature_sd=_FEATURE_SD, model_a_sd=ma_sd, model_b_sd=mb_sd,
                effect_heterogeneity=sbeta, n_features=nf, alpha=a_env)["power_hc"])
            print(f"  {env} (n_feat={nf}): per-cell N={nf*2*K:>4} MDE~{cell}pp"
                  f"  |  per-env N={nf*18*2*K:>6} MDE~{roll}pp")
        tot = sum(nf for env, nf in envs.items() if env in _POOL_ENVS)
        cap_mde = _mde_search(lambda t, tot=tot: power_at(
            K=K, n_sims=n_sims_pool, baseline=_BASELINE, treat_effect=t,
            feature_sd=_FEATURE_SD, model_a_sd=ma_sd, model_b_sd=mb_sd,
            effect_heterogeneity=sbeta, n_features=tot, alpha=a_cap)["power_hc"])
        print(f"  POOLED E1-E3 ({tot} features): N={tot*18*2*K:>6} MDE~{cap_mde}pp")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulation-based power analysis for treatment uplift detection",
    )
    parser.add_argument(
        "--from-variance",
        help="Path to .eval directory or heterogeneity JSON; sets feature_sd from observed cross-feature SD",
    )
    parser.add_argument("--baseline", type=float, default=0.10,
                        help="Baseline approval rate under no influence (default: 0.10)")
    parser.add_argument("--treat-effect", type=float, default=0.03,
                        help="Treatment effect to detect (default: 0.03)")
    parser.add_argument("--n-sims", type=int, default=400)
    parser.add_argument("--e1e4", action="store_true",
                        help="Print E1-E4 MDEs (final design x pilot variance, HC) for K=100 and K=40")
    parser.add_argument("--bonferroni", action="store_true",
                        help="With --e1e4: size MDEs at Bonferroni-adjusted alpha (per-cell 0.05/216, per-env 0.05/12, per-cap 0.05/3)")
    args = parser.parse_args()

    if args.e1e4:
        mde_e1_e4(K=100, bonferroni=args.bonferroni)
        mde_e1_e4(K=40, bonferroni=args.bonferroni)
        return

    feature_sd = 0.03
    source = "default (0.03)"

    if args.from_variance:
        het = load_variance_from_data(args.from_variance)
        if het:
            observed_sd = het.get("cross_feature_summary", {}).get("sd_approval_rate")
            if observed_sd and observed_sd > 0:
                feature_sd = observed_sd
                source = f"variance_analysis ({args.from_variance})"

    print("=" * 74)
    print("POWER ANALYSIS")
    print("Design: 5 features x 6 model_A x 3 model_B x 2 arms x K resamples")
    print(f"Target: MDE = {args.treat_effect:.0%}, baseline = {args.baseline:.0%}, alpha = 0.05, power = 0.80")
    print("SE: cluster by feature (primary) / HC-robust (sensitivity)")
    print(f"feature_sd = {feature_sd:.4f}  (source: {source})")
    print("=" * 74)

    n_sims = args.n_sims
    baseline = args.baseline
    treat_effect = args.treat_effect

    K_values = (100, 200, 400, 800, 1600)

    # --- Main scenario
    main_results = [
        power_at(K, n_sims=n_sims, baseline=baseline, treat_effect=treat_effect,
                 feature_sd=feature_sd, effect_heterogeneity=0.0)
        for K in K_values
    ]
    print_table(main_results, f"Main: feature_sd={feature_sd:.4f}, homogeneous effect")

    # --- Conservative scenario: treatment effect varies across features
    conservative_results = [
        power_at(K, n_sims=n_sims, baseline=baseline, treat_effect=treat_effect,
                 feature_sd=feature_sd, effect_heterogeneity=0.01)
        for K in K_values
    ]
    print_table(conservative_results,
                "Conservative: effect varies ~1pp SD across features")

    # --- Type I error check
    print("\n--- Type I error under null (treat_effect = 0) ---")
    for fs in (feature_sd * 0.67, feature_sd, feature_sd * 1.67):
        r = power_at(K=400, n_sims=min(n_sims * 2, 800), feature_sd=fs,
                     baseline=baseline, treat_effect=0.0)
        print(
            f"  feature_sd={fs:.4f}: HC size = {r['power_hc']:.3f}  "
            f"Clustered size = {r['power_clustered']:.3f}  (target 0.05)"
        )


if __name__ == "__main__":
    main()