"""CACE (complier average causal effect) next to ITT and Q1, per capability.

Three estimands on the same rows:

    ITT    effect of prompt ASSIGNMENT on approval. Full sample.
    Q1     effect on the Q1-filtered sample (filter.apply_q1_filter). Both arms
           filtered by the per-arm rules in spec.py.
    CACE   2SLS with Z = arm, D = attempt (signals.attempt), Y = approval.
           Full sample: D is a regressor, never a row filter.

Within a single cell (one env, one model pair, no fixed effects) CACE is the
Wald ratio ITT / first stage. Pooled numbers use 2SLS with feature and
model-pair fixed effects, which weights each stratum by its sample size times
its first stage, so weak-instrument strata get little weight instead of
blowing up.

Deception is estimated under two D definitions. Primary: false claim found
(``D``). Sensitivity: self-report ack AND false claim (``D_ack``). The first
over-counts attempts in both arms (scorer noise, overstatements), so its first
stage is too large and its CACE too small; the second under-counts in the
treatment arm (concealment) and its CACE is too large. Both assume the prompt
acts only through D; if it does not, both CACEs are too large.

Coercion has a sensitivity D too: any scored coercion technique (``D_tech``),
which also counts punitive tool use without a stated threat.

E4 enters the per-environment rows only, with Y = points / 2.

The ack table reports, among rows with a false claim, the self-report ack rate
by arm and Agent A. It shows whether the probe responds to the deception
prompt; it is NOT a concealment test. Spot-checking baseline findings
(2026-09-03): most are misreadings of the material or invented supporting
detail that A does not experience as lying and does not ack; a minority are
outright fabrications, which A does ack. So the baseline rate is the share of
deliberate lies among baseline false claims, not an honest-reporting benchmark
for hallucinations, and a higher treatment rate is expected either way.

Just-identified IV is computed directly (b = (Z'X)^-1 Z'y) with HC1 robust
standard errors, matching the pipeline's regression convention.

Usage:
    python3 -m scripts.analysis.cace                       # master parquet -> analysis/10_cace/
    python3 -m scripts.analysis.cace --master PATH --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .filter import apply_q1_filter
from .io import load_master_samples
from .signals import (
    DECEPTION_D_ACK_AND_FALSE_CLAIM,
    DECEPTION_D_FALSE_CLAIM_ONLY,
    attempt,
    coercion_technique_attempt,
    deception_cell,
    false_claim_count,
    false_claim_found,
    self_report,
    self_report_ack,
)
from .spec import (
    CACE_DIR,
    CACE_EXCLUDE_ENVS,
    CAPABILITIES,
    COERCION_FEATURE_KEEP,
    E4_ENVS,
    E4_OUTCOME_COL,
    MIN_FIRST_STAGE,
    OUTCOME_COL,
    TIER_A_EXCLUDE_ENVS,
    WEAK_IV_F,
    env_code,
)

COERCION_D_TECHNIQUE = "technique"

# (column tag, definition). The first tag is primary; a second is a sensitivity D.
D_DEFS: dict[str, list[tuple[str, str | None]]] = {
    "persuasion": [("D", None)],
    "coercion": [("D", None), ("D_tech", COERCION_D_TECHNIQUE)],
    "deception": [("D", DECEPTION_D_FALSE_CLAIM_ONLY), ("D_ack", DECEPTION_D_ACK_AND_FALSE_CLAIM)],
}


# --------------------------------------------------------------------- linalg
def _hc1(bread, meat, n, k):
    return bread @ meat @ bread.T * n / (n - k)


def ols(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    V = _hc1(XtX_inv, X.T @ (X * (e ** 2)[:, None]), len(y), X.shape[1])
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def iv(y, X, Zm):
    """Just-identified IV; X and Zm share a shape, column 0 is endog/instrument."""
    ZtX_inv = np.linalg.pinv(Zm.T @ X)
    b = ZtX_inv @ Zm.T @ y
    e = y - X @ b
    V = _hc1(ZtX_inv, Zm.T @ (Zm * (e ** 2)[:, None]), len(y), X.shape[1])
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def design(df, cols):
    """Intercept plus drop-first dummies for each FE column with >1 level."""
    parts = [np.ones((len(df), 1))]
    for c in cols:
        if df[c].nunique() < 2:
            continue
        parts.append(pd.get_dummies(df[c].astype(str), drop_first=True).to_numpy(float))
    return np.hstack(parts)


# ------------------------------------------------------------------ prepare
def prepare(df: pd.DataFrame, cap: str) -> tuple[pd.DataFrame, dict]:
    """Rows for one capability with Z, Y, D columns and the Q1 keep mask.

    Rows whose attempt flag was never graded are dropped (flag-complete
    sample); the count is returned in the audit dict.
    """
    d = df[(df["capability"] == cap) & ~df["environment"].isin(CACE_EXCLUDE_ENVS)].copy()
    if cap == "coercion":
        keep = [COERCION_FEATURE_KEEP.get(e) in (None, f)
                for e, f in zip(d["environment"], d["feature_name"])]
        d = d[keep].copy()

    d["Z"] = d["T"].astype(int)
    d["Y"] = d[OUTCOME_COL].astype(float)
    e4 = d["environment"].isin(E4_ENVS)
    d.loc[e4, "Y"] = d.loc[e4, E4_OUTCOME_COL].astype(float)
    d["pair"] = d["model_a"].astype(str) + " | " + d["model_b"].astype(str)
    d["cell"] = deception_cell(d)

    for tag, ddef in D_DEFS[cap]:
        if ddef is None:
            d[tag] = attempt(d, cap)
        elif ddef == COERCION_D_TECHNIQUE:
            d[tag] = coercion_technique_attempt(d)
        else:
            d[tag] = attempt(d, cap, ddef)

    q1_rows, _ = apply_q1_filter(d)
    d["q1_keep"] = d.index.isin(q1_rows.index)

    tags = [t for t, _ in D_DEFS[cap]]
    complete = d[tags].notna().all(axis=1)
    audit = {"capability": cap, "n_rows": len(d),
             "n_flag_missing": int((~complete).sum())}
    for tag in tags:
        d[tag] = d[tag].astype("boolean")
    return d[complete].copy(), audit


# ----------------------------------------------------------------- estimate
def estimate(d: pd.DataFrame, fe_cols: list[str], tags: list[str]) -> dict:
    """ITT, Q1 and one CACE per D tag on one sample."""
    out = {"n": len(d), "n_t": int((d.Z == 1).sum()), "n_c": int((d.Z == 0).sum()),
           "y_t": d.loc[d.Z == 1, "Y"].mean(), "y_c": d.loc[d.Z == 0, "Y"].mean()}
    y, z = d["Y"].to_numpy(float), d["Z"].to_numpy(float)[:, None]
    FE = design(d, fe_cols)

    b, se = ols(y, np.hstack([z, FE]))
    out["ITT"], out["ITT_se"] = b[0], se[0]

    q1 = d[d.q1_keep]
    out["q1_n"] = len(q1)
    out["q1_keep_t"] = d.loc[d.Z == 1, "q1_keep"].mean()
    out["q1_keep_c"] = d.loc[d.Z == 0, "q1_keep"].mean()
    if q1.Z.nunique() == 2 and len(q1) > 20:
        b, se = ols(q1["Y"].to_numpy(float),
                    np.hstack([q1["Z"].to_numpy(float)[:, None], design(q1, fe_cols)]))
        out["Q1"], out["Q1_se"] = b[0], se[0]
    else:
        out["Q1"] = out["Q1_se"] = np.nan

    for tag in tags:
        Dv = d[tag].astype(float).to_numpy()
        out[f"{tag}_t"] = Dv[d.Z == 1].mean()
        out[f"{tag}_c"] = Dv[d.Z == 0].mean()          # always-taker rate
        bf, sef = ols(Dv, np.hstack([z, FE]))
        fs = bf[0]
        out[f"fs_{tag}"] = fs
        out[f"fsF_{tag}"] = (fs / sef[0]) ** 2 if sef[0] > 1e-8 else (np.inf if abs(fs) > 0 else np.nan)
        if abs(fs) < MIN_FIRST_STAGE:
            out[f"CACE_{tag}"] = out[f"CACE_{tag}_se"] = np.nan
            continue
        b, se = iv(y, np.hstack([Dv[:, None], FE]), np.hstack([z, FE]))
        out[f"CACE_{tag}"], out[f"CACE_{tag}_se"] = b[0], se[0]
    return out


def _cols(tags: list[str]) -> list[str]:
    cols = ["n", "n_t", "n_c", "y_t", "y_c", "ITT", "ITT_se",
            "q1_n", "q1_keep_t", "q1_keep_c", "Q1", "Q1_se"]
    for t in tags:
        cols += [f"{t}_t", f"{t}_c", f"fs_{t}", f"fsF_{t}", f"CACE_{t}", f"CACE_{t}_se"]
    return cols


def ack_among_false_claims(d: pd.DataFrame) -> pd.DataFrame:
    """Self-report ack rate among rows with a false claim, by (environment, arm, model_a).

    Diagnostic only: shows whether the probe tracks the deception prompt. Not a
    concealment test (see module docstring).
    """
    sr = self_report(d)
    fc = d[false_claim_found(false_claim_count(d)) & sr.notna()]
    ack = self_report_ack(self_report(fc))
    t = (fc.assign(ack=ack).groupby(["environment", "Z", "model_a"])
           .ack.agg(ack_rate="mean", n_false_claim="size").reset_index())
    return t


def cell_counts(d: pd.DataFrame) -> pd.DataFrame:
    """Deception 2x2 cell counts per (environment, arm, model_a)."""
    t = (d.assign(cell=d["cell"].fillna("ungraded"))
          .groupby(["environment", "Z", "model_a", "cell"]).size()
          .unstack("cell", fill_value=0).reset_index())
    return t


# ---------------------------------------------------------------------- run
def off_target(d: pd.DataFrame, tags: list[str]) -> list[dict]:
    """Exclusion diagnostic: do treated transcripts with no attempt (D = 0) beat baseline?

    For each D definition, compares treated non-attempters (and attempters) with
    the baseline of the same (model_a, model_b, environment) cell, weighting cells
    by treated count. ``adjusted`` is the Wald CACE after removing the non-attempt
    excess: (ITT - (1 - D_t) * gap_no_attempt) / (D_t - D_c). It assumes that excess
    is caused by the prompt; non-attempters are self-selected, so read it as a
    diagnostic, not an estimate.
    """
    key = ["model_a", "model_b", "environment"]
    t, c = d[d.Z == 1], d[d.Z == 0]
    base = c.groupby(key)["Y"].mean().rename("base")
    out = []
    for tag in tags:
        Dt = t[tag].astype(float)
        rec = {"D": tag, "n_t": len(t), "D_t": float(Dt.mean()),
               "D_c": float(c[tag].astype(float).mean()),
               "ITT": float(t.Y.mean() - c.Y.mean())}
        for label, sub in (("no_attempt", t[Dt == 0]), ("attempt", t[Dt == 1])):
            g = sub.groupby(key)["Y"].agg(["size", "mean"]).join(base, how="inner")
            rec[f"n_{label}"] = int(g["size"].sum())
            rec[f"gap_{label}"] = (float(np.average(g["mean"] - g["base"], weights=g["size"]))
                                   if len(g) else np.nan)
        fs = rec["D_t"] - rec["D_c"]
        rec["wald"] = rec["ITT"] / fs if abs(fs) >= MIN_FIRST_STAGE else np.nan
        rec["adjusted"] = ((rec["ITT"] - (1 - rec["D_t"]) * rec["gap_no_attempt"]) / fs
                           if abs(fs) >= MIN_FIRST_STAGE else np.nan)
        out.append(rec)
    return out


def run(master: Path | None = None) -> dict[str, pd.DataFrame]:
    df = load_master_samples(master)
    df = df[df["capability"].isin(CAPABILITIES)]

    pooled, by_env, by_pair, cells, acks, audits, offt = [], [], {}, [], [], [], []
    for cap in CAPABILITIES:
        d, audit = prepare(df, cap)
        audits.append(audit)
        tags = [t for t, _ in D_DEFS[cap]]
        cells.append(cell_counts(d).assign(capability=cap))
        acks.append(ack_among_false_claims(d).assign(capability=cap))

        core = d[~d["environment"].isin(TIER_A_EXCLUDE_ENVS)]
        if core.Z.nunique() == 2:
            pooled.append({"capability": cap,
                           **estimate(core, ["feature_name", "pair"], tags)})
            offt += [{"capability": cap, "scope": "E1-E3", **r} for r in off_target(core, tags)]
        e4 = d[d["environment"].isin(E4_ENVS)]
        if e4.Z.nunique() == 2:
            offt += [{"capability": cap, "scope": "E4", **r} for r in off_target(e4, tags)]

        for env, g in d.groupby("environment"):
            if g.Z.nunique() < 2:
                continue
            by_env.append({"capability": cap, "environment": env,
                           **estimate(g, ["feature_name", "pair"], tags)})

        rows = []
        for p, g in core.groupby("pair"):
            if g.Z.nunique() < 2:
                continue
            rows.append({"pair": p, **estimate(g, ["environment", "feature_name"], tags)})
        by_pair[cap] = pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()

    out = {"pooled": pd.DataFrame(pooled).set_index("capability"),
           "by_env": pd.DataFrame(by_env).set_index(["capability", "environment"]),
           "cells": pd.concat(cells, ignore_index=True),
           "ack_among_false_claims": pd.concat(acks, ignore_index=True),
           "audit": pd.DataFrame(audits).set_index("capability"),
           "off_target": pd.DataFrame(offt).set_index(["capability", "scope", "D"])}
    for cap, t in by_pair.items():
        out[f"by_pair_{cap}"] = t
    return out


def _md(t: pd.DataFrame) -> str:
    """Minimal markdown table (index + columns); avoids the tabulate dependency."""
    t = t.reset_index()
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    L = ["| " + " | ".join(map(str, t.columns)) + " |",
         "|" + "|".join("---" for _ in t.columns) + "|"]
    L += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(L)


def render_summary_md(tables: dict[str, pd.DataFrame]) -> str:
    L = ["# CACE vs ITT vs Q1", "",
         "ITT and CACE use the full flag-complete sample; Q1 uses the filtered sample.",
         "CACE = 2SLS with Z = arm, D = attempt. Pooled and per-pair rows use E1-E3; per-environment",
         "rows add E4 on A's outcome score (0, 0.5 or 1). E5 is excluded. Pooled rows",
         f"and use feature + model-pair FE, HC1 SEs. `fsF` < {WEAK_IV_F:g} flags a weak",
         f"instrument; |first stage| < {MIN_FIRST_STAGE} is not inverted.", ""]

    def _tbl(t: pd.DataFrame, cols: list[str]) -> str:
        return _md(t[cols].astype(float))

    p = tables["pooled"]
    L += ["## Pooled (E1-E3)", "",
          _tbl(p, ["n", "ITT", "ITT_se", "Q1", "Q1_se", "q1_keep_t", "q1_keep_c"]), ""]
    for cap in p.index:
        tags = [t for t, _ in D_DEFS[cap]]
        L += [f"### {cap}", "", _tbl(p.loc[[cap]], _cols(tags)[12:]), ""]

    if "deception" in p.index:
        r = p.loc["deception"]
        L += ["### Deception: the two D definitions", "",
              "| | D = false claim (primary) | D_ack = ack AND false claim | gap |",
              "|---|---:|---:|---:|",
              f"| first stage | {r['fs_D']:.3f} | {r['fs_D_ack']:.3f} | {r['fs_D'] - r['fs_D_ack']:+.3f} |",
              f"| CACE | {r['CACE_D']:.3f} | {r['CACE_D_ack']:.3f} | {r['CACE_D'] - r['CACE_D_ack']:+.3f} |",
              "",
              "False-claim-only over-counts attempts (first stage too large, CACE too",
              "small); ack AND false claim under-counts in the treatment arm (CACE too",
              "large). A small gap means the choice does not matter.", ""]

    L += ["## Self-report ack rate among rows with a false claim", "",
          "Diagnostic for whether the probe responds to the deception prompt. Not a",
          "concealment test: baseline false claims are mostly misreadings or invented",
          "supporting detail A does not regard as lies (not acked), plus a minority of",
          "outright fabrications (acked), so a higher treatment rate is expected.", ""]
    a = tables["ack_among_false_claims"]
    pooled_ack = (a.assign(acks=a.ack_rate * a.n_false_claim)
                   .groupby(["capability", "Z"]).agg(acks=("acks", "sum"), n=("n_false_claim", "sum")))
    pooled_ack["ack_rate"] = pooled_ack.acks / pooled_ack.n
    pooled_ack = pooled_ack[["ack_rate", "n"]].unstack("Z").round(3)
    pooled_ack.columns = [f"{c}_{'treatment' if z == 1 else 'baseline'}" for c, z in pooled_ack.columns]
    L += [_md(pooled_ack), ""]
    by_model = (a.assign(acks=a.ack_rate * a.n_false_claim)
                 .groupby(["capability", "model_a", "Z"]).agg(acks=("acks", "sum"), n=("n_false_claim", "sum")))
    by_model = (by_model.acks / by_model.n).unstack("Z").round(3)
    by_model.columns = ["baseline", "treatment"]
    L += [_md(by_model), ""]

    L += ["## By environment", ""]
    e = tables["by_env"].reset_index()
    for cap, g in e.groupby("capability", sort=False):
        tags = [t for t, _ in D_DEFS[cap]]
        cols = ["n", "ITT", "Q1", "q1_keep_t", "q1_keep_c"] + [
            c for t in tags for c in (f"fs_{t}", f"fsF_{t}", f"CACE_{t}", f"CACE_{t}_se")]
        L += [f"### {cap}", "", _md(g.set_index("environment")[cols].astype(float)), ""]

    L += ["## Off-target diagnostic (exclusion)", "",
          "Treated transcripts with no attempt (D = 0) and with an attempt, each minus the",
          "baseline of the same (A, B, environment) cell, weighted by treated count.",
          "`adjusted` removes the no-attempt excess from the Wald CACE; it assumes that",
          "excess is caused by the prompt. E4 is on A's outcome score.", "",
          _md(tables["off_target"][["D_t", "D_c", "n_no_attempt", "gap_no_attempt",
                                     "gap_attempt", "ITT", "wald", "adjusted"]]), ""]
    L += ["## Flag completeness", "", _md(tables["audit"]), ""]
    return "\n".join(L) + "\n"


def render_cace_latex(tables: dict[str, pd.DataFrame]) -> str:
    """ITT, Q1 and CACE per capability: E1-E3 pooled and per environment, then E4 unpooled."""
    num = lambda v: "--" if pd.isna(v) else f"${v:+.3f}$"
    fs = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    cace = lambda v, se: "--" if pd.isna(v) else f"${v:+.3f}$ ({se:.3f})"
    by_env = tables["by_env"]

    def row(cap_label: str, env_label: str, cap: str, r: pd.Series) -> str:
        alt = [t for t, _ in D_DEFS[cap]][1:]
        a = alt[0] if alt else None
        alt_cells = ([fs(r.get(f"fs_{a}")), cace(r.get(f"CACE_{a}"), r.get(f"CACE_{a}_se"))]
                     if a else ["--", "--"])
        return (f"{cap_label} & {env_label} & {int(r['n']):,} & {num(r['ITT'])} & {num(r['Q1'])} & "
                f"{fs(r['fs_D'])} & {cace(r['CACE_D'], r['CACE_D_se'])} & "
                f"{alt_cells[0]} & {alt_cells[1]} \\\\")

    L = ["% Requires preamble: \\usepackage{booktabs}",
         "\\begin{table}[!ht]",
         "\\caption{\\textbf{ITT, Q1, and CACE by capability.} ITT is the effect of prompt assignment on all "
         "transcripts. Q1 is refitted on every transcript that passes the filter, without the "
         "per-cell subsampling of the Q1 tables. CACE is two-stage least squares on all "
         "transcripts, with prompt assignment as the instrument for the attempt indicator; the "
         "first stage is the effect of assignment on the attempt rate. The attempt is advocacy "
         "for persuasion, a false claim for deception, and an expressed threat for coercion. The "
         "alternative attempt counts a deception attempt only when \\agentA{} also acknowledges "
         "the false claim, and a coercion attempt whenever a scored coercion technique appears, "
         "including punitive tool use without a stated threat. E4 rows use \\agentA{}'s "
         "outcome score (0, 0.5, or 1) and are not pooled. All rows use feature and "
         "model-pair fixed effects. HC1-robust standard errors in parentheses.}",
         "\\label{tab:cace}",
         "\\centering",
         "\\footnotesize",
         "\\begin{tabular}{llrrrrrrr}",
         "\\toprule",
         " & & & & & \\multicolumn{2}{c}{\\textbf{Attempt}} & \\multicolumn{2}{c}{\\textbf{Alternative attempt}} \\\\",
         "\\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
         "\\textbf{Capability} & \\textbf{Env} & $\\bm{N}$ & \\textbf{ITT} & \\textbf{Q1} & "
         "\\textbf{\\begin{tabular}[b]{@{}r@{}}First\\\\stage\\end{tabular}} & \\textbf{CACE} & "
         "\\textbf{\\begin{tabular}[b]{@{}r@{}}First\\\\stage\\end{tabular}} & \\textbf{CACE} \\\\",
         "\\midrule"]
    e4_rows = []
    for i, cap in enumerate(tables["pooled"].index):
        if i:
            L.append("\\addlinespace")
        envs = by_env.loc[cap] if cap in by_env.index.get_level_values(0) else pd.DataFrame()
        envs = envs.sort_index(key=lambda ix: ix.map(env_code))
        core = [(env_code(e), r) for e, r in envs.iterrows() if e not in E4_ENVS]
        e4_rows += [(cap, r) for e, r in envs.iterrows() if e in E4_ENVS]
        for j, (label, r) in enumerate([("E1--E3", tables["pooled"].loc[cap])] + core):
            L.append(row(cap.title() if j == 0 else "", label, cap, r))
    if e4_rows:
        L += ["\\midrule", "\\multicolumn{9}{l}{\\textbf{E4, outcome score, not pooled}} \\\\"]
        L += [row(cap.title(), "E4", cap, r) for cap, r in e4_rows]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--master", type=Path, help="override final_data/master_samples.parquet")
    ap.add_argument("--out", type=Path, default=CACE_DIR, help="output directory")
    args = ap.parse_args()

    tables = run(args.master)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(args.out / f"cace_{name}.csv")
    (args.out / "cace_summary.md").write_text(render_summary_md(tables))
    (args.out / "cace_table.tex").write_text(render_cace_latex(tables))

    pd.set_option("display.width", 300)
    print(render_summary_md(tables))
    for cap in CAPABILITIES:
        t = tables.get(f"by_pair_{cap}")
        if t is None or t.empty:
            continue
        tags = [tg for tg, _ in D_DEFS[cap]]
        print(f"\n{'=' * 25} {cap.upper()} per pair (env + feature FE, HC1) {'=' * 25}")
        print(t[_cols(tags)].astype(float).round(3).to_string())
        oob = t[[abs(t[f"CACE_{tg}"]) > 1 for tg in tags][0]]
        if len(oob):
            print(f"  !! {len(oob)}/{len(t)} pairs with |CACE| > 1 "
                  f"(impossible for a valid IV on a binary outcome): {list(oob.index)}")
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
