"""Stage 2: principled Q1 filter.

Q2 sample = full validity-passing dataset (no behavioral filter).
Q1 sample = Q2 ∩ behavioral filter (per env-group baseline/treatment rules).

Predicates come from ``signals.py`` (shared with the CACE estimator) and are
NaN-strict: a flag with NaN value does not equal False/True, so the row
drops. The audit table records exactly how many rows are lost to each
condition.

Outputs (analysis/02_filter/):
  filter_audit.csv        per (capability, T, rule, condition) drop counts
  filter_audit.md         human-readable summary
  filter_attrition.png    attrition diagram per capability
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import load_master_samples
from .signals import (
    false_claim_count,
    false_claim_found,
    flag_is_false,
    flag_is_true,
    no_false_claim,
    self_report,
    self_report_ack,
    self_report_no_ack,
)
from .spec import (
    BASELINE_RULES,
    CAPABILITIES,
    DATA_DIR,
    FILTER_DIR,
    FilterRule,
    TIER_A_EXCLUDE_ENVS,
    TREATMENT_RULES,
    _env_rule_group,
)


# ---------------------------------------------------------------------------
# Apply a rule, returning kept-mask + per-condition drop counts
# ---------------------------------------------------------------------------


@dataclass
class RuleAudit:
    capability: str
    T: int
    rule_label: str
    n_in: int
    n_out: int
    drops: dict[str, int]    # condition -> rows that fail it (non-exclusive)
    drops_first_failure: dict[str, int]  # exclusive: counted once per row at the first failing condition


def apply_rule(df: pd.DataFrame, capability: str, T: int, rule: FilterRule) -> tuple[pd.Series, RuleAudit]:
    """Return a boolean keep-mask aligned to ``df`` plus a RuleAudit."""
    n_in = int(len(df))

    # Per-condition keep masks. Names mirror the spec.
    masks: dict[str, pd.Series] = {}

    for flag_col in rule.forbidden_flags:
        masks[f"{flag_col}=False"] = flag_is_false(df[flag_col])

    for flag_col in rule.required_true_flags:
        masks[f"{flag_col}=True"] = flag_is_true(df[flag_col])

    if rule.require_self_report is False:
        masks["self_report=false"] = self_report_no_ack(self_report(df))
    elif rule.require_self_report is True:
        masks["self_report=true"] = self_report_ack(self_report(df))

    if rule.require_false_claim is False:
        masks["no_false_claim"] = no_false_claim(false_claim_count(df))
    elif rule.require_false_claim is True:
        masks["false_claim_found"] = false_claim_found(false_claim_count(df))

    # Aggregate.
    keep = pd.Series(True, index=df.index)
    drops_each: dict[str, int] = {}
    for name, m in masks.items():
        drops_each[name] = int((~m).sum())
        keep = keep & m
    n_out = int(keep.sum())

    # Exclusive drop counts: for each row, the FIRST condition (in spec order)
    # that fails. Sums match (n_in - n_out).
    drops_first: dict[str, int] = {n: 0 for n in masks}
    rolling_keep = pd.Series(True, index=df.index)
    for name, m in masks.items():
        first_fail = rolling_keep & ~m
        drops_first[name] = int(first_fail.sum())
        rolling_keep = rolling_keep & m

    return keep, RuleAudit(
        capability=capability, T=T, rule_label=rule.label,
        n_in=n_in, n_out=n_out,
        drops=drops_each,
        drops_first_failure=drops_first,
    )


# ---------------------------------------------------------------------------
# Apply Q1 filter to the full dataset
# ---------------------------------------------------------------------------


def apply_q1_filter(
    df: pd.DataFrame,
    baseline_rules: dict | None = None,
    treatment_rules: dict | None = None,
) -> tuple[pd.DataFrame, list[RuleAudit]]:
    """Run the Q1 behavioral filter; return kept rows + per-capability audits.

    The rule dicts default to the spec's. `filter_rule_robustness` passes
    variants so an alternative rule can be scored without reassigning module
    state.
    """
    audits: list[RuleAudit] = []
    keep = pd.Series(False, index=df.index)
    baseline_rules = BASELINE_RULES if baseline_rules is None else baseline_rules
    treatment_rules = TREATMENT_RULES if treatment_rules is None else treatment_rules

    env_groups = df["environment"].map(_env_rule_group)

    for cap in CAPABILITIES:
        for grp in ("default", "e4"):
            bl_rule = baseline_rules[grp]
            tx_rules = treatment_rules[grp]

            # T = 0 (baseline).
            sub_idx = df[
                (df["capability"] == cap) & (df["T"] == 0) & (env_groups == grp)
            ].index
            if len(sub_idx):
                sub = df.loc[sub_idx]
                sub_keep, audit = apply_rule(sub, capability=cap, T=0, rule=bl_rule)
                keep.loc[sub_idx] = sub_keep
                audits.append(audit)

            # T = 1 (treatment) — per-capability rule.
            rule = tx_rules.get(cap)
            if rule is None:
                continue
            sub_idx = df[
                (df["capability"] == cap) & (df["T"] == 1) & (env_groups == grp)
            ].index
            if len(sub_idx):
                sub = df.loc[sub_idx]
                sub_keep, audit = apply_rule(sub, capability=cap, T=1, rule=rule)
                keep.loc[sub_idx] = sub_keep
                audits.append(audit)

    return df.loc[keep].copy(), audits


# ---------------------------------------------------------------------------
# Audit rendering
# ---------------------------------------------------------------------------


def audits_to_dataframe(audits: list[RuleAudit]) -> pd.DataFrame:
    rows: list[dict] = []
    for a in audits:
        # One summary row per (capability, T, rule).
        rows.append({
            "capability": a.capability, "T": a.T, "rule": a.rule_label,
            "condition": "TOTAL", "n_in": a.n_in, "n_out": a.n_out,
            "n_dropped_total": a.n_in - a.n_out,
            "n_dropped_at_this_condition": a.n_in - a.n_out,
            "n_failing_this_condition": a.n_in - a.n_out,
        })
        for cond, n_first in a.drops_first_failure.items():
            rows.append({
                "capability": a.capability, "T": a.T, "rule": a.rule_label,
                "condition": cond, "n_in": a.n_in, "n_out": a.n_out,
                "n_dropped_total": a.n_in - a.n_out,
                "n_dropped_at_this_condition": n_first,
                "n_failing_this_condition": a.drops[cond],
            })
    return pd.DataFrame(rows)


def render_audit_markdown(audit_df: pd.DataFrame) -> str:
    L = []
    L.append("# Q1 filter audit")
    L.append("")
    L.append("- `n_in` / `n_out`: rows entering / surviving the rule.")
    L.append("- `n_failing_this_condition`: rows that fail this single condition (non-exclusive — a row may fail multiple conditions).")
    L.append("- `n_dropped_at_this_condition`: rows lost AT this condition in spec order (exclusive — sums to `n_dropped_total`).")
    L.append("")
    for (capability, T), g in audit_df.groupby(["capability", "T"]):
        rule_label = g["rule"].iloc[0]
        n_in = int(g["n_in"].iloc[0])
        n_out = int(g["n_out"].iloc[0])
        L.append(f"## {capability} · T={T} · `{rule_label}`")
        L.append(f"- in: {n_in:,}  →  out: {n_out:,}  ({n_out / n_in:.1%} retention)")
        L.append("")
        L.append("| Condition | Failing this condition | Dropped at this condition |")
        L.append("|---|---:|---:|")
        for _, r in g[g["condition"] != "TOTAL"].iterrows():
            L.append(
                f"| {r['condition']} "
                f"| {int(r['n_failing_this_condition']):,} "
                f"| {int(r['n_dropped_at_this_condition']):,} |"
            )
        L.append("")
    return "\n".join(L) + "\n"


def _tex_escape(s: str) -> str:
    """Escape LaTeX special chars in arbitrary strings (rule names, conditions)."""
    return (
        s.replace("\\", "\\\\")
         .replace("_", "\\_")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("$", "\\$")
         .replace("#", "\\#")
         .replace("≥", "$\\geq$")
    )


def render_audit_latex(audit_df: pd.DataFrame) -> str:
    """Paper-ready LaTeX: per-(capability, T, condition) drop counts."""
    L: list[str] = []
    L.append("% Requires preamble: \\usepackage{booktabs}")
    L.append("\\begin{table}[!ht]")
    L.append("\\caption{Q1 filter: per-condition drop counts, by capability "
             "and treatment indicator. Failing $=$ rows that fail this single "
             "condition (non-exclusive). Dropped at $=$ rows lost AT this "
             "condition in spec order (exclusive); column sums to "
             "$n_{\\text{in}} - n_{\\text{out}}$ within each rule.}")
    L.append("\\label{tab:filter_audit}")
    L.append("\\centering")
    L.append("\\small")
    L.append("\\begin{tabular}{llrr}")
    L.append("\\toprule")
    L.append("Group & Condition & Failing this cond.\\ & Dropped at this cond.\\ \\\\")
    L.append("\\midrule")
    grouped = audit_df.groupby(["capability", "T"], dropna=False)
    first = True
    for (cap, t), g in grouped:
        rule = g["rule"].iloc[0]
        n_in = int(g["n_in"].iloc[0])
        n_out = int(g["n_out"].iloc[0])
        if not first:
            L.append("\\midrule")
        first = False
        retention_pct = 100.0 * n_out / max(n_in, 1)
        L.append(
            f"\\multicolumn{{4}}{{l}}{{\\emph{{{cap}, $T={int(t)}$ — "
            f"\\texttt{{{_tex_escape(rule)}}}: "
            f"in $\\to$ out $=$ {n_in:,} $\\to$ {n_out:,} "
            f"({retention_pct:.1f}\\% retention)}}}} \\\\"
        )
        for _, r in g[g["condition"] != "TOTAL"].iterrows():
            L.append(
                f" & \\texttt{{{_tex_escape(str(r['condition']))}}} & "
                f"{int(r['n_failing_this_condition']):,} & "
                f"{int(r['n_dropped_at_this_condition']):,} \\\\"
            )
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


def plot_attrition(audit_df: pd.DataFrame, out_path: Path) -> None:
    """One stacked bar per filter rule: rows surviving + rows dropped at each condition."""
    grouped = audit_df[audit_df["condition"] != "TOTAL"].copy()
    if grouped.empty:
        return

    keys = ["capability", "T", "rule"]
    pivot_drops = grouped.pivot_table(
        index=keys, columns="condition",
        values="n_dropped_at_this_condition",
        aggfunc="sum", fill_value=0,
    )
    survives = audit_df[audit_df["condition"] == "TOTAL"].set_index(keys)["n_out"]
    pivot_drops["surviving"] = survives

    # Reorder so "surviving" is first (bottom of stack).
    cols = ["surviving"] + [c for c in pivot_drops.columns if c != "surviving"]
    pivot_drops = pivot_drops[cols]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.6 * len(pivot_drops) + 1.5)))
    pivot_drops.plot(kind="barh", stacked=True, ax=ax,
                     colormap="tab20")
    ax.set_xlabel("rows")
    ax.set_title("Q1 filter — attrition by rule", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-protocol validity check
# ---------------------------------------------------------------------------


def per_protocol_summary(full: pd.DataFrame, kept_mask: pd.Series) -> pd.DataFrame:
    """Per-capability approval rates for three groups:

      T=0 kept           (the Q1 baseline term)
      T=1 dropped        (failed treatment: A refused / modified / no canonical
                          technique fired)
      T=1 kept           (the Q1 treatment term)

    Under a clean per-protocol interpretation of Q1, T=1 dropped should
    approximate T=0 kept: rows where the treatment did not actually fire
    revert to baseline behaviour. E1-E3 only.
    """
    core = ~full["environment"].isin(TIER_A_EXCLUDE_ENVS)
    full, kept_mask = full[core], kept_mask[core]
    rows: list[dict] = []
    groups = [
        ("T=0 kept (Q1 baseline)", lambda d, k: (d["T"] == 0) & k),
        ("T=1 dropped (failed treatment)", lambda d, k: (d["T"] == 1) & ~k),
        ("T=1 kept (Q1 treatment)", lambda d, k: (d["T"] == 1) & k),
    ]
    for cap in CAPABILITIES:
        mask_cap = full["capability"] == cap
        for label, mask_fn in groups:
            g = full[mask_cap & mask_fn(full, kept_mask)]
            n = int(len(g))
            p = float(g["approved"].mean()) if n > 0 else float("nan")
            se = float(np.sqrt(p * (1 - p) / n)) if n > 0 else float("nan")
            rows.append({
                "capability": cap, "group": label,
                "approval": p, "n": n, "se": se,
            })
    return pd.DataFrame(rows)


def plot_per_protocol(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: per-capability approval for the three protocol groups."""
    from . import style
    from .spec import BASELINE_COLOR, CAPABILITY_DISPLAY, Q1_COLOR
    style.apply()
    caps = list(CAPABILITIES)
    group_order = [
        "T=0 kept (Q1 baseline)",
        "T=1 dropped (failed treatment)",
        "T=1 kept (Q1 treatment)",
    ]
    labels = {
        "T=0 kept (Q1 baseline)":         "T = 0, Kept (Q1 Baseline)",
        "T=1 dropped (failed treatment)": "T = 1, Dropped (Failed Treatment)",
        "T=1 kept (Q1 treatment)":        "T = 1, Kept (Q1 Treatment)",
    }
    colors = {
        "T=0 kept (Q1 baseline)":         BASELINE_COLOR,
        "T=1 dropped (failed treatment)": Q1_COLOR,
        "T=1 kept (Q1 treatment)":        Q1_COLOR,
    }
    alphas = {
        "T=0 kept (Q1 baseline)":         1.0,
        "T=1 dropped (failed treatment)": 0.45,
        "T=1 kept (Q1 treatment)":        1.0,
    }

    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH_IN * 0.9, 2.5))
    x = np.arange(len(caps))
    width = 0.26
    for i, label in enumerate(group_order):
        s = summary[summary["group"] == label].set_index("capability").reindex(caps)
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, s["approval"], width=width * 0.94,
            color=colors[label], alpha=alphas[label], label=labels[label], lw=0, zorder=3,
        )
        ax.errorbar(x + offset, s["approval"], yerr=1.96 * s["se"], fmt="none",
                    ecolor="#3a3a3a", elinewidth=0.7, capsize=2, capthick=0.7, zorder=4)
        for bar, p, n, se in zip(bars, s["approval"], s["n"], s["se"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.96 * se + 0.012,
                f"{p:.3f}\nn = {int(n):,}",
                ha="center", va="bottom", fontsize=style.FS_SMALL - 0.5, zorder=5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([CAPABILITY_DISPLAY.get(c, c.title()) for c in caps])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Approval Rate")
    ax.set_ylim(0, max(0.62, float(summary["approval"].max()) + 0.2))
    ax.legend(loc="upper left", ncol=1, fontsize=style.FS_BASE - 0.7,
              handlelength=1.0, handleheight=1.0, handletextpad=0.5, labelspacing=0.35)
    style.tidy_axes(ax, grid_axis="y")
    style.save(fig, out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(out_dir: Path | None = None,
        data_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or FILTER_DIR
    data_dir = data_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = load_master_samples()
    df = df[df["capability"].isin(CAPABILITIES)].reset_index(drop=True)

    # Q2 = full sample (in-scope capabilities). Q1 = behavioral-filter survivors.
    q2 = df.copy()
    q1, audits = apply_q1_filter(df)

    written: dict[str, Path] = {}
    q1_path = data_dir / "q1_filtered.parquet"
    q2_path = data_dir / "q2_filtered.parquet"
    q1.to_parquet(q1_path, index=False)
    q2.to_parquet(q2_path, index=False)
    written["q1_filtered"] = q1_path
    written["q2_filtered"] = q2_path

    audit_df = audits_to_dataframe(audits)
    audit_path = out_dir / "filter_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    written["filter_audit_csv"] = audit_path

    md = render_audit_markdown(audit_df)
    md_path = out_dir / "filter_audit.md"
    md_path.write_text(md)
    written["filter_audit_md"] = md_path

    tex_path = out_dir / "filter_audit.tex"
    tex_path.write_text(render_audit_latex(audit_df))
    written["filter_audit_tex"] = tex_path

    plot_attrition(audit_df, out_dir / "filter_attrition")
    written["filter_attrition_png"] = out_dir / "filter_attrition.png"

    # Per-protocol validity check: approval rates for T=0 kept, T=1 dropped,
    # T=1 kept. Bypasses load_master_samples since `df` is already in the
    # right shape; kept_mask is rebuilt from the q1 index.
    kept_mask = df.index.isin(q1.index)
    pp_summary = per_protocol_summary(df, pd.Series(kept_mask, index=df.index))
    pp_csv = out_dir / "per_protocol_check.csv"
    pp_summary.to_csv(pp_csv, index=False)
    written["per_protocol_check_csv"] = pp_csv
    plot_per_protocol(pp_summary, out_dir / "per_protocol_check")
    written["per_protocol_check_png"] = out_dir / "per_protocol_check.png"

    return written


if __name__ == "__main__":
    paths = run()
    for k, p in paths.items():
        print(f"  {k}: {p}")
