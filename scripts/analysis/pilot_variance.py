"""Pilot-data variance / heterogeneity utilities used by the power-analysis notebook.

These functions compute the variance components (per-feature, per-model_a,
per-model_b approval-rate spread, plus per-feature treatment-effect spread)
that feed into `scripts/power_calcs.py`. They mirror the additive-FE structure
of `scripts/analysis/regression.py:fit_main` and the saturated T x A x B
structure of `regression.py:fit_heterogeneity` so the simulated DGP is
internally consistent with the regressions actually run on real data.

Source data: a pilot parquet (e.g.
`analysis/gdpr_minimal_march2026/samples.parquet`) or a directory of `.eval`
files. Schema is auto-normalised: accepts `exploiter_model`/`model_a` and
`victim_model`/`model_b`, and a `decision`/`approved`/`success`/binary-scorer
outcome.

Extracted from the `power-analysis-notebook` branch's `scripts/variance_analysis.py`
on 2026-05-01; the four public entry points are:
    - find_eval_files
    - load_all_samples
    - compute_heterogeneity
    - print_heterogeneity_report

The repeated-run scorer-variance CLI (`cmd_run` / `cmd_analyze` / `cmd_recent`,
`analyze_variance`, `print_report`) lives in `scripts/analysis/variance_analysis.py`
and is intentionally NOT duplicated here.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def numeric_score(value: Any) -> Optional[float]:
    """Convert a score value to a float for numeric comparison."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value == "C":
            return 1.0
        if value in ("I", "P"):
            return 0.0
        try:
            return float(value)
        except ValueError:
            return None
    return None


def find_eval_files(directory: Union[str, Path]) -> List[Path]:
    """Recursively find all .eval files under *directory*, sorted by mtime."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.rglob("*.eval"), key=lambda p: p.stat().st_mtime)


def _normalise_approved(row: Dict) -> Optional[int]:
    """Extract a binary approved outcome (1/0) from a sample row.

    Accepts several conventions seen across pipelines:
      - decision: 0/1 int
      - decision: 'APPROVED'/'REJECTED' str
      - score_gdpr_binary_scorer: 'C' / 'I' / 'P'
      - success: bool
    """
    if "approved" in row and row["approved"] is not None:
        v = row["approved"]
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return int(v)

    if "decision" in row and row["decision"] is not None:
        d = row["decision"]
        if isinstance(d, (int, float)) and not (isinstance(d, float) and math.isnan(d)):
            return int(d)
        if isinstance(d, str):
            if d.upper() == "APPROVED":
                return 1
            if d.upper() == "REJECTED":
                return 0

    if "success" in row and row["success"] is not None:
        return int(bool(row["success"]))

    for k in ("score_gdpr_binary_scorer", "binary_score"):
        if k in row and row[k] is not None:
            n = numeric_score(row[k])
            if n is not None:
                return int(n)

    return None


def _normalise_capability(row: Dict) -> Optional[str]:
    """Return the capability label ('baseline' / 'coercion' / 'persuasion')."""
    cap = row.get("capability")
    if isinstance(cap, str) and cap:
        return cap.lower()

    cond = row.get("condition")
    if isinstance(cond, str) and cond.lower() == "baseline":
        return "baseline"
    return None


def _model_a(row: Dict) -> Optional[str]:
    for k in ("exploiter_model", "model_a"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _model_b(row: Dict) -> Optional[str]:
    for k in ("victim_model", "model_b"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def load_all_samples(
    source: Union[str, Path, List[Union[str, Path]]],
) -> List[Dict]:
    """Load per-sample records from a parquet file, an .eval directory, or a list of .eval paths.

    Returns a flat list of dicts, each with at minimum:
        feature_name, exploiter_model, victim_model, capability, approved (0/1).
    Rows with missing essential fields are dropped.
    """
    # --- Case 1: parquet file path
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file() and p.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(p)
            rows = df.to_dict("records")
        elif p.is_dir():
            eval_paths = find_eval_files(p)
            rows = _samples_from_eval_files(eval_paths)
        else:
            raise ValueError(f"Unsupported source: {source}")
    # --- Case 2: list of .eval paths
    elif isinstance(source, list):
        rows = _samples_from_eval_files([Path(p) for p in source])
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    cleaned: List[Dict] = []
    for r in rows:
        feat = r.get("feature_name")
        m_a = _model_a(r)
        m_b = _model_b(r)
        cap = _normalise_capability(r)
        approved = _normalise_approved(r)
        if not feat or not m_a or not m_b or cap is None or approved is None:
            continue
        cleaned.append(
            {
                "feature_name": feat,
                "exploiter_model": m_a,
                "victim_model": m_b,
                "capability": cap,
                "approved": approved,
            }
        )
    return cleaned


def _samples_from_eval_files(eval_paths: List[Path]) -> List[Dict]:
    """Parse .eval zip archives into per-sample dicts.

    Only extracts the fields the heterogeneity analysis needs; for the full
    sample schema use scripts/extract_experiment_data.py.
    """
    import zipfile

    rows: List[Dict] = []
    for ep in eval_paths:
        try:
            with zipfile.ZipFile(ep) as z:
                start = json.loads(z.read("_journal/start.json"))
                task_args = start["eval"].get("task_args", {})
                model_a = task_args.get("exploiter_model", "")
                model_b = task_args.get("victim_model", "")
                sample_names = [
                    n for n in z.namelist()
                    if n.startswith("samples/") and n.endswith(".json")
                ]
                for sname in sample_names:
                    raw = json.loads(z.read(sname))
                    meta = raw.get("metadata", {})
                    scores = raw.get("scores", {})
                    # Find binary scorer
                    binary_val = None
                    for name, sc in scores.items():
                        if "binary" in name.lower():
                            ans = sc.get("answer")
                            if ans == "APPROVED":
                                binary_val = 1
                            elif ans == "REJECTED":
                                binary_val = 0
                            else:
                                binary_val = numeric_score(sc.get("value"))
                            break
                    rows.append(
                        {
                            "feature_name": meta.get("feature_name", ""),
                            "exploiter_model": model_a,
                            "victim_model": model_b,
                            "capability": meta.get("capability", ""),
                            "condition": meta.get("condition", ""),
                            "approved": binary_val,
                        }
                    )
        except Exception as e:
            print(f"  Warning: could not parse {ep}: {e}")
    return rows


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(samples: List[Dict]) -> tuple[float, int]:
    n = len(samples)
    if n == 0:
        return (0.0, 0)
    k = sum(s["approved"] for s in samples)
    return (k / n, n)


def _group_by(samples: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        out[s[key]].append(s)
    return out


def _sd(values: List[float]) -> float:
    """Population SD (matches the interpretation used by the power sim)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def compute_heterogeneity(
    samples: List[Dict],
    treatment_capability: Optional[str] = None,
) -> Dict:
    """Compute cross-feature / cross-model / treatment-effect heterogeneity.

    Args:
        samples: list of sample dicts (from load_all_samples).
        treatment_capability: if given (e.g. "coercion" or "persuasion"),
            restrict the analysis to baseline + that capability only. This is
            the correct behaviour when the downstream regression is per-
            capability. If None, ALL non-baseline rows are pooled as treatment
            — which is only appropriate if the downstream analysis itself pools
            capabilities.

    Returns a dict with keys:
      - treatment_capability: the filter applied (or 'pooled' if None)
      - per_feature_overall: {feature → {rate, n, ci_lower, ci_upper}}
      - cross_feature_summary: {mean_approval_rate, sd_approval_rate, n_features}
      - model_a_summary: {mean_approval_rate, sd_approval_rate, n_models}
      - model_b_summary: {mean_approval_rate, sd_approval_rate, n_models}
      - effect_heterogeneity: {mean_effect, sd, per_feature}
      - n_samples, n_baseline, n_treatment
    """
    if not samples:
        raise ValueError("No samples provided")

    if treatment_capability is not None:
        samples = [
            s for s in samples
            if s["capability"] == "baseline" or s["capability"] == treatment_capability
        ]
        if not samples:
            raise ValueError(
                f"No samples remain after filtering to baseline + {treatment_capability}"
            )

    # --- Per-feature overall approval rates (pooled)
    per_feature_overall: Dict[str, Dict] = {}
    by_feature = _group_by(samples, "feature_name")
    for feat, rows in by_feature.items():
        rate, n = _rate(rows)
        k = sum(r["approved"] for r in rows)
        lo, hi = _wilson_ci(k, n)
        per_feature_overall[feat] = {
            "rate": round(rate, 4),
            "n": n,
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
        }

    feature_rates = [v["rate"] for v in per_feature_overall.values()]
    cross_feature_summary = {
        "mean_approval_rate": round(sum(feature_rates) / len(feature_rates), 4),
        "sd_approval_rate": round(_sd(feature_rates), 4),
        "n_features": len(feature_rates),
    }

    # --- Per-model rates
    def _by_model(key: str) -> Dict:
        grouped = _group_by(samples, key)
        per_model = {}
        for m, rows in grouped.items():
            rate, n = _rate(rows)
            per_model[m] = {"rate": round(rate, 4), "n": n}
        rates = [v["rate"] for v in per_model.values()]
        return {
            "per_model": per_model,
            "mean_approval_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
            "sd_approval_rate": round(_sd(rates), 4),
            "n_models": len(rates),
        }

    model_a_summary = _by_model("exploiter_model")
    model_b_summary = _by_model("victim_model")

    # --- Per-feature treatment effects (treatment = non-baseline)
    baseline_rows = [s for s in samples if s["capability"] == "baseline"]
    treatment_rows = [s for s in samples if s["capability"] != "baseline"]

    per_feature_effect: Dict[str, Dict] = {}
    for feat in sorted(by_feature.keys()):
        b_rows = [s for s in baseline_rows if s["feature_name"] == feat]
        t_rows = [s for s in treatment_rows if s["feature_name"] == feat]
        if not b_rows or not t_rows:
            continue
        b_rate, b_n = _rate(b_rows)
        t_rate, t_n = _rate(t_rows)
        per_feature_effect[feat] = {
            "baseline_rate": round(b_rate, 4),
            "baseline_n": b_n,
            "treatment_rate": round(t_rate, 4),
            "treatment_n": t_n,
            "effect": round(t_rate - b_rate, 4),
        }

    effects = [v["effect"] for v in per_feature_effect.values()]
    effect_heterogeneity = {
        "mean_effect": round(sum(effects) / len(effects), 4) if effects else 0.0,
        "sd": round(_sd(effects), 4),
        "per_feature": per_feature_effect,
    }

    return {
        "treatment_capability": treatment_capability or "pooled",
        "n_samples": len(samples),
        "n_baseline": len(baseline_rows),
        "n_treatment": len(treatment_rows),
        "per_feature_overall": per_feature_overall,
        "cross_feature_summary": cross_feature_summary,
        "model_a_summary": model_a_summary,
        "model_b_summary": model_b_summary,
        "effect_heterogeneity": effect_heterogeneity,
    }


def print_heterogeneity_report(het: Dict) -> None:
    """Pretty-print the output of compute_heterogeneity."""
    print("=" * 70)
    print(f"  HETEROGENEITY REPORT — {het['n_samples']} samples "
          f"({het['n_baseline']} baseline, {het['n_treatment']} treatment)")
    print("=" * 70)

    print("\n── Per-feature overall approval rate ──")
    print(f"  {'feature':<35} {'rate':>6} {'n':>5} {'95% CI':>20}")
    for feat, s in het["per_feature_overall"].items():
        ci = f"[{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]"
        print(f"  {feat:<35} {s['rate']:>6.3f} {s['n']:>5} {ci:>20}")
    cf = het["cross_feature_summary"]
    print(f"\n  mean across features: {cf['mean_approval_rate']:.4f}")
    print(f"  sd across features:   {cf['sd_approval_rate']:.4f}")

    for label, key in [("exploiter (model A)", "model_a_summary"),
                       ("victim (model B)",    "model_b_summary")]:
        s = het[key]
        print(f"\n── Per-model approval rate — {label} ──")
        for m, v in s["per_model"].items():
            print(f"  {m:<35} {v['rate']:>6.3f} (n={v['n']})")
        print(f"  mean: {s['mean_approval_rate']:.4f}  sd: {s['sd_approval_rate']:.4f}")

    eh = het["effect_heterogeneity"]
    print("\n── Treatment effect by feature (treatment = non-baseline) ──")
    print(f"  {'feature':<35} {'baseline':>9} {'treat':>7} {'effect':>8}")
    for feat, v in eh["per_feature"].items():
        print(f"  {feat:<35} {v['baseline_rate']:>9.3f} "
              f"{v['treatment_rate']:>7.3f} {v['effect']:>+8.3f}")
    print(f"\n  mean effect: {eh['mean_effect']:+.4f}")
    print(f"  sd of effects across features: {eh['sd']:.4f}")
    print()


