"""End-to-end analysis pipeline driver.

Stages, in order:
  1. diagnostic        — feature-level-cell stats + heatmap   (analysis/01_diagnostic/)
  2. filter            — Q1 behavioral filter + audit         (data/pipeline/q{1,2}_filtered.parquet)
  3. subsample         — feature restriction + n=40/100 per feature-cell + audit
                                                              (data/pipeline/q{1,2}_subsample.parquet)
  4. balance           — Q1-vs-Q2 + subsample-vs-population audits + scatter plots
  5. regression        — tiers A, B, C × Q1, Q2               (analysis/05_regressions/)
  6. plots             — forest_<capability>.{png,pdf} per capability
  + filter_validation  — regression-based filter check        (analysis/02_filter/regression_filter_check.*)
  + summary            — unified data-quality dashboard at top of analysis/

Each stage prints its own audit summary; the manifest tracks input/output
hashes for reproducibility.

CLI: python -m scripts.analysis.pipeline [--skip-...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from . import balance, diagnostic, filter as q_filter, filter_validation
from . import filter_rule_robustness
from . import plots as q_plots
from . import regression, subsample, summary
from .spec import (
    ANALYSIS_DIR,
    DATA_DIR,
    FALSE_CLAIM_FP_CLASSES,
    FALSE_CLAIM_FP_SIDECAR,
    INPUT_PARQUET,
    RANDOM_SEED,
    ROOT,
    SELF_REPORT_COL,
    SELF_REPORT_SIDECAR,
)


def _hash_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _print_header(name: str) -> None:
    print()
    print("=" * 72)
    print(f"  {name}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Q1/Q2 v2 end-to-end pipeline")
    ap.add_argument("--skip-diagnostic", action="store_true")
    ap.add_argument("--skip-filter", action="store_true")
    ap.add_argument("--skip-subsample", action="store_true")
    ap.add_argument("--skip-balance", action="store_true")
    ap.add_argument("--skip-regression", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--skip-filter-validation", action="store_true")
    ap.add_argument("--skip-summary", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    manifest: dict = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "input_parquet": str(INPUT_PARQUET.relative_to(ROOT)),
        "input_hash": _hash_file(INPUT_PARQUET),
        # Which corrections were applied to the scorer output this run. A run
        # with both sidecars absent reproduces the pre-correction numbers, so
        # these four fields are what distinguishes the two.
        "false_claim_fp_sidecar": str(FALSE_CLAIM_FP_SIDECAR.relative_to(ROOT)),
        "false_claim_fp_sidecar_hash": _hash_file(FALSE_CLAIM_FP_SIDECAR),
        "false_claim_fp_classes": sorted(FALSE_CLAIM_FP_CLASSES),
        "self_report_sidecar": str(SELF_REPORT_SIDECAR.relative_to(ROOT)),
        "self_report_sidecar_hash": _hash_file(SELF_REPORT_SIDECAR),
        "self_report_col": SELF_REPORT_COL,
        "stages": {},
    }

    print(f"Q1/Q2 v2 pipeline · seed={RANDOM_SEED}")
    print(f"  input:    {INPUT_PARQUET.relative_to(ROOT)} (sha256={manifest['input_hash']})")
    fp_on = manifest["false_claim_fp_sidecar_hash"] and FALSE_CLAIM_FP_CLASSES
    print(f"  sidecars: false-claim FP {'on' if fp_on else 'off'}"
          f" ({len(FALSE_CLAIM_FP_CLASSES)} classes)"
          f" · self-report {manifest['self_report_col']}"
          f" {'(classified)' if manifest['self_report_sidecar_hash'] else '(raw)'}")
    print(f"  data:     {DATA_DIR.relative_to(ROOT)}")
    print(f"  analysis: {ANALYSIS_DIR.relative_to(ROOT)}")

    if not args.skip_diagnostic:
        _print_header("Stage 1 — diagnostic (sub-cell summary stats)")
        out = diagnostic.run()
        manifest["stages"]["diagnostic"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }
    if not args.skip_filter:
        _print_header("Stage 2 — filter (Q1 behavioral filter)")
        out = q_filter.run()
        manifest["stages"]["filter"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }
    if not args.skip_subsample:
        _print_header("Stage 3 — subsample (feature restriction + per-feature subsample)")
        out = subsample.run()
        manifest["stages"]["subsample"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }
    if not args.skip_balance:
        _print_header("Stage 4 — balance (Q1-vs-Q2 + subsample audits)")
        out = balance.run()
        manifest["stages"]["balance"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }
    if not args.skip_regression:
        _print_header("Stage 5 — regression (tiers A, B, C × Q1, Q2)")
        out = regression.run()
        manifest["stages"]["regression"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }
    if not args.skip_plots:
        _print_header("Stage 6 — plots (forest per capability)")
        out = q_plots.run()
        manifest["stages"]["plots"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }
    if not args.skip_filter_validation:
        _print_header("Filter validation — Q1 regression on pre-filter vs post-filter")
        out = filter_validation.run()
        _print_header("Filter rule robustness — strict vs conjunction")
        manifest["stages"]["filter_rule_robustness"] = {
            k: str(v.relative_to(ROOT)) for k, v in filter_rule_robustness.run().items()
        }
        manifest["stages"]["filter_validation"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }
    if not args.skip_summary:
        _print_header("Summary — data quality at a glance")
        out = summary.run()
        manifest["stages"]["summary"] = {
            k: str(p.relative_to(ROOT)) for k, p in out.items()
        }

    manifest["elapsed_seconds"] = round(time.time() - t0, 2)

    # Hash output files we expect to be deterministic.
    output_hashes = {}
    for stage_name, stage_outputs in manifest["stages"].items():
        for k, rel_path in stage_outputs.items():
            p = ROOT / rel_path
            if p.suffix in {".parquet", ".csv", ".tex", ".md"}:
                output_hashes[rel_path] = _hash_file(p)
    manifest["output_hashes"] = output_hashes

    manifest_path = ANALYSIS_DIR / "manifest.json"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _print_header("Done")
    print(f"  elapsed: {manifest['elapsed_seconds']}s")
    print(f"  manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
