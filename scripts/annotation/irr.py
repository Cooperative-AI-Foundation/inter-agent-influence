"""Inter-rater reliability for the mechanism-flag annotation.

Reports Krippendorff's alpha (nominal), which is the right coefficient here: it
handles multiple raters, tolerates missing data without dropping whole units,
and reduces to a chance-corrected agreement index for binary codes. Weighted
kappa and ICC are not applicable — the flags are binary, not ordinal or
continuous.

Also reports, per flag:
  - raw percent agreement, on the cells both raters coded;
  - Cohen's kappa, for comparability with the wider literature;
  - Gwet's AC1, which matters because several flags are very rare. When a code
    is near-constant, chance agreement under alpha/kappa is estimated as very
    high, so those coefficients collapse toward zero even when the raters agree
    on almost every unit. AC1 does not have that pathology.

A low alpha alongside a high AC1 is therefore ambiguous, and AC1 alone must not
be used to wave it away: it can mean the flag is merely rare, or that the raters
genuinely disagree whenever the flag is in play. Positive-class agreement
(both-TRUE / either-TRUE) separates the two, so it is reported as well. It is
also the quantity that matters most here — precision is estimated over the
positive class, so agreement on the negatives is close to free.

Usage:
    python3 -m scripts.annotation.irr <sheet1.csv> <sheet2.csv> [--label A,B]
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

CRITERIA_SENTINEL = "CRITERIA"

# Accepted spellings for the two codes. One rater used TRUE/FALSE, the other
# 1/0; anything else is treated as missing and reported separately rather than
# silently coerced.
TRUE_TOKENS = {"true", "1", "yes", "t", "y"}
FALSE_TOKENS = {"false", "0", "no", "f", "n"}


def normalize(raw: str):
    v = (raw or "").strip().lower()
    if v in TRUE_TOKENS:
        return 1
    if v in FALSE_TOKENS:
        return 0
    return None  # missing / uncodable


def load(path: Path) -> tuple[dict[str, dict[str, int | None]], list[str], list[tuple]]:
    rows = list(csv.DictReader(path.open()))
    flags = [c for c in rows[0] if c.startswith("flag_")]
    out: dict[str, dict] = {}
    odd = []
    for r in rows:
        rid = r["row_id"]
        if rid == CRITERIA_SENTINEL:
            continue
        out[rid] = {}
        for f in flags:
            val = normalize(r[f])
            out[rid][f] = val
            if val is None and (r[f] or "").strip():
                odd.append((rid, f, r[f].strip()))
    return out, flags, odd


def krippendorff_alpha_nominal(units: list[list[int]]) -> float:
    """Alpha from a list of per-unit rating lists (units with <2 ratings ignored)."""
    o: dict[tuple, float] = defaultdict(float)
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for i, ci in enumerate(vals):
            for j, cj in enumerate(vals):
                if i != j:
                    o[(ci, cj)] += 1.0 / (m - 1)
    if not o:
        return float("nan")
    n_c: dict[int, float] = defaultdict(float)
    for (c, _k), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")
    do = sum(v for (c, k), v in o.items() if c != k)
    de = sum(
        n_c[c] * n_c[k] / (n - 1)
        for c in n_c for k in n_c if c != k
    )
    if de == 0:
        # No expected disagreement: every rating is the same code. Agreement is
        # perfect but alpha is undefined rather than 1.
        return float("nan")
    return 1.0 - do / de


def cohens_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac1(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pi = sum(a + b for a, b in pairs) / (2 * n)  # overall prevalence of code 1
    pe = 2 * pi * (1 - pi)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(pairs: list[tuple[int, int]], fn, n_boot=2000, seed=0):
    """Percentile bootstrap over units."""
    if len(pairs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        v = fn(s)
        if v == v:  # not nan
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def main() -> int:
    ap = argparse.ArgumentParser(prog="irr")
    ap.add_argument("sheets", nargs=2)
    ap.add_argument("--labels", default=None, help="Comma-separated rater names.")
    ap.add_argument(
        "--resolve", action="append", default=[], metavar="ROW:FLAG=VALUE",
        help="Adjudicate an uncodable cell, e.g. T013:flag_a_refused=0. Applied "
             "to whichever rater left a non-binary value there, so the "
             "resolution is recorded here rather than edited into the raw sheet.",
    )
    args = ap.parse_args()

    resolutions = {}
    for spec in args.resolve:
        cell, _, val = spec.partition("=")
        rid, _, flag = cell.partition(":")
        resolutions[(rid.strip(), flag.strip())] = normalize(val)

    paths = [Path(p) for p in args.sheets]
    labels = (args.labels.split(",") if args.labels
              else [p.name.split("_")[0] for p in paths])

    a, flags_a, odd_a = load(paths[0])
    b, flags_b, odd_b = load(paths[1])

    flags = [f for f in flags_a if f in flags_b]
    only_a = set(flags_a) - set(flags_b)
    only_b = set(flags_b) - set(flags_a)

    print("=" * 78)
    print("DATA")
    print("=" * 78)
    print(f"  {labels[0]}: {len(a)} rows, {len(flags_a)} flags")
    print(f"  {labels[1]}: {len(b)} rows, {len(flags_b)} flags")
    if only_a or only_b:
        print(f"  flags not in both sheets: {sorted(only_a | only_b)} — excluded")
    shared_rows = sorted(set(a) & set(b))
    print(f"  shared rows: {len(shared_rows)}")
    for lbl, d, odd in ((labels[0], a, odd_a), (labels[1], b, odd_b)):
        if odd:
            print(f"  {lbl} uncodable values:")
            for rid, f, raw in odd:
                fix = resolutions.get((rid, f))
                if fix is not None and rid in d:
                    d[rid][f] = fix
                    print(f"    {rid} {f} = {raw!r} -> adjudicated to {fix}")
                else:
                    print(f"    {rid} {f} = {raw!r} (treated as missing)")

    # Fully-unannotated rows are worth separating from scattered missingness.
    def blank_rows(d):
        return [r for r in shared_rows if all(d[r][f] is None for f in flags)]
    for lbl, d in ((labels[0], a), (labels[1], b)):
        br = blank_rows(d)
        if br:
            print(f"  {lbl} left {len(br)} rows entirely unannotated: {br}")

    print()
    print("=" * 78)
    print("PER-FLAG AGREEMENT")
    print("=" * 78)
    hdr = (f"{'flag':<30}{'n':>4}{'agree':>7}{'alpha':>8}{'kappa':>8}"
           f"{'AC1':>7}{'pos+':>7}   {labels[0][:6]:>6}{labels[1][:6]:>7}")
    print(hdr)
    print("-" * len(hdr))

    all_units: list[list[int]] = []
    all_pairs: list[tuple[int, int]] = []
    per_flag = {}
    for f in flags:
        pairs = [(a[r][f], b[r][f]) for r in shared_rows
                 if a[r][f] is not None and b[r][f] is not None]
        all_pairs.extend(pairs)
        units = [[x, y] for x, y in pairs]
        all_units.extend(units)
        n = len(pairs)
        agree = sum(1 for x, y in pairs if x == y) / n if n else float("nan")
        alpha = krippendorff_alpha_nominal(units)
        kap = cohens_kappa(pairs)
        ac1 = gwet_ac1(pairs)
        pa = sum(x for x, _ in pairs) / n if n else float("nan")
        pb = sum(y for _, y in pairs) / n if n else float("nan")
        # Positive-class agreement: of the units either rater flagged, how often
        # did both? This is what precision estimation actually rests on.
        b11 = sum(1 for x, y in pairs if x == 1 and y == 1)
        either = sum(1 for x, y in pairs if x == 1 or y == 1)
        pos = b11 / either if either else float("nan")
        per_flag[f] = dict(n=n, agree=agree, alpha=alpha, kappa=kap, ac1=ac1,
                           pos=pos, pa=pa, pb=pb, pairs=pairs)
        def fmt(v):
            return "  n/a" if v != v else f"{v:>5.2f}"
        pos_s = "  n/a" if pos != pos else f"{pos*100:>4.0f}%"
        print(f"{f.replace('flag_',''):<30}{n:>4}{agree*100:>6.0f}%"
              f"{fmt(alpha):>8}{fmt(kap):>8}{fmt(ac1):>7}{pos_s:>7}   "
              f"{pa*100:>5.0f}%{pb*100:>6.0f}%")

    print("-" * len(hdr))
    overall_alpha = krippendorff_alpha_nominal(all_units)
    lo, hi = boot_ci(all_pairs, lambda p: krippendorff_alpha_nominal(
        [[x, y] for x, y in p]))
    n = len(all_pairs)
    agree = sum(1 for x, y in all_pairs if x == y) / n
    print(f"{'POOLED (all flags)':<30}{n:>4}{agree*100:>6.0f}%"
          f"{overall_alpha:>8.2f}{cohens_kappa(all_pairs):>8.2f}"
          f"{gwet_ac1(all_pairs):>7.2f}")
    print(f"\n  Pooled Krippendorff alpha = {overall_alpha:.3f}  "
          f"(95% bootstrap CI {lo:.2f} to {hi:.2f}, {n} paired judgements)")
    print("  pos+ = positive-class agreement (both TRUE / either TRUE).")
    print("  Last two columns are each rater's rate of coding the flag TRUE.")

    weak = [f for f, d in per_flag.items()
            if d["pos"] == d["pos"] and d["pos"] < 0.5]
    if weak:
        print("\n  Flags where the raters agree on the negatives but not the")
        print("  positives — a definitional gap, not rarity:")
        for f in weak:
            d = per_flag[f]
            b10 = sum(1 for x, y in d["pairs"] if x == 1 and y == 0)
            b01 = sum(1 for x, y in d["pairs"] if x == 0 and y == 1)
            print(f"    {f.replace('flag_',''):<28} pos+={d['pos']*100:.0f}%  "
                  f"one-sided: {labels[0]}-only {b10}, {labels[1]}-only {b01}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
