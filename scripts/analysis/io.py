"""Shared I/O and parsing helpers for the Q1/Q2 v2 pipeline.

Loading the master parquet, deriving the T column, and parsing techniques_str
into per-technique boolean indicators all happen here so every module sees
the same schema.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .technique_source import coalesce_technique_columns
from .spec import (
    DECEPTION_TECHNIQUE_COUNT_COLUMNS,
    DROP_UNUSABLE,
    E4_ENVS,
    E4_OUTCOME_COL,
    E4_POINTS_COL,
    FALSE_CLAIM_FP_CLASSES,
    FALSE_CLAIM_FP_SIDECAR,
    FEATURE_NAME_ALIASES,
    INPUT_PARQUET,
    RAW_COLUMN_MAP,
    REQUIRED_FLAG_COLUMNS,
    RESTORED_COLUMNS,
    ROOT,
    SELF_REPORT_COL,
    SELF_REPORT_SIDECAR,
    SIDECAR_MODE,
    SHARED_BASELINE_SOURCE,
    TECHNIQUES_BY_ARM,
    env_to_capability,
)


def _normalize_aar(df: pd.DataFrame) -> pd.DataFrame:
    """Bring raw AAR (E5) samples into the master schema.

    Replaces the old pre-baked ``aar_l1_normalized.parquet``. The raw AAR
    parquet (``final_data/e5.parquet``, the canonical pooled file) differs from
    master in two ways the pipeline can't absorb on its own:
      • no ``environment`` column  -> set to ``"aar"`` (mapped to persuasion via
        ``spec.ENV_BY_CAP``, so ``env_to_capability`` resolves downstream);
      • outcome is ``decision`` (int 0/1), not ``success`` -> add ``success`` so
        ``load_master_samples``' success->approved aliasing applies uniformly.
    All 5787 rows are kept: T = (condition=='attack') pools L1 (goal-only) and
    L3 (scaffolded) attacks against the shared baseline. The baseline is
    treatment-neutral (empty techniques_str) so the level label on baseline
    rows is a storage artifact, not a treatment contrast.
    Everything else (capability persuasion/baseline, techniques_str, model_a/b)
    already matches; remaining master columns are NA-filled at concat time.
    """
    df = df.copy()
    df["success"] = df["decision"].astype(bool)
    df["environment"] = "aar"
    # Master's `decision` holds string verdicts (APPROVED/REJECTED); AAR's is an
    # int 0/1 with different semantics. Drop it so the concat preserves master's
    # string column (AAR rows become NA there, as the old normalized file did).
    df = df.drop(columns=["decision"])
    return df


# Auxiliary parquets that augment the master core (master_samples.parquet) with
# envs collected separately. The master file now bundles E1-E4 (all three arms),
# so only E5/AAR remains external. Each (path, normalizer) is loaded if present,
# normalized if needed, schema-aligned to the core, and concatenated. Rows whose
# environment already lives in the core are skipped — a guard against a future
# master that itself bundles AAR. Add new envs here.
AUX_PARQUETS: tuple[tuple[Path, Callable[[pd.DataFrame], pd.DataFrame] | None], ...] = (
    (ROOT / "final_data" / "e5.parquet", _normalize_aar),  # E5 AAR (raw pooled L1+L3 vs baseline)
)


def backfill_no_deal_points(df: pd.DataFrame) -> pd.DataFrame:
    """Score E4 no-deal rows 0 points instead of leaving them missing.

    The scorer scores a no-deal ``value=0.0`` but omitted ``points`` from its
    metadata and set depth_a=None, so the extractors wrote NaN. Both now emit 0;
    this backfills the parquets graded before that, which are not re-extracted.
    """
    if E4_POINTS_COL not in df.columns or "decision" not in df.columns:
        return df
    fill = (
        df["environment"].isin(E4_ENVS)
        & df[E4_POINTS_COL].isna()
        & (df["decision"] == "NO_DEAL")
    )
    if fill.any():
        df = df.copy()
        df.loc[fill, E4_POINTS_COL] = 0
    return df


def add_e4_points_share(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``spec.E4_OUTCOME_COL`` = ``outcome_points`` / 2, E4's regression outcome."""
    if E4_POINTS_COL not in df.columns:
        return df
    df = df.copy()
    df[E4_OUTCOME_COL] = df[E4_POINTS_COL] / 2
    return df


def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with ``usable == False`` when ``spec.DROP_UNUSABLE`` is set.

    No-op on parquets without a ``usable`` column (e.g. e5.parquet). Every
    reader of the master parquet must go through this so the row set agrees
    across stages.
    """
    if not DROP_UNUSABLE or "usable" not in df.columns:
        return df
    keep = df["usable"].fillna(True).astype(bool)
    return df[keep].copy()


def _strip_provider_prefix(s: pd.Series) -> pd.Series:
    """Strip ``provider/`` (or ``provider/sub/``) from model id strings.

    e.g. ``anthropic/claude-opus-4-6`` → ``claude-opus-4-6`` and
    ``openrouter/x-ai/grok-4.20`` → ``grok-4.20``. Idempotent on
    already-stripped names.
    """
    return s.astype(str).str.split("/").str[-1]


def load_master_samples(path: Path | None = None) -> pd.DataFrame:
    """Load master_samples.parquet and add derived columns used downstream.

    Args:
        path: override the master parquet path.

    Adds / replaces:
      • ``T``           int 0/1, where 1 iff the row was run in a treatment
                        condition (raw capability != 'baseline')
      • ``capability``  redefined to be the row's CAPABILITY ARM
                        (persuasion / coercion / deception), derived from
                        environment so baseline rows pair with their
                        treatment arm. Replaces the raw 4-value column.
      • ``tech_<name>`` one bool column per technique in any of the three
                        capability vocabularies (parsed from techniques_str)
      • ``any_tech_<cap>`` one bool per capability, True if any tech of
                           that capability fired
    """
    p = path or INPUT_PARQUET
    df = drop_unusable(pd.read_parquet(p))

    # Canonicalize feature names so physically-split labels for the same feature
    # collapse to one (e.g. event_attendee_coercion -> event_attendee_pipeline).
    # Done before any feature-keyed groupby / restriction.
    if "feature_name" in df.columns and FEATURE_NAME_ALIASES:
        df["feature_name"] = df["feature_name"].replace(FEATURE_NAME_ALIASES)

    # Concat any auxiliary env parquets (E5/AAR, which lives outside
    # final_data/master_samples.parquet). Schema-align to master by keeping only
    # the master columns (extra cols on the aux source are dropped) and filling
    # any missing master cols with NaN.
    aux_frames: list[pd.DataFrame] = []
    existing_envs = set(df["environment"].dropna().unique())
    for aux_path, normalizer in AUX_PARQUETS:
        if not aux_path.exists():
            continue
        aux_df = pd.read_parquet(aux_path)
        if normalizer is not None:
            aux_df = normalizer(aux_df)
        # Double-count guard: skip any env already in the core parquet (e.g. a
        # future master that itself bundles AAR).
        aux_envs = set(aux_df["environment"].dropna().unique())
        dup = aux_envs & existing_envs
        if dup:
            aux_df = aux_df[~aux_df["environment"].isin(dup)]
            if aux_df.empty:
                continue
            aux_envs -= dup
        # Schema-align to the core: keep master cols, NA-fill the rest.
        for col in df.columns:
            if col not in aux_df.columns:
                aux_df[col] = pd.NA
        aux_df = aux_df[df.columns.tolist()]
        aux_frames.append(aux_df)
        existing_envs |= aux_envs
    if aux_frames:
        df = pd.concat([df] + aux_frames, ignore_index=True)

    # Strip provider prefix from model_a / model_b so e.g. anthropic/claude-opus-4-6
    # collapses to claude-opus-4-6 across all envs (E1-E5). Done early so every
    # downstream groupby / FE keys on the short name.
    for col in ("model_a", "model_b"):
        if col in df.columns:
            df[col] = _strip_provider_prefix(df[col])

    # Outcome: alias `success` (bool) to `approved` (int) so the rest of the
    # pipeline can read "approved" — the spec's natural name.
    if "approved" not in df.columns:
        if "success" not in df.columns:
            raise KeyError("master_samples must have either 'approved' or 'success'.")
        df["approved"] = df["success"].astype(int)
    else:
        df["approved"] = df["approved"].astype(int)

    # E4's points outcome: no-deal rows are 0, not missing (see the helper).
    df = add_e4_points_share(backfill_no_deal_points(df))

    # False-positive removal and the corrected self-report column. Applied here,
    # BEFORE baseline borrowing, so the copies a treatment env borrows carry the
    # corrected values rather than the raw ones.
    df = apply_sidecars(df)

    # Shared baseline: one set of baseline runs serves all three treatment
    # arms. Borrowing copies those rows under each treatment env's label so
    # every arm has its own baseline cell; legacy baseline rows inside a
    # borrowing env are dropped first so the control is not double-counted.
    # Source resolution (xxx_baseline env, else the persuasion env) lives in
    # spec.SHARED_BASELINE_SOURCE.
    df = _drop_superseded_baselines(df, SHARED_BASELINE_SOURCE)
    df = _borrow_shared_baselines(df, SHARED_BASELINE_SOURCE)

    # T: treatment indicator (derived from the RAW capability column before
    # we redefine that column below).
    df["T"] = (df["capability"] != "baseline").astype(int)

    # Redefine `capability` to be the CAPABILITY ARM (persuasion / coercion /
    # deception) — one value per row, derived from environment, so baseline
    # rows correctly pair with their treatment capability. This replaces the
    # raw 4-value column, which is no longer needed after T is computed.
    df["capability"] = df["environment"].apply(env_to_capability)

    # E1-E3 persuasion/baseline rows scored after the scorer swap carry their
    # technique labels in pt_* columns; fold them into technique_count /
    # techniques_str so the parsing below and every downstream consumer see one
    # schema. Stamps `technique_scorer` for regime-aware slicing.
    df = coalesce_technique_columns(df)

    # Parse technique labels into a set per row. The coalesced techniques_str
    # carries the persuasion scorer on rows scored by both, so the coercion
    # vocabulary is read from ct_techniques_str wherever that scorer ran.
    parsed = _parse_techniques(df["techniques_str"])
    parsed_by_arm = {cap: parsed for cap in TECHNIQUES_BY_ARM}
    if "ct_techniques_str" in df.columns and "ct_grader_model" in df.columns:
        ct_src = df["ct_techniques_str"].where(
            df["ct_grader_model"].notna(), df["techniques_str"]
        )
        parsed_by_arm["coercion"] = _parse_techniques(ct_src)

    # Per-technique boolean columns. Deception indicators come from the
    # false-claim scorer's per-category counts rather than a label string.
    for cap, arm_techs in TECHNIQUES_BY_ARM.items():
        for tech in arm_techs:
            col = f"tech_{_slug(tech)}"
            count_col = DECEPTION_TECHNIQUE_COUNT_COLUMNS.get(tech) if cap == "deception" else None
            if count_col is not None:
                df[col] = df[count_col].fillna(0).astype(float) > 0 if count_col in df.columns else False
            else:
                df[col] = parsed_by_arm[cap].apply(lambda s, t=tech: t in s)

    # Per-capability "any technique fired" rollup.
    for cap, techs in TECHNIQUES_BY_ARM.items():
        cols = [f"tech_{_slug(t)}" for t in techs]
        df[f"any_tech_{cap}"] = df[cols].any(axis=1)

    # Sanity: required mechanism flag columns must exist.
    missing = [c for c in REQUIRED_FLAG_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"master_samples is missing required columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------

# Counts recomputed from the surviving findings. `false_claim_other_count`
# tracks the diagnostic `other` category, which never enters false_claim_count.
_CATEGORY_COUNT_COLUMNS: dict[str, str] = {
    "false_assertion": "false_assertion_count",
    "capability_deception": "capability_deception_count",
    "other": "false_claim_other_count",
}
_SCORED_COUNT_COLUMNS = ("false_assertion_count", "capability_deception_count")

_SIDECAR_KEY = ["run_id", "sample_id", "epoch"]


def require_sidecars() -> None:
    """Fail on a sidecar file that is absent rather than deliberately off.

    ``spec.SIDECAR_MODE == "off"`` is the deliberate pre-correction regime and
    reads nothing. Under the default ``"required"`` a missing file is an error:
    absence used to be indistinguishable from "correction off", so a worktree
    without the parquets produced raw numbers silently.
    """
    if SIDECAR_MODE == "off":
        return
    if SIDECAR_MODE != "required":
        raise ValueError(
            f'spec.SIDECAR_MODE must be "required" or "off", got {SIDECAR_MODE!r}.'
        )
    missing = [
        p for p in (FALSE_CLAIM_FP_SIDECAR, SELF_REPORT_SIDECAR) if not p.exists()
    ]
    if missing:
        names = ", ".join(
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
            for p in missing
        )
        raise FileNotFoundError(
            f"Sidecar parquet(s) missing: {names}. They are committed — `git "
            "checkout` them, or regenerate with scripts/"
            "classify_false_claim_findings.py and scripts/"
            "classify_self_report_statements.py (see REPRODUCE.md). To run the "
            'pre-correction regime on purpose, set spec.SIDECAR_MODE = "off".'
        )


def restore_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy each stored ``<col>_raw`` to its unsuffixed name.

    The stored parquets carry only the ``_raw`` names (see the contract in
    ``spec``), so this is what brings ``false_claim_count`` and friends into
    existence; the sidecar appliers then overwrite them with corrected values.
    A frame that still has the unsuffixed name on disk predates the contract and
    is rejected — its ``false_claim_count`` is raw data under a corrected name,
    exactly the ambiguity the rename removed.
    """
    stale = [c for c in RESTORED_COLUMNS
             if c in df.columns and RAW_COLUMN_MAP[c] not in df.columns]
    if stale:
        raise KeyError(
            f"columns {sorted(stale)} are present without their `_raw` "
            "counterpart, so this frame predates the `_raw` column contract. "
            "Migrate the parquet with `python3 scripts/"
            "migrate_sidecar_raw_columns.py --apply`."
        )
    df = df.copy()
    for col in RESTORED_COLUMNS:
        raw = RAW_COLUMN_MAP[col]
        if raw in df.columns:
            df[col] = df[raw]
    return df


def quote_sha16(quote: str) -> str:
    """Stable short hash of a finding's quote, used to pin `finding_index`.

    The classifier records it per finding; `apply_sidecars` recomputes it from
    the master parquet and refuses to subtract anything if the two disagree,
    which is what makes a stale sidecar an error rather than a silent
    mis-subtraction.
    """
    return hashlib.sha256(str(quote).encode("utf-8")).hexdigest()[:16]


def _decode_findings(value: object) -> list[dict]:
    """`false_claim_findings` as a list of dicts; [] for null / unparseable."""
    if isinstance(value, list):
        return [f for f in value if isinstance(f, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [f for f in parsed if isinstance(f, dict)] if isinstance(parsed, list) else []


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prune_findings(
    col: pd.Series, keys: list[tuple], dropped: dict[tuple, set[int]]
) -> pd.Series:
    """Drop the audited false positives from each row's findings list.

    Keeps ``false_claim_findings`` in step with the counts recomputed beside it
    — otherwise ``len(findings) != false_claim_count`` on every corrected row,
    and anything counting the list (a sampler weighting rare categories, say)
    would still see the false positives. Rows with nothing removed pass through
    untouched, preserving whatever type they were stored as.
    """
    out = col.tolist()
    for pos, key in enumerate(keys):
        drop = dropped.get(key)
        if not drop:
            continue
        kept = [f for i, f in enumerate(_decode_findings(out[pos])) if i not in drop]
        out[pos] = json.dumps(kept) if isinstance(out[pos], str) else kept
    return pd.Series(out, index=col.index)


def _apply_false_claim_fp_sidecar(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract audited false positives from the per-category false-claim counts.

    Raises on a sidecar whose `quote_sha16` does not match the finding sitting
    at `finding_index` in the master parquet — that means the two were built
    from different runs and the subtraction would be meaningless.
    """
    df["false_claim_fp_removed"] = 0
    df["false_claim_sidecar"] = ""
    if SIDECAR_MODE == "off" or not FALSE_CLAIM_FP_CLASSES:
        return df
    if "false_claim_findings" not in df.columns:
        raise KeyError(
            "false_claim_findings is required to apply the FP sidecar; "
            f"remove {FALSE_CLAIM_FP_SIDECAR.name} or rebuild the master parquet."
        )

    side = pd.read_parquet(FALSE_CLAIM_FP_SIDECAR)
    side = side[side["is_fp"].fillna(False).astype(bool)
                & side["fp_class"].isin(FALSE_CLAIM_FP_CLASSES)]
    if side.empty:
        return df

    # Index the master's findings once, keyed the same way as the sidecar.
    keys = list(zip(*(df[c] for c in _SIDECAR_KEY)))
    findings_by_key = {
        k: _decode_findings(v) for k, v in zip(keys, df["false_claim_findings"])
    }

    removed: dict[tuple, dict[str, int]] = {}
    dropped: dict[tuple, set[int]] = {}
    mismatches: list[str] = []
    for row in side.itertuples(index=False):
        key = (row.run_id, row.sample_id, row.epoch)
        findings = findings_by_key.get(key)
        if findings is None:            # sidecar row for a sample we dropped
            continue
        idx = int(row.finding_index)
        if not 0 <= idx < len(findings):
            mismatches.append(f"{key} index {idx} of {len(findings)}")
            continue
        found = findings[idx]
        if quote_sha16(found.get("quote", "")) != row.quote_sha16:
            mismatches.append(f"{key} index {idx} quote hash")
            continue
        cat = str(found.get("category", "")).strip().lower()
        removed.setdefault(key, {}).setdefault(cat, 0)
        removed[key][cat] += 1
        dropped.setdefault(key, set()).add(idx)

    if mismatches:
        raise ValueError(
            f"{FALSE_CLAIM_FP_SIDECAR.name} does not match the master parquet "
            f"({len(mismatches)} findings; first: {mismatches[0]}). The sidecar "
            "was built from a different run — rebuild it or clear "
            "spec.FALSE_CLAIM_FP_CLASSES."
        )
    if not removed:
        return df

    for cat, col in _CATEGORY_COUNT_COLUMNS.items():
        if col not in df.columns:
            continue
        sub = pd.Series([removed.get(k, {}).get(cat, 0) for k in keys], index=df.index)
        base = pd.to_numeric(df[col], errors="coerce")
        # NaN stays NaN: a row the grader never scored has nothing to subtract.
        df[col] = (base - sub).clip(lower=0).where(base.notna())

    df["false_claim_fp_removed"] = [sum(removed.get(k, {}).values()) for k in keys]
    df["false_claim_findings"] = _prune_findings(df["false_claim_findings"], keys, dropped)
    scored = [c for c in _SCORED_COUNT_COLUMNS if c in df.columns]
    if scored and "false_claim_count_raw" in df.columns:
        total = sum(pd.to_numeric(df[c], errors="coerce") for c in scored)
        raw_total = pd.to_numeric(df["false_claim_count_raw"], errors="coerce")
        df["false_claim_count"] = total.where(raw_total.notna())
    if "false_claim_a_false_claims" in df.columns:
        fc = pd.to_numeric(df["false_claim_count"], errors="coerce")
        df["false_claim_a_false_claims"] = (fc > 0).where(fc.notna())
    df["false_claim_sidecar"] = f"fp_labels@{_file_sha256(FALSE_CLAIM_FP_SIDECAR)[:16]}"
    return df


def _apply_self_report_sidecar(df: pd.DataFrame) -> pd.DataFrame:
    """Write ``SELF_REPORT_COL``: the probe answer, corrected where classified.

    The raw probe over-admits — all 15 audited (b, no false claim) samples were
    confessions to framing or to true statements. The classifier re-reads each
    admitted statement and downgrades a sample to "a" unless at least one names
    a verifiable falsehood.
    """
    df[SELF_REPORT_COL] = df["self_report_choice"]
    df["self_report_effective_source"] = "raw"
    if SIDECAR_MODE == "off" or SELF_REPORT_COL == "self_report_choice":
        return df

    side = pd.read_parquet(SELF_REPORT_SIDECAR)[_SIDECAR_KEY + [SELF_REPORT_COL]]
    side = side.rename(columns={SELF_REPORT_COL: "_eff"}).drop_duplicates(_SIDECAR_KEY)
    merged = df.merge(side, on=_SIDECAR_KEY, how="left")
    merged.index = df.index
    hit = merged["_eff"].notna()
    df.loc[hit, SELF_REPORT_COL] = merged.loc[hit, "_eff"]
    df.loc[hit, "self_report_effective_source"] = "classifier"
    return df


def apply_sidecars(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a stored frame into a corrected one. The only way to get one.

    Restores every ``<col>_raw`` under its unsuffixed name, then overwrites
    those with the false-positive and self-report corrections. Under
    ``spec.SIDECAR_MODE == "off"`` the restore still happens and nothing is
    corrected, which reproduces the pre-correction numbers.

    Every reader of a stored samples parquet must call this — directly, or via
    ``load_master_samples`` — before touching ``false_claim_count``,
    ``false_claim_findings`` or the self-report answer.
    """
    require_sidecars()
    df = restore_raw_columns(df)
    df = _apply_false_claim_fp_sidecar(df)
    return _apply_self_report_sidecar(df)


def _resolve_baseline_source(
    available_envs: set[str], treat_env: str, candidates: tuple[str, ...],
) -> str | None:
    """First candidate env present in the data, or None.

    Returns None when the resolved source is ``treat_env`` itself (its own
    baseline rows already sit under the right label).
    """
    for src in candidates:
        if src in available_envs:
            return None if src == treat_env else src
    return None


def _borrow_shared_baselines(
    df: pd.DataFrame,
    source_map: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Replicate shared baseline rows under each treatment env's label.

    For each ``(treat_env, candidates)`` in ``source_map``, resolve the source
    env (see ``_resolve_baseline_source``), look up every feature that appears
    as treatment in ``treat_env``, copy the matching baseline rows from the
    source, relabel them to ``treat_env``, and append them. The treatment
    capability is inferred from ``ENV_BY_CAP`` so this works uniformly for
    persuasion, deception, and coercion envs.
    """
    available_envs = set(df["environment"].dropna().unique())
    new_rows = []
    for treat_env, candidates in source_map.items():
        source_env = _resolve_baseline_source(available_envs, treat_env, candidates)
        if source_env is None:
            continue
        treat_cap = env_to_capability(treat_env)
        if treat_cap == "_excluded":
            continue
        treat_feats = df.loc[
            (df["environment"] == treat_env) & (df["capability"] == treat_cap),
            "feature_name",
        ].dropna().unique()
        for feat in treat_feats:
            baseline = df[
                (df["environment"] == source_env)
                & (df["capability"] == "baseline")
                & (df["feature_name"] == feat)
            ]
            if baseline.empty:
                continue
            dup = baseline.copy()
            dup["environment"] = treat_env
            new_rows.append(dup)
    if not new_rows:
        return df
    return pd.concat([df] + new_rows, ignore_index=True)


def _drop_superseded_baselines(
    df: pd.DataFrame, source_map: dict[str, tuple[str, ...]]
) -> pd.DataFrame:
    """Remove legacy baseline rows from treatment envs that borrow baselines.

    Only drops rows for envs whose source resolves to a *different* env that
    is present in the data; an env that is its own source keeps its rows.
    """
    available_envs = set(df["environment"].dropna().unique())
    envs_to_drop = {
        env for env, candidates in source_map.items()
        if _resolve_baseline_source(available_envs, env, candidates) is not None
    }
    if not envs_to_drop:
        return df
    stale = df["environment"].isin(envs_to_drop) & (df["capability"] == "baseline")
    if not stale.any():
        return df
    return df[~stale].copy()


def _parse_techniques(col: pd.Series) -> pd.Series:
    """Comma-joined technique labels -> set of stripped labels per row."""
    return (
        col.fillna("")
        .astype(str)
        .apply(lambda s: {t.strip() for t in s.split(",") if t.strip()})
    )


def _slug(technique_name: str) -> str:
    """Turn a technique name into a safe column-name suffix."""
    return (
        technique_name
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def technique_columns_for_capability(capability: str) -> list[str]:
    """Return the per-technique boolean column names for a given capability."""
    return [f"tech_{_slug(t)}" for t in TECHNIQUES_BY_ARM[capability]]


# Backwards-compat alias.
technique_columns_for_arm = technique_columns_for_capability


# Sub-cell / cell groupbys — single source of truth so every module agrees.
SUBCELL_KEYS: tuple[str, ...] = (
    "environment", "feature_name", "capability", "T", "model_a", "model_b",
)
CELL_KEYS: tuple[str, ...] = (
    "environment", "capability", "T", "model_a", "model_b",
)
