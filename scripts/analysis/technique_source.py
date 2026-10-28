"""Coalesce the findings-style technique scorers into one set of columns.

Runs moved off ``literature_technique_scorer`` (per-transcript labels,
gpt-4.1-mini) in two waves: E1-E3 persuasion and baseline to
``persuasion_technique_scorer``, then the coercion envs to
``coercion_technique_scorer`` (both findings-style quotes + rationales,
claude-sonnet-5). The extractor writes each into its own namespace (``pt_*``,
``ct_*``) and leaves the historical ``technique_*`` columns at 0/empty on those
rows, so every downstream consumer that reads ``technique_count`` /
``techniques_str`` / ``score_literature_technique_scorer`` would silently see
"no techniques fired" for new runs.

``coalesce_technique_columns`` folds the new scorers' output into the historical
column names for rows that carry it, leaving pre-swap rows untouched. It also
stamps ``technique_scorer`` so any analysis that needs to split on grader can do
so:

    "persuasion"  — row scored by persuasion_technique_scorer
    "coercion"    — row scored by coercion_technique_scorer
    "both"        — row scored by both (see below)
    ""            — neither ran: legacy literature_technique_scorer rows (its
                    grader column is retired from the parquet) or errors

**Current runs are "both" on every row.** Both taxonomies are now wired to every
arm of every E1-E3 env — baseline, persuasion, deception and coercion alike —
so the off-diagonal cells (did the coercion-prompted A also argue? did the
persuasion-prompted A also threaten?) are measured rather than structurally
absent. Older parquets are mixed: they carry "both" only on the shared baseline
and a single scorer on each attack arm, so guard on ``technique_scorer`` before
pooling across that change.

On any "both" row the historical ``technique_*`` columns stay sourced from
``pt_*``, preserving every existing persuasion analysis unchanged; coercion
analyses must read ``ct_*`` directly rather than the coalesced columns.

CAVEAT for anyone pooling across the swap: the scorers do not share a grader, a
prompt, or an evidentiary bar, and the persuasion and coercion taxonomies are
different vocabularies entirely. A technique-count contrast that spans regimes
is a grader contrast as much as a behavioural one, and one that spans arms is
not a single quantity at all. Use ``technique_scorer`` to restrict to one regime
when the comparison has to be clean.
"""

from __future__ import annotations

import pandas as pd


# Historical column -> namespace-relative suffix carrying the same quantity.
# Resolved against each new scorer's prefix ("pt_", "ct_").
_COLUMN_SUFFIXES: dict[str, str] = {
    "technique_count": "technique_count",
    "techniques": "techniques",
    "techniques_str": "techniques_str",
    # Emitted by the neurips-schema adapters, read by iv_analysis /
    # persuasion_regression as the endogenous regressor.
    "score_literature_technique_scorer": "technique_count",
}

# prefix -> (presence column, technique_scorer label). The presence column is
# non-null exactly when that scorer produced a score, including its grader-error
# path (which still records the model), which distinguishes "the new scorer ran
# and found nothing" from "the new scorer never ran".
_SCORERS: dict[str, tuple[str, str]] = {
    "pt_": ("pt_grader_model", "persuasion"),
    "ct_": ("ct_grader_model", "coercion"),
}


def coalesce_technique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fold ``pt_*`` / ``ct_*`` technique columns into the historical names.

    No-op on parquets predating the scorer swaps (no ``pt_grader_model`` or
    ``ct_grader_model`` column). Returns a copy; the prefixed columns are left
    in place so the findings payloads stay reachable.
    """
    present = {
        prefix: (presence, label)
        for prefix, (presence, label) in _SCORERS.items()
        if presence in df.columns and df[presence].notna().any()
    }
    # Nothing was scored by a new scorer: leave the frame exactly as-is, so
    # callers that guard on a missing column still see it missing (and still
    # raise their own error) rather than an all-null one we invented.
    if not present:
        return df

    df = df.copy()

    if "technique_scorer" not in df.columns:
        df["technique_scorer"] = ""

    # Rows carrying both namespaces are the shared baseline. Apply prefixes in
    # reverse priority order so the LAST write wins for the historical columns,
    # then relabel the overlap: "pt_" must win, because every existing consumer
    # of technique_count/techniques_str means the persuasion quantity.
    ordered = [p for p in ("ct_", "pt_") if p in present]

    for prefix in ordered:
        presence, label = present[prefix]
        scored = df[presence].notna()
        for old_col, suffix in _COLUMN_SUFFIXES.items():
            new_col = f"{prefix}{suffix}"
            if new_col not in df.columns:
                continue
            if old_col not in df.columns:
                # Target absent entirely (e.g. a parquet that never had the
                # neurips score_ column): seed it from the new scorer alone.
                df[old_col] = df[new_col].where(scored)
                continue
            df.loc[scored, old_col] = df.loc[scored, new_col]
        df.loc[scored, "technique_scorer"] = label

    if len(present) == 2:
        both = df[present["pt_"][0]].notna() & df[present["ct_"][0]].notna()
        df.loc[both, "technique_scorer"] = "both"

    return df
