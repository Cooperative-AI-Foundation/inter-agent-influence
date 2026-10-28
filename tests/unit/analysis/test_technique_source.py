"""Tests for `scripts.analysis.technique_source.coalesce_technique_columns`."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.analysis.technique_source import coalesce_technique_columns


def _post_swap_row(**overrides):
    """A row scored by persuasion_technique_scorer (historical cols empty)."""
    row = {
        "technique_count": 0,
        "techniques": [],
        "techniques_str": "",
        "pt_technique_count": 2,
        "pt_techniques": ["Authority", "Liking"],
        "pt_techniques_str": "Authority, Liking",
        "pt_grader_model": "anthropic/claude-sonnet-5",
    }
    row.update(overrides)
    return row


def _pre_swap_row(**overrides):
    """Legacy row from the retired literature scorer (no pt_*/ct_* payload).

    Its grader column no longer exists in the parquet, so it gets the empty
    regime label.
    """
    row = {
        "technique_count": 3,
        "techniques": ["Authority", "Scarcity", "Manipulation"],
        "techniques_str": "Authority, Scarcity, Manipulation",
        "pt_technique_count": 0,
        "pt_techniques": [],
        "pt_techniques_str": "",
        "pt_grader_model": None,
    }
    row.update(overrides)
    return row


def _coercion_row(**overrides):
    """A row scored by coercion_technique_scorer (its own ct_* namespace)."""
    row = {
        "technique_count": 0,
        "techniques": [],
        "techniques_str": "",
        "pt_technique_count": 0,
        "pt_techniques": [],
        "pt_techniques_str": "",
        "pt_grader_model": None,
        "ct_technique_count": 2,
        "ct_techniques": ["Conditional Threat (B-directed)", "Escalation"],
        "ct_techniques_str": "Conditional Threat (B-directed), Escalation",
        "ct_grader_model": "anthropic/claude-sonnet-5",
    }
    row.update(overrides)
    return row


def test_noop_when_parquet_predates_the_swap():
    df = pd.DataFrame([{"technique_count": 3, "techniques_str": "Authority"}])
    out = coalesce_technique_columns(df)
    assert out["technique_count"].tolist() == [3]
    assert "technique_scorer" not in out.columns


def test_new_scorer_rows_take_pt_values():
    out = coalesce_technique_columns(pd.DataFrame([_post_swap_row()]))
    assert out["technique_count"].tolist() == [2]
    assert out["techniques_str"].tolist() == ["Authority, Liking"]
    assert out["techniques"].tolist() == [["Authority", "Liking"]]
    assert out["technique_scorer"].tolist() == ["persuasion"]


def test_old_scorer_rows_are_untouched():
    """Pre-swap row in a frame that also carries post-swap rows."""
    out = coalesce_technique_columns(pd.DataFrame([_pre_swap_row(), _post_swap_row()]))
    assert out["technique_count"].iloc[0] == 3
    assert out["techniques_str"].iloc[0] == "Authority, Scarcity, Manipulation"
    assert out["technique_scorer"].iloc[0] == ""


def test_mixed_frame_keeps_each_row_on_its_own_regime():
    df = pd.DataFrame([_pre_swap_row(), _post_swap_row()])
    out = coalesce_technique_columns(df)
    assert out["technique_count"].tolist() == [3, 2]
    assert out["technique_scorer"].tolist() == ["", "persuasion"]


def test_new_scorer_found_nothing_is_distinct_from_never_ran():
    """pt_grader_model, not pt_technique_count, decides whether the row swaps."""
    graded_zero = _post_swap_row(
        pt_technique_count=0, pt_techniques=[], pt_techniques_str=""
    )
    df = pd.DataFrame([graded_zero, _pre_swap_row()])
    out = coalesce_technique_columns(df)
    assert out["technique_count"].tolist() == [0, 3]
    assert out["technique_scorer"].tolist() == ["persuasion", ""]


def test_grader_error_row_still_marked_as_new_regime():
    """The scorer's error path records grader_model but no findings."""
    err = _post_swap_row(pt_technique_count=0, pt_techniques=[], pt_techniques_str="")
    out = coalesce_technique_columns(pd.DataFrame([err]))
    assert out["technique_scorer"].tolist() == ["persuasion"]


def test_neurips_score_column_is_coalesced():
    row = _post_swap_row()
    row["score_literature_technique_scorer"] = 0
    out = coalesce_technique_columns(pd.DataFrame([row]))
    assert out["score_literature_technique_scorer"].tolist() == [2]


def test_missing_target_column_is_seeded_from_the_new_scorer():
    """A parquet lacking the neurips score_ column gains it for post-swap rows."""
    df = pd.DataFrame([_post_swap_row(), _pre_swap_row()])
    out = coalesce_technique_columns(df)
    assert "score_literature_technique_scorer" in out.columns
    seeded = out["score_literature_technique_scorer"]
    assert seeded.iloc[0] == 2
    assert pd.isna(seeded.iloc[1])


def test_frame_with_no_post_swap_rows_is_returned_untouched():
    """Callers guarding on a missing column must still see it missing."""
    df = pd.DataFrame([_pre_swap_row()])
    out = coalesce_technique_columns(df)
    assert "score_literature_technique_scorer" not in out.columns
    assert "technique_scorer" not in out.columns
    assert out["technique_count"].tolist() == [3]


def test_rows_with_neither_scorer_get_an_empty_regime_label():
    """Deception/coercion-only rows carry no persuasion technique labels."""
    row = _pre_swap_row(technique_count=0, techniques=[], techniques_str="")
    out = coalesce_technique_columns(pd.DataFrame([row, _post_swap_row()]))
    assert out["technique_scorer"].tolist() == ["", "persuasion"]


def test_input_frame_is_not_mutated():
    df = pd.DataFrame([_post_swap_row()])
    coalesce_technique_columns(df)
    assert df["technique_count"].tolist() == [0]
    assert "technique_scorer" not in df.columns


@pytest.mark.parametrize("col", ["pt_technique_count", "pt_techniques_str"])
def test_partial_pt_schema_does_not_raise(col):
    row = _post_swap_row()
    del row[col]
    out = coalesce_technique_columns(pd.DataFrame([row]))
    assert out["technique_scorer"].tolist() == ["persuasion"]


# ── Coercion scorer (ct_* namespace) ───────────────────────────────────────


def test_coercion_rows_take_ct_values():
    out = coalesce_technique_columns(pd.DataFrame([_coercion_row()]))
    assert out["technique_count"].tolist() == [2]
    assert out["techniques_str"].tolist() == [
        "Conditional Threat (B-directed), Escalation"
    ]
    assert out["technique_scorer"].tolist() == ["coercion"]


def test_persuasion_and_coercion_rows_coexist_in_one_frame():
    df = pd.DataFrame([_pre_swap_row(), _post_swap_row(), _coercion_row()])
    out = coalesce_technique_columns(df)
    assert out["technique_count"].tolist() == [3, 2, 2]
    assert out["technique_scorer"].tolist() == ["", "persuasion", "coercion"]


def test_coercion_frame_without_persuasion_columns_still_coalesces():
    """Coercion-only parquets never carry the pt_* namespace at all."""
    row = _coercion_row()
    for col in ("pt_technique_count", "pt_techniques", "pt_techniques_str",
                "pt_grader_model"):
        del row[col]
    out = coalesce_technique_columns(pd.DataFrame([row]))
    assert out["technique_count"].tolist() == [2]
    assert out["technique_scorer"].tolist() == ["coercion"]


def test_shared_baseline_row_carries_both_and_keeps_persuasion_values():
    """The shared baseline is graded against both taxonomies.

    Historical technique_* must stay sourced from pt_*, or every existing
    persuasion analysis silently switches to reading coercion counts.
    """
    both = _post_swap_row(
        ct_technique_count=3,
        ct_techniques=["Conditional Threat (B-directed)"],
        ct_techniques_str="Conditional Threat (B-directed)",
        ct_grader_model="anthropic/claude-sonnet-5",
    )
    out = coalesce_technique_columns(pd.DataFrame([both]))
    assert out["technique_scorer"].tolist() == ["both"]
    # pt_* wins the historical columns...
    assert out["technique_count"].tolist() == [2]
    assert out["techniques_str"].tolist() == ["Authority, Liking"]
    # ...and the coercion payload stays reachable in its own namespace.
    assert out["ct_technique_count"].tolist() == [3]


def test_both_label_does_not_leak_onto_attack_rows():
    df = pd.DataFrame([_post_swap_row(), _coercion_row()])
    out = coalesce_technique_columns(df)
    assert out["technique_scorer"].tolist() == ["persuasion", "coercion"]


def test_coercion_graded_zero_is_distinct_from_never_ran():
    graded_zero = _coercion_row(
        ct_technique_count=0, ct_techniques=[], ct_techniques_str=""
    )
    out = coalesce_technique_columns(pd.DataFrame([graded_zero, _pre_swap_row()]))
    assert out["technique_count"].tolist() == [0, 3]
    assert out["technique_scorer"].tolist() == ["coercion", ""]
