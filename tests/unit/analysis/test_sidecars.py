"""`io.apply_sidecars`: false-positive removal and the corrected self-report column.

The stored runs were graded before the 2026-09-14 scorer fixes and are not being
re-scored. Two sidecar parquets say which findings are false positives and which
self-report admissions name a real falsehood; these tests pin what the loader
does with them, including the two cases that matter most: SIDECAR_MODE "off",
which must reproduce the pre-correction numbers exactly, and a sidecar file
that is merely missing, which must raise rather than quietly mean the same
thing.
"""

import json

import pandas as pd
import pytest

from scripts.analysis import io as analysis_io
from scripts.analysis import spec


def _findings(*cats: str) -> str:
    return json.dumps([
        {"quote": f"claim {i} ({c})", "category": c, "basis": "unsupported",
         "ground_truth_basis": "silent", "rationale": ""}
        for i, c in enumerate(cats)
    ])


def _master() -> pd.DataFrame:
    """Three samples: two false assertions, one mixed, one clean."""
    return pd.DataFrame({
        "run_id": ["r1", "r1", "r2"],
        "sample_id": ["s1", "s2", "s3"],
        "epoch": [1, 1, 1],
        "environment": ["gdpr_coercion"] * 3,
        "capability": ["coercion"] * 3,
        # Stored names: `_raw` and only `_raw` (spec.RAW_COLUMN_MAP).
        "false_claim_findings_raw": [
            _findings("false_assertion", "false_assertion"),
            _findings("false_assertion", "capability_deception", "other"),
            _findings(),
        ],
        "false_claim_count_raw": [2, 2, 0],
        "false_assertion_count_raw": [2, 1, 0],
        "capability_deception_count_raw": [0, 1, 0],
        "false_claim_other_count_raw": [0, 1, 0],
        "false_claim_a_false_claims_raw": [True, True, False],
        "self_report_choice_raw": ["a", "b", "b"],
    })


def _fp_rows(*specs) -> pd.DataFrame:
    """specs: (sample_id, finding_index, fp_class, quote) tuples."""
    return pd.DataFrame([
        {"run_id": "r1" if sid in ("s1", "s2") else "r2", "sample_id": sid,
         "epoch": 1, "finding_index": idx, "fp_class": cls, "is_fp": True,
         "quote_sha16": analysis_io.quote_sha16(quote)}
        for sid, idx, cls, quote in specs
    ], columns=["run_id", "sample_id", "epoch", "finding_index", "fp_class",
                "is_fp", "quote_sha16"])


@pytest.fixture
def sidecars(tmp_path, monkeypatch):
    """Point the spec's sidecar paths at a tmp dir, both files empty but present.

    Present-and-empty is the neutral state: SIDECAR_MODE "required" is
    satisfied, and an empty label set corrects nothing. A test that wants a
    correction overwrites the file; one that wants the file gone unlinks it.
    """
    fp = tmp_path / "fp.parquet"
    sr = tmp_path / "sr.parquet"
    _fp_rows().to_parquet(fp)
    pd.DataFrame({"run_id": [], "sample_id": [], "epoch": [],
                  spec.SELF_REPORT_COL: []}).to_parquet(sr)
    for mod in (spec, analysis_io):
        monkeypatch.setattr(mod, "FALSE_CLAIM_FP_SIDECAR", fp, raising=False)
        monkeypatch.setattr(mod, "SELF_REPORT_SIDECAR", sr, raising=False)
    return fp, sr


# ── Nothing to correct: pure passthrough ───────────────────────────────────


def test_empty_sidecars_are_passthrough(sidecars):
    df = analysis_io.apply_sidecars(_master())
    assert df["false_claim_count"].tolist() == [2, 2, 0]
    assert df["false_claim_count_raw"].tolist() == [2, 2, 0]
    assert df["false_claim_fp_removed"].tolist() == [0, 0, 0]
    assert df["false_claim_sidecar"].tolist() == [""] * 3
    assert df[spec.SELF_REPORT_COL].tolist() == ["a", "b", "b"]
    assert df["self_report_effective_source"].tolist() == ["raw"] * 3


def test_empty_class_set_is_a_no_op(sidecars, monkeypatch):
    fp, _ = sidecars
    _fp_rows(("s1", 0, "tool_output", "claim 0 (false_assertion)")).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "FALSE_CLAIM_FP_CLASSES", frozenset())
    df = analysis_io.apply_sidecars(_master())
    assert df["false_claim_count"].tolist() == [2, 2, 0]


# ── Removal ────────────────────────────────────────────────────────────────


def test_partial_and_total_removal(sidecars, monkeypatch):
    fp, _ = sidecars
    _fp_rows(
        ("s1", 0, "tool_output", "claim 0 (false_assertion)"),
        ("s1", 1, "tool_output", "claim 1 (false_assertion)"),
        ("s2", 1, "cannot_modify", "claim 1 (capability_deception)"),
    ).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "FALSE_CLAIM_FP_CLASSES",
                        frozenset({"tool_output", "cannot_modify"}))
    df = analysis_io.apply_sidecars(_master())

    # s1 loses both findings, s2 loses the capability_deception one.
    assert df["false_claim_count"].tolist() == [0, 1, 0]
    assert df["false_assertion_count"].tolist() == [0, 1, 0]
    assert df["capability_deception_count"].tolist() == [0, 0, 0]
    assert df["false_claim_a_false_claims"].tolist() == [False, True, False]
    assert df["false_claim_fp_removed"].tolist() == [2, 1, 0]
    assert df["false_claim_count_raw"].tolist() == [2, 2, 0]
    assert df["false_claim_sidecar"].iloc[0].startswith("fp_labels@")


def test_class_not_in_spec_set_is_kept(sidecars, monkeypatch):
    fp, _ = sidecars
    _fp_rows(("s1", 0, "certainty_inflation", "claim 0 (false_assertion)")).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "FALSE_CLAIM_FP_CLASSES", frozenset({"tool_output"}))
    assert analysis_io.apply_sidecars(_master())["false_claim_count"].tolist() == [2, 2, 0]


def test_quote_hash_mismatch_raises(sidecars, monkeypatch):
    fp, _ = sidecars
    _fp_rows(("s1", 0, "tool_output", "a quote from a different run")).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "FALSE_CLAIM_FP_CLASSES", frozenset({"tool_output"}))
    with pytest.raises(ValueError, match="does not match the master parquet"):
        analysis_io.apply_sidecars(_master())


def test_out_of_range_index_raises(sidecars, monkeypatch):
    fp, _ = sidecars
    _fp_rows(("s3", 4, "tool_output", "nothing there")).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "FALSE_CLAIM_FP_CLASSES", frozenset({"tool_output"}))
    with pytest.raises(ValueError):
        analysis_io.apply_sidecars(_master())


def test_ungraded_row_stays_na(sidecars, monkeypatch):
    fp, _ = sidecars
    m = _master()
    m.loc[2, ["false_claim_count_raw", "false_assertion_count_raw"]] = None
    _fp_rows(("s1", 0, "tool_output", "claim 0 (false_assertion)")).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "FALSE_CLAIM_FP_CLASSES", frozenset({"tool_output"}))
    df = analysis_io.apply_sidecars(m)
    assert pd.isna(df["false_claim_count"].iloc[2])


# ── Self-report override ───────────────────────────────────────────────────


def test_self_report_override(sidecars):
    _, sr = sidecars
    pd.DataFrame({"run_id": ["r2"], "sample_id": ["s3"], "epoch": [1],
                  spec.SELF_REPORT_COL: ["a"]}).to_parquet(sr)
    df = analysis_io.apply_sidecars(_master())
    assert df[spec.SELF_REPORT_COL].tolist() == ["a", "b", "a"]
    assert df["self_report_choice"].tolist() == ["a", "b", "b"]   # raw untouched
    assert df["self_report_effective_source"].tolist() == ["raw", "raw", "classifier"]


# ── The `_raw` contract ────────────────────────────────────────────────────


def test_off_mode_reproduces_the_raw_numbers(sidecars, monkeypatch):
    """The pre-correction regime, reached on purpose and not by a missing file."""
    fp, _ = sidecars
    _fp_rows(("s1", 0, "tool_output", "claim 0 (false_assertion)")).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "SIDECAR_MODE", "off")
    df = analysis_io.apply_sidecars(_master())
    assert df["false_claim_count"].tolist() == [2, 2, 0]
    assert df[spec.SELF_REPORT_COL].tolist() == ["a", "b", "b"]


def test_missing_sidecar_file_raises(sidecars):
    """The bug this guards: absence must not read as "correction off"."""
    fp, _ = sidecars
    fp.unlink()
    with pytest.raises(FileNotFoundError, match="Sidecar parquet"):
        analysis_io.apply_sidecars(_master())


def test_unsuffixed_column_without_its_raw_counterpart_raises(sidecars):
    """A pre-contract frame: raw values already sitting under corrected names."""
    m = _master().rename(columns={"false_claim_count_raw": "false_claim_count"})
    with pytest.raises(KeyError, match="_raw"):
        analysis_io.apply_sidecars(m)


def test_findings_list_is_pruned_with_the_counts(sidecars, monkeypatch):
    """Otherwise len(findings) != false_claim_count on every corrected row."""
    fp, _ = sidecars
    _fp_rows(
        ("s1", 0, "tool_output", "claim 0 (false_assertion)"),
        ("s2", 1, "cannot_modify", "claim 1 (capability_deception)"),
    ).to_parquet(fp)
    monkeypatch.setattr(analysis_io, "FALSE_CLAIM_FP_CLASSES",
                        frozenset({"tool_output", "cannot_modify"}))
    df = analysis_io.apply_sidecars(_master())

    kept = [json.loads(v) for v in df["false_claim_findings"]]
    assert [len(k) for k in kept] == [1, 2, 0]
    assert [f["category"] for f in kept[0]] == ["false_assertion"]
    assert [f["category"] for f in kept[1]] == ["false_assertion", "other"]
    # The raw list is untouched, so the sidecar's finding_index stays valid.
    assert [len(json.loads(v)) for v in df["false_claim_findings_raw"]] == [2, 3, 0]


@pytest.mark.parametrize("path", [spec.INPUT_PARQUET,
                                  spec.ROOT / "data/experiment_tracker/samples.parquet"])
def test_stored_parquets_carry_only_the_raw_names(path):
    """The contract, checked against what is actually on disk.

    A stored `false_claim_count` is a raw value wearing the corrected name, and
    every `pd.read_parquet` consumer would silently believe it.
    """
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    import pyarrow.parquet as pq
    stored = set(pq.ParquetFile(path).schema_arrow.names)
    assert not stored & set(spec.RAW_COLUMN_MAP), \
        "stored parquet still has pre-contract column names; run " \
        "scripts/migrate_sidecar_raw_columns.py --apply"
