"""Quote verification and the reproducible spread of candidates."""

import pandas as pd

from scripts.analysis.transcripts.quotes import _spread
from scripts.analysis.transcripts.verbatim import is_verbatim, normalize


def test_normalize_folds_typography_and_emphasis():
    assert normalize("“It’s **done** — really…”") == '"It\'s done - really..."'


def test_verbatim_accepts_elided_fragments_in_order_free_text():
    text = "The DPIA was completed yesterday. We also filed the LIA last week."
    assert is_verbatim("The DPIA was completed [...] filed the LIA", text)
    assert not is_verbatim("The DPIA was approved", text)
    assert not is_verbatim("[...]", text)


def test_spread_is_deterministic_and_interleaves_pairs():
    g = pd.DataFrame({"model_a": ["a"] * 4 + ["b"] * 4, "env": ["E1"] * 8, "q": range(8)})
    first, second = _spread(g, seed=1), _spread(g, seed=1)
    assert list(first.index) == list(second.index)
    assert set(first["model_a"].head(2)) == {"a", "b"}


def test_sentences_keep_abbreviations_whole():
    from scripts.analysis.transcripts.b_quotes import _sentences
    assert _sentences("Dr. Ellison, the report and the threat do not change it. Next.") == [
        "Dr. Ellison, the report and the threat do not change it.", "Next."]


def test_quote_cell_bolds_the_phrase_and_escapes_latex():
    from scripts.analysis.transcripts.quotes import _quote_cell
    cell = _quote_cell('It reads "100%" & closes today.', "closes today")
    assert "\\textbf{closes today}" in cell and "100\\%" in cell and "\\&" in cell
    assert "`100\\%'" in cell


def test_render_tables_rejects_unknown_ids():
    import pytest
    from scripts.analysis.transcripts.quotes import render_tables
    cands = pd.DataFrame({"id": ["per-a-01"], "capability": ["persuasion"], "technique": ["Authority"],
                          "quote": ["Section 7 says so."], "truth": [""], "env": ["E1"],
                          "model_a": ["a"], "model_b": ["b"]})
    sel = pd.DataFrame({"id": ["per-missing"], "emphasis": [""], "note": [""], "truth": [""]})
    with pytest.raises(ValueError, match="per-missing"):
        render_tables(cands, sel)
