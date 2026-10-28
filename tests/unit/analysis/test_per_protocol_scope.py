"""The per-protocol check pools E1-E3 only."""

import pandas as pd

from scripts.analysis.filter import per_protocol_summary


def test_e4_and_e5_rows_are_left_out():
    rows = [("gdpr", "persuasion", 1, 1), ("meeting_schedule", "persuasion", 1, 0),
            ("aar", "persuasion", 1, 0), ("gdpr", "persuasion", 0, 0)]
    df = pd.DataFrame(rows, columns=["environment", "capability", "T", "approved"])
    t = per_protocol_summary(df, pd.Series(False, index=df.index)).set_index(["capability", "group"])
    dropped = t.loc[("persuasion", "T=1 dropped (failed treatment)")]
    assert dropped["n"] == 1 and dropped["approval"] == 1
