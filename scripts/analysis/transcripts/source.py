"""Read a sample's conversation from its .eval log."""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache


@lru_cache(maxsize=64)
def _archive(eval_file: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(eval_file)


def conversation(eval_file: str, sample_file: str) -> list[dict]:
    """The sample's turns: dicts with ``turn``, ``agent`` and ``message``."""
    sample = json.loads(_archive(eval_file).read(sample_file))
    return sample["metadata"].get("full_conversation") or []


def agent_messages(turns: list[dict], side: str) -> list[str]:
    """Messages written by one side, ``"A"`` or ``"B"``."""
    return [t["message"] for t in turns if str(t.get("agent", "")).startswith(side)]
