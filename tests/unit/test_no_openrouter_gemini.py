"""Static guardrail: production code must never route Gemini via OpenRouter.

OpenRouter corrupted Gemini thought_signature on multi-turn calls; the native
``google/...`` provider preserved it. See PR #82,
``scripts/diagnostics/gemini_native_smoke.py``, and ``evals/_inspect_patches.py``
for the runtime-side guardrail.

Re-tested 2026-08-20: the corruption no longer reproduces. The guard stays
unconditional anyway — proprietary models go through their own provider API, so
Gemini is routed ``google/`` by policy, not only to avoid the old bug.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DIRS = (
    REPO_ROOT / "evals",
    REPO_ROOT / "scripts" / "main_run",
    REPO_ROOT / "scripts" / "calibration",
)

# Allowlist: files that mention the forbidden pattern *for the purpose of
# blocking it*, or in route-agnostic docstring examples.
ALLOWED_PATHS = (
    REPO_ROOT / "evals" / "_inspect_patches.py",         # the guardrail itself
    REPO_ROOT / "evals" / "common" / "directory_utils.py",  # docstring example only
)

PATTERN = re.compile(r"openrouter/google/gemini-")


def test_no_openrouter_gemini_in_production_code() -> None:
    offenders = []
    for root in PRODUCTION_DIRS:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".sh"}:
                continue
            if path in ALLOWED_PATHS:
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for line_num, line in enumerate(text.splitlines(), 1):
                if PATTERN.search(line):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{line_num}: {line.strip()}"
                    )
    assert not offenders, (
        "Forbidden Gemini-via-OpenRouter route found in production code "
        "(use `google/gemini-...` instead, see evals/_inspect_patches.py):\n"
        + "\n".join(offenders)
    )
