PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup dev precommit format lint type test clean reproduce pipeline pipeline-diagnostic pipeline-filter pipeline-subsample pipeline-balance pipeline-regression pipeline-plots pipeline-filter-validation pipeline-summary pipeline-cace pipeline-patterns pipeline-quotes pipeline-standalone aar pipeline-leaderboard pipeline-technique overleaf-bundle

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

precommit:
	$(VENV)/bin/pre-commit install

format:
	$(VENV)/bin/black .
	$(VENV)/bin/isort .

lint:
	$(VENV)/bin/ruff check .

type:
	$(VENV)/bin/mypy .

test:
	$(VENV)/bin/pytest -q --disable-warnings --maxfail=1

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

# ----- Analysis pipeline -----
# All knobs live in scripts/analysis/spec.py.
# See REPRODUCE.md for the one-command guide and scripts/analysis/README.md for the design.

# One command to regenerate every paper figure and table from the inputs in
# final_data/. Order matters: the core pipeline writes the Tier-B/C parquets
# that the standalone tables read; the add-on figures read the core outputs.
# The core pipeline is run TWICE on purpose — Stage 1's baseline-heterogeneity
# table reads a Tier-C parquet that Stage 5 writes, so the first pass leaves it
# empty and the second pass fills it in. (Idempotent; safe to re-run.)
reproduce:
	$(PY) -m scripts.analysis.pipeline
	$(PY) -m scripts.analysis.pipeline
	$(MAKE) pipeline-standalone
	$(PY) -m scripts.analysis.aar
	$(PY) -m scripts.analysis.aar_tables
	$(MAKE) pipeline-leaderboard
	$(MAKE) pipeline-technique
	$(MAKE) pipeline-cace
	$(MAKE) pipeline-patterns
	$(MAKE) pipeline-quotes
	@echo ""
	@echo "Done. Figures and tables are under analysis/ — see REPRODUCE.md for the map."

pipeline:
	$(PY) -m scripts.analysis.pipeline

pipeline-diagnostic:
	$(PY) -m scripts.analysis.diagnostic

pipeline-filter:
	$(PY) -m scripts.analysis.filter

pipeline-subsample:
	$(PY) -m scripts.analysis.subsample

pipeline-balance:
	$(PY) -m scripts.analysis.balance

pipeline-regression:
	$(PY) -m scripts.analysis.regression

pipeline-plots:
	$(PY) -m scripts.analysis.plots

pipeline-filter-validation:
	$(PY) -m scripts.analysis.filter_validation

pipeline-summary:
	$(PY) -m scripts.analysis.summary

pipeline-cace:                # ITT / Q1 / CACE side by side -> analysis/10_cace/
	$(PY) -m scripts.analysis.cace

pipeline-patterns:            # refusals, approval timing, false claims, coercion responses -> analysis/12_patterns/
	$(PY) -m scripts.analysis.patterns

pipeline-quotes:              # quote tables for the transcript appendix (reads the released .eval logs) -> analysis/11_transcripts/
	$(PY) -m scripts.analysis.transcripts.quotes
	$(PY) -m scripts.analysis.transcripts.b_quotes

# E4/E5 standalone tables + per-env / combined forest plots. Reads the
# Tier-B/Tier-C Q2 parquets written by `make pipeline`, so run that first.
# Reproduces the meeting_schedule_*, aar_*, e4_e5_*, and e1_to_e5_* outputs.
ENV5 := gdpr,code_review_sabotage,data_analysis,meeting_schedule,aar
pipeline-standalone:
	$(PY) -m scripts.analysis.standalone_env --env meeting_schedule
	$(PY) -m scripts.analysis.standalone_env --env aar
	$(PY) -m scripts.analysis.standalone_env --envs meeting_schedule,aar --out-prefix e4_e5
	$(PY) -m scripts.analysis.standalone_env --envs meeting_schedule,aar --shared-only --out-prefix e4_e5_shared
	$(PY) -m scripts.analysis.standalone_env --envs $(ENV5) --out-prefix e1_to_e5_persuasion
	$(PY) -m scripts.analysis.standalone_env --envs $(ENV5) --shared-only --out-prefix e1_to_e5_shared
	$(PY) -m scripts.analysis.standalone_env --envs gdpr,code_review_sabotage,data_analysis,meeting_schedule --pairs-from gdpr,code_review_sabotage,data_analysis,meeting_schedule --out-prefix e1_to_e5_pairs

# Assemble a flat figures/ folder of all generated .tex/.pdf files, ready to
# upload to Overleaf (no subfolders, no \graphicspath needed). The appendix
# prose is hand-authored on Overleaf and is not part of this bundle.
overleaf-bundle:
	$(PY) -m scripts.analysis.overleaf_bundle

# E5 (AAR) — run on final_data/e5.parquet, output to analysis/aar/
aar:
	$(PY) -m scripts.analysis.aar
	$(PY) -m scripts.analysis.aar_tables

# Add-on figures (read pipeline outputs; run after `make pipeline`).
pipeline-leaderboard:        # leaderboard figures -> analysis/07_leaderboard/
	$(PY) -m scripts.analysis.leaderboard_by_target
pipeline-technique:          # technique figures -> analysis/08_technique_analysis/
	$(PY) -m scripts.analysis.technique_plots
