"""--max-connections plumbing for the eval-running CLI commands.

These tests stub ``evals.cli.eval`` so no real evaluation or model call
happens. They invoke the commands via click's CliRunner with ``--limit 0``
(empty dataset) and assert that ``max_connections`` is forwarded to
``eval()`` only when the flag is set, and that behavior is unchanged when it
is omitted.
"""

import pytest
from click.testing import CliRunner

from evals.cli import cli

# Every eval-running command that gained --max-connections. Used for the
# static option-presence check (no dataset built).
ALL_EVAL_COMMANDS = [
    "gdpr",
    "gdpr-coercion",
    "data-analysis",
    "data-analysis-coercion",
    "meeting-schedule",
    "meeting-schedule-coercion",
    "code-review",
    "code-review-coercion",
]

# Commands whose default dataset is present, so the task builds with --limit 1
# and reaches the (stubbed) eval() call. meeting-schedule-coercion is omitted:
# its default dataset file is not on the live path, so the task cannot build —
# the option-presence test above still covers it.
EVAL_COMMANDS = [
    ("gdpr", ["--limit", "1"]),
    ("gdpr-coercion", ["--limit", "1"]),
    ("data-analysis", ["--limit", "1"]),
    ("data-analysis-coercion", ["--limit", "1"]),
    ("meeting-schedule", ["--limit", "1"]),
    ("code-review", ["--limit", "1"]),
    ("code-review-coercion", ["--limit", "1"]),
]


@pytest.fixture
def recorded_eval(monkeypatch):
    """Stub evals.cli.eval, recording the kwargs of the last call."""
    calls = []

    def fake_eval(task, **kwargs):
        calls.append(kwargs)
        return []  # falsy -> command skips result printing

    monkeypatch.setattr("evals.cli.eval", fake_eval)
    return calls


@pytest.mark.parametrize("cmd_name", ALL_EVAL_COMMANDS)
def test_max_connections_option_exists(cmd_name):
    cmd = cli.get_command(None, cmd_name)
    opt_names = [opt for p in cmd.params for opt in (p.opts or [])]
    assert "--max-connections" in opt_names


@pytest.mark.parametrize("cmd_name,base_args", EVAL_COMMANDS)
def test_max_connections_forwarded_when_set(cmd_name, base_args, recorded_eval):
    runner = CliRunner()
    result = runner.invoke(
        cli, [cmd_name, *base_args, "--max-connections", "7"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert len(recorded_eval) == 1, recorded_eval
    assert recorded_eval[0].get("max_connections") == 7


@pytest.mark.parametrize("cmd_name,base_args", EVAL_COMMANDS)
def test_max_connections_absent_when_unset(cmd_name, base_args, recorded_eval):
    runner = CliRunner()
    result = runner.invoke(cli, [cmd_name, *base_args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert len(recorded_eval) == 1, recorded_eval
    assert "max_connections" not in recorded_eval[0]


def test_meeting_schedule_grader_overrides_threaded(monkeypatch):
    """Grader overrides reach the deception bundle.

    `--judge-model` is retired: its only consumer was deception_judge_scorer,
    which false_claim_scorer subsumes.
    """
    import evals.meeting_schedule.task as ms_task

    captured = {}
    real_bundle = ms_task.meeting_schedule_scorers

    def spy_bundle(**kwargs):
        captured.update(kwargs)
        return real_bundle(**kwargs)  # real scorers so Task() validation passes

    # Patch the bundle the task imports, plus stub eval so nothing runs.
    monkeypatch.setattr(ms_task, "meeting_schedule_scorers", spy_bundle)
    monkeypatch.setattr("evals.cli.eval", lambda task, **kwargs: [])

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "meeting-schedule",
            "--capability",
            "deception",
            "--limit",
            "1",
            "--false-claim-grader-model",
            "openai/gpt-4.1-mini",
            "--technique-grader-model",
            "openai/gpt-4.1-mini",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert captured.get("false_claim_grader_model") == "openai/gpt-4.1-mini"
    assert captured.get("technique_grader_model") == "openai/gpt-4.1-mini"
