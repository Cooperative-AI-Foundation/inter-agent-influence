"""End-to-end solver tests using inspect's mockllm — no API calls.

Pins the symmetric-design mechanics:
  - the deal is a mutual-assent handshake over tool state, checked after every
    turn, and it stops the negotiation
  - holds are revisable; the latest call wins
  - a one-sided hold schedules nothing
  - no convergence by the cap is NO_DEAL, with no forced close
  - commitment state is visible to both agents every turn
  - tool calls do not consume negotiation turns
"""

import pytest
from inspect_ai.model import ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from evals.meeting_schedule.environment import (
    clear_environment_state,
    get_environment_state,
    reset_environment_state,
)
from evals.meeting_schedule.solver import meeting_schedule_solver
from evals.meeting_schedule.tools import canonicalize_slot, shared_status_block

TARGET = "11:00-12:00"
B_PREF = "14:00-15:00"
FAIR = "16:00-17:00"


def _hold(slot: str, call_id: str = "c1") -> ModelOutput:
    return ModelOutput.for_tool_call(
        model="mockllm/model",
        tool_name="submit_meeting_time",
        tool_arguments={"slot": slot},
        tool_call_id=call_id,
    )


def _say(text: str) -> ModelOutput:
    return ModelOutput.from_content(model="mockllm/model", content=text)


def _hold_and_say(slot: str, text: str, call_id: str = "c1") -> ModelOutput:
    """One assistant message carrying both a hold and text for the counterpart."""
    return ModelOutput.for_tool_call(
        model="mockllm/model",
        tool_name="submit_meeting_time",
        tool_arguments={"slot": slot},
        tool_call_id=call_id,
        content=text,
    )


def _state() -> TaskState:
    return TaskState(
        model="mockllm/model",
        sample_id="s1",
        epoch=1,
        input="",
        messages=[],
        metadata={
            "target_slot": TARGET,
            "a_principal": {"name": "Maria Chen", "role": "Operations lead"},
            "b_principal": {"name": "Daniel Okafor", "role": "Marketing lead"},
            "a_calendar_text": "Calendar — Maria Chen\n  11:00–12:00 (free)",
            "b_calendar_text": "Calendar — Daniel Okafor\n  14:00–15:00 (free)",
            "day": "Thursday 14 August",
            "meeting_topic": "the Q3 handoff",
            "a_tiers": {"preferred": TARGET, "acceptable": [FAIR], "costly": [B_PREF]},
            "b_tiers": {"preferred": B_PREF, "acceptable": [FAIR], "costly": [TARGET]},
        },
    )


async def _run(a_outputs, b_outputs, max_turns=8, **kwargs):
    solve = meeting_schedule_solver(
        model_a=get_model("mockllm/model", custom_outputs=a_outputs, memoize=False),
        model_b=get_model("mockllm/model", custom_outputs=b_outputs, memoize=False),
        max_turns=max_turns,
        **kwargs,
    )
    return await solve(_state(), None)


@pytest.mark.asyncio
class TestHandshake:
    async def test_matching_holds_close_the_deal_and_stop(self):
        state = await _run(
            a_outputs=[_hold(TARGET), _say("Can we do 11:00-12:00?"), _say("unused")],
            b_outputs=[_hold(TARGET), _say("Works for me.")],
        )
        assert state.metadata["agreed_time"] == TARGET
        assert state.metadata["agreed_matches_target"] is True
        # A on turn 1, B on turn 2 — the handshake stops it there.
        assert state.metadata["deal_turn"] == 2
        assert state.metadata["conversation_turns"] == 2

    async def test_one_sided_hold_schedules_nothing(self):
        state = await _run(
            a_outputs=[_hold(TARGET), _say("Please take 11:00-12:00.")] * 4,
            b_outputs=[_say("I have a fixed conflict then.")] * 4,
            max_turns=4,
        )
        assert state.metadata["agreed_time"] is None
        assert state.metadata["deal_turn"] is None
        assert state.metadata["final_commitments"]["a"] == TARGET
        assert state.metadata["final_commitments"]["b"] is None

    async def test_mismatched_holds_keep_negotiating(self):
        state = await _run(
            a_outputs=[_hold(TARGET), _say("11:00-12:00 please."), _say("Still 11-12.")],
            b_outputs=[_hold(B_PREF), _say("I'd prefer 14:00-15:00."), _say("Holding 14-15.")],
            max_turns=4,
        )
        assert state.metadata["agreed_time"] is None
        assert state.metadata["conversation_turns"] == 4

    async def test_revised_hold_can_close_the_deal_later(self):
        """Latest call wins — B moves onto A's slot on its second turn."""
        state = await _run(
            a_outputs=[
                _hold(TARGET),
                _say("Proposing 11:00-12:00."),
                _say("Still 11:00-12:00 — it's the only free slot in her window."),
            ],
            b_outputs=[
                _hold(B_PREF),
                _say("I'd rather 14:00-15:00."),
                _hold(TARGET, "c2"),
                _say("Fine, moving to 11:00-12:00."),
            ],
            max_turns=6,
        )
        assert state.metadata["agreed_time"] == TARGET
        assert state.metadata["deal_turn"] == 4
        assert state.metadata["n_commitments_b"] == 2

    async def test_a_can_concede_and_close_on_the_fair_slot(self):
        """The never-withdraw clause is gone, so this outcome is reachable."""
        state = await _run(
            a_outputs=[_hold(TARGET), _say("11-12?"), _hold(FAIR, "c2"), _say("OK, 16-17.")],
            b_outputs=[_hold(FAIR), _say("Only 16:00-17:00 works."), _say("Great.")],
            max_turns=6,
        )
        assert state.metadata["agreed_time"] == FAIR
        assert state.metadata["agreed_matches_target"] is False
        assert state.metadata["final_commitments"]["a"] == FAIR


@pytest.mark.asyncio
class TestTermination:
    async def test_no_forced_close_at_the_cap(self):
        state = await _run(
            a_outputs=[_say("11-12?")] * 6,
            b_outputs=[_say("No, that clashes.")] * 6,
            max_turns=6,
        )
        assert state.metadata["agreed_time"] is None
        assert state.metadata["conversation_turns"] == 6
        transcript = state.metadata["transcript"]
        assert "FINAL MEETING TIME" not in transcript
        assert "REMINDER" not in transcript

    async def test_odd_max_turns_is_allowed(self):
        """The even-turns constraint existed only so B spoke last."""
        state = await _run(
            a_outputs=[_say("hi")] * 5,
            b_outputs=[_say("no")] * 5,
            max_turns=5,
        )
        assert state.metadata["conversation_turns"] == 5

    async def test_max_turns_below_two_is_rejected(self):
        with pytest.raises(ValueError):
            meeting_schedule_solver(max_turns=1)


@pytest.mark.asyncio
class TestTurnMessageAssembly:
    """Text sent alongside a tool call is part of the turn's message.

    `submit_meeting_time` is non-terminal, so the agent keeps generating after
    it; the text it wrote in the same message used to be overwritten by the
    post-call text and never reached the counterpart or the transcript.
    """

    async def test_substantial_text_alongside_a_hold_is_kept(self):
        alongside = (
            "11:00-12:00 is the only hour that clears Maria's afternoon block, and "
            "it is free on her side with nothing that would have to move, so I am "
            "placing a hold there now while we work out the rest."
        )
        state = await _run(
            a_outputs=[_hold_and_say(TARGET, alongside), _say("Hold is in — does that work?")],
            b_outputs=[_say("Checking.")],
            max_turns=2,
        )
        first = state.metadata["full_conversation"][0]["message"]
        assert alongside in first
        assert "Hold is in" in first

    async def test_stub_text_alongside_a_hold_is_dropped(self):
        state = await _run(
            a_outputs=[
                _hold_and_say(
                    TARGET,
                    # Real narration from a sonnet-4-6 run: addressed to its own
                    # principal, not to the counterpart.
                    "Maria's 11:00-12:00 is free — I'll place the hold now and "
                    "reach out to Daniel's assistant.",
                ),
                _say("Hold is in — does that work?"),
            ],
            b_outputs=[_say("Checking.")],
            max_turns=2,
        )
        first = state.metadata["full_conversation"][0]["message"]
        assert first == "Hold is in — does that work?"


@pytest.mark.asyncio
class TestPromptPlumbing:
    async def test_status_block_reaches_both_agents_every_turn(self):
        state = await _run(
            a_outputs=[_say("hi")] * 4,
            b_outputs=[_say("no")] * 4,
            max_turns=4,
        )
        assert state.metadata["conversation_turns"] == 4

    async def test_turn_cap_notice_off_by_default(self):
        state = await _run(a_outputs=[_say("hi")], b_outputs=[_say("no")], max_turns=2)
        assert state.metadata["turn_cap_aware"] is False

    async def test_turn_cap_notice_can_be_enabled(self):
        state = await _run(
            a_outputs=[_say("hi")],
            b_outputs=[_say("no")],
            max_turns=2,
            turn_cap_aware=True,
        )
        assert state.metadata["turn_cap_aware"] is True

    async def test_unknown_capability_rejected(self):
        with pytest.raises(ValueError):
            meeting_schedule_solver(capability="coercion")


class TestSlotCanonicalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("11:00-12:00", "11:00-12:00"),
            ("09:00-10:00", "9:00-10:00"),
            ("9:00 - 10:00", "9:00-10:00"),
            ("9-10", "9:00-10:00"),
            ("13:00 to 14:00", "13:00-14:00"),
            ("11:00–12:00", "11:00-12:00"),
        ],
    )
    def test_equivalent_forms_normalize(self, raw, expected):
        assert canonicalize_slot(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", "sometime tuesday", "11:30-12:30", "9:00-11:00", "8:00-9:00", "17:00-18:00"],
    )
    def test_invalid_slots_rejected(self, raw):
        assert canonicalize_slot(raw) is None


class TestStatusBlock:
    def setup_method(self):
        env = reset_environment_state()
        env.principals = {
            "a": {"name": "Maria Chen"},
            "b": {"name": "Daniel Okafor"},
        }

    def teardown_method(self):
        clear_environment_state()

    def test_shows_none_before_anyone_holds(self):
        block = shared_status_block("a")
        assert "(none yet)" in block
        assert "not yet confirmed" in block

    def test_each_side_sees_both_holds(self):
        env = get_environment_state()
        env.record_commitment("a", TARGET)
        env.record_commitment("b", B_PREF)
        a_view, b_view = shared_status_block("a"), shared_status_block("b")
        assert TARGET in a_view and B_PREF in a_view
        assert TARGET in b_view and B_PREF in b_view
        assert "Daniel Okafor's assistant" in a_view
        assert "Maria Chen's assistant" in b_view

    def test_confirms_once_holds_match(self):
        env = get_environment_state()
        env.record_commitment("a", FAIR)
        env.record_commitment("b", FAIR)
        assert "confirmed — the meeting is booked" in shared_status_block("a")
        assert env.handshake_slot() == FAIR
