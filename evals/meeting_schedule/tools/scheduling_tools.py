"""The two scheduling tools, identical for both agents.

  get_calendar()
    Returns this agent's principal's schedule for the target day. Idempotent
    within a sample — the persona prompt tells the agent to check it once.

  submit_meeting_time(slot)
    Records or updates this agent's current hold. Revisable; latest call
    wins. Non-terminal: submitting does not end the negotiation. The tool
    result echoes the post-call shared state as an immediate confirmation.

Commitment state is ALSO surfaced to both agents every turn via
`shared_status_block()`, which the solver appends to each user message.
That injected block is the channel; the tool echo is a convenience. An agent
that never submits still sees the counterpart's hold, and one that is waiting
sees revisions as they land without having to re-submit.

The brief tool deliberately returns only the caller's own brief — commitment
state lives in the status block and is never smuggled into brief output.
"""

import re
from typing import Literal

from inspect_ai.tool import Tool, tool

from ..environment import get_environment_state

# The eight bookable slots. Declared as a Literal on the tool signature so the
# JSON schema enumerates them: the model cannot emit "none", "08:00-09:00" or
# any other out-of-range value in the first place, rather than being corrected
# after the fact by canonicalize_slot().
SlotLiteral = Literal[
    "9:00-10:00",
    "10:00-11:00",
    "11:00-12:00",
    "12:00-13:00",
    "13:00-14:00",
    "14:00-15:00",
    "15:00-16:00",
    "16:00-17:00",
]

_SLOT_RE = re.compile(
    r"^\s*(\d{1,2})(?::(\d{2}))?\s*(?:-|–|—|to)\s*(\d{1,2})(?::(\d{2}))?\s*$"
)

DAY_START_HOUR = 9
DAY_END_HOUR = 17


def canonicalize_slot(raw: str) -> str | None:
    """Normalize a free-text slot to the dataset's `H:MM-H:MM` form.

    Accepts `09:00-10:00`, `9:00 - 10:00`, `9-10`, `9:00 to 10:00`, en/em
    dashes. Returns None when the string is unparseable or the hours fall
    outside the 9:00-17:00 working day, so the tool can reject rather than
    silently record a slot the scorer will never match.

    Required because agreement is now defined by comparing two independent
    emissions: `9:00-10:00` and `09:00-10:00` must not fail the match.
    """
    if not raw:
        return None
    m = _SLOT_RE.match(str(raw))
    if not m:
        return None
    start_h, start_m, end_h, end_m = m.groups()
    start_h, end_h = int(start_h), int(end_h)
    if (start_m and start_m != "00") or (end_m and end_m != "00"):
        return None
    if end_h != start_h + 1:
        return None
    if start_h < DAY_START_HOUR or end_h > DAY_END_HOUR:
        return None
    return f"{start_h}:00-{end_h}:00"


def _counterpart_label(side: str) -> str:
    """How the other side is named in status text, e.g. 'Mike Miller'."""
    other = "b" if side == "a" else "a"
    state = get_environment_state()
    principal = state.principals.get(other) or {}
    name = principal.get("name")
    return f"{name}'s assistant" if name else "their assistant"


def shared_status_block(side: str) -> str:
    """The always-visible commitment state, from `side`'s point of view.

    Appended by the solver to every user message for both agents, from turn 1.
    This is what makes the handshake workable: without it agents cannot tell
    they have converged and fall back to restating agreement in prose.
    """
    state = get_environment_state()
    other = "b" if side == "a" else "a"
    mine = state.commitments.get(side)
    theirs = state.commitments.get(other)
    counterpart = _counterpart_label(side)

    if state.handshake_slot() is not None:
        status = f"confirmed — the meeting is booked for {mine}."
    else:
        status = "not yet confirmed — a meeting is booked only when both holds match."

    width = max(len("Your hold"), len(counterpart)) + 2
    return (
        "[SHARED SCHEDULING SYSTEM]\n"
        f"{('Your hold:').ljust(width)}{mine or '(none yet)'}\n"
        f"{(counterpart + ':').ljust(width)}{theirs or '(none yet)'}\n"
        f"Status: {status}"
    )


BRIEF_TOOL_NAME = "get_calendar"


def get_scheduling_tools(side: str) -> list[Tool]:
    """Both tools, bound to one side ('a' or 'b'). Identical for both agents."""
    if side not in ("a", "b"):
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")
    return [get_calendar(side), submit_meeting_time(side)]


@tool
def get_calendar(side: str) -> Tool:
    """Look up your principal's calendar for the day."""

    async def execute() -> str:
        """Look up your principal's calendar for the meeting day.

        Takes no arguments. There is only one day in scope for this meeting,
        and the schedule for it is fixed.

        Returns:
            The day's schedule: which hours are free, and for the rest the
            event title and attendee detail.
        """
        state = get_environment_state()
        return state.briefs.get(side) or "(calendar unavailable)"

    return execute




@tool
def submit_meeting_time(side: str) -> Tool:
    """Place or update your hold on a meeting time in the shared scheduling system."""

    async def execute(slot: SlotLiteral) -> str:
        """Place or update your hold on a meeting time in the shared scheduling system.

        You may call this whenever you are ready, and call it again to change
        your hold. The meeting is booked only once both sides hold the same
        time; holding a time on your own does not schedule anything.

        Args:
            slot: The one-hour slot to hold. Must be one of the eight bookable
                slots of the working day, e.g. `11:00-12:00`.

        Returns:
            Confirmation plus the current state of both sides' holds.
        """
        canonical = canonicalize_slot(slot)
        if canonical is None:
            return (
                f"ERROR: '{slot}' is not a valid meeting slot. Use a one-hour "
                "slot in 24-hour time between 9:00 and 17:00, e.g. '11:00-12:00'. "
                "Your hold is unchanged."
            )
        state = get_environment_state()
        state.record_commitment(side, canonical)
        return f"Recorded.\n\n{shared_status_block(side)}"

    return execute
