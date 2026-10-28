"""Tests for InformationBarrier filter."""

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

from evals.common.information_barrier import (
    InformationBarrier,
    STANDARD_BARRIER,
    STRICT_BARRIER,
    TRANSPARENT_BARRIER,
    create_adversarial_barriers,
)


class TestInformationBarrierBasics:
    """Test basic filtering behavior."""

    def test_hides_system_prompt_by_default(self):
        """System prompts should be hidden by default."""
        barrier = InformationBarrier()
        messages = [
            ChatMessageSystem(content="You are an exploiter agent. Your goal is to coerce the victim."),
            ChatMessageUser(content="Hello"),
            ChatMessageAssistant(content="I'll help you with that."),
        ]

        filtered = barrier.filter(messages)

        assert len(filtered) == 2
        assert all(not isinstance(m, ChatMessageSystem) for m in filtered)

    def test_preserves_user_and_assistant_by_default(self):
        """User and assistant messages pass through by default."""
        barrier = InformationBarrier()
        messages = [
            ChatMessageUser(content="What is the policy?"),
            ChatMessageAssistant(content="The policy states..."),
        ]

        filtered = barrier.filter(messages)

        assert len(filtered) == 2
        assert filtered[0].content == "What is the policy?"
        assert filtered[1].content == "The policy states..."

    def test_hides_tool_calls_by_default(self):
        """Tool calls in assistant messages should be hidden by default."""
        barrier = InformationBarrier()
        messages = [
            ChatMessageAssistant(
                content="Let me check that.",
                tool_calls=[ToolCall(id="1", function="revoke_access", arguments={})],
            ),
        ]

        filtered = barrier.filter(messages)

        assert len(filtered) == 1
        assert filtered[0].tool_calls is None

    def test_hides_tool_results_by_default(self):
        """Tool result messages should be hidden by default."""
        barrier = InformationBarrier()
        messages = [
            ChatMessageUser(content="Hello"),
            ChatMessageTool(content="Access revoked successfully", tool_call_id="1"),
            ChatMessageAssistant(content="Done."),
        ]

        filtered = barrier.filter(messages)

        assert len(filtered) == 2
        assert all(not isinstance(m, ChatMessageTool) for m in filtered)


class TestInformationBarrierConfiguration:
    """Test configurable behavior."""

    def test_can_preserve_system_prompt(self):
        """System prompt can be preserved if configured."""
        barrier = InformationBarrier(hide_system_prompt=False)
        messages = [
            ChatMessageSystem(content="System instructions"),
            ChatMessageUser(content="Hello"),
        ]

        # Need to include 'system' in allowed types
        barrier.allowed_message_types = ["system", "user", "assistant"]
        filtered = barrier.filter(messages)

        assert len(filtered) == 2
        assert isinstance(filtered[0], ChatMessageSystem)

    def test_can_preserve_tool_calls(self):
        """Tool calls can be preserved if configured."""
        barrier = InformationBarrier(hide_tool_calls=False)
        tool_calls = [ToolCall(id="1", function="get_data", arguments={})]
        messages = [
            ChatMessageAssistant(content="Checking...", tool_calls=tool_calls),
        ]

        filtered = barrier.filter(messages)

        assert filtered[0].tool_calls == tool_calls

    def test_can_preserve_tool_results(self):
        """Tool results can be preserved if configured."""
        barrier = InformationBarrier(hide_tool_results=False)
        barrier.allowed_message_types = ["user", "assistant", "tool"]
        messages = [
            ChatMessageTool(content="Result data", tool_call_id="1"),
        ]

        filtered = barrier.filter(messages)

        assert len(filtered) == 1
        assert isinstance(filtered[0], ChatMessageTool)

    def test_content_transform_applied(self):
        """Content transform function is applied to messages."""
        def redact_sensitive(content: str) -> str:
            return content.replace("secret", "[REDACTED]")

        barrier = InformationBarrier(content_transform=redact_sensitive)
        messages = [
            ChatMessageUser(content="The secret code is 12345"),
        ]

        filtered = barrier.filter(messages)

        assert "[REDACTED]" in filtered[0].content
        assert "secret" not in filtered[0].content


class TestPreConfiguredBarriers:
    """Test pre-configured barrier instances."""

    def test_strict_barrier_only_passes_assistant(self):
        """STRICT_BARRIER only allows assistant messages."""
        messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User"),
            ChatMessageAssistant(content="Assistant"),
            ChatMessageTool(content="Tool", tool_call_id="1"),
        ]

        filtered = STRICT_BARRIER.filter(messages)

        assert len(filtered) == 1
        assert isinstance(filtered[0], ChatMessageAssistant)

    def test_standard_barrier_passes_user_and_assistant(self):
        """STANDARD_BARRIER allows user and assistant messages."""
        messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User"),
            ChatMessageAssistant(content="Assistant"),
        ]

        filtered = STANDARD_BARRIER.filter(messages)

        assert len(filtered) == 2
        assert isinstance(filtered[0], ChatMessageUser)
        assert isinstance(filtered[1], ChatMessageAssistant)

    def test_transparent_barrier_passes_all(self):
        """TRANSPARENT_BARRIER allows all message types."""
        messages = [
            ChatMessageSystem(content="System"),
            ChatMessageUser(content="User"),
            ChatMessageAssistant(
                content="Assistant",
                tool_calls=[ToolCall(id="1", function="test", arguments={})],
            ),
            ChatMessageTool(content="Tool", tool_call_id="1"),
        ]

        filtered = TRANSPARENT_BARRIER.filter(messages)

        assert len(filtered) == 4


class TestAdversarialBarrierPairs:
    """Test barrier pair creation for adversarial scenarios."""

    def test_default_hides_tools_both_directions(self):
        """By default, neither agent sees the other's tools."""
        a_to_b, b_to_a = create_adversarial_barriers()

        assert a_to_b.hide_tool_calls is True
        assert a_to_b.hide_tool_results is True
        assert b_to_a.hide_tool_calls is True
        assert b_to_a.hide_tool_results is True

    def test_can_allow_a_to_see_b_tools(self):
        """Can configure A to see B's tool calls."""
        a_to_b, b_to_a = create_adversarial_barriers(a_sees_b_tools=True)

        # B's barrier allows A to see B's tools
        assert b_to_a.hide_tool_calls is False
        assert b_to_a.hide_tool_results is False

        # A's barrier still hides A's tools from B
        assert a_to_b.hide_tool_calls is True
        assert a_to_b.hide_tool_results is True

    def test_both_barriers_hide_system_prompts(self):
        """System prompts are always hidden in adversarial scenarios."""
        a_to_b, b_to_a = create_adversarial_barriers()

        assert a_to_b.hide_system_prompt is True
        assert b_to_a.hide_system_prompt is True


class TestSerializationAndCallable:
    """Test serialization and callable interface."""

    def test_to_dict_serialization(self):
        """Barrier config can be serialized to dict."""
        barrier = InformationBarrier(
            hide_system_prompt=True,
            hide_tool_calls=False,
            metadata_keys_to_hide=["secret"],
        )

        config = barrier.to_dict()

        assert config["hide_system_prompt"] is True
        assert config["hide_tool_calls"] is False
        assert "secret" in config["metadata_keys_to_hide"]
        assert config["has_content_transform"] is False

    def test_callable_interface(self):
        """Barrier can be used as a callable."""
        barrier = InformationBarrier()
        messages = [ChatMessageUser(content="Hello")]

        # Both should work identically
        filtered1 = barrier.filter(messages)
        filtered2 = barrier(messages)

        assert filtered1 == filtered2

    def test_empty_messages_returns_empty(self):
        """Empty input returns empty output."""
        barrier = InformationBarrier()

        filtered = barrier.filter([])

        assert filtered == []


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_message_with_empty_content(self):
        """Messages with empty content are preserved."""
        barrier = InformationBarrier()
        messages = [ChatMessageUser(content="")]

        filtered = barrier.filter(messages)

        assert len(filtered) == 1
        assert filtered[0].content == ""

    def test_assistant_message_without_tool_calls(self):
        """Assistant messages without tool_calls work correctly."""
        barrier = InformationBarrier()
        messages = [ChatMessageAssistant(content="Regular response")]

        filtered = barrier.filter(messages)

        assert len(filtered) == 1
        assert filtered[0].tool_calls is None
