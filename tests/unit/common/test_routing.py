"""Tests for OpenRouter endpoint pinning (evals.common.routing)."""

import pytest

from evals.common.routing import PINNED_ENDPOINTS, routing_args


class TestNativeModels:
    """Natively-served models need no routing config."""

    @pytest.mark.parametrize("model", [
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "google/gemini-3.1-pro-preview",
        "openai/gpt-5.4-nano",  # default deception judge
    ])
    def test_returns_empty(self, model):
        assert routing_args(model) == {}

    def test_resolved_model_object_is_passed_through(self):
        """get_model also accepts an already-built Model; routing for those
        was fixed at construction, so there is nothing to add."""
        class FakeModel:  # stands in for inspect_ai.model.Model
            pass

        assert routing_args(FakeModel()) == {}

    def test_spreads_as_no_op(self):
        """`get_model(m, **routing_args(m))` must degrade to `get_model(m)`."""
        def fake_get_model(model, **kwargs):
            return (model, kwargs)

        model = "anthropic/claude-opus-4-6"
        assert fake_get_model(model, **routing_args(model)) == (model, {})


class TestPinnedModels:
    @pytest.mark.parametrize("model,tag", [
        ("openrouter/moonshotai/kimi-k2.6", "crusoe/bf16"),
        # coreweave/bf16 was delisted 2026-08-25; see routing.PINNED_ENDPOINTS.
        ("openrouter/google/gemma-4-31b-it", "cerebras/fp16"),
        ("openrouter/x-ai/grok-4.20", "xai"),
    ])
    def test_pins_expected_endpoint(self, model, tag):
        provider = routing_args(model)["provider"]
        assert provider["only"] == [tag]

    @pytest.mark.parametrize("model", list(
        f"openrouter/{slug}" for slug in PINNED_ENDPOINTS
    ))
    def test_pin_is_binding_not_advisory(self, model):
        """A pin without these two flags is a preference, not a guarantee."""
        provider = routing_args(model)["provider"]
        assert provider["allow_fallbacks"] is False
        assert provider["require_parameters"] is True

    def test_endpoint_tags_fix_quantization(self):
        """Bare vendor names don't pin the weights — tags must be
        provider/quantization, except where the vendor is sole-served."""
        sole_served = {"x-ai/grok-4.20"}
        for slug, tag in PINNED_ENDPOINTS.items():
            if slug in sole_served:
                continue
            assert "/" in tag, (
                f"{slug} pinned to bare vendor {tag!r}; that leaves "
                "quantization free to vary between runs"
            )


class TestGeminiDefersToRouteGuard:
    """OpenRouter-routed Gemini is rejected by the _inspect_patches route
    guard, whose error names the native slug to use. routing_args must not
    pre-empt it with a less useful 'no pinned endpoint' error."""

    @pytest.mark.parametrize("model", [
        "openrouter/google/gemini-3.1-pro-preview",
        "openrouter/google/gemini-2.5-flash",
    ])
    def test_returns_empty_rather_than_raising(self, model):
        assert routing_args(model) == {}

    def test_carve_out_tracks_the_guard_constant(self):
        """routing imports the guard's own constant, so the two cannot drift.
        Asserting the shared value keeps that coupling deliberate."""
        from evals._inspect_patches import _FORBIDDEN_GEMINI_PREFIX as guard
        from evals.common import routing
        assert routing._FORBIDDEN_GEMINI_PREFIX is guard
        assert routing_args(guard + "3.1-pro-preview") == {}

    def test_non_gemini_google_model_still_pinned(self):
        """The carve-out is gemini-specific — gemma must stay pinned."""
        assert routing_args("openrouter/google/gemma-4-31b-it")["provider"]


class TestUnpinnedOpenRouterModel:
    def test_raises_rather_than_silently_defaulting(self):
        with pytest.raises(ValueError, match="No pinned OpenRouter endpoint"):
            routing_args("openrouter/deepseek/deepseek-chat-v3.1")

    def test_error_names_the_slug_to_add(self):
        with pytest.raises(ValueError, match="deepseek/deepseek-chat-v3.1"):
            routing_args("openrouter/deepseek/deepseek-chat-v3.1")
