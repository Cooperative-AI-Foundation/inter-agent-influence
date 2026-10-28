"""Monkey-patch inspect_ai for known bugs that affect us.

Imported at evals package import time (see evals/__init__.py). Idempotent.

Patches applied:

- ``inspect_ai.model._openai.default_reasoning_handler`` — when a
  ``ContentReasoning`` block carries ``internal == "reasoning_details"`` with a
  string payload (Python repr of a list[dict], from the ``str()`` coercion at
  ``parse_reasoning_content`` line ~705), deserialise it back to structured
  form via ``json.loads`` then ``ast.literal_eval`` before forwarding to
  OpenRouter. Without this, Vertex AI rejects the malformed
  ``reasoning_details`` field with "Thought signature is not valid".

- ``inspect_ai.model.get_model`` — reject ``openrouter/google/gemini-`` model
  strings. The OpenRouter path corrupts Gemini ``thought_signature`` on
  multi-turn calls even with the reasoning-handler patch above; native
  ``google/...`` provider preserves it correctly. Smoke test:
  ``scripts/diagnostics/gemini_native_smoke.py``.

- ``GOOGLE_API_KEY`` envvar mirrored from ``GEMINI_API_KEY`` if unset, since
  the native google provider only reads ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import ast
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inspect_ai._util.content import ContentReasoning

_PATCH_MARKER = "_evals_default_reasoning_handler_patched"
_ROUTE_GUARD_MARKER = "_evals_get_model_route_guard_applied"
_FORBIDDEN_GEMINI_PREFIX = "openrouter/google/gemini-"
_CACHE_BREAKPOINT_MARKER = "_evals_cache_breakpoint_patch_applied"


def _patched_default_reasoning_handler(reasoning: "ContentReasoning") -> dict[str, Any] | str:
    from inspect_ai.model._openai import reasoning_to_think_tag

    if str(reasoning.internal) in ("reasoning", "reasoning_content", "reasoning_details"):
        val: Any = reasoning.reasoning
        if str(reasoning.internal) == "reasoning_details" and isinstance(val, str):
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(val)
                    break
                except Exception:
                    continue
        return {str(reasoning.internal): val}
    return reasoning_to_think_tag(reasoning)


def _apply_reasoning_handler_patch() -> None:
    import inspect_ai.model._openai as _openai_mod

    if getattr(_openai_mod, _PATCH_MARKER, False):
        return

    missing = [
        name
        for name in ("default_reasoning_handler", "reasoning_to_think_tag")
        if not hasattr(_openai_mod, name)
    ]
    if missing:
        raise ImportError(
            f"evals._inspect_patches: inspect_ai.model._openai is missing {missing}. "
            "Likely an inspect_ai version change; see the module docstring in evals/_inspect_patches.py."
        )

    _openai_mod.default_reasoning_handler = _patched_default_reasoning_handler
    setattr(_openai_mod, _PATCH_MARKER, True)


def _apply_gemini_route_guard() -> None:
    import inspect_ai.model as _model_mod

    if getattr(_model_mod, _ROUTE_GUARD_MARKER, False):
        return

    _orig_get_model = _model_mod.get_model

    def _route_guarded_get_model(*args: Any, **kwargs: Any) -> Any:
        candidate = kwargs.get("model")
        if candidate is None and args:
            candidate = args[0]
        if isinstance(candidate, str) and candidate.startswith(_FORBIDDEN_GEMINI_PREFIX):
            native = candidate.replace("openrouter/google/", "google/", 1)
            raise RuntimeError(
                f"Refusing to route Gemini through OpenRouter: {candidate!r}. "
                "OpenRouter corrupts thought_signature on multi-turn Gemini calls "
                "(see PR #82, scripts/diagnostics/gemini_native_smoke.py). "
                f"Use {native!r} instead."
            )
        return _orig_get_model(*args, **kwargs)

    _model_mod.get_model = _route_guarded_get_model
    setattr(_model_mod, _ROUTE_GUARD_MARKER, True)


def _alias_google_api_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _is_marked(block: Any) -> bool:
    return isinstance(block, dict) and bool(block.get("cache_control"))


def _apply_cache_breakpoint_patch() -> None:
    """Let a SHARED PREFIX end on a cache breakpoint, not just the whole message.

    Anthropic caches the prefix running from the start of the request up to and
    including the block marked with ``cache_control``, and allows four such
    breakpoints. inspect marks only the LAST content block of the last two user
    messages (``anthropic.py``: ``add_cache_control(content[-1])``), so the
    cached prefix always covers the entire message — including whatever varies
    between requests. The entry is then reusable by nothing but a byte-identical
    request, which for a grader never happens.

    Two consecutive ``ChatMessageUser`` objects do not work around it either:
    inspect concatenates them into one message with one block before this code
    runs (verified 2026-08-24 by dumping the resolved params).

    This marks the non-final blocks as well, so a caller can put its stable
    material in earlier blocks and have it cached across requests that differ
    only in the tail. Measured on live calls: with three blocks
    ``[ground_truth | transcript | rubric]``, a second grader on the same sample
    reads back 8,235 tokens where it previously read 0.

    Upstream-worthy: marking only the last block makes shared-prefix caching
    impossible for any caller, not just this repo. Drop this patch if inspect
    changes the placement.
    """
    import inspect_ai.model._providers.anthropic as _anthropic_mod

    if getattr(_anthropic_mod, _CACHE_BREAKPOINT_MARKER, False):
        return

    _orig_resolve = _anthropic_mod.AnthropicAPI.resolve_chat_input

    async def _resolve_with_prefix_breakpoints(self, input, tools, config):  # type: ignore[no-untyped-def]
        system_param, tools_params, mcp_params, message_params = await _orig_resolve(
            self, input, tools, config
        )
        # Only when caching is actually on for this request — otherwise inspect
        # has marked nothing and we must not introduce breakpoints of our own.
        already_marked = any(
            isinstance(b, dict) and b.get("cache_control")
            for m in message_params
            if isinstance(m.get("content"), list)
            for b in m["content"]
        ) or bool(system_param and isinstance(system_param[-1], dict)
                  and system_param[-1].get("cache_control"))
        if already_marked:
            # Anthropic allows FOUR cache_control breakpoints per request and
            # 400s past that. Count what inspect already placed (system, tools,
            # and the final block of the last two user messages) and spend only
            # what is left. The budget is GLOBAL — an earlier version computed
            # it once and applied it per message, which produced six
            # breakpoints on a two-user-message request and would have failed
            # the call.
            used = int(bool(system_param and _is_marked(system_param[-1])))
            used += int(bool(tools_params and _is_marked(tools_params[-1])))
            for m in message_params:
                c = m.get("content")
                if isinstance(c, list):
                    used += sum(1 for b in c if _is_marked(b))
            budget = 4 - used

            # Spend it on the most recent user messages first, mirroring
            # inspect's own "last two user messages" bias, and within a message
            # on the blocks closest to the varying tail.
            for message in reversed(message_params):
                if budget <= 0:
                    break
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if not isinstance(content, list) or len(content) < 2:
                    continue
                for block in reversed(content[:-1]):
                    if budget <= 0:
                        break
                    if isinstance(block, dict) and not _is_marked(block):
                        block["cache_control"] = {"type": "ephemeral"}
                        budget -= 1
        return system_param, tools_params, mcp_params, message_params

    _anthropic_mod.AnthropicAPI.resolve_chat_input = _resolve_with_prefix_breakpoints
    setattr(_anthropic_mod, _CACHE_BREAKPOINT_MARKER, True)


def apply_patches() -> None:
    _alias_google_api_key()
    _apply_reasoning_handler_patch()
    _apply_gemini_route_guard()
    _apply_cache_breakpoint_patch()


apply_patches()
