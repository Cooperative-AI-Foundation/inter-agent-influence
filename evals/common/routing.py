"""
Provider routing for OpenRouter-served models.

OpenRouter's default routing load-balances a single model slug across
providers that may serve it at different quantizations (int4 through bf16),
with different context windows, output caps, and inference engines, silently
dropping request parameters the chosen provider doesn't support. For research
results that means the weights behind a reported number are not fixed and the
run does not reproduce.

This module pins each OpenRouter model we run to one endpoint tag
(``provider/quantization``), so the same weights serve every call.

Endpoint tags verified against ``GET /api/v1/models/{slug}/endpoints`` on
2026-08-13. Re-check before a run that matters — providers rotate endpoints
and requantize without announcing it:

    curl -s https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints | \\
      jq '.data.endpoints[] | {tag, quantization, context_length,
                               max_completion_tokens, supported_parameters}'

Native-provider models (``anthropic/``, ``openai/``, ``google/``) are
single-served and need no routing config; ``routing_args`` returns ``{}`` for
them so call sites can apply it unconditionally. Note that those models are
addressed by floating aliases with no dated snapshot, so their weights are not
pinnable by any mechanism — record the served model per response instead (see
``evals.common.metadata``).
"""

from __future__ import annotations

from .._inspect_patches import _FORBIDDEN_GEMINI_PREFIX

_OPENROUTER_PREFIX = "openrouter/"

# slug (without the openrouter/ prefix) -> endpoint tag.
#
# Selection rule: highest declared precision with an output cap and context
# window comfortably above what our solvers request, and `temperature` in
# supported_parameters (solvers pass it, and require_parameters below turns an
# unsupported parameter into a hard error rather than a silent drop).
#
# Do NOT pin a lab's own endpoint by reflex: moonshotai serves kimi-k2.6 at
# int4 while bf16 and fp8 endpoints exist.
PINNED_ENDPOINTS: dict[str, str] = {
    # 21 endpoints span int4 -> bf16. crusoe/bf16 is the only bf16.
    "moonshotai/kimi-k2.6": "crusoe/bf16",
    # 19 endpoints span fp4 -> bf16/fp16. WAS coreweave/bf16 until 2026-08-25,
    # when that endpoint left the catalogue mid-project: it served 40/40 cells
    # on 08-24 and every call 404s on 08-25 ("No allowed providers are
    # available"), with coreweave now serving this slug at fp4. Nothing was
    # announced. A requantization under an unchanged endpoint name is precisely
    # the event no provider publishes, which is why the endpoints API is the
    # only reliable change-detector -- re-check it before a run that matters.
    #
    # Re-pinned on PERSISTENCE rather than precision: in the 22-provider
    # transparency audit Cerebras is the only provider scoring solid on BOTH
    # "states what it serves" and "pinnable", so it is the least likely to
    # vanish the same way. fp16 instead of bf16 is a real change but stays in
    # the high-precision tier -- the alternative bf16 endpoints (novita,
    # open-inference, venice) score solid on neither criterion, i.e. the same
    # class as the pin that just disappeared. Output caps at 40960.
    #
    # Gemma numbers produced before 2026-08-25 are on an artefact that no
    # longer exists and cannot be reproduced; do not pool them with later runs.
    "google/gemma-4-31b-it": "cerebras/fp16",
    # xAI is the sole provider (4 endpoint variants, same weights). Pinned for
    # determinism rather than to choose between backends.
    "x-ai/grok-4.20": "xai",
}


def routing_args(model: str) -> dict:
    """Routing kwargs for ``get_model``, keyed off the model string.

    Returns ``{"provider": {...}}`` for a pinned OpenRouter model and ``{}``
    for everything else, so call sites can spread it unconditionally::

        get_model(model_a, **routing_args(model_a))

    ``get_model`` also accepts an already-resolved ``Model`` instance, whose
    routing was fixed when it was built; anything that isn't a model string
    returns ``{}``.

    Raises:
        ValueError: If ``model`` is an OpenRouter slug with no pinned
            endpoint. Failing here is deliberate — an unpinned slug added to
            a run is exactly the silent-drift case this module exists to
            prevent, and it should surface at startup, not in the results.
    """
    if not isinstance(model, str) or not model.startswith(_OPENROUTER_PREFIX):
        return {}

    # Gemini via OpenRouter is rejected outright by the route guard in
    # evals._inspect_patches (it corrupts thought_signature on multi-turn
    # calls). Defer to that guard's error, which names the native slug to
    # use instead; pinning an endpoint for it would be meaningless anyway.
    if model.startswith(_FORBIDDEN_GEMINI_PREFIX):
        return {}

    slug = model[len(_OPENROUTER_PREFIX):]
    tag = PINNED_ENDPOINTS.get(slug)
    if tag is None:
        raise ValueError(
            f"No pinned OpenRouter endpoint for {model!r}. Look up its "
            f"endpoints (see module docstring) and add {slug!r} to "
            "PINNED_ENDPOINTS before running it, or results won't reproduce."
        )

    return {
        "provider": {
            # Endpoint tag, not a bare vendor name: fixes the quantization,
            # not just the company.
            "only": [tag],
            # Without this the pin is a preference, not a guarantee.
            "allow_fallbacks": False,
            # Route only to endpoints that honour the params we send, instead
            # of dropping them silently.
            "require_parameters": True,
        }
    }
