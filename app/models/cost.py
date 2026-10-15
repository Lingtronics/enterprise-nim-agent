def estimate_cost(usage, settings):
    """Estimate tokens only using explicit operator-provided pricing."""
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        prompt is None
        or completion is None
        or settings.price_input_per_million is None
        or settings.price_output_per_million is None
    ):
        return {
            "amount": None,
            "currency": settings.price_currency,
            "reason": "Provider token usage and configured token prices are both required.",
        }
    return {
        "amount": (
            prompt * settings.price_input_per_million
            + completion * settings.price_output_per_million
        )
        / 1_000_000,
        "currency": settings.price_currency,
        "basis": "configured_token_prices",
    }
