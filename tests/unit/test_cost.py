from app.models.cost import estimate_cost


def test_no_cost_is_invented_when_usage_or_pricing_is_missing(settings):
    assert estimate_cost({}, settings)["amount"] is None
    assert estimate_cost({"prompt_tokens": 10, "completion_tokens": 10}, settings)["amount"] is None


def test_explicit_token_prices_and_provider_usage(settings):
    configuration = settings.model_copy(
        update={"price_input_per_million": 2, "price_output_per_million": 4}
    )
    cost = estimate_cost({"prompt_tokens": 1000, "completion_tokens": 500}, configuration)
    assert cost["amount"] == 0.004
    assert cost["currency"] == "USD"
