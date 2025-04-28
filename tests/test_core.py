from PowerQuant import get_spot_prices

def test_get_spot_prices_runs():
    api_key = "your-api-key"
    prices = get_spot_prices(api_key, "FR", "2025-04-24", "2025-04-25")
    assert prices is not None
    assert not prices.empty


