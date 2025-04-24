# test_spot_price_script.py
from PowerQuant import get_spot_prices


api_key = "fa36b49d-3d4d-4198-ae43-e830c4d280d3"
country = "FR"
start_date = "2025-04-24"
end_date = "2025-04-25"

prices = get_spot_prices(api_key, country, start_date, end_date)
print(prices.head())
