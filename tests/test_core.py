from PowerQuant import get_spot_prices
from PowerQuant import get_temp_smoothed_fr
from PowerQuant import eval_forecast
from PowerQuant import plot_forecast
from PowerQuant import calculate_mape
from PowerQuant import calculate_prem_risk_vol


load_dotenv()
api_key = os.environ["api_key"]
input_data = pd.read_csv('tests/data/cdc_historical.csv', index_col=0)


country = "FR"
start_date = "2025-04-24"
end_date = "2026-04-25"

prices = get_spot_prices(api_key, country, start_date, end_date)
print(prices.head())


eval_forecast(input_data, datetime_col='interval_start', target_col='interval_value_W')

plot_forecast(input_data, datetime_col='interval_start', target_col='interval_value_W')
