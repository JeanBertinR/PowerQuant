from PowerQuant import (
    get_spot_prices,
    get_temp_smoothed_fr,
    eval_forecast,
    plot_forecast,
    calculate_prem_risk_vol,
    consumption_price_correlation,
    forecast_performance_report,
    monte_carlo_consumption_simulation
)

import pandas as pd
import numpy as np

api_key = "fa36b49d-3d4d-4198-ae43-e830c4d280d3"
country_code = "FR"
start_date = '2024-01-01'
end_date = '2024-12-31'
datetime_col = 'datetime'

# --- Données d'exemple pour tester eval_forecast, plot_forecast, etc. ---
date_rng = pd.date_range(start=start_date, end=end_date, freq='H')
consumption_data = np.random.randint(1000, 1500, size=len(date_rng))
df_example = pd.DataFrame({datetime_col: date_rng, 'consumption': consumption_data})

print("=== Test get_temp_smoothed_fr ===")
try:
    temp_df = get_temp_smoothed_fr(start_date, end_date)
    print(temp_df.head())
except Exception as e:
    print(f"Erreur get_temp_smoothed_fr: {e}")

print("\n=== Test eval_forecast ===")
try:
    forecast_df = eval_forecast(df_example, datetime_col=datetime_col, target_col='consumption')
    print(forecast_df[[datetime_col, 'consumption', 'forecast']].head())
except Exception as e:
    print(f"Erreur eval_forecast: {e}")

print("\n=== Test plot_forecast ===")
try:
    plot_forecast(forecast_df, datetime_col=datetime_col, target_col='consumption')
    print("Plot affiché avec succès")
except Exception as e:
    print(f"Erreur plot_forecast: {e}")

print("\n=== Test get_spot_prices ===")
try:
    spot_prices = get_spot_prices(api_key, country_code, start_date, end_date)
    spot_prices = spot_prices.reset_index()
    spot_prices.rename(columns={'date': datetime_col, 'price': 'spot_price'}, inplace=True)
    print(spot_prices.head())
except Exception as e:
    print(f"Erreur get_spot_prices: {e}")

print("\n=== Test calculate_prem_risk_vol ===")
try:
    premium = calculate_prem_risk_vol(api_key, df_example, datetime_col, 'consumption', plot_chart=True, quantile=50)
    print(f"Risk premium (quantile 50): {premium}")
except Exception as e:
    print(f"Erreur calculate_prem_risk_vol: {e}")

print("\n=== Test consumption_price_correlation ===")
try:
    corr = consumption_price_correlation(df_example, api_key, datetime_col=datetime_col, consumption_col='consumption', country_code=country_code)
    print(f"Correlation consommation-prix: {corr}")
except Exception as e:
    print(f"Erreur consumption_price_correlation: {e}")

print("\n=== Test forecast_performance_report ===")
try:
    report = forecast_performance_report(forecast_df, target_col='consumption', forecast_col='forecast')
    print(report)
except Exception as e:
    print(f"Erreur forecast_performance_report: {e}")

print("\n=== Test monte_carlo_consumption_simulation ===")
try:
    sim_df = monte_carlo_consumption_simulation(df_example, n_simulations=5, datetime_col=datetime_col, consumption_col='consumption')
    print(sim_df.head())
except Exception as e:
    print(f"Erreur monte_carlo_consumption_simulation: {e}")

print("\n=== Colonnes df_example ===")
print(df_example.columns)
