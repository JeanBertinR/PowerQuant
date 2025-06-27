"""
This module implements the main functionality of PowerQuant

Author: Jean Bertin
"""

__author__ = "Jean Bertin"
__email__ = "jeanbertin.ensam@gmail.com"
__status__ = "planning"

from entsoe import EntsoePandasClient
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import holidays
import matplotlib.pyplot as plt
import holidays
import requests
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_temp_smoothed_fr(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves smoothed hourly average temperatures across several major French cities.

    :param start_date: Start date (format 'YYYY-MM-DD')
    :param end_date: End date (format 'YYYY-MM-DD')
    :return: DataFrame with columns ['datetime', 'temperature']
    """
    # Selected cities to smooth data at a national scale
    cities = {
        "Paris": (48.85, 2.35),
        "Lyon": (45.76, 4.84),
        "Marseille": (43.30, 5.37),
        "Lille": (50.63, 3.07),
        "Toulouse": (43.60, 1.44),
        "Strasbourg": (48.58, 7.75),
        "Nantes": (47.22, -1.55),
        "Bordeaux": (44.84, -0.58)
    }

    city_dfs = []

    for city, (lat, lon) in tqdm(cities.items(), desc="Fetching city data"):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": "Europe/Paris"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame({
                'datetime': data['hourly']['time'],
                city: data['hourly']['temperature_2m']
            })
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            city_dfs.append(df)
        except Exception as e:
            print(f"Error with {city}: {e}")

    # Merge all city data and compute the mean temperature
    df_all = pd.concat(city_dfs, axis=1)
    df_all['temperature'] = df_all.mean(axis=1)

    # Return only datetime and the averaged temperature
    return df_all[['temperature']].reset_index()

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import holidays

def eval_forecast(df, datetime_col='datetime', target_col='consumption'):
    import pandas as pd
    import holidays
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import GradientBoostingRegressor

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=[datetime_col, target_col])

    df = df.sort_values(datetime_col)
    midpoint = len(df) // 2
    train_df = df.iloc[:midpoint]
    test_df = df.iloc[midpoint:]

    full_start = df[datetime_col].min().strftime('%Y-%m-%d')
    full_end = df[datetime_col].max().strftime('%Y-%m-%d')

    temp_df = get_temp_smoothed_fr(full_start, full_end)
    temp_df = temp_df.rename(columns={'datetime': datetime_col})
    temp_df[datetime_col] = pd.to_datetime(temp_df[datetime_col], errors='coerce')

    train_df = pd.merge(train_df, temp_df, on=datetime_col, how='left')
    test_df = pd.merge(test_df, temp_df, on=datetime_col, how='left')

    def add_time_and_extreme_features(df):
        df['hour'] = df[datetime_col].dt.hour
        df['dayofweek'] = df[datetime_col].dt.dayofweek
        df['week'] = df[datetime_col].dt.isocalendar().week.astype(int)
        df['month'] = df[datetime_col].dt.month
        df['year'] = df[datetime_col].dt.year
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        fr_holidays = holidays.country_holidays('FR')
        df['is_holiday'] = df[datetime_col].dt.date.apply(lambda d: 1 if d in fr_holidays else 0)

        # Variables météo extrêmes basées sur la température
        df['temp_sq'] = df['temperature'] ** 2
        df['temp_sqrt'] = df['temperature'].apply(lambda x: x ** 0.5 if x >= 0 else 0)
        df['temp_canicule'] = (df['temperature'] > 30).astype(int)  # canicule au-dessus de 30°C
        df['temp_froid'] = (df['temperature'] < 5).astype(int)      # froid intense en dessous de 5°C

        # Indicateurs de plages de température (exemples)
        df['temp_5_15'] = ((df['temperature'] >= 5) & (df['temperature'] < 15)).astype(int)
        df['temp_15_25'] = ((df['temperature'] >= 15) & (df['temperature'] < 25)).astype(int)
        df['temp_25_30'] = ((df['temperature'] >= 25) & (df['temperature'] < 30)).astype(int)

        return df

    train_df = add_time_and_extreme_features(train_df)
    test_df = add_time_and_extreme_features(test_df)

    # Ajout des nouvelles features dans la liste
    features = [
        'hour', 'dayofweek', 'week', 'month', 'year',
        'is_weekend', 'is_holiday', 'temperature',
        'temp_sq', 'temp_sqrt', 'temp_canicule', 'temp_froid',
        'temp_5_15', 'temp_15_25', 'temp_25_30'
    ]

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_df = test_df.copy()
    test_df['forecast'] = y_pred

    return test_df



def get_spot_prices(api_key: str, country_code: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch day-ahead electricity spot prices from ENTSO-E for a given country and time range.

    :param api_key: ENTSO-E API key
    :param country_code: Country code (e.g., 'FR' for France)
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :return: Pandas Series with hourly spot prices
    """
    client = EntsoePandasClient(api_key=api_key)
    start = pd.Timestamp(start_date, tz='Europe/Brussels')
    end = pd.Timestamp(end_date, tz='Europe/Brussels')

    try:
        prices = client.query_day_ahead_prices(country_code, start=start, end=end)
        return prices
    except Exception as e:
        print(f"Error while retrieving spot prices: {e}")
        return pd.Series()

def plot_forecast(df, datetime_col='datetime', target_col='consumption'):
    # 1. Call eval_forecast
    forecast_df = eval_forecast(df, datetime_col=datetime_col, target_col=target_col)

    # 2. Calculate MAPE
    y_true = forecast_df[target_col].values
    y_pred = forecast_df['forecast'].values
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 3. Create the interactive plot
    fig = go.Figure()

    # Actual values series
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col],
        y=forecast_df[target_col],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    # Forecast series
    fig.add_trace(go.Scatter(
        x=forecast_df[datetime_col],
        y=forecast_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # 4. Layout with black background
    fig.update_layout(
        title='Forecast vs Actual Consumption',
        xaxis_title='Date',
        yaxis=dict(title='Consumption', color='white', gridcolor='gray'),
        xaxis=dict(color='white', gridcolor='gray'),
        legend=dict(x=0.01, y=0.99, font=dict(color='white')),
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(t=120)  # increase top margin
    )

    # 5. Add MAPE at the top center
    fig.add_annotation(
        text=f"MAPE: {mape:.2f}%",
        xref="paper", yref="paper",
        x=0.5, y=1,  # slightly above the title
        showarrow=False,
        font=dict(size=14, color="white"),
        align="center"
    )

    # 6. Show the plot
    fig.show()

def calculate_prem_risk_vol(token: str, input_df: pd.DataFrame, datetime_col: str, target_col: str, plot_chart: bool = False, quantile: int = 50) -> float:
    """
    Calculates a risk premium based on multiple forward prices,
    and returns the value corresponding to the specified quantile.

    :param token: Databricks token.
    :param input_df: DataFrame containing consumption data.
    :param datetime_col: Name of the datetime column in input_df.
    :param target_col: Name of the actual consumption column in input_df.
    :param plot_chart: If True, displays the premium distribution chart.
    :param quantile: Quantile to return (between 1 and 100).
    :return: Risk premium corresponding to the requested quantile.
    """
    # 1. Forecast evaluation
    forecast_df = eval_forecast(input_df, datetime_col=datetime_col, target_col=target_col)
    forecast_df[datetime_col] = pd.to_datetime(forecast_df[datetime_col])

    # 2. Determine the dominant year
    year_counts = forecast_df[datetime_col].dt.year.value_counts()
    if year_counts.empty:
        raise ValueError("No valid data in eval_forecast.")
    major_year = year_counts.idxmax()
    print(f"Dominant year: {major_year} with {year_counts.max()} occurrences")

    # 3. Retrieve spot prices for the covered period
    start_date = forecast_df[datetime_col].min().strftime('%Y-%m-%d')
    end_date = forecast_df[datetime_col].max().strftime('%Y-%m-%d')
    spot_df = get_spot_prices(token, "FR", start_date, end_date)
    spot_df['delivery_from'] = pd.to_datetime(spot_df['delivery_from'])

    # 4. Retrieve forward prices for the dominant year
    forward_df = get_forward_price_fr(token, major_year)
    if forward_df.empty:
        raise ValueError(f"No forward prices found for year {major_year}")
    forward_prices = forward_df['forward_price'].tolist()

    # 5. Prepare dataframe for merging
    forecast_df = forecast_df.rename(columns={datetime_col: 'datetime', target_col: 'consommation_realisee'})
    forecast_df = forecast_df[['datetime', 'consommation_realisee', 'forecast']]
    merged_df = pd.merge(forecast_df, spot_df, left_on='datetime', right_on='delivery_from', how='inner')
    if merged_df.empty:
        raise ValueError("No match between consumption and spot prices")

    # 6. Compute annual consumption once
    merged_df['diff_conso'] = merged_df['consommation_realisee'] - merged_df['forecast']
    conso_totale_MWh = merged_df['consommation_realisee'].sum()
    if conso_totale_MWh == 0:
        raise ValueError("Annual consumption is zero, division not possible")

    # 7. Calculate premiums for each forward price
    premiums = []
    for fwd_price in forward_prices:
        merged_df['diff_price'] = merged_df['price_eur_per_mwh'] - fwd_price
        merged_df['produit'] = merged_df['diff_conso'] * merged_df['diff_price']
        premium = abs(merged_df['produit'].sum()) / conso_totale_MWh
        premiums.append(premium)

    # 8. Optional: display chart
    if plot_chart:
        premiums_sorted = sorted(premiums)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=premiums_sorted,
            x=list(range(1, len(premiums_sorted)+1)),
            mode='lines+markers',
            name='Premiums',
            line=dict(color='cyan')
        ))
        fig.update_layout(
            title="Risk premium distribution (volume)",
            xaxis_title="Index (sorted)",
            yaxis_title="Premium",
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            hovermode='closest'
        )
        fig.show()

    # 9. Return the requested quantile
    if not (1 <= quantile <= 100):
        raise ValueError("Quantile must be an integer between 1 and 100.")
    quantile_value = np.percentile(premiums, quantile)
    return quantile_value

def consumption_price_correlation(input_df: pd.DataFrame, api_key: str, datetime_col='datetime', consumption_col='consumption', country_code='FR') -> float:
    """
    Calculates the correlation between electricity consumption and hourly spot prices.

    :param input_df: DataFrame containing hourly consumption data (with datetime_col and consumption_col)
    :param api_key: ENTSO-E API key to fetch spot prices
    :param datetime_col: Name of the datetime column
    :param consumption_col: Name of the consumption column
    :param country_code: Country code for price retrieval (e.g., 'FR' for France)
    :return: Pearson correlation coefficient
    """
    # Format datetime column
    df = input_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    start_date = df[datetime_col].min().strftime('%Y-%m-%d')
    end_date = df[datetime_col].max().strftime('%Y-%m-%d')

    # Fetch spot prices
    spot_prices = get_spot_prices(api_key, country_code, start_date, end_date)
    spot_prices = spot_prices.reset_index()
    spot_prices.rename(columns={'date': datetime_col, 'price': 'spot_price'}, inplace=True)

    # Merge consumption and price data
    merged = pd.merge(df, spot_prices, on=datetime_col, how='inner').dropna(subset=[consumption_col, 'spot_price'])

    # Compute correlation
    corr = merged[consumption_col].corr(merged['spot_price'])
    return corr

def forecast_performance_report(forecast_df: pd.DataFrame, target_col='consumption', forecast_col='forecast') -> dict:
    """
    Generates a performance report of the forecast using multiple metrics.

    :param forecast_df: DataFrame containing actual (target_col) and predicted (forecast_col) values
    :param target_col: Name of the actual value column
    :param forecast_col: Name of the predicted value column
    :return: Dictionary containing metrics (MAPE, MAE, RMSE, R2)
    """
    y_true = forecast_df[target_col].values
    y_pred = forecast_df[forecast_col].values
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAPE (%)": mape,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def monte_carlo_consumption_simulation(df: pd.DataFrame, n_simulations: int = 100, datetime_col='datetime', consumption_col='consumption', temp_std_dev=2.0):
    """
    Simulates multiple consumption trajectories by introducing uncertainty on temperature,
    adding Gaussian noise to the temperatures used for forecasting.

    :param df: DataFrame with initial data (datetime_col, consumption_col)
    :param n_simulations: Number of Monte Carlo simulations
    :param datetime_col: Name of the datetime column
    :param consumption_col: Name of the consumption column
    :param temp_std_dev: Standard deviation of Gaussian noise added to temperature (in degrees Celsius)
    :return: DataFrame containing datetime and simulated consumption columns ['sim_1', 'sim_2', ...]
    """
    base_forecast_df = eval_forecast(df, datetime_col, consumption_col)
    
    sim_results = pd.DataFrame({datetime_col: base_forecast_df[datetime_col]})

    for i in range(n_simulations):
        # Perturb temperature with Gaussian noise
        perturbed_temp = base_forecast_df['temperature'] + np.random.normal(0, temp_std_dev, size=len(base_forecast_df))
        sim_df = base_forecast_df.copy()
        sim_df['temperature'] = perturbed_temp

        # Recalculate features and predictions with the existing model
        # For simplicity, retrain the model on each simulation (can be optimized)
        features = ['hour', 'dayofweek', 'week', 'month', 'year', 'is_weekend', 'is_holiday', 'temperature']

        X = sim_df[features]
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        y_train = sim_df[consumption_col]
        model.fit(X, y_train)
        y_sim = model.predict(X)
        sim_results[f'sim_{i+1}'] = y_sim

    return sim_results



