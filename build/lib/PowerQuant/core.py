"""
This module implements the main functionality of PowerQuantAnalysis.

Author: Jean Bertin
"""

__author__ = "Jean Bertin"
__email__ = "jeanbertin.ensam@gmail.com"
__status__ = "planning"

from entsoe import EntsoePandasClient
import pandas as pd

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
