"""
This module implements the main functionality of PowerQuantAnalysis.

Author: Jean Bertin
"""

__author__ = "Jean Bertin"
__email__ = "jeanbertin.ensam@gmail.com"
__status__ = "planning"


import pandas as pd

class PowerQuantAnalysis:
    """
    Une classe contenant des fonctions basiques d'analyse quantitative
    pour des données potentiellement liées au secteur de l'électricité.
    """

    @staticmethod
    def calculate_simple_moving_average(data: pd.Series, window: int) -> pd.Series:
        """
        Calcule la moyenne mobile simple d'une série temporelle.

        Args:
            data (pd.Series): La série temporelle de données.
            window (int): La taille de la fenêtre pour le calcul de la moyenne mobile.

        Returns:
            pd.Series: La série temporelle de la moyenne mobile.
                      Retourne une série vide si les données sont vides ou la fenêtre invalide.
        """
        if data.empty or window <= 0 or window > len(data):
            return pd.Series()
        return data.rolling(window=window).mean()

    @staticmethod
    def calculate_percentage_change(data: pd.Series) -> pd.Series:
        """
        Calcule le pourcentage de changement d'une série temporelle par rapport à la période précédente.

        Args:
            data (pd.Series): La série temporelle de données.

        Returns:
            pd.Series: La série temporelle du pourcentage de changement.
                      Retourne une série vide si les données contiennent moins de deux points.
        """
        if len(data) < 2:
            return pd.Series()
        return data.pct_change() * 100
