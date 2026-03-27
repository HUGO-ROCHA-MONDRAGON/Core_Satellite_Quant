"""
Corrig - Niveau 1 Beta Filter Inline (pour notebook)

Insérer directement ce code dans le notebook pour bypasser le problème de module import.
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress

# CONFIG - Paramètres du filtre niveau 1 CORRIGES du pipeline original
BETA_WINDOW = 63  # Doit être en jours pour le rolling beta
CALIB_START_DATE = "2019-01-01"
CALIB_END_DATE = "2020-12-31"

# Conditions statistiques multi (ancien pipeline original)
MEDIAN_BETA_MAX = 0.35
Q75_BETA_MAX = 0.55
PASS_RATIO_MIN = 0.80


def calculate_rolling_beta_simple(prices_df, core_returns, window=63):
    """
    Calcule beta rolling (fenêtre = window jours) pour tous les tickers vs Core.
    
    Returns: DataFrame wide (index=date, cols=tickers)
    """
    # Récupérer colonnes qui matchent entre prices et core
    fund_rets = np.log(prices_df).diff()
    
    # Créer series temporaire du core pour l'alignment
    core_ret_series = core_returns.copy()
    
    # One-liner optimization: calcul vectorisé si possible
    betas_rolling = {}
    
    for ticker in fund_rets.columns:
        # Récupérer les rendements du fonds
        fund_col = fund_rets[ticker].dropna()
        
        # Aligner avec Core
        aligned_pairs = pd.concat([
            fund_col,
            core_ret_series
        ], axis=1, keys=['fund', 'core']).dropna()
        
        if len(aligned_pairs) < window:
            continue
            
        # Calculer rolling beta
        fund_vals = aligned_pairs['fund'].values
        core_vals = aligned_pairs['core'].values
        dates = aligned_pairs.index
        
        rolling_betas = []
        rolling_dates = []
        
        for i in range(window, len(fund_vals)+1):
            window_fund = fund_vals[i-window:i]
            window_core = core_vals[i-window:i]
            
            # OLS: fund = alpha + beta * core + eps
            slope, intercept, r_val, p_val, std_err = linregress(window_core, window_fund)
            rolling_betas.append(slope)
            rolling_dates.append(dates[i-1])
        
        if rolling_betas:
            betas_rolling[ticker] = pd.Series(rolling_betas, index=rolling_dates)
    
    return pd.DataFrame(betas_rolling)


print("[SNIPPET] Code Ready - à insérer dans le notebook après la cellule de chargement prices")
