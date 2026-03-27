"""
FILTRE NIVEAU 1 - VERSION SIMPLE & CORRECTE

D'après l'analyse du pipeline original, voici la VRAIE logique:

1. Calculer beta rolling avec fenêtre = 126 jours
2. Appliquer 3 conditions statistiques SIMULTANÉES sur la série de betas:
    - median(|beta|) <= 0.32
    - Q75(|beta|) <= 0.50
    - Ratio(|beta| <= 0.32) >= 0.85

IMPORTANT: Version harmonisée sur 126j pour rester cohérente avec le reste du pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def _detect_ticker_col(df: pd.DataFrame) -> str:
    """Retourne le nom de colonne ticker ('Ticker' ou 'ticker')."""
    for col in ["ticker", "Ticker"]:
        if col in df.columns:
            return col
    raise ValueError("Colonne ticker introuvable (attendu: 'Ticker' ou 'ticker').")


def apply_level1_filter_corrected(
    level0_df: pd.DataFrame,
    prices_aligned: pd.DataFrame,
    core_returns_aligned: pd.Series,
    calib_start: str = "2019-01-01",
    calib_end: str = "2020-12-31",
    rolling_window: int = 126,
    median_beta_max: float = 0.20,
    q75_beta_max: float = 0.30,
    pass_ratio_min: float = 0.95,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique le filtre niveau 1 CORRECT avec les conditions multi-stat du pipeline original.
    
    Parameters:
    -----------
    level0_df: DataFrame avec colonnes ['ticker', 'Strat', ...]
    prices_aligned: DataFrame wide (dates × tickers) avec prix alignés à calib window
    core_returns_aligned: Series avec rendements Core alignés
    calib_start/end: fenêtre de calibration (e.g., "2019-01-01" à "2020-12-31")
    rolling_window: fenêtre de rolling beta = 126j
    verbose: afficher logs
    
    Returns:
    --------
    (level1_df_filtered, results_diagnostics)
    """
    ticker_col = _detect_ticker_col(level0_df)
    
    if verbose:
        print("\n" + "="*80)
        print("🔹 FILTRE NIVEAU 1 – BETA ROLLING (PIPELINE ORIGINAL - CORRECTED)")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Rolling window: {rolling_window} jours")
        print(f"  Conditions: median(|β|)≤{median_beta_max}, Q75(|β|)≤{q75_beta_max}, ratio≥{pass_ratio_min}")
        print(f"  Calib: {calib_start} à {calib_end}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Étape 1: Calculer rendements journaliers
    # ════════════════════════════════════════════════════════════════════════════
    
    if verbose:
        print(f"\n1️⃣  Calcul rendements journaliers pour {prices_aligned.shape[1]} tickers...")
    
    fund_rets = np.log(prices_aligned).diff().dropna(how='all')
    
    if verbose:
        print(f"   Shape rendements: {fund_rets.shape}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Étape 2: Calculer rolling beta (126j) pour TOUS les tickers
    # ════════════════════════════════════════════════════════════════════════════
    
    if verbose:
        print(f"\n2️⃣  Calcul beta rolling ({rolling_window}j)...")
    
    # Aligner fund_rets avec core_returns
    aligned_returns = pd.concat(
        [fund_rets, core_returns_aligned.rename('__CORE__')],
        axis=1, sort=True
    ).dropna(how='all')
    
    # Calculer rolling beta pour chaque fonds
    betas_rolling = {}
    for ticker in fund_rets.columns:
        if ticker not in aligned_returns.columns:
            continue
        
        # Récupérer les séries alignées
        fund_col = aligned_returns[ticker].dropna()
        core_col = aligned_returns['__CORE__'].loc[fund_col.index].dropna()
        
        # Aligner à nouveau
        both = pd.concat([fund_col, core_col], axis=1, keys=['fund', 'core']).dropna()
        
        if len(both) < rolling_window + 1:
            continue
        
        # Calculer rolling covariance et variance
        fund_vals = both['fund'].values
        core_vals = both['core'].values
        dates_vals = both.index
        
        rolling_betas_list = []
        rolling_betas_dates = []
        
        for i in range(rolling_window, len(fund_vals)):
            # Fenêtre[i-rolling_window:i]
            fund_window = fund_vals[i-rolling_window:i]
            core_window = core_vals[i-rolling_window:i]
            
            # Beta = Cov(fund, core) / Var(core)
            cov = np.cov(fund_window, core_window)[0, 1]
            var = np.var(core_window, ddof=1)
            
            if var > 1e-12:
                beta = cov / var
                rolling_betas_list.append(beta)
                rolling_betas_dates.append(dates_vals[i])
        
        if rolling_betas_list:
            betas_rolling[ticker] = pd.Series(
                rolling_betas_list,
                index=pd.DatetimeIndex(rolling_betas_dates),
                name=ticker
            )
    
    if verbose:
        print(f"   ✅ Beta rolling calculés pour {len(betas_rolling)} tickers")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Étape 3: Extraire fenêtre de calibration et appliquer conditions
    # ════════════════════════════════════════════════════════════════════════════
    
    if verbose:
        print(f"\n3️⃣  Application des conditions statistiques sur calib window...")
    
    results = []
    passed_tickers = []
    
    for ticker, beta_series in betas_rolling.items():
        # Extraire fenêtre de calibration
        betas_calib = beta_series.loc[calib_start:calib_end]
        
        if len(betas_calib) < 50:  # Minimum observations
            # Fonds avec données insuffisantes
            result = {
                'ticker': ticker,
                'strat': level0_df[level0_df[ticker_col] == ticker]['Strat'].values[0] if ticker in level0_df[ticker_col].values else 'Unknown',
                'status': 'insufficient_data',
                'n_obs': len(betas_calib),
                'passed': False,
                'median_abs_beta': np.nan,
                'q75_abs_beta': np.nan,
                'pass_ratio': np.nan,
            }
            results.append(result)
            continue
        
        # Calculer statistiques
        abs_beta = betas_calib.abs()
        median_abs_beta = float(abs_beta.median())
        q75_abs_beta = float(abs_beta.quantile(0.75))
        pass_ratio = float((abs_beta <= median_beta_max).mean())
        
        # Appliquer les 3 conditions SIMULTANÉMENT
        cond1_median = median_abs_beta <= median_beta_max
        cond2_q75 = q75_abs_beta <= q75_beta_max
        cond3_ratio = pass_ratio >= pass_ratio_min
        passed = cond1_median and cond2_q75 and cond3_ratio
        
        # Récupérer stratégie
        strat = "Unknown"
        if ticker in level0_df[ticker_col].values:
            strat = level0_df[level0_df[ticker_col] == ticker]['Strat'].values[0]
        
        result = {
            'ticker': ticker,
            'strat': strat,
            'status': 'passed' if passed else 'failed',
            'n_obs': len(betas_calib),
            'passed': passed,
            'median_abs_beta': median_abs_beta,
            'q75_abs_beta': q75_abs_beta,
            'pass_ratio': pass_ratio,
            'cond1_median': cond1_median,
            'cond2_q75': cond2_q75,
            'cond3_ratio': cond3_ratio,
        }
        results.append(result)
        
        if passed:
            passed_tickers.append(ticker)
            if verbose and len(passed_tickers) <= 10:
                print(f"   ✅ {ticker:25s} | med={median_abs_beta:.3f} q75={q75_abs_beta:.3f} ratio={pass_ratio:.1%}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Résultats
    # ════════════════════════════════════════════════════════════════════════════
    
    results_df = pd.DataFrame(results)
    
    # Filtrer level0_df pour garder seulement les tickers passés
    level1_df = level0_df[level0_df[ticker_col].isin(passed_tickers)].copy()
    
    if verbose:
        n_total = len(level0_df)
        n_passed = len(level1_df)
        print(f"\n" + "-"*80)
        print(f"RÉSULTATS: {n_passed} / {n_total} fonds PASSED")
        
        if len(results_df) > 0:
            cond1 = results_df['cond1_median'].fillna(False).astype(bool)
            cond2 = results_df['cond2_q75'].fillna(False).astype(bool)
            cond3 = results_df['cond3_ratio'].fillna(False).astype(bool)
            failed_cond1 = (~cond1).sum()
            failed_cond2 = (~cond2).sum()
            failed_cond3 = (~cond3).sum()
            print(f"Échechs par condition:")
            print(f"  • median(|β|) > {median_beta_max}: {failed_cond1} fonds")
            print(f"  • Q75(|β|) > {q75_beta_max}: {failed_cond2} fonds")
            print(f"  • ratio < {pass_ratio_min}: {failed_cond3} fonds")
        
        if n_passed > 0:
            summary = level1_df.groupby('Strat').size()
            print(f"\nComposition par Strat:")
            for strat, count in summary.items():
                print(f"  {strat}: {count}")
    
    return level1_df, results_df


# ==============================================================================
# VERSION SNIPPET POUR NOTEBOOK (à copier directement dans une cellule)
# ==============================================================================

SNIPPET_CODE = """
# 🔧 FILTRE NIVEAU 1 CORRIGÉ - À exécuter après le chargement des données

from src.satellite_level1_beta_filter_final import apply_level1_filter_corrected

# Appliquer le filtre
df_satellite_level1, results_level1 = apply_level1_filter_corrected(
    df_satellite_level0,
    df_satellite_prices_aligned,
    core_returns_aligned,
    calib_start=CALIB_START,
    calib_end=CALIB_END,
    rolling_window=126,
    median_beta_max=0.20,
    q75_beta_max=0.30,
    pass_ratio_min=0.95,
    verbose=True
)

# Afficher résultats
print(f"\\n✅ Filtre appliqué: {len(df_satellite_level1)} / {len(df_satellite_level0)} fonds passent")
print(results_level1[results_level1['passed']].head(10))
"""

if __name__ == "__main__":
    print(SNIPPET_CODE)
