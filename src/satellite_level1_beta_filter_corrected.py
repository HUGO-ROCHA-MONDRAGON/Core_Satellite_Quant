"""
Filtre Niveau 1 – Beta Rolling (Corrected)

Version harmonisée 126j :
    - Rolling window: 126 jours
  - Tests statistiques multi-conditions:
        * median(|beta|) <= 0.32
        * Q75(|beta|) <= 0.50
        * Ratio(|beta| <= 0.32) >= 0.85 (85% des jours)

Cette version corrigée :
1. Utilise les données PRE-CHARGÉES et ALIGNÉES
2. Applique la logique statistique exacte du pipeline original
3. Retourne les fonds qui passent TOUS les 3 tests simultanément
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def calculate_rolling_beta_126d(
    prices_wide: pd.DataFrame,
    core_returns: pd.Series,
    window: int = 126
) -> pd.DataFrame:
    """
    Calcule le beta rolling (126 jours) pour chaque fonds vs Core.
    
    Args:
        prices_wide: DataFrame wide (dates × tickers) avec prix
        core_returns: Series avec rendements journaliers du Core
        window: fenêtre de rolling (défaut: 126 jours)
    
    Returns:
        DataFrame wide (dates × tickers) avec beta rolling
    """
    # Calculer les rendements journaliers
    fund_returns = np.log(prices_wide).diff().dropna(how='all')
    
    # Aligner avec les rendements Core
    aligned = pd.concat([fund_returns, core_returns.rename('core_ret')], 
                        axis=1, sort=True).dropna(how='all')
    
    # Supprimer colonnes vides après alignment
    aligned = aligned.dropna(axis=1, how='all')
    
    # Extraire colonnes fonds (tout sauf 'core_ret')
    core_col = 'core_ret'
    if core_col not in aligned.columns:
        raise ValueError("Core returns not found in aligned data")
    
    fund_cols = [c for c in aligned.columns if c != core_col]
    
    # Calculer beta rolling pour chaque fonds
    betas = {}
    for ticker in fund_cols:
        subset = aligned[[ticker, core_col]].dropna()
        if len(subset) < window:
            continue
        
        ticker_ret = subset[ticker].values
        core_ret = subset[core_col].values
        
        # Rolling covariance et variance
        cov = pd.Series(ticker_ret).rolling(window).cov(pd.Series(core_ret))
        var = pd.Series(core_ret).rolling(window).var()
        
        # Beta = Cov / Var
        beta = (cov / var).where(var > 1e-12)
        beta.index = subset.index
        betas[ticker] = beta
    
    return pd.DataFrame(betas).sort_index()


def filter_level1_beta_corrected(
    level0_tickers: List[str],
    prices_wide: pd.DataFrame,
    core_returns: pd.Series,
    info_df: pd.DataFrame,
    calib_start: str = "2019-01-01",
    calib_end: str = "2020-12-31",
    median_max: float = 0.20,
    q75_max: float = 0.30,
    pass_ratio_min: float = 0.95,
    verbose: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Filtre niveau 1 avec les conditions CORRECTES du pipeline initial:
    - median(|beta|) <= 0.20
    - Q75(|beta|) <= 0.30
    - Ratio(|beta| <= 0.20) >= 0.95
    
    Args:
        level0_tickers: liste des tickers du filtre niveau 0
        prices_wide: DataFrame wide avec prix (dates × tickers)
        core_returns: Series avec rendements benchmark
        info_df: DataFrame info avec métadonnées des fonds
        calib_start/calib_end: fenêtre de calibration
        median_max: seuil pour condition 1
        q75_max: seuil pour condition 2
        pass_ratio_min: seuil pour condition 3
        verbose: afficher les logs
    
    Returns:
        (tickers_passed, results_df)
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🔹 FILTRE NIVEAU 1 – BETA ROLLING 126J (CORRECTED)")
        print("="*80)
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 1: Calculer le beta rolling
    # ────────────────────────────────────────────────────────────────────────────
    
    if verbose:
        print(f"\n1️⃣  Calcul beta rolling (window=126j) pour {len(level0_tickers)} tickers...")
    
    # Filtrer les prix pour les tickers level0 uniquement
    prices_available = [t for t in level0_tickers if t in prices_wide.columns]
    if not prices_available:
        raise ValueError("Aucun ticker level0 trouvé dans les prix")
    
    prices_subset = prices_wide[prices_available]
    
    # Calculer le beta rolling
    betas_rolling = calculate_rolling_beta_126d(
        prices_subset,
        core_returns,
        window=126
    )
    
    if verbose:
        print(f"   ✅ Beta rolling calculés pour {len(betas_rolling.columns)} tickers")
        print(f"   Période: {betas_rolling.index.min().date()} à {betas_rolling.index.max().date()}")
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 2: Extraire la fenêtre de calibration
    # ────────────────────────────────────────────────────────────────────────────
    
    if verbose:
        print(f"\n2️⃣  Extraction fenêtre de calibration: {calib_start} à {calib_end}")
    
    betas_calib = betas_rolling.loc[calib_start:calib_end]
    
    if verbose:
        print(f"   Observations: {len(betas_calib)} jours")
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 3: Appliquer les 3 conditions statistiques simultanément
    # ────────────────────────────────────────────────────────────────────────────
    
    if verbose:
        print(f"\n3️⃣  Application des 3 conditions statistiques:")
        print(f"    • median(|β|) ≤ {median_max}")
        print(f"    • Q75(|β|) ≤ {q75_max}")
        print(f"    • Ratio(|β| ≤ {median_max}) ≥ {pass_ratio_min*100:.0f}%")
    
    results = []
    passed_tickers = []
    
    for ticker in betas_calib.columns:
        betas_series = betas_calib[ticker].dropna()
        
        if len(betas_series) == 0:
            continue
        
        # Calcul des statistiques
        abs_beta = betas_series.abs()
        median_abs_beta = float(abs_beta.median())
        q75_abs_beta = float(abs_beta.quantile(0.75))
        pass_ratio = float((abs_beta <= median_max).mean())
        
        # Vérification simultanée des 3 conditions
        cond1 = median_abs_beta <= median_max
        cond2 = q75_abs_beta <= q75_max
        cond3 = pass_ratio >= pass_ratio_min
        passed = cond1 and cond2 and cond3
        
        # Récupérer stratégie depuis info
        strat = "Unknown"
        if ticker in info_df.index:
            strat = str(info_df.loc[ticker, 'Strat'])
        
        result_row = {
            'ticker': ticker,
            'strat': strat,
            'median_abs_beta': median_abs_beta,
            'q75_abs_beta': q75_abs_beta,
            'pass_ratio': pass_ratio,
            'cond1_median': cond1,
            'cond2_q75': cond2,
            'cond3_ratio': cond3,
            'passed': passed,
            'n_obs': len(betas_series)
        }
        results.append(result_row)
        
        if passed:
            passed_tickers.append(ticker)
            if verbose and len(passed_tickers) <= 5:
                print(f"    ✅ {ticker:20s} | med={median_abs_beta:.3f} q75={q75_abs_beta:.3f} ratio={pass_ratio:.1%}")
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n" + "-"*80)
        n_passed = len(passed_tickers)
        n_failed = len(betas_calib.columns) - n_passed
        print(f"   Résultats: {n_passed} PASSÉ / {n_failed} ÉCHOUÉ")
        
        # Statisques des conditions échouées
        if len(results_df) > 0:
            failed_cond1 = (~results_df['cond1_median']).sum()
            failed_cond2 = (~results_df['cond2_q75']).sum()
            failed_cond3 = (~results_df['cond3_ratio']).sum()
            print(f"   Échechs par condition:")
            print(f"     • Condition 1 (median): {failed_cond1} fonds")
            print(f"     • Condition 2 (Q75): {failed_cond2} fonds")
            print(f"     • Condition 3 (ratio): {failed_cond3} fonds")
    
    return passed_tickers, results_df


def filter_level1_all_strats_corrected(
    level0_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    core_returns: pd.Series,
    calib_start: str = "2019-01-01",
    calib_end: str = "2020-12-31",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique le filtre niveau 1 sur tous les tickers level0,
    en gardant la structure (métadonnées) du DataFrame d'entrée.
    
    Args:
        level0_df: DataFrame avec colonnes ['ticker', 'Strat', ...]
        prices_wide: DataFrame wide (dates × tickers)
        core_returns: Series avec rendements benchmark
        calib_start/end: fenêtre de calibration
        verbose: afficher logs
    
    Returns:
        (level1_df, results_df)
    """
    
    level0_tickers = level0_df['ticker'].tolist()
    
    # Appliquer le filtre
    passed_tickers, results_df = filter_level1_beta_corrected(
        level0_tickers,
        prices_wide,
        core_returns,
        level0_df.set_index('ticker') if 'ticker' in level0_df.columns else level0_df,
        calib_start=calib_start,
        calib_end=calib_end,
        verbose=verbose
    )
    
    # Filtrer le DataFrame level0 pour garder seulement les tickers passés
    level1_df = level0_df[level0_df['ticker'].isin(passed_tickers)].copy()
    
    if verbose:
        print(f"\n✅ Structure préservée: {len(level1_df)} / {len(level0_df)} fonds avancent au niveau 2")
    
    return level1_df, results_df
