"""
Satellite Level 1 Filter
Filtrage sur le beta rolling par rapport au benchmark Core
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_satellite_prices(strat_num: int, data_dir: str = "data") -> dict:
    """
    Charge les prix d'un fichier STRAT_price.xlsx.
    Format: colonnes alternées [Ticker1, Dates1, Ticker2, Dates2, ...]
    
    Parameters
    ----------
    strat_num : int
        Numéro du strat (1, 2, ou 3)
    data_dir : str
        Répertoire contenant les fichiers Excel
        
    Returns
    -------
    dict
        Dictionnaire {ticker: pd.Series de prix}
    """
    filepath = Path(data_dir) / f"STRAT{strat_num}_price.xlsx"
    
    prices_dict = {}
    
    xls = pd.ExcelFile(filepath)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        # Les colonnes alternent : Ticker, Dates, Ticker, Dates, ...
        # Colonnes pairs (0, 2, 4, ...) = Ticker ou Date
        # Colonnes impairs (1, 3, 5, ...) = Date ou Prix
        
        # Parcourir les paires de colonnes
        for i in range(0, df.shape[1] - 1, 2):
            col_ticker = df.columns[i]
            col_prix = df.columns[i + 1]
            
            # Vérifier que c'est un ticker (contient "Equity")
            if 'Equity' in str(col_ticker):
                ticker = col_ticker
                try:
                    # Extraire les prix et nettoyer
                    prices = pd.to_numeric(df[col_prix], errors='coerce')
                    prices = prices.dropna()
                    
                    if len(prices) > 0:
                        prices_dict[ticker] = prices
                except:
                    pass
    
    return prices_dict


def load_core_benchmark_returns(
    core_tickers: list,
    price_file: str = "data/ETF EUR Bloomberg Cross Asset.xlsx"
) -> pd.Series:
    """
    Charge les rendements du benchmark Core.
    
    Parameters
    ----------
    core_tickers : list
        Liste des tickers Core (ex: ["XDWD GY Equity", ...])
    price_file : str
        Chemin du fichier des prix Core
        
    Returns
    -------
    pd.Series
        Série des rendements cumulés/pondérés du Core
    """
    # Cette fonction sera appelée après chargement du Core
    # Pour l'instant, elle sera simplifiée
    pass


def calculate_rolling_beta(
    returns_fund: pd.Series,
    returns_benchmark: pd.Series,
    window: int = 252  # k jours
) -> pd.Series:
    """
    Calcule le beta rolling d'un fonds par rapport à un benchmark.
    
    Parameters
    ----------
    returns_fund : pd.Series
        Rendements journaliers du fonds
    returns_benchmark : pd.Series
        Rendements journaliers du benchmark
    window : int
        Fenêtre de calcul en jours (paramètre k)
        
    Returns
    -------
    pd.Series
        Série du beta rolling
    """
    # Aligner les index
    df = pd.DataFrame({
        'fund': returns_fund,
        'bench': returns_benchmark
    }).dropna()
    
    betas = []
    for i in range(len(df) - window + 1):
        window_data = df.iloc[i:i+window]
        
        # Calcul du beta
        covariance = window_data['fund'].cov(window_data['bench'])
        benchmark_variance = window_data['bench'].var()
        
        if benchmark_variance != 0:
            beta = covariance / benchmark_variance
        else:
            beta = np.nan
        
        betas.append(beta)
    
    # Créer une série avec les dates correspondantes
    beta_series = pd.Series(
        betas,
        index=df.index[window-1:],
        name='beta_rolling'
    )
    
    return beta_series


def filter_level1_rolling_beta(
    df_funds: pd.DataFrame,
    core_returns: pd.Series,
    strat_num: int,
    price_data_dir: str = "data",
    rolling_window: int = 126,  # k jours
    beta_threshold: float = 1.0,  # s seuil
    percentile_threshold: float = 70.0,  # j pourcentage
    verbose: bool = True
) -> tuple:
    """
    Filtre les fonds sur le beta rolling.
    
    Un fonds passe le filtre si son beta rolling est < s (seuil)
    dans au moins j% des périodes.
    
    Parameters
    ----------
    df_funds : pd.DataFrame
        DataFrame des fonds (niveau 0 filtré)
    core_returns : pd.Series
        Rendements journaliers du Core (benchmark)
    strat_num : int
        Numéro du strat
    price_data_dir : str
        Répertoire des données price
    rolling_window : int
        Fenêtre de calcul du beta (k jours)
    beta_threshold : float
        Seuil du beta (s)
    percentile_threshold : float
        Pourcentage des périodes où beta < s (j%)
    verbose : bool
        Afficher les détails
        
    Returns
    -------
    tuple
        (df_filtered, results_summary)
    """
    
    # Charger les prix du strat
    try:
        prices_dict = load_satellite_prices(strat_num, data_dir=price_data_dir)
    except Exception as e:
        if verbose:
            print(f"⚠️  Erreur chargement prix STRAT{strat_num}: {e}")
        return df_funds.copy(), pd.DataFrame()
    
    if len(prices_dict) == 0:
        if verbose:
            print(f"⚠️  Aucun prix disponible pour STRAT{strat_num}")
        return df_funds.copy(), pd.DataFrame()
    
    # Filtrer les fonds du strat courant
    df_strat = df_funds[df_funds['Strat'] == f'STRAT{strat_num}'].copy()
    
    results = []
    
    for idx, row in df_strat.iterrows():
        ticker = row['Ticker']
        
        # Chercher le ticker dans les prix
        matching_ticker = None
        for available_ticker in prices_dict.keys():
            if available_ticker.startswith(ticker) or available_ticker == ticker:
                matching_ticker = available_ticker
                break
        
        if matching_ticker is None:
            # Pas de données de prix, exclure
            results.append({
                'Ticker': ticker,
                'Pass_Level1': False,
                'Reason': 'No price data'
            })
            continue
        
        try:
            # Extraire les prix
            prices = prices_dict[matching_ticker]
            
            # Convertir en rendements log
            returns_fund = np.log(prices / prices.shift(1)).dropna()
            
            # Aligner avec les rendements du Core
            common_index = returns_fund.index.intersection(core_returns.index)
            
            if len(common_index) < rolling_window:
                results.append({
                    'Ticker': ticker,
                    'Pass_Level1': False,
                    'Reason': f'Insufficient data ({len(common_index)} < {rolling_window})'
                })
                continue
            
            returns_fund_aligned = returns_fund[common_index]
            core_returns_aligned = core_returns[common_index]
            
            # Calculer le beta rolling
            beta_rolling = calculate_rolling_beta(
                returns_fund_aligned,
                core_returns_aligned,
                window=rolling_window
            )
            
            # Calculer le pourcentage de périodes où beta < seuil
            pct_under_threshold = (beta_rolling < beta_threshold).sum() / len(beta_rolling) * 100
            
            # Passe le filtre si le pourcentage est >= j
            passes_filter = pct_under_threshold >= percentile_threshold
            
            results.append({
                'Ticker': ticker,
                'Pass_Level1': passes_filter,
                'Beta_Mean': beta_rolling.mean(),
                'Beta_Median': beta_rolling.median(),
                'Beta_Std': beta_rolling.std(),
                'Pct_Under_Threshold': pct_under_threshold,
                'Threshold': beta_threshold,
                'Reason': 'OK' if passes_filter else f'{pct_under_threshold:.1f}% < {percentile_threshold}%'
            })
            
        except Exception as e:
            results.append({
                'Ticker': ticker,
                'Pass_Level1': False,
                'Reason': f'Error: {str(e)[:50]}'
            })
    
    # Créer le résumé
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    
    # Filtrer les fonds qui passent
    if len(results_df) > 0:
        passed_tickers = results_df[results_df['Pass_Level1'] == True]['Ticker'].unique()
        df_filtered = df_funds[df_funds['Ticker'].isin(passed_tickers)].copy()
    else:
        df_filtered = pd.DataFrame(columns=df_funds.columns)
    
    return df_filtered, results_df


def filter_level1_all_strats(
    df_funds: pd.DataFrame,
    core_returns: pd.Series,
    rolling_window: int = 126,
    beta_threshold: float = 1.0,
    percentile_threshold: float = 70.0,
    verbose: bool = True
) -> tuple:
    """
    Applique le filtre niveau 1 à tous les strats.
    
    Parameters
    ----------
    df_funds : pd.DataFrame
        DataFrame des fonds (niveau 0 filtré)
    core_returns : pd.Series
        Rendements journaliers du Core (benchmark)
    rolling_window : int
        Fenêtre de calcul du beta (k jours)
    beta_threshold : float
        Seuil du beta (s)
    percentile_threshold : float
        Pourcentage des périodes où beta < s (j%)
    verbose : bool
        Afficher les détails
        
    Returns
    -------
    tuple
        (df_filtered, results_summary)
    """
    
    all_results = []
    df_filtered_list = []
    
    if verbose:
        print("="*80)
        print("SATELLITE LEVEL 1 FILTER - ROLLING BETA")
        print("="*80)
        print(f"\nParamètres:")
        print(f"  - Rolling window (k) : {rolling_window} jours")
        print(f"  - Beta threshold (s) : {beta_threshold}")
        print(f"  - Percentile threshold (j) : {percentile_threshold}%")
        print(f"\nCritère: Beta rolling < {beta_threshold} dans au moins {percentile_threshold}% des périodes")
    
    for strat_num in [1, 2, 3]:
        if verbose:
            print(f"\n--- STRAT{strat_num} ---")
        
        df_strat_filtered, results_strat = filter_level1_rolling_beta(
            df_funds,
            core_returns,
            strat_num=strat_num,
            rolling_window=rolling_window,
            beta_threshold=beta_threshold,
            percentile_threshold=percentile_threshold,
            verbose=verbose
        )
        
        if len(results_strat) > 0:
            passed = results_strat['Pass_Level1'].sum()
            failed = (~results_strat['Pass_Level1']).sum()
            if verbose:
                print(f"Résultat: {passed} passent, {failed} exclus")
        
        df_filtered_list.append(df_strat_filtered)
        all_results.append(results_strat)
    
    # Combiner tous les résultats
    df_final = pd.concat(df_filtered_list, ignore_index=True)
    results_final = pd.concat(all_results, ignore_index=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Fonds restants après filtre niveau 1: {len(df_final)}")
        summary = df_final.groupby('Strat').size().reset_index(name='Nombre de fonds')
        print(f"\nComposition par bloc:")
        print(summary.to_string(index=False))
        print(f"\nTotal: {summary['Nombre de fonds'].sum()} fonds")
    
    return df_final, results_final
