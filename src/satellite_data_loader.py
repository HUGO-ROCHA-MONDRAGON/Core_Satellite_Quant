"""
Satellite Data Loader & Preprocessor
Charge les données (prix + info), filtre sur 2019-2020, aligne les dates correctement
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def load_satellite_prices_wide(price_file: str) -> pd.DataFrame:
    """
    Charge un fichier STRAT_price.xlsx.
    Format: colonnes alternées [Ticker | Dates | Ticker | Dates | ...]
    où Ticker est le nom (Equity), et Dates contient les dates.
    Retourne un DataFrame avec dates en index et tickers en colonnes (prix).
    """
    xls = pd.ExcelFile(price_file)
    all_tickers = {}  # {ticker: [(date, price), ...]}
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(price_file, sheet_name=sheet_name)
        
        # Extraire les paires ticker/prix
        # Format: Col0=Ticker, Col1=Prix_ou_Date, Col2=Ticker, Col3=Prix_ou_Date, ...
        for i in range(0, df.shape[1] - 1, 2):
            col_ticker = df.columns[i]
            col_data = df.columns[i + 1]
            
            if 'Equity' not in str(col_ticker):
                continue
            
            ticker = str(col_ticker).strip()
            
            try:
                # Colonne i contient des dates (header=col_ticker)
                # Colonne i+1 contient les prix
                dates = pd.to_datetime(df[col_ticker], errors='coerce')
                prices = pd.to_numeric(df[col_data], errors='coerce')
                
                # Créer une série avec dates valides et prices non-NaN
                valid_mask = dates.notna() & prices.notna()
                if valid_mask.sum() > 0:
                    series = pd.Series(
                        prices[valid_mask].values,
                        index=dates[valid_mask]
                    )
                    series.index.name = 'Date'
                    all_tickers[ticker] = series.sort_index()
            except Exception as e:
                pass
    
    # Créer un DataFrame avec réindexation
    if not all_tickers:
        return pd.DataFrame()
    
    df_prices = pd.DataFrame(all_tickers)
    df_prices.index.name = 'Date'
    df_prices = df_prices.sort_index()
    
    return df_prices


def load_all_satellite_prices(data_dir: str = "data") -> pd.DataFrame:
    """
    Charge tous les fichiers STRAT_price et les combine.
    """
    all_prices_list = []
    
    for strat_num in [1, 2, 3]:
        filepath = f"{data_dir}/STRAT{strat_num}_price.xlsx"
        try:
            df_prices = load_satellite_prices_wide(filepath)
            all_prices_list.append(df_prices)
            print(f"✅ Chargé {filepath}: {df_prices.shape[1]} tickers")
        except Exception as e:
            print(f"⚠️  Erreur {filepath}: {e}")
    
    # Combine tous les fichiers (union des tickers)
    df_all = pd.concat(all_prices_list, axis=1)
    df_all = df_all.sort_index()
    
    print(f"\n✅ Total prix chargées: {df_all.shape[1]} tickers uniques, {df_all.shape[0]} dates")
    print(f"   Période: {df_all.index[0].date()} à {df_all.index[-1].date()}")
    
    return df_all


def preprocess_prices(
    df_prices: pd.DataFrame,
    start_date: str = "2019-01-01",
    end_date: str = "2020-12-31",
    ffill_limit: int = 5,
    min_obs: int = 50
) -> Tuple[pd.DataFrame, list]:
    """
    Prétraite les prix:
    1. Filtre sur la période [start_date, end_date]
    2. Forward fill limité à ffill_limit jours
    3. Exclut les tickers avec < min_obs observations
    
    Retourne (df_processed, list_tickers_kept)
    """
    # S'assurer que l'index est trié
    df_sorted = df_prices.sort_index()
    
    # Filtre dates
    df_period = df_sorted.loc[start_date:end_date].copy()
    
    if len(df_period) == 0:
        print(f"❌ Aucune donnée dans la période {start_date} à {end_date}")
        return pd.DataFrame(), []
    
    print(f"\n📊 Prétraitement des prix (2019-2020):")
    print(f"   Période: {df_period.index[0].date()} à {df_period.index[-1].date()}")
    print(f"   Observations: {len(df_period)}")
    
    # Forward fill limité
    df_ffilled = df_period.fillna(method='ffill', limit=ffill_limit)
    
    # Compter les observations
    n_obs_per_ticker = df_ffilled.notna().sum()
    
    # Garder les tickers avec suffisamment de données
    valid_tickers = n_obs_per_ticker[n_obs_per_ticker >= min_obs].index.tolist()
    df_clean = df_ffilled[valid_tickers]
    
    excluded_tickers = [t for t in df_period.columns if t not in valid_tickers]
    
    print(f"   Tickers valides: {len(valid_tickers)} / {len(df_period.columns)}")
    print(f"   Tickers exclus (< {min_obs} obs): {len(excluded_tickers)}")
    
    return df_clean, valid_tickers


def calculate_daily_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements log journaliers.
    Supprime les NaN.
    """
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    
    print(f"\n📈 Rendements log calculés: {df_returns.shape}")
    
    return df_returns


def align_prices_with_core(
    df_prices: pd.DataFrame,
    core_returns: pd.Series,
    ffill_limit: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aligne les prix des fonds avec les rendements du Core.
    - Forward fill limité des prix des fonds
    - Intersection des dates
    - Retourne (df_aligned_prices, core_aligned_returns)
    """
    # FFill des prix satellite
    df_ffilled = df_prices.fillna(method='ffill', limit=ffill_limit)
    
    # Intersection des dates
    common_dates = df_ffilled.index.intersection(core_returns.index)
    
    df_aligned = df_ffilled.loc[common_dates]
    core_aligned = core_returns.loc[common_dates]
    
    print(f"\n🔗 Alignement avec Core:")
    print(f"   Dates communes: {len(common_dates)}")
    print(f"   Tickers: {df_aligned.shape[1]}")
    
    return df_aligned, core_aligned


def get_satellite_prices_in_period(
    ticker: str,
    df_prices: pd.DataFrame,
    start_date: str = "2019-01-01",
    end_date: str = "2020-12-31"
) -> pd.Series:
    """
    Récupère les prix d'un fonds pour une période donnée.
    """
    if ticker not in df_prices.columns:
        return pd.Series([], dtype=float)
    
    prices = df_prices.loc[start_date:end_date, ticker].copy()
    prices = prices.dropna()
    
    return prices


def get_satellite_returns_aligned(
    ticker: str,
    df_prices: pd.DataFrame,
    core_returns: pd.Series,
    ffill_limit: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Récupère les rendements log d'un fonds, alignés avec le Core.
    """
    if ticker not in df_prices.columns:
        return pd.Series([], dtype=float), pd.Series([], dtype=float)
    
    prices = df_prices[ticker].copy()
    
    # FFill limité
    prices_ffilled = prices.fillna(method='ffill', limit=ffill_limit)
    
    # Rendements
    returns_fund = np.log(prices_ffilled / prices_ffilled.shift(1)).dropna()
    
    # Aligner avec le Core
    common_index = returns_fund.index.intersection(core_returns.index)
    
    returns_fund_aligned = returns_fund.loc[common_index]
    core_returns_aligned = core_returns.loc[common_index]
    
    return returns_fund_aligned, core_returns_aligned


def load_and_preprocess(
    data_dir: str = "data",
    core_returns_file: str = "outputs/core3_etf_daily_log_returns.csv",
    start_date: str = "2019-01-01",
    end_date: str = "2020-12-31",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list]:
    """
    Pipeline complet de chargement et prétraitement.
    
    Retourne:
    - df_prices: tous les prix
    - df_returns: rendements journaliers
    - core_returns: rendements du Core équipondéré
    - valid_tickers: liste des tickers valides
    """
    
    print("\n" + "="*80)
    print("SATELLITE DATA LOADER & PREPROCESSOR")
    print("="*80)
    
    # 1. Charger les prix
    df_prices = load_all_satellite_prices(data_dir)
    
    # 2. Charger les rendements du Core
    try:
        core_returns_raw = pd.read_csv(core_returns_file, index_col=0, parse_dates=True)
        core_returns = core_returns_raw.iloc[:, 0].dropna()
    except FileNotFoundError:
        print(f"\n❌ Fichier Core {core_returns_file} non trouvé")
        return None
    
    # 3. Prétraiter les prix (filtrer dates, ffill)
    df_prices_clean, valid_tickers = preprocess_prices(
        df_prices,
        start_date=start_date,
        end_date=end_date,
        ffill_limit=5,
        min_obs=50
    )
    
    # 4. Calculer les rendements
    df_returns = calculate_daily_returns(df_prices_clean)
    
    # 5. Aligner avec le Core
    df_aligned, core_aligned = align_prices_with_core(
        df_prices_clean,
        core_returns,
        ffill_limit=5
    )
    
    print(f"\n✅ Pipeline complet terminé")
    
    return df_aligned, df_returns, core_aligned, valid_tickers
