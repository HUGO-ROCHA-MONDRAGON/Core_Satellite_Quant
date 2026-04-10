#!/usr/bin/env python3
"""
Script de test: Filtre Niveau 1 Beta Corrigé (63j avec conditions multi-stat)

Ce script valide que la nouvelle implémentation fonctionne correctement
en utilisant les données chargées via le data_loader.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from satellite_data_loader import load_all_satellite_prices, preprocess_prices, align_prices_with_core
from satellite_level0_filter import filter_satellite_level0
from satellite_level1_beta_filter_corrected import filter_level1_all_strats_corrected


def main():
    print("="*80)
    print(" TEST FILTRE NIVEAU 1 BETA CORRIGÉ (63j Multi-Statistics)")
    print("="*80)
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 1: Charger et préparer les données LEVEL 0
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n Préparation Level 0 (Devise + AUM > 50M)...")
    CALIB_START = "2019-01-01"
    CALIB_END = "2020-12-31"
    
    level0_df, _ = filter_satellite_level0(
        data_dir="data",
        devise=["Euro"],
        min_aum_usd=50,  # $50M
        verbose=True
    )
    
    print(f"\n  Level 0: {len(level0_df)} fonds")
    summary = level0_df.groupby('Strat').size()
    for strat, count in summary.items():
        print(f"      {strat}: {count}")
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 2: Charger et préparer les prix
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n Chargement et prétraitement des prix (2019-2020)...")
    df_prices_all = load_all_satellite_prices(data_dir="data")
    print(f"   Prices loaded: {df_prices_all.shape}")
    
    df_prices_calib, valid_tickers = preprocess_prices(
        df_prices_all,
        start_date=CALIB_START,
        end_date=CALIB_END,
        ffill_limit=5,
        min_obs=50
    )
    print(f"  Prices calibration: {df_prices_calib.shape}")
    print(f"      Valid tickers: {len(valid_tickers)}")
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 3: Charger Core returns (utilisé pour l'alignement)
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n Chargement rendements Core...")
    try:
        core_returns_raw = pd.read_csv("outputs/core3_etf_daily_log_returns.csv", 
                                       index_col=0, parse_dates=True)
        # Moyenne simple
        core_returns_benchmark = core_returns_raw.mean(axis=1)
        print(f"  Core returns: {len(core_returns_benchmark)} jours")
        print(f"      Période: {core_returns_benchmark.index[0].date()} à {core_returns_benchmark.index[-1].date()}")
    except FileNotFoundError:
        print("   Core3 log returns non trouvés, génération dummy...")
        date_range = pd.date_range(CALIB_START, CALIB_END, freq='B')
        core_returns_benchmark = pd.Series(
            np.random.normal(0.0002, 0.01, len(date_range)),
            index=date_range,
            name='core_return'
        )
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 4: Aligner les prix avec Core
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n Alignement prix avec Core...")
    df_prices_aligned, core_returns_aligned = align_prices_with_core(
        df_prices_calib,
        core_returns_benchmark,
        ffill_limit=5
    )
    print(f"  Prices aligned: {df_prices_aligned.shape}")
    print(f"      Core aligned: {len(core_returns_aligned)}")
    
    # ────────────────────────────────────────────────────────────────────────────
    # Étape 5: APPLIQUER LE FILTRE NIVEAU 1 CORRIGÉ
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n APPLICATION FILTRE NIVEAU 1 CORRIGÉ (Beta 63j)...\n")
    
    level1_df, results_df = filter_level1_all_strats_corrected(
        level0_df,
        df_prices_aligned,
        core_returns_aligned,
        calib_start=CALIB_START,
        calib_end=CALIB_END,
        verbose=True
    )
    
    # ────────────────────────────────────────────────────────────────────────────
    # Résultats
    # ────────────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print(" RÉSULTATS FILTRE NIVEAU 1")
    print("="*80)
    
    if len(level1_df) > 0:
        print(f"\n {len(level1_df)} / {len(level0_df)} fonds PASSED")
        
        summary = level1_df.groupby('Strat').size()
        print(f"\nPar bloc:")
        for strat, count in summary.items():
            pct = 100.0 * count / len(level0_df[level0_df['Strat'] == strat])
            print(f"   {strat}: {count} ({pct:.0f}%)")
        
        # Afficher statistiques des fonds passés
        print(f"\nExemples de fonds PASSED (détails conditions):")
        print(results_df[results_df['passed']][
            ['ticker', 'strat', 'median_abs_beta', 'q75_abs_beta', 'pass_ratio']
        ].head(10).to_string(index=False))
        
    else:
        print(f"\n  AUCUN FOND N'A PASSÉ LE FILTRE")
        
        # Afficher statistiques des échechs
        print(f"\nAnalyse des tests échoués:")
        print(results_df[[
            'ticker', 'median_abs_beta', 'q75_abs_beta', 'pass_ratio',
            'cond1_median', 'cond2_q75', 'cond3_ratio'
        ]].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    
    # Sauvegarder les résultats
    results_df.to_csv("outputs/level1_beta_filter_results.csv", index=False)
    print(f" Résultats sauvegardés dans outputs/level1_beta_filter_results.csv")
    
    if len(level1_df) > 0:
        level1_df.to_csv("outputs/level1_filtered_funds.csv", index=False)
        print(f" Fonds passés sauvegardés dans outputs/level1_filtered_funds.csv")


if __name__ == "__main__":
    main()
