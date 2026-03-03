"""
Pipeline de construction de la poche Satellite (projet Core/Satellite).

Objectifs :
- Télécharger des proxys Satellite via Yahoo Finance (yfinance)
- Passer en rendements mensuels
- Construire un portefeuille Satellite (equal-weight ou inverse-vol)
- Calculer des métriques clés et exporter les séries pour les étapes suivantes

Notes :
- On travaille en mensuel pour avoir un backtest plus robuste (moins de bruit).
- Les poids sont appliqués avec un décalage d'une période pour éviter le look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class SatelliteConfig:
    """Configuration de la poche Satellite."""
    tickers: List[str]
    start: str = "2006-01-01"
    end: str | None = None
    frequency: str = "ME"  # Mensuel
    weighting: str = "equal"  # "equal" ou "inv_vol"
    vol_lookback_months: int = 12  # utilisé si inv_vol
    output_returns_csv: str = "satellite_returns_monthly.csv"
    output_weights_csv: str = "satellite_weights_monthly.csv"


def telecharger_prix_adj_close(
    tickers: List[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Télécharge les prix 'Adjusted Close' depuis Yahoo Finance."""
    print("[1/6] Téléchargement des prix (Adjusted Close) depuis Yahoo Finance...")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError("Aucune donnée téléchargée. Vérifie les tickers et la connexion.")

    # Quand on télécharge plusieurs tickers, yfinance renvoie souvent des colonnes MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prix = data["Adj Close"].copy()
        else:
            # Fallback si Yahoo ne fournit pas Adj Close dans ce format
            prix = data["Close"].copy()
    else:
        # Cas 1 ticker : on renomme proprement la colonne
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        prix = data[[col]].rename(columns={col: tickers[0]})

    # On nettoie et on comble les trous (ffill)
    prix = prix.dropna(how="all").ffill()

    print(f"  -> Période: {prix.index.min().date()} à {prix.index.max().date()}")
    print(f"  -> Tickers: {list(prix.columns)}")
    return prix


def convertir_en_prix_mensuels(prix_journalier: pd.DataFrame) -> pd.DataFrame:
    """Convertit des prix journaliers en prix fin de mois (month-end)."""
    print("[2/6] Conversion en prix mensuels (fin de mois)...")

    prix_m = prix_journalier.resample("ME").last()
    prix_m = prix_m.dropna(how="all").ffill()

    print(f"  -> Nombre de mois: {len(prix_m)}")
    return prix_m


def calculer_rendements(prix: pd.DataFrame) -> pd.DataFrame:
    """Calcule les rendements simples à partir des prix."""
    print("[3/6] Calcul des rendements mensuels...")

    rets = prix.pct_change().dropna(how="all")
    rets = rets.dropna(axis=1, how="all")

    print(f"  -> Dimensions (mois x actifs): {rets.shape}")
    return rets


def poids_equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    """Poids égaux chaque mois (long-only, somme = 1)."""
    print("[4/6] Construction des poids 'equal-weight'...")

    w = pd.DataFrame(
        np.ones_like(returns.values),
        index=returns.index,
        columns=returns.columns,
    )
    w = w.div(w.sum(axis=1), axis=0)
    return w


def poids_inverse_volatilite(
    returns: pd.DataFrame,
    lookback_months: int,
) -> pd.DataFrame:
    """
    Poids inverse-vol (rolling).

    Logique :
    - On estime la volatilité réalisée sur les N derniers mois
    - On met un poids proportionnel à 1/vol
    - On normalise pour que la somme des poids fasse 1 (long-only)

    Remarque :
    - Vol annualisée = std mensuelle * sqrt(12)
    """
    print("[4/6] Construction des poids 'inverse-vol' (rolling)...")

    rolling_std = returns.rolling(lookback_months).std()
    rolling_vol = rolling_std * np.sqrt(12)

    inv_vol = 1.0 / rolling_vol.replace(0.0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

    w = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    w = w.dropna(how="all")

    print(f"  -> Poids disponibles à partir de: {w.index.min().date()}")
    return w


def backtester_portefeuille(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Backtest portefeuille.

    Important (anti look-ahead) :
    - Les poids décidés à la date t sont appliqués aux rendements de t+1
    - D'où le shift(1) sur les poids.
    """
    print("[5/6] Backtest de la poche Satellite...")

    weights = weights.reindex(returns.index).ffill()
    w_lag = weights.shift(1)

    port_rets = (w_lag * returns).sum(axis=1)
    port_rets = port_rets.dropna()

    print(f"  -> Le backtest commence le: {port_rets.index.min().date()}")
    return port_rets


def calculer_metriques(returns: pd.Series) -> Dict[str, float]:
    """Calcule quelques métriques classiques à partir de rendements mensuels."""
    # Performance annualisée
    ann_ret = (1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0

    # Vol annualisée
    ann_vol = returns.std() * np.sqrt(12)

    # Sharpe (rf=0)
    sharpe = np.nan
    if ann_vol > 0:
        sharpe = ann_ret / ann_vol

    # Max Drawdown
    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    max_dd = dd.min()

    # Calmar
    calmar = np.nan
    if max_dd < 0:
        calmar = ann_ret / abs(max_dd)

    return {
        "Rendement annualisé": float(ann_ret),
        "Volatilité annualisée": float(ann_vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Calmar": float(calmar),
        "Nombre de mois": float(len(returns)),
    }


def sauvegarder_sorties(
    port_returns: pd.Series,
    weights: pd.DataFrame,
    cfg: SatelliteConfig,
) -> None:
    """Sauvegarde rendements et poids au format CSV pour la fusion Core + Satellite."""
    print("[6/6] Sauvegarde des sorties (CSV)...")

    out_rets = port_returns.to_frame("satellite_portfolio_return")
    out_rets.to_csv(cfg.output_returns_csv, index=True)
    weights.to_csv(cfg.output_weights_csv, index=True)

    print(f"  -> Rendements sauvegardés: {cfg.output_returns_csv}")
    print(f"  -> Poids sauvegardés: {cfg.output_weights_csv}")


def construire_satellite(cfg: SatelliteConfig) -> Tuple[pd.Series, pd.DataFrame]:
    """Pipeline complet de construction de la poche Satellite."""
    prix_d = telecharger_prix_adj_close(cfg.tickers, cfg.start, cfg.end)
    prix_m = convertir_en_prix_mensuels(prix_d)
    rets_m = calculer_rendements(prix_m)

    if cfg.weighting == "equal":
        weights = poids_equal_weight(rets_m)
    elif cfg.weighting == "inv_vol":
        weights = poids_inverse_volatilite(rets_m, cfg.vol_lookback_months)
    else:
        raise ValueError("weighting doit être 'equal' ou 'inv_vol'.")

    port_rets = backtester_portefeuille(rets_m, weights)
    return port_rets, weights


def main() -> None:
    """
    Exemple d'utilisation.

    Proxies Satellite simples et pratiques :
    - Managed Futures / Trend : DBMF ou KMLM
    - Or : GLD ou IAU
    """
    cfg = SatelliteConfig(
        tickers=["DBMF", "GLD"],  # si DBMF ne marche pas, tester KMLM
        start="2006-01-01",
        weighting="inv_vol",  # "equal" ou "inv_vol"
        vol_lookback_months=12,
        output_returns_csv="outputs/satellite_returns_monthly.csv",
        output_weights_csv="outputs/satellite_weights_monthly.csv",
    )

    port_rets, weights = construire_satellite(cfg)

    metriques = calculer_metriques(port_rets)
    print("\n=== MÉTRIQUES : Poche Satellite (mensuel) ===")
    for k, v in metriques.items():
        if "Nombre" in k:
            print(f"{k}: {v:.0f}")
        else:
            print(f"{k}: {v:.4f}")

    sauvegarder_sorties(port_rets, weights, cfg)


if __name__ == "__main__":
    main()