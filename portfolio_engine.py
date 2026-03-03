"""
Moteur de portefeuille Core + Satellite avec volatility targeting (mensuel).

Entrées attendues (CSV) :
- outputs/core_returns_monthly.csv avec colonne 'core_portfolio_return'
- outputs/satellite_returns_monthly.csv avec colonne 'satellite_portfolio_return'

Sorties :
- outputs/portfolio_returns_monthly.csv (rendements du portefeuille total)
- outputs/portfolio_weights_monthly.csv (poids Core/Sat après vol targeting)

Contraintes :
- poids Satellite <= 30% (w_sat_max)
- vol ciblée dans une zone 8-12% (on vise typiquement 10%)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration du portefeuille global."""
    project_root: Path = Path(__file__).resolve().parent  # fichier à la racine du projet
    input_dir: Path = project_root / "outputs"
    output_dir: Path = project_root / "outputs"

    core_csv: Path = input_dir / "core_returns_monthly.csv"
    sat_csv: Path = input_dir / "satellite_returns_monthly.csv"

    # Construction Core/Satellite
    w_sat: float = 0.30         # poids Satellite "structurel" (<= 0.30)
    w_sat_max: float = 0.30     # plafond imposé par le mandat

    # Vol targeting
    target_vol: float = 0.10    # 10% annualisé (au centre de 8-12%)
    vol_lookback_months: int = 12
    scale_min: float = 0.50     # bornes sur le scaling (évite levier excessif)
    scale_max: float = 2.0

    # Sorties
    output_returns_csv: Path = output_dir / "portfolio_returns_monthly.csv"
    output_weights_csv: Path = output_dir / "portfolio_weights_monthly.csv"


def lire_serie_returns(csv_path: Path, colname: str) -> pd.Series:
    """Lit une série de rendements depuis un CSV (index dates)."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if colname not in df.columns:
        raise ValueError(f"Colonne '{colname}' introuvable dans {csv_path}. Colonnes: {list(df.columns)}")
    s = df[colname].copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    s = s.sort_index()
    return s


def aligner_series(core: pd.Series, sat: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Aligne Core et Satellite sur les dates communes."""
    idx = core.index.intersection(sat.index)
    core2 = core.reindex(idx).dropna()
    sat2 = sat.reindex(idx).dropna()
    idx2 = core2.index.intersection(sat2.index)
    return core2.reindex(idx2), sat2.reindex(idx2)


def construire_portefeuille_brut(core: pd.Series, sat: pd.Series, w_sat: float) -> pd.Series:
    """Construit le rendement mensuel brut (avant vol targeting)."""
    w_sat = float(min(w_sat, 0.30))
    w_core = 1.0 - w_sat
    return w_core * core + w_sat * sat


def vol_annualisee_mensuelle(returns: pd.Series) -> pd.Series:
    """Vol réalisée annualisée à partir de rendements mensuels (rolling)."""
    return returns.std() * np.sqrt(12)


def appliquer_vol_targeting(
    returns_brut: pd.Series,
    target_vol: float,
    lookback_months: int,
    scale_min: float,
    scale_max: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Applique un scaling mensuel pour viser target_vol.
    - Vol réalisée estimée sur lookback_months (mensuel) -> annualisée
    - Scale_t = target_vol / vol_realisee_t, borné [scale_min, scale_max]
    - Pour éviter look-ahead : scale calculé à t est appliqué à t+1 (shift)
    """
    # Vol réalisée rolling (annualisée)
    rolling_vol = returns_brut.rolling(lookback_months).std() * np.sqrt(12)

    # Scale (borné)
    scale = target_vol / rolling_vol.replace(0.0, np.nan)
    scale = scale.clip(lower=scale_min, upper=scale_max)

    # Décalage d'une période : scale(t) appliqué à returns(t+1)
    scale_lag = scale.shift(1)

    returns_scaled = scale_lag * returns_brut
    returns_scaled = returns_scaled.dropna()
    scale_lag = scale_lag.reindex(returns_scaled.index)

    return returns_scaled, scale_lag


def metriques_base(returns: pd.Series) -> dict:
    """Métriques classiques (mensuel)."""
    ann_ret = (1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = np.nan if ann_vol <= 0 else ann_ret / ann_vol

    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    max_dd = dd.min()

    calmar = np.nan if max_dd >= 0 else ann_ret / abs(max_dd)

    return {
        "Rendement annualisé": float(ann_ret),
        "Volatilité annualisée": float(ann_vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Calmar": float(calmar),
        "Nombre de mois": float(len(returns)),
    }


def sauvegarder_outputs(
    cfg: PortfolioConfig,
    returns_port: pd.Series,
    scale_lag: pd.Series,
) -> None:
    """Sauvegarde les rendements et les poids effectifs Core/Sat après scaling."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Rendements
    returns_port.to_frame("portfolio_return").to_csv(cfg.output_returns_csv, index=True)

    # Poids effectifs : (1-w_sat)*scale et w_sat*scale
    w_sat_eff = cfg.w_sat * scale_lag
    w_core_eff = (1.0 - cfg.w_sat) * scale_lag
    weights = pd.DataFrame(
        {"w_core_effectif": w_core_eff, "w_sat_effectif": w_sat_eff, "scale": scale_lag},
        index=returns_port.index,
    )
    weights.to_csv(cfg.output_weights_csv, index=True)

    print(f"  -> Export: {cfg.output_returns_csv}")
    print(f"  -> Export: {cfg.output_weights_csv}")


def main() -> None:
    cfg = PortfolioConfig(
        # Tu peux changer w_sat ici : 0.20 ou 0.30
        w_sat=0.30,
        target_vol=0.10,
        vol_lookback_months=12,
        scale_min=0.50,
        scale_max=2.0,
    )

    print("[1/5] Lecture des CSV Core & Satellite...")
    core = lire_serie_returns(cfg.core_csv, "core_portfolio_return")
    sat = lire_serie_returns(cfg.sat_csv, "satellite_portfolio_return")

    core, sat = aligner_series(core, sat)
    print(f"  -> Période commune: {core.index.min().date()} à {core.index.max().date()} | Nb mois: {len(core)}")

    if cfg.w_sat > cfg.w_sat_max:
        raise ValueError(f"w_sat={cfg.w_sat} dépasse le max autorisé {cfg.w_sat_max}.")

    print("[2/5] Construction portefeuille brut (avant vol targeting)...")
    brut = construire_portefeuille_brut(core, sat, cfg.w_sat)

    print("[3/5] Application du volatility targeting...")
    port, scale_lag = appliquer_vol_targeting(
        returns_brut=brut,
        target_vol=cfg.target_vol,
        lookback_months=cfg.vol_lookback_months,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    )

    print("[4/5] Métriques portefeuille (après vol targeting)...")
    mets = metriques_base(port)
    for k, v in mets.items():
        if "Nombre" in k:
            print(f"{k}: {v:.0f}")
        else:
            print(f"{k}: {v:.4f}")

    print("[5/5] Sauvegarde outputs...")
    sauvegarder_outputs(cfg, port, scale_lag)
    print("Terminé")


if __name__ == "__main__":
    main()