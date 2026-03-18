"""
Moteur de portefeuille Core + Satellite avec volatility targeting (mensuel).

Modes Core disponibles :
- optimized : utilise outputs/core_returns_monthly.csv (core_portfolio_return)
- policy    : utilise outputs/core3_etf_returns_monthly.csv (3 ETF) + poids fixes

Entrées (outputs/) :
- core_returns_monthly.csv : core_portfolio_return (mode optimized)
- core3_etf_returns_monthly.csv : 3 colonnes (mode policy)
- satellite_returns_monthly.csv : satellite_portfolio_return

Sorties :
- outputs/portfolio_returns_monthly.csv
- outputs/portfolio_weights_monthly.csv
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
    project_root: Path = Path(__file__).resolve().parent.parent
    input_dir: Path = project_root / "outputs"
    output_dir: Path = project_root / "outputs"

    # Choix Core
    core_mode: str = "policy"  # "optimized" ou "policy"

    # Fichiers Core
    core_optimized_csv: Path = input_dir / "core_returns_monthly.csv"
    core3_etf_csv: Path = input_dir / "core3_etf_returns_monthly.csv"

    # Satellite
    sat_csv: Path = input_dir / "satellite_returns_monthly.csv"

    # Policy weights (ordre = colonnes du core3_etf_csv)
    # Ici : [Equity, Rates, Credit] = [SWDA, CBE3, ICOV]
    policy_w_equity: float = 0.60
    policy_w_rates: float = 0.20
    policy_w_credit: float = 0.20

    # Construction Core/Satellite
    w_sat: float = 0.30
    w_sat_max: float = 0.30

    # Vol targeting
    target_vol: float = 0.10
    vol_lookback_months: int = 12
    scale_min: float = 0.50
    scale_max: float = 2.00

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
    return s.sort_index()


def lire_core_policy(csv_path: Path, w_eq: float, w_rt: float, w_cr: float) -> pd.Series:
    """
    Construit un Core policy à partir des rendements mensuels des 3 ETF.
    On suppose que le CSV contient 3 colonnes dans l'ordre :
    Equity / Rates / Credit (ex: SWDA, CBE3, ICOV).
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df.sort_index().dropna(how="any")

    if df.shape[1] != 3:
        raise ValueError(f"core3_etf_returns_monthly.csv doit contenir 3 colonnes, trouvé {df.shape[1]}.")

    weights = np.array([w_eq, w_rt, w_cr], dtype=float)
    if abs(weights.sum() - 1.0) > 1e-9:
        raise ValueError("Les poids policy doivent sommer à 1.")

    core = df.values @ weights
    s = pd.Series(core, index=df.index, name="core_portfolio_return_policy")
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


def appliquer_vol_targeting(
    returns_brut: pd.Series,
    target_vol: float,
    lookback_months: int,
    scale_min: float,
    scale_max: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Scaling mensuel pour viser target_vol (annualisé).
    - scale_t calculé à t est appliqué à t+1 (shift) pour éviter le look-ahead.
    """
    rolling_vol = returns_brut.rolling(lookback_months).std() * np.sqrt(12)

    scale = target_vol / rolling_vol.replace(0.0, np.nan)
    scale = scale.clip(lower=scale_min, upper=scale_max)

    scale_lag = scale.shift(1)

    returns_scaled = (scale_lag * returns_brut).dropna()
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
    w_core_eff: pd.Series | None = None,
    w_sat_eff: pd.Series | None = None,
) -> None:
    """Sauvegarde rendements et poids effectifs après scaling asymétrique."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    returns_port.to_frame("portfolio_return").to_csv(cfg.output_returns_csv, index=True)

    # Poids effectifs : fournis explicitement (vol targeting asymétrique)
    # ou calculés symétriquement par défaut pour compatibilité descendante.
    if w_core_eff is None:
        w_core_eff = (1.0 - cfg.w_sat) * scale_lag
    if w_sat_eff is None:
        w_sat_eff = pd.Series(cfg.w_sat, index=scale_lag.index)

    weights = pd.DataFrame(
        {"w_core_effectif": w_core_eff, "w_sat_effectif": w_sat_eff, "scale": scale_lag},
        index=returns_port.index,
    )
    weights.to_csv(cfg.output_weights_csv, index=True)

    print(f"  -> Export: {cfg.output_returns_csv}")
    print(f"  -> Export: {cfg.output_weights_csv}")


def main() -> None:
    cfg = PortfolioConfig(
        core_mode="policy",          # <-- policy par défaut (calibration prof)
        policy_w_equity=0.60,
        policy_w_rates=0.20,
        policy_w_credit=0.20,
        w_sat=0.30,
        target_vol=0.10,
        vol_lookback_months=12,
        scale_min=0.50,
        scale_max=2.00,
    )

    print("[1/6] Lecture des CSV Core & Satellite...")
    if cfg.core_mode == "optimized":
        core = lire_serie_returns(cfg.core_optimized_csv, "core_portfolio_return")
        print("  -> Core mode: optimized (core_portfolio_return)")
    elif cfg.core_mode == "policy":
        core = lire_core_policy(cfg.core3_etf_csv, cfg.policy_w_equity, cfg.policy_w_rates, cfg.policy_w_credit)
        print("  -> Core mode: policy (weights fixes)")
        print(f"     weights policy = [{cfg.policy_w_equity:.0%}, {cfg.policy_w_rates:.0%}, {cfg.policy_w_credit:.0%}]")
    else:
        raise ValueError("core_mode doit être 'optimized' ou 'policy'.")

    sat = lire_serie_returns(cfg.sat_csv, "satellite_portfolio_return")

    core, sat = aligner_series(core, sat)
    print(f"  -> Période commune: {core.index.min().date()} à {core.index.max().date()} | Nb mois: {len(core)}")

    if cfg.w_sat > cfg.w_sat_max:
        raise ValueError(f"w_sat={cfg.w_sat} dépasse le max autorisé {cfg.w_sat_max}.")

    print("[2/6] Construction portefeuille brut (avant vol targeting)...")
    brut = construire_portefeuille_brut(core, sat, cfg.w_sat)
    vol_brut = brut.std() * np.sqrt(12)
    print(f"  -> Volatilité brute (avant targeting): {vol_brut:.4f}")

    print("[3/6] Application du volatility targeting asymétrique...")
    # Calcul du scale_lag à partir de la volatilité du portefeuille brut (signal de risk)
    _, scale_lag = appliquer_vol_targeting(
        returns_brut=brut,
        target_vol=cfg.target_vol,
        lookback_months=cfg.vol_lookback_months,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    )

    # Vol targeting asymétrique : ne réduit que le Core pour préserver la
    # décorrélation structurelle du Satellite
    w_core_base = 1.0 - cfg.w_sat
    w_core_eff = w_core_base * scale_lag          # Core scalé par le signal de vol
    w_sat_eff  = pd.Series(cfg.w_sat, index=scale_lag.index)  # Satellite fixe

    # Renormaliser si l'exposition totale dépasse 100 % (cas scale_lag > 1)
    total_eff = w_core_eff + w_sat_eff
    norm_mask = total_eff > 1.0
    w_core_eff[norm_mask] = w_core_eff[norm_mask] / total_eff[norm_mask]
    w_sat_eff[norm_mask]  = w_sat_eff[norm_mask]  / total_eff[norm_mask]

    # Rendements du portefeuille avec poids effectifs asymétriques
    core_aligned = core.reindex(scale_lag.index)
    sat_aligned  = sat.reindex(scale_lag.index)
    port = (w_core_eff * core_aligned + w_sat_eff * sat_aligned).dropna()
    scale_lag  = scale_lag.reindex(port.index)
    w_core_eff = w_core_eff.reindex(port.index)
    w_sat_eff  = w_sat_eff.reindex(port.index)

    print(f"  -> Scale moyen: {scale_lag.mean():.3f}")
    print(f"  -> Scale min observé: {scale_lag.min():.3f}")
    print(f"  -> Scale max observé: {scale_lag.max():.3f}")
    print(f"  -> % du temps au plancher: {(scale_lag <= cfg.scale_min + 1e-9).mean():.1%}")
    print(f"  -> % du temps au plafond: {(scale_lag >= cfg.scale_max - 1e-9).mean():.1%}")

    print("[4/6] Métriques portefeuille (après vol targeting)...")
    mets = metriques_base(port)
    for k, v in mets.items():
        if "Nombre" in k:
            print(f"{k}: {v:.0f}")
        else:
            print(f"{k}: {v:.4f}")

    print("[5/6] Sauvegarde outputs...")
    sauvegarder_outputs(cfg, port, scale_lag, w_core_eff=w_core_eff, w_sat_eff=w_sat_eff)

    print("[6/6] Terminé")


if __name__ == "__main__":
    main()