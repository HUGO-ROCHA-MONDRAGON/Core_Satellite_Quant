"""
attribution.py
==============

Attribution alpha / bêta cohérente avec le pipeline final Core-Satellite.

Logique retenue
---------------
Ce script est aligné avec :
- core_pipeline_corrected.py
- satellite_pipeline_corrected.py
- fond_construction_corrected.py

Il lit directement le fichier final de backtest :
    outputs/fond_returns_daily.csv

Colonnes attendues :
- core_ret
- sat_pocket_ret
- portfolio_ret

Objectifs
---------
1. Estimer l'alpha et le bêta globaux :
   - de la poche Satellite vs Core
   - du portefeuille total vs Core

2. Calculer les versions rolling :
   - alpha rolling 63 jours
   - bêta rolling 63 jours

3. Exporter des fichiers cohérents avec le projet :
   - outputs/attribution_summary.csv
   - outputs/attribution_rolling_daily.csv

Convention
----------
Régression OLS simple :
    y_t = alpha_daily + beta * x_t + eps_t

avec :
- y_t = rendement Satellite (ou portefeuille)
- x_t = rendement Core

Annualisation de l'alpha :
    alpha_ann = (1 + alpha_daily)^252 - 1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AttributionConfig:
    """Configuration du module d'attribution."""

    input_csv: Path = PROJECT_ROOT / "outputs" / "fond_returns_daily.csv"
    output_summary_csv: Path = PROJECT_ROOT / "outputs" / "attribution_summary.csv"
    output_rolling_csv: Path = PROJECT_ROOT / "outputs" / "attribution_rolling_daily.csv"

    rolling_window_days: int = 63
    min_obs_global: int = 30
    min_obs_rolling: int = 30


def lire_backtest_final(path: Path) -> pd.DataFrame:
    """Lit le fichier final de backtest journalier."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df.sort_index()

    required_cols = ["core_ret", "sat_pocket_ret", "portfolio_ret"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path.name} : {missing}. "
            f"Colonnes trouvées : {list(df.columns)}"
        )

    return df[required_cols].dropna(how="all")


def aligner_series(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Aligne deux séries sur leurs dates communes."""
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    return data["x"], data["y"]


def regression_alpha_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float]:
    """
    Régression OLS simple : y = alpha + beta * x + eps.

    Retourne :
    - alpha_daily
    - beta
    - r_squared
    """
    if len(x) != len(y):
        raise ValueError("x et y doivent avoir la même longueur.")

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    x_centered = x - x_mean
    y_centered = y - y_mean

    ss_x = float(np.sum(x_centered ** 2))
    if ss_x <= 1e-14:
        return np.nan, np.nan, np.nan

    beta = float(np.sum(x_centered * y_centered) / ss_x)
    alpha = float(y_mean - beta * x_mean)

    y_hat = alpha + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))

    if ss_tot <= 1e-14:
        r_squared = np.nan
    else:
        r_squared = 1.0 - ss_res / ss_tot

    return alpha, beta, r_squared


def annualiser_alpha_daily(alpha_daily: float) -> float:
    """Annualise un alpha journalier."""
    if pd.isna(alpha_daily):
        return np.nan
    return float((1.0 + alpha_daily) ** 252 - 1.0)


def annualiser_retour_journalier(r: pd.Series) -> float:
    """Rendement annualisé à partir de rendements journaliers."""
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def annualiser_vol_journaliere(r: pd.Series) -> float:
    """Volatilité annualisée à partir de rendements journaliers."""
    if len(r) == 0:
        return np.nan
    return float(r.std() * np.sqrt(252.0))


def construire_resume_pair(
    y: pd.Series,
    x: pd.Series,
    label: str,
    min_obs: int,
) -> dict:
    """Construit le résumé global alpha / bêta pour une paire (y, x)."""
    x_aligned, y_aligned = aligner_series(x, y)

    if len(x_aligned) < min_obs:
        raise ValueError(
            f"Observations insuffisantes pour {label}: "
            f"{len(x_aligned)} < {min_obs}"
        )

    alpha_daily, beta, r2 = regression_alpha_beta(
        y_aligned.values,
        x_aligned.values,
    )

    corr = float(y_aligned.corr(x_aligned))
    ret_ann = annualiser_retour_journalier(y_aligned)
    vol_ann = annualiser_vol_journaliere(y_aligned)

    return {
        "serie": label,
        "n_obs": len(y_aligned),
        "ret_ann": ret_ann,
        "vol_ann": vol_ann,
        "corr_vs_core": corr,
        "alpha_daily": alpha_daily,
        "alpha_ann": annualiser_alpha_daily(alpha_daily),
        "beta": beta,
        "r_squared": r2,
    }


def rolling_attribution(
    y: pd.Series,
    x: pd.Series,
    window: int,
    min_obs: int,
    prefix: str,
) -> pd.DataFrame:
    """Calcule alpha / bêta rolling sur fenêtre fixe."""
    x_aligned, y_aligned = aligner_series(x, y)

    alphas_daily = []
    alphas_ann = []
    betas = []
    r2_list = []
    dates = []

    for i in range(window - 1, len(x_aligned)):
        x_win = x_aligned.iloc[i - window + 1 : i + 1]
        y_win = y_aligned.iloc[i - window + 1 : i + 1]

        valid = pd.concat([x_win, y_win], axis=1).dropna()
        if len(valid) < min_obs:
            continue

        x_vals = valid.iloc[:, 0].values
        y_vals = valid.iloc[:, 1].values

        alpha_daily, beta, r2 = regression_alpha_beta(y_vals, x_vals)

        dates.append(valid.index[-1])
        alphas_daily.append(alpha_daily)
        alphas_ann.append(annualiser_alpha_daily(alpha_daily))
        betas.append(beta)
        r2_list.append(r2)

    out = pd.DataFrame(
        {
            f"{prefix}_alpha_daily": alphas_daily,
            f"{prefix}_alpha_ann": alphas_ann,
            f"{prefix}_beta": betas,
            f"{prefix}_r_squared": r2_list,
        },
        index=pd.DatetimeIndex(dates),
    )
    out.index.name = "Date"
    return out


def main() -> None:
    """Exécute l'attribution finale cohérente avec le projet."""
    cfg = AttributionConfig()

    print("=" * 68)
    print("  ATTRIBUTION FINALE – ALPHA / BÊTA")
    print("  Cohérente avec fond_construction_corrected.py")
    print("=" * 68)

    print("\n[1] Lecture du backtest final...")
    df = lire_backtest_final(cfg.input_csv)
    print(
        f"  -> {len(df)} observations | "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )

    core = df["core_ret"]
    sat = df["sat_pocket_ret"]
    port = df["portfolio_ret"]

    print("\n[2] Attribution globale...")
    sat_summary = construire_resume_pair(
        y=sat,
        x=core,
        label="Satellite_vs_Core",
        min_obs=cfg.min_obs_global,
    )
    port_summary = construire_resume_pair(
        y=port,
        x=core,
        label="Portfolio_vs_Core",
        min_obs=cfg.min_obs_global,
    )

    summary_df = pd.DataFrame([sat_summary, port_summary])

    print("\n  Résultats globaux :")
    for _, row in summary_df.iterrows():
        print(f"\n  {row['serie']}")
        print(f"    n_obs          : {int(row['n_obs'])}")
        print(f"    Rendement ann. : {row['ret_ann']:+.2%}")
        print(f"    Volatilité ann.: {row['vol_ann']:.2%}")
        print(f"    Corrélation    : {row['corr_vs_core']:+.3f}")
        print(f"    Alpha daily    : {row['alpha_daily']:+.6f}")
        print(f"    Alpha ann.     : {row['alpha_ann']:+.2%}")
        print(f"    Bêta           : {row['beta']:+.3f}")
        print(f"    R²             : {row['r_squared']:.3f}")

    print("\n[3] Attribution rolling...")
    sat_roll = rolling_attribution(
        y=sat,
        x=core,
        window=cfg.rolling_window_days,
        min_obs=cfg.min_obs_rolling,
        prefix="satellite",
    )
    port_roll = rolling_attribution(
        y=port,
        x=core,
        window=cfg.rolling_window_days,
        min_obs=cfg.min_obs_rolling,
        prefix="portfolio",
    )

    rolling_df = pd.concat([sat_roll, port_roll], axis=1).sort_index()

    if rolling_df.empty:
        print("  ⚠ Aucun résultat rolling exploitable.")
    else:
        print(
            f"  -> Fenêtre rolling : {cfg.rolling_window_days} jours"
        )
        print(
            f"  -> Série rolling   : {rolling_df.index.min().date()} "
            f"→ {rolling_df.index.max().date()}"
        )

        sat_beta_mean = rolling_df["satellite_beta"].dropna().mean()
        sat_alpha_mean = rolling_df["satellite_alpha_ann"].dropna().mean()

        print(f"  -> Beta rolling moyen Satellite : {sat_beta_mean:+.3f}")
        print(f"  -> Alpha rolling ann. moyen     : {sat_alpha_mean:+.2%}")

    print("\n[4] Exports...")
    cfg.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(cfg.output_summary_csv, index=False)
    rolling_df.to_csv(cfg.output_rolling_csv, index=True)

    print(f"  -> {cfg.output_summary_csv}")
    print(f"  -> {cfg.output_rolling_csv}")

    print("\n  ✓ Attribution terminée.")


if __name__ == "__main__":
    main()