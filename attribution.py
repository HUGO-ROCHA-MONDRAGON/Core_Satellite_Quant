"""
Attribution : estimation de l'alpha et du bêta de la poche Satellite vs Core.

Conforme au sujet :
- Estimer l'alpha généré par la poche Satellite
- Suivre le bêta au cours du temps (rolling)

Méthode :
- Régression (mensuelle) : R_sat,t = alpha + beta * R_core,t + eps_t
- Rolling window (par défaut 36 mois)
- Alpha annualisé : (1 + alpha_mensuel)^12 - 1

Entrées :
- outputs/core_returns_monthly.csv : colonne 'core_portfolio_return'
- outputs/satellite_returns_monthly.csv : colonne 'satellite_portfolio_return'

Sorties :
- outputs/attribution_rolling.csv : alpha/beta rolling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttributionConfig:
    """Configuration attribution."""
    project_root: Path = Path(__file__).resolve().parent
    input_dir: Path = project_root / "outputs"
    output_dir: Path = project_root / "outputs"

    core_csv: Path = input_dir / "core_returns_monthly.csv"
    sat_csv: Path = input_dir / "satellite_returns_monthly.csv"

    window_months: int = 36  # rolling window (3 ans)

    output_csv: Path = output_dir / "attribution_rolling.csv"


def lire_serie(csv_path: Path, colname: str) -> pd.Series:
    """Lit une série de rendements depuis un CSV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if colname not in df.columns:
        raise ValueError(f"Colonne '{colname}' introuvable dans {csv_path}. Colonnes: {list(df.columns)}")
    s = df[colname].copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    return s.sort_index()


def aligner(core: pd.Series, sat: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Aligne Core et Satellite sur les dates communes."""
    idx = core.index.intersection(sat.index)
    core2 = core.reindex(idx).dropna()
    sat2 = sat.reindex(idx).dropna()
    idx2 = core2.index.intersection(sat2.index)
    return core2.reindex(idx2), sat2.reindex(idx2)


def regression_alpha_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    Régression OLS simple y = alpha + beta * x.

    Retourne :
    - alpha (mensuel)
    - beta
    """
    # X = [1, x]
    X = np.column_stack([np.ones_like(x), x])
    # beta_hat = (X'X)^(-1) X'y
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = float(coef[0])
    beta = float(coef[1])
    return alpha, beta


def rolling_attribution(
    core: pd.Series,
    sat: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Calcule alpha/beta rolling sur fenêtre fixe (en mois)."""
    alphas = []
    betas = []
    dates = []

    core_vals = core.values
    sat_vals = sat.values
    idx = core.index

    for i in range(window - 1, len(idx)):
        sl = slice(i - window + 1, i + 1)
        x = core_vals[sl]
        y = sat_vals[sl]

        alpha_m, beta = regression_alpha_beta(y, x)

        alphas.append(alpha_m)
        betas.append(beta)
        dates.append(idx[i])

    out = pd.DataFrame(
        {
            "alpha_monthly": alphas,
            "alpha_annualized": (1.0 + np.array(alphas)) ** 12 - 1.0,
            "beta": betas,
        },
        index=pd.DatetimeIndex(dates),
    )
    out.index.name = "Date"
    return out


def alpha_beta_global(core: pd.Series, sat: pd.Series) -> Tuple[float, float]:
    """Alpha/bêta sur toute la période (mensuel)."""
    alpha_m, beta = regression_alpha_beta(sat.values, core.values)
    return alpha_m, beta


def main() -> None:
    cfg = AttributionConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Lecture Core & Satellite (mensuel brut)...")
    core = lire_serie(cfg.core_csv, "core_portfolio_return")
    sat = lire_serie(cfg.sat_csv, "satellite_portfolio_return")
    core, sat = aligner(core, sat)
    print(f"  -> Période commune: {core.index.min().date()} à {core.index.max().date()} | Nb mois: {len(core)}")

    print("[2/4] Attribution globale (OLS sur toute la période)...")
    alpha_m, beta = alpha_beta_global(core, sat)
    alpha_a = (1.0 + alpha_m) ** 12 - 1.0
    print(f"  -> Alpha mensuel: {alpha_m:.5f}")
    print(f"  -> Alpha annualisé: {alpha_a:.4%}")
    print(f"  -> Bêta: {beta:.4f}")

    print("[3/4] Attribution rolling...")
    rolling = rolling_attribution(core, sat, cfg.window_months)
    print(f"  -> Rolling window: {cfg.window_months} mois")
    print(f"  -> Début rolling: {rolling.index.min().date()} | Fin rolling: {rolling.index.max().date()}")

    print("[4/4] Export CSV...")
    rolling.to_csv(cfg.output_csv, index=True)
    print(f"  -> Export: {cfg.output_csv}")
    print("Terminé")


if __name__ == "__main__":
    main()