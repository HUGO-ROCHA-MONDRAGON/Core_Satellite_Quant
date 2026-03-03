"""
Génération des graphiques pour la soutenance (Core/Satellite).

Entrées (outputs/) :
- core_returns_monthly.csv : core_portfolio_return
- portfolio_returns_monthly.csv : portfolio_return
- attribution_rolling.csv : alpha_annualized, beta

Sorties :
- outputs/figures/01_cum_performance.png
- outputs/figures/02_drawdown_portfolio.png
- outputs/figures/03_beta_rolling.png
- outputs/figures/04_alpha_rolling_annualized.png

Remarque :
- Graphiques simples, lisibles, sans styles exotiques.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotConfig:
    """Configuration des chemins et options."""
    project_root: Path = Path(__file__).resolve().parent
    input_dir: Path = project_root / "outputs"
    fig_dir: Path = input_dir / "figures"

    core_csv: Path = input_dir / "core_returns_monthly.csv"
    portfolio_csv: Path = input_dir / "portfolio_returns_monthly.csv"
    attribution_csv: Path = input_dir / "attribution_rolling.csv"
    weights_csv: Path = input_dir / "portfolio_weights_monthly.csv"

    dpi: int = 160


def lire_serie(csv_path: Path, col: str) -> pd.Series:
    """Lit une série depuis CSV (dates en index)."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if col not in df.columns:
        raise ValueError(f"Colonne '{col}' introuvable dans {csv_path}. Colonnes: {list(df.columns)}")
    s = df[col].copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    return s.sort_index()


def cumuler(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Transforme une série de rendements en valeur cumulée."""
    return base * (1.0 + returns).cumprod()


def drawdown_from_returns(returns: pd.Series) -> pd.Series:
    """Calcule le drawdown à partir de rendements."""
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd


def align_two(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Aligne deux séries sur les dates communes."""
    idx = a.index.intersection(b.index)
    a2 = a.reindex(idx).dropna()
    b2 = b.reindex(idx).dropna()
    idx2 = a2.index.intersection(b2.index)
    return a2.reindex(idx2), b2.reindex(idx2)


def save_fig(path: Path, dpi: int) -> None:
    """Sauvegarde la figure courante proprement."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_cum_performance(cfg: PlotConfig, core: pd.Series, port: pd.Series) -> None:
    """Courbe de performance cumulée : portefeuille vs core."""
    core_a, port_a = align_two(core, port)

    core_c = cumuler(core_a, base=100.0)
    port_c = cumuler(port_a, base=100.0)

    plt.figure()
    plt.plot(core_c.index, core_c.values, label="Core (cumulé)")
    plt.plot(port_c.index, port_c.values, label="Portefeuille (ciblé, cumulé)")
    plt.title("Performance cumulée (base 100)")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend()

    out = cfg.fig_dir / "01_cum_performance.png"
    save_fig(out, cfg.dpi)
    print(f"  -> Figure: {out}")


def plot_drawdown(cfg: PlotConfig, port: pd.Series) -> None:
    """Drawdown du portefeuille."""
    dd = drawdown_from_returns(port)

    plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown du portefeuille (après vol targeting)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    out = cfg.fig_dir / "02_drawdown_portfolio.png"
    save_fig(out, cfg.dpi)
    print(f"  -> Figure: {out}")


def plot_beta_rolling(cfg: PlotConfig, attribution: pd.DataFrame) -> None:
    """Bêta rolling."""
    plt.figure()
    plt.plot(attribution.index, attribution["beta"].values)
    plt.title("Bêta rolling (36 mois) : Satellite vs Core")
    plt.xlabel("Date")
    plt.ylabel("Bêta")

    out = cfg.fig_dir / "03_beta_rolling.png"
    save_fig(out, cfg.dpi)
    print(f"  -> Figure: {out}")


def plot_alpha_rolling(cfg: PlotConfig, attribution: pd.DataFrame) -> None:
    """Alpha rolling annualisé."""
    plt.figure()
    plt.plot(attribution.index, attribution["alpha_annualized"].values)
    plt.title("Alpha rolling annualisé (36 mois) : Satellite vs Core")
    plt.xlabel("Date")
    plt.ylabel("Alpha annualisé")

    out = cfg.fig_dir / "04_alpha_rolling_annualized.png"
    save_fig(out, cfg.dpi)
    print(f"  -> Figure: {out}")


def plot_scale(cfg: PlotConfig, weights: pd.DataFrame) -> None:
    """Graphique du scaling (vol targeting)."""
    if "scale" not in weights.columns:
        raise ValueError("Colonne 'scale' manquante dans portfolio_weights_monthly.csv")

    s = weights["scale"].copy().dropna()

    plt.figure()
    plt.plot(s.index, s.values)
    plt.title("Volatility targeting : facteur d'exposition (scale)")
    plt.xlabel("Date")
    plt.ylabel("Scale")

    out = cfg.fig_dir / "05_scale_vol_targeting.png"
    save_fig(out, cfg.dpi)
    print(f"  -> Figure: {out}")


def main() -> None:
    cfg = PlotConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Lecture des données...")
    core = lire_serie(cfg.core_csv, "core_portfolio_return")
    port = lire_serie(cfg.portfolio_csv, "portfolio_return")

    attrib = pd.read_csv(cfg.attribution_csv, index_col=0, parse_dates=True)
    attrib.index = pd.DatetimeIndex(attrib.index).tz_localize(None)
    attrib = attrib.sort_index()

    # Vérifications colonnes
    for col in ("alpha_annualized", "beta"):
        if col not in attrib.columns:
            raise ValueError(f"Colonne '{col}' manquante dans {cfg.attribution_csv}.")

    weights = pd.read_csv(cfg.weights_csv, index_col=0, parse_dates=True)
    weights.index = pd.DatetimeIndex(weights.index).tz_localize(None)
    weights = weights.sort_index()

    print("[2/3] Génération des figures...")
    plot_cum_performance(cfg, core, port)
    plot_drawdown(cfg, port)
    plot_beta_rolling(cfg, attrib)
    plot_alpha_rolling(cfg, attrib)
    plot_scale(cfg, weights)

    print("[3/3] Terminé")
    scale = weights["scale"].dropna()
    print(f"  -> Scale moyen: {scale.mean():.3f}")
    print(f"  -> Scale max: {scale.max():.3f}")
    print(f"  -> % du temps au plafond (approx): {(scale >= scale.max() - 1e-9).mean():.1%}")


if __name__ == "__main__":
    main()