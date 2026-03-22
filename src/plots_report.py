"""
plot_reports.py – Génération des graphiques Core-Satellite (daily).

Version corrigée et alignée avec :
- core_pipeline_corrected.py
- satellite_pipeline_corrected.py
- fond_construction_corrected.py

Sections :
  A. Poche Core
  B. Poche Satellite
  C. Core vs Satellite
  D. Portefeuille total

Entrées (outputs/) :
  - core_returns_daily_oos.csv
  - core3_etf_daily_simple_returns.csv
  - core3_etf_daily_log_returns.csv (fallback)
  - Core_finaux.csv
  - fond_returns_daily.csv
  - fond_weights.csv
  - fond_metrics.csv
  - fond_annual_perf.csv
  - fond_beta_rolling.csv
  - satellite_selected.csv
  - satellite_individual_returns.csv

Sorties :
  - outputs/figures/*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


matplotlib.use("Agg")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PlotConfig:
    """Configuration des graphiques du rapport."""

    fig_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    dpi: int = 150

    core_oos_csv: Path = PROJECT_ROOT / "outputs" / "core_returns_daily_oos.csv"
    core_etf_simple_csv: Path = (
        PROJECT_ROOT / "outputs" / "core3_etf_daily_simple_returns.csv"
    )
    core_etf_log_csv: Path = (
        PROJECT_ROOT / "outputs" / "core3_etf_daily_log_returns.csv"
    )
    core_finaux_csv: Path = PROJECT_ROOT / "outputs" / "Core_finaux.csv"

    fond_returns_csv: Path = PROJECT_ROOT / "outputs" / "fond_returns_daily.csv"
    fond_weights_csv: Path = PROJECT_ROOT / "outputs" / "fond_weights.csv"
    fond_metrics_csv: Path = PROJECT_ROOT / "outputs" / "fond_metrics.csv"
    fond_annual_csv: Path = PROJECT_ROOT / "outputs" / "fond_annual_perf.csv"
    fond_beta_roll_csv: Path = PROJECT_ROOT / "outputs" / "fond_beta_rolling.csv"

    sat_selected_csv: Path = PROJECT_ROOT / "outputs" / "satellite_selected.csv"
    sat_indiv_csv: Path = (
        PROJECT_ROOT / "outputs" / "satellite_individual_returns.csv"
    )

    core_meta_excel: Path = (
        PROJECT_ROOT / "data" / "univers_core_etf_eur_daily_wide.xlsx"
    )

    vol_target_min: float = 0.08
    vol_target_max: float = 0.12
    beta_target_abs: float = 0.25


BLOC_COLORS = {
    "Bloc1": "#e6194b",
    "Bloc2": "#3cb44b",
    "Bloc3": "#4363d8",
}
ETF_COLORS = ["#e6194b", "#3cb44b", "#4363d8"]


def _cum(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Calcule la performance cumulée base 100."""
    return base * (1.0 + returns).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    """Calcule le drawdown courant."""
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def _roll_vol(returns: pd.Series, window: int = 63) -> pd.Series:
    """Volatilité rolling annualisée."""
    return returns.rolling(window).std() * np.sqrt(252)


def _roll_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Sharpe rolling annualisé."""
    roll_ret = returns.rolling(window).mean() * 252
    roll_vol = returns.rolling(window).std() * np.sqrt(252)
    return roll_ret / roll_vol.replace(0.0, np.nan)


def _roll_alpha_beta(
    y: pd.Series,
    x: pd.Series,
    window: int = 63,
) -> tuple[pd.Series, pd.Series]:
    """Calcule alpha et bêta rolling."""
    cov_ = y.rolling(window).cov(x)
    var_ = x.rolling(window).var()
    beta = cov_ / var_.replace(0.0, np.nan)
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    alpha_ann = (1.0 + alpha) ** 252 - 1.0
    return alpha_ann.rename("alpha_ann_roll"), beta.rename("beta_roll")


def _save(fig_dir: Path, name: str, dpi: int) -> None:
    """Sauvegarde la figure courante."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"  -> {fig_dir / name}")


def _read_returns_csv(path: Path) -> pd.Series:
    """Lit un CSV de rendements en détectant log/simple."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"CSV vide : {path}")

    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").dropna().sort_index()

    is_log = "log" in col.lower() or "log" in path.stem.lower()
    if is_log:
        series = np.exp(series) - 1.0

    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    return series


def _read_dataframe_returns(path: Path, is_log: bool) -> pd.DataFrame:
    """Lit un DataFrame de rendements simples ou log."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df.apply(pd.to_numeric, errors="coerce")

    if is_log:
        df = np.exp(df) - 1.0

    return df


def _load_core_etf_returns(cfg: PlotConfig) -> pd.DataFrame:
    """Charge les rendements simples des 3 ETF Core."""
    if cfg.core_etf_simple_csv.exists():
        return _read_dataframe_returns(cfg.core_etf_simple_csv, is_log=False)

    if cfg.core_etf_log_csv.exists():
        return _read_dataframe_returns(cfg.core_etf_log_csv, is_log=True)

    raise FileNotFoundError(
        "Aucun fichier des 3 ETF Core trouvé "
        "(simple ou log returns)."
    )


def _load_core_fees_bps(core_finaux: pd.DataFrame, cfg: PlotConfig) -> Dict[str, float]:
    """Charge les TER des ETF Core depuis le fichier metadata."""
    fee_map: Dict[str, float] = {}
    if not cfg.core_meta_excel.exists():
        return fee_map

    for sheet in ["Equity", "Rates", "Credit"]:
        try:
            df = pd.read_excel(cfg.core_meta_excel, sheet_name=sheet, header=None)
            headers = df.iloc[4].tolist()
            data = df.iloc[5:]

            ticker_idx = None
            ter_idx = None
            for i, header in enumerate(headers):
                text = str(header).strip().lower() if pd.notna(header) else ""
                if "bloomberg" in text:
                    ticker_idx = i
                elif "ter" in text:
                    ter_idx = i

            if ticker_idx is None or ter_idx is None:
                continue

            for _, row in data.iterrows():
                ticker = (
                    str(row[ticker_idx]).strip() if pd.notna(row[ticker_idx]) else ""
                )
                ter = pd.to_numeric(row[ter_idx], errors="coerce")
                if not ticker or pd.isna(ter):
                    continue

                ter_value = float(ter)
                ter_pct = ter_value * 100.0 if ter_value < 0.05 else ter_value
                fee_map[ticker] = ter_pct * 100.0
        except Exception:
            continue

    return fee_map


# ══════════════════════════════════════════════════════════════════════
# Section A — Poche Core
# ══════════════════════════════════════════════════════════════════════

def plot_A01_core_etf_cum(
    core_etf_rets: pd.DataFrame,
    core_finaux: pd.DataFrame,
    cfg: PlotConfig,
) -> None:
    """A01 – Performance cumulée des 3 ETF Core individuels."""
    oos = core_etf_rets.loc["2021":"2025"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(oos.columns):
        theme = core_finaux.loc[core_finaux["Ticker"] == col, "Theme"].values
        label = f"{col} [{theme[0] if len(theme) else '?'}]"
        _cum(oos[col]).plot(ax=ax, label=label, color=ETF_COLORS[i % 3])

    ax.set_title("A01 – Performance cumulée des 3 ETF Core (OOS 2021-2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "A01_core_etf_cum.png", cfg.dpi)


def plot_A02_core_portfolio_cum(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A02 – Performance cumulée du portefeuille Core."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _cum(r_core).plot(ax=ax, color="#e6194b", lw=2)
    ax.set_title("A02 – Performance cumulée du Core (Risk Parity + Tilt rolling)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    _save(cfg.fig_dir, "A02_core_portfolio_cum.png", cfg.dpi)


def plot_A03_core_drawdown(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A03 – Drawdown du portefeuille Core."""
    dd = _drawdown(r_core).mul(100)

    fig, ax = plt.subplots(figsize=(10, 4))
    dd.plot(ax=ax, color="#e6194b", lw=1.5)
    ax.fill_between(dd.index, dd, 0, alpha=0.25, color="#e6194b")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("A03 – Drawdown du portefeuille Core")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    _save(cfg.fig_dir, "A03_core_drawdown.png", cfg.dpi)


def plot_A04_core_rolling_vol(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A04 – Volatilité rolling du Core."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _roll_vol(r_core).mul(100).plot(ax=ax, color="#e6194b")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("A04 – Volatilité rolling 63j du Core (annualisée)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vol annualisée (%)")
    _save(cfg.fig_dir, "A04_core_rolling_vol.png", cfg.dpi)


def plot_A05_core_rolling_sharpe(r_core: pd.Series, cfg: PlotConfig) -> None:
    """A05 – Sharpe rolling du Core."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _roll_sharpe(r_core).plot(ax=ax, color="#e6194b")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("A05 – Sharpe rolling 63j du Core")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    _save(cfg.fig_dir, "A05_core_rolling_sharpe.png", cfg.dpi)


def plot_A06_core_annual_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """A06 – Performance annuelle du Core."""
    fig, ax = plt.subplots(figsize=(9, 5))
    values = annual["core"].mul(100)
    bars = ax.bar(
        annual.index.astype(str),
        values,
        color=["#e6194b" if v >= 0 else "#888888" for v in values],
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.3 if value >= 0 else -0.8),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    ax.set_title("A06 – Performance annuelle du Core")
    ax.set_xlabel("Année")
    ax.set_ylabel("Rendement (%)")
    _save(cfg.fig_dir, "A06_core_annual_bar.png", cfg.dpi)


def plot_A07_core_fees_bar(core_finaux: pd.DataFrame, cfg: PlotConfig) -> None:
    """A07 – Frais des ETF Core."""
    fee_map = _load_core_fees_bps(core_finaux, cfg)

    tickers = core_finaux["Ticker"].tolist()
    themes = core_finaux["Theme"].tolist()
    fees = [fee_map.get(ticker, 25.0) for ticker in tickers]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(tickers)), fees, color=ETF_COLORS[: len(tickers)])

    for bar, value in zip(bars, fees):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.0f} bps",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(
        [f"{ticker}\n({theme})" for ticker, theme in zip(tickers, themes)],
        fontsize=8,
    )
    ax.set_ylabel("Expense ratio (bps/an)")
    ax.set_title("A07 – Frais des ETF Core")
    _save(cfg.fig_dir, "A07_core_fees_bar.png", cfg.dpi)


# ══════════════════════════════════════════════════════════════════════
# Section B — Poche Satellite
# ══════════════════════════════════════════════════════════════════════

def plot_B01_sat_fund_cum(
    sat_indiv: pd.DataFrame,
    sat_info: pd.DataFrame,
    sat_weights: pd.Series,
    cfg: PlotConfig,
) -> None:
    """B01 – Performance cumulée individuelle des fonds satellite."""
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")

    for i, ticker in enumerate(sat_weights.index):
        if ticker not in sat_indiv.columns:
            continue
        bloc = sat_info.loc[ticker, "bloc"] if ticker in sat_info.index else "?"
        _cum(sat_indiv[ticker]).plot(
            ax=ax,
            label=f"{ticker} [{bloc}]",
            color=cmap(i),
            lw=1.5,
        )

    ax.set_title("B01 – Performance cumulée individuelle des fonds satellite")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=8, ncol=2)
    _save(cfg.fig_dir, "B01_sat_fund_cum.png", cfg.dpi)


def plot_B02_sat_pocket_cum(r_sat: pd.Series, cfg: PlotConfig) -> None:
    """B02 – Performance cumulée de la poche satellite."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _cum(r_sat).plot(ax=ax, color="#3cb44b", lw=2)
    ax.set_title("B02 – Performance cumulée de la poche satellite")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    _save(cfg.fig_dir, "B02_sat_pocket_cum.png", cfg.dpi)


def plot_B03_sat_weights_bar(
    sat_weights: pd.Series,
    sat_info: pd.DataFrame,
    w_sat: float,
    cfg: PlotConfig,
) -> None:
    """B03 – Poids satellite."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    colors = [
        BLOC_COLORS.get(
            sat_info.at[ticker, "bloc"] if ticker in sat_info.index else "Inconnu",
            "#aaaaaa",
        )
        for ticker in sat_weights.index
    ]
    bars = ax.bar(range(len(sat_weights)), sat_weights.mul(100), color=colors)

    for bar, value in zip(bars, sat_weights.mul(100)):
        if value > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.2,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(range(len(sat_weights)))
    ax.set_xticklabels(sat_weights.index, rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("B03a – Poids dans la poche satellite")
    ax.set_ylabel("Poids (%)")

    ax2 = axes[1]
    abs_weights = pd.concat(
        [
            pd.Series({"Core (total)": 1.0 - w_sat}),
            sat_weights.mul(w_sat).rename(lambda x: x.split(" ")[0]),
        ]
    )
    wedge_colors = ["#cccccc"] + [
        BLOC_COLORS.get(
            sat_info.at[ticker, "bloc"] if ticker in sat_info.index else "",
            "#aaaaaa",
        )
        for ticker in sat_weights.index
    ]
    abs_weights.plot.pie(
        ax=ax2,
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax2.set_title("B03b – Allocation totale du portefeuille")
    ax2.set_ylabel("")

    _save(cfg.fig_dir, "B03_sat_weights.png", cfg.dpi)


def plot_B04_sat_rolling_alpha(
    r_sat: pd.Series,
    r_core: pd.Series,
    cfg: PlotConfig,
) -> None:
    """B04 – Alpha rolling de la poche satellite."""
    alpha_roll, _ = _roll_alpha_beta(r_sat, r_core, window=63)

    fig, ax = plt.subplots(figsize=(10, 4))
    alpha_roll.mul(100).plot(ax=ax, color="#3cb44b")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("B04 – Alpha rolling 63j de la poche satellite vs Core")
    ax.set_xlabel("Date")
    ax.set_ylabel("Alpha annualisé (%)")
    _save(cfg.fig_dir, "B04_sat_rolling_alpha.png", cfg.dpi)


def plot_B05_sat_rolling_beta(beta_roll: pd.Series, cfg: PlotConfig) -> None:
    """B05 – Bêta rolling de la poche satellite."""
    fig, ax = plt.subplots(figsize=(10, 4))
    beta_roll.plot(ax=ax, color="#4363d8", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(cfg.beta_target_abs, color="orange", lw=1, ls=":", label="+β max")
    ax.axhline(-cfg.beta_target_abs, color="orange", lw=1, ls=":", label="-β max")
    ax.fill_between(beta_roll.index, beta_roll, 0, alpha=0.15, color="#4363d8")
    ax.set_title("B05 – Beta rolling 63j de la poche satellite vs Core")
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "B05_sat_rolling_beta.png", cfg.dpi)


def plot_B06_sat_perf_annual(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """B06 – Performance annuelle de la poche satellite."""
    fig, ax = plt.subplots(figsize=(9, 5))
    values = annual["satellite"].mul(100)
    bars = ax.bar(
        annual.index.astype(str),
        values,
        color=["#3cb44b" if v >= 0 else "#888888" for v in values],
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.2 if value >= 0 else -0.8),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    ax.set_title("B06 – Performance annuelle de la poche satellite")
    ax.set_xlabel("Année")
    ax.set_ylabel("Rendement (%)")
    _save(cfg.fig_dir, "B06_sat_annual.png", cfg.dpi)


def plot_B07_sat_fees_bar(
    sat_weights: pd.Series,
    sat_info: pd.DataFrame,
    cfg: PlotConfig,
) -> None:
    """B07 – Frais des fonds satellite."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    expense_bps = []
    for ticker in sat_weights.index:
        if ticker in sat_info.index and "expense_pct" in sat_info.columns:
            value = sat_info.at[ticker, "expense_pct"]
        else:
            value = np.nan

        if pd.isna(value):
            expense_bps.append(200.0)
        else:
            expense_bps.append(float(value) * 100.0)

    colors = [
        BLOC_COLORS.get(
            sat_info.at[ticker, "bloc"] if ticker in sat_info.index else "",
            "#aaaaaa",
        )
        for ticker in sat_weights.index
    ]

    ax = axes[0]
    ax.bar(range(len(sat_weights)), expense_bps, color=colors)
    ax.set_xticks(range(len(sat_weights)))
    ax.set_xticklabels(sat_weights.index, rotation=35, ha="right", fontsize=8)
    ax.set_title("B07a – Expense ratio de chaque fonds satellite")
    ax.set_ylabel("bps/an")

    ax2 = axes[1]
    contributions = [w * e for w, e in zip(sat_weights.values, expense_bps)]
    ax2.bar(range(len(sat_weights)), contributions, color=colors)
    ax2.set_xticks(range(len(sat_weights)))
    ax2.set_xticklabels(sat_weights.index, rotation=35, ha="right", fontsize=8)
    ax2.set_title("B07b – Contribution aux frais de la poche satellite")
    ax2.set_ylabel("Contribution (poids × bps)")

    _save(cfg.fig_dir, "B07_sat_fees.png", cfg.dpi)


# ══════════════════════════════════════════════════════════════════════
# Section C — Core vs Satellite
# ══════════════════════════════════════════════════════════════════════

def plot_C01_core_vs_sat_cum(
    r_core: pd.Series,
    r_sat: pd.Series,
    r_port: pd.Series,
    cfg: PlotConfig,
) -> None:
    """C01 – Performance cumulée comparée."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _cum(r_core).plot(ax=ax, label="Core (Risk Parity + Tilt)", color="#e6194b", lw=2)
    _cum(r_sat).plot(ax=ax, label="Satellite", color="#3cb44b", lw=2, ls="--")
    _cum(r_port).plot(ax=ax, label="Portefeuille total", color="#4363d8", lw=2.5)
    ax.set_title("C01 – Performance cumulée : Core vs Satellite vs Portefeuille")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=10)
    _save(cfg.fig_dir, "C01_core_vs_sat_cum.png", cfg.dpi)


def plot_C02_excess_cum(r_port: pd.Series, r_core: pd.Series, cfg: PlotConfig) -> None:
    """C02 – Excès cumulé du portefeuille vs Core."""
    excess = r_port - r_core
    cum_excess = excess.cumsum().mul(100)

    fig, ax = plt.subplots(figsize=(10, 4))
    cum_excess.plot(ax=ax, color="#4363d8", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("C02 – Rendements excédentaires cumulés (portefeuille – core)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Excès cumulé (points de %)")
    _save(cfg.fig_dir, "C02_excess_cum.png", cfg.dpi)


def plot_C04_annual_grouped_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """C04 – Performance annuelle groupée."""
    years = annual.index.astype(str).tolist()
    x = np.arange(len(years))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, annual["core"].mul(100), width, label="Core", color="#e6194b")
    ax.bar(
        x,
        annual["portfolio"].mul(100),
        width,
        label="Portefeuille",
        color="#4363d8",
    )
    ax.bar(
        x + width,
        annual["satellite"].mul(100),
        width,
        label="Satellite",
        color="#3cb44b",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("C04 – Performance annuelle : Core vs Portefeuille vs Satellite")
    ax.set_ylabel("Rendement (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "C04_annual_grouped_bar.png", cfg.dpi)


def plot_C05_excess_annual_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """C05 – Excès annuel du portefeuille vs Core."""
    excess = (annual["portfolio"] - annual["core"]).mul(100)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        excess.index.astype(str),
        excess,
        color=["#3cb44b" if v >= 0 else "#e6194b" for v in excess],
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, value in zip(bars, excess):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.1 if value >= 0 else -0.3),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    ax.set_title("C05 – Rendement excédentaire annuel (portefeuille – core)")
    ax.set_ylabel("Excès (%)")
    _save(cfg.fig_dir, "C05_excess_annual.png", cfg.dpi)


# ══════════════════════════════════════════════════════════════════════
# Section D — Portefeuille total
# ══════════════════════════════════════════════════════════════════════

def plot_D01_portfolio_cum(
    r_port: pd.Series,
    r_core: pd.Series,
    cfg: PlotConfig,
) -> None:
    """D01 – Performance cumulée du portefeuille."""
    fig, ax = plt.subplots(figsize=(11, 6))
    cum_port = _cum(r_port)
    cum_core = _cum(r_core)

    ax.fill_between(
        cum_port.index,
        cum_port,
        cum_core,
        where=cum_port >= cum_core,
        alpha=0.15,
        color="#3cb44b",
    )
    ax.fill_between(
        cum_port.index,
        cum_port,
        cum_core,
        where=cum_port < cum_core,
        alpha=0.15,
        color="#e6194b",
    )

    cum_core.plot(ax=ax, color="#e6194b", lw=1.5, ls="--", label="Core")
    cum_port.plot(ax=ax, color="#4363d8", lw=2.5, label="Portefeuille total")

    ax.set_title("D01 – Performance cumulée du portefeuille vs Core")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=10)
    _save(cfg.fig_dir, "D01_portfolio_cum.png", cfg.dpi)


def plot_D02_portfolio_drawdown(
    r_port: pd.Series,
    r_core: pd.Series,
    cfg: PlotConfig,
) -> None:
    """D02 – Drawdown portefeuille vs Core."""
    dd_core = _drawdown(r_core).mul(100)
    dd_port = _drawdown(r_port).mul(100)

    fig, ax = plt.subplots(figsize=(10, 4))
    dd_core.plot(ax=ax, color="#e6194b", lw=1, ls="--", label="Core")
    dd_port.plot(ax=ax, color="#4363d8", lw=1.5, label="Portefeuille")
    ax.fill_between(dd_port.index, dd_port, 0, alpha=0.2, color="#4363d8")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("D02 – Drawdown : Portefeuille vs Core")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D02_portfolio_drawdown.png", cfg.dpi)


def plot_D03_portfolio_annual_bar(annual: pd.DataFrame, cfg: PlotConfig) -> None:
    """D03 – Performance annuelle portefeuille vs Core."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(annual))

    ax.bar(
        x - 0.2,
        annual["core"].mul(100),
        0.35,
        label="Core",
        color="#e6194b",
        alpha=0.7,
    )
    ax.bar(
        x + 0.2,
        annual["portfolio"].mul(100),
        0.35,
        label="Portefeuille",
        color="#4363d8",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(annual.index.astype(str))
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("D03 – Performance annuelle : Portefeuille vs Core")
    ax.set_ylabel("Rendement (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D03_portfolio_annual.png", cfg.dpi)


def plot_D04_portfolio_vol_target(r_port: pd.Series, cfg: PlotConfig) -> None:
    """D04 – Volatilité rolling vs bande cible."""
    roll_vol = _roll_vol(r_port).mul(100)

    fig, ax = plt.subplots(figsize=(10, 4))
    roll_vol.plot(ax=ax, color="#4363d8", lw=1.5, label="Vol rolling 63j")
    ax.axhline(
        cfg.vol_target_min * 100,
        color="orange",
        ls="--",
        lw=1,
        label=f"Min cible {cfg.vol_target_min:.0%}",
    )
    ax.axhline(
        cfg.vol_target_max * 100,
        color="orange",
        ls="--",
        lw=1,
        label=f"Max cible {cfg.vol_target_max:.0%}",
    )
    ax.fill_between(
        roll_vol.index,
        cfg.vol_target_min * 100,
        cfg.vol_target_max * 100,
        alpha=0.1,
        color="orange",
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("D04 – Volatilité rolling 63j vs bande cible")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vol annualisée (%)")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D04_portfolio_vol_target.png", cfg.dpi)


def plot_D05_beta_sat_rolling(beta_roll: pd.Series, cfg: PlotConfig) -> None:
    """D05 – Bêta rolling satellite vs Core."""
    fig, ax = plt.subplots(figsize=(10, 4))
    beta_roll.plot(ax=ax, color="#4363d8", lw=1.5, label="Beta rolling 63j")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(
        cfg.beta_target_abs,
        color="orange",
        lw=1,
        ls=":",
        label=f"±β max = {cfg.beta_target_abs:.2f}",
    )
    ax.axhline(-cfg.beta_target_abs, color="orange", lw=1, ls=":")
    ax.fill_between(
        beta_roll.index,
        -cfg.beta_target_abs,
        cfg.beta_target_abs,
        alpha=0.07,
        color="green",
        label=f"Zone cible |β| ≤ {cfg.beta_target_abs:.2f}",
    )

    ax.set_title("D05 – Beta rolling 63j de la poche satellite vs Core")
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D05_beta_sat_rolling.png", cfg.dpi)


def plot_D06_portfolio_dist(
    r_port: pd.Series,
    r_core: pd.Series,
    cfg: PlotConfig,
) -> None:
    """D06 – Distribution des rendements journaliers."""
    fig, ax = plt.subplots(figsize=(9, 5))
    r_core.mul(100).hist(
        ax=ax,
        bins=80,
        alpha=0.5,
        color="#e6194b",
        label="Core",
        density=True,
    )
    r_port.mul(100).hist(
        ax=ax,
        bins=80,
        alpha=0.5,
        color="#4363d8",
        label="Portefeuille",
        density=True,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("D06 – Distribution des rendements journaliers")
    ax.set_xlabel("Rendement journalier (%)")
    ax.set_ylabel("Densité")
    ax.legend(fontsize=9)
    _save(cfg.fig_dir, "D06_portfolio_dist.png", cfg.dpi)


def main() -> None:
    """Lance la génération complète des graphiques."""
    cfg = PlotConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    print("=" * 60)
    print("  GÉNÉRATION DES GRAPHIQUES – RAPPORT CORE-SATELLITE")
    print("=" * 60)

    print("\n[1] Chargement des données...")

    r_core_oos = _read_returns_csv(cfg.core_oos_csv).sort_index()
    core_etf_rets = _load_core_etf_returns(cfg)
    core_finaux = pd.read_csv(cfg.core_finaux_csv)

    bt = pd.read_csv(cfg.fond_returns_csv, index_col=0, parse_dates=True).sort_index()
    bt.index = pd.DatetimeIndex(bt.index).tz_localize(None)
    r_port = pd.to_numeric(bt["portfolio_ret"], errors="coerce").dropna()
    r_core = pd.to_numeric(bt["core_ret"], errors="coerce").dropna()
    r_sat = pd.to_numeric(bt["sat_pocket_ret"], errors="coerce").dropna()

    weights_df = pd.read_csv(cfg.fond_weights_csv, index_col=0)
    sat_weights = pd.to_numeric(
        weights_df["theta_satellite"], errors="coerce"
    ).dropna()
    sat_weights = sat_weights[sat_weights > 0]
    w_sat = float(pd.to_numeric(weights_df["w_sat"], errors="coerce").iloc[0])

    annual = pd.read_csv(cfg.fond_annual_csv, index_col=0)
    annual.index = annual.index.astype(int)

    beta_roll = pd.read_csv(
        cfg.fond_beta_roll_csv,
        index_col=0,
        parse_dates=True,
    ).sort_index()
    beta_roll.index = pd.DatetimeIndex(beta_roll.index).tz_localize(None)
    beta_series = pd.to_numeric(beta_roll.iloc[:, 0], errors="coerce").dropna()

    sat_info = pd.read_csv(cfg.sat_selected_csv).set_index("ticker")
    sat_indiv = pd.read_csv(
        cfg.sat_indiv_csv,
        index_col=0,
        parse_dates=True,
    ).sort_index()
    sat_indiv.index = pd.DatetimeIndex(sat_indiv.index).tz_localize(None)
    sat_indiv = sat_indiv.apply(pd.to_numeric, errors="coerce")

    print(
        f"    Backtest : {bt.index.min().date()} → {bt.index.max().date()} "
        f"({len(bt)} obs)"
    )

    print("\n[2] Section A – Poche Core...")
    plot_A01_core_etf_cum(core_etf_rets, core_finaux, cfg)
    plot_A02_core_portfolio_cum(r_core_oos.loc[r_port.index.min() :], cfg)
    plot_A03_core_drawdown(r_core, cfg)
    plot_A04_core_rolling_vol(r_core, cfg)
    plot_A05_core_rolling_sharpe(r_core, cfg)
    plot_A06_core_annual_bar(annual, cfg)
    plot_A07_core_fees_bar(core_finaux, cfg)

    print("\n[3] Section B – Poche Satellite...")
    plot_B01_sat_fund_cum(sat_indiv, sat_info, sat_weights, cfg)
    plot_B02_sat_pocket_cum(r_sat, cfg)
    plot_B03_sat_weights_bar(sat_weights, sat_info, w_sat, cfg)
    plot_B04_sat_rolling_alpha(r_sat, r_core, cfg)
    plot_B05_sat_rolling_beta(beta_series, cfg)
    plot_B06_sat_perf_annual(annual, cfg)
    plot_B07_sat_fees_bar(sat_weights, sat_info, cfg)

    print("\n[4] Section C – Core vs Satellite...")
    plot_C01_core_vs_sat_cum(r_core, r_sat, r_port, cfg)
    plot_C02_excess_cum(r_port, r_core, cfg)
    plot_C04_annual_grouped_bar(annual, cfg)
    plot_C05_excess_annual_bar(annual, cfg)

    print("\n[5] Section D – Portefeuille total...")
    plot_D01_portfolio_cum(r_port, r_core, cfg)
    plot_D02_portfolio_drawdown(r_port, r_core, cfg)
    plot_D03_portfolio_annual_bar(annual, cfg)
    plot_D04_portfolio_vol_target(r_port, cfg)
    plot_D05_beta_sat_rolling(beta_series, cfg)
    plot_D06_portfolio_dist(r_port, r_core, cfg)

    n_figs = len(list(cfg.fig_dir.glob("*.png")))
    print(f"\n  ✓ {n_figs} figures générées dans {cfg.fig_dir}")


if __name__ == "__main__":
    main()