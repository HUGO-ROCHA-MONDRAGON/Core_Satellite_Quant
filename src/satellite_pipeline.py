"""
Pipeline Satellite v3 – Shortlist 19 fonds, filtres IS renforcés, scoring décorrélation-first.

Sources :
  - STRAT1/2/3_info.xlsx  : métadonnées Bloomberg (AUM, frais, stratégie, etc.)
  - STRAT1/2/3_price.xlsx : prix daily (paires date/prix par fonds)
  - outputs/core_returns_daily_oos.csv : rendements journaliers du Core (benchmark)
    - outputs/core_selected_etfs.csv + outputs/core3_etf_daily_log_returns.csv
        : benchmark équipondéré dynamique des 3 ETF Core sélectionnés

Structure des 3 blocs :
  Bloc 1 – Décorrélation / Convexité  : CTA Trend, Short Vol
  Bloc 2 – Alpha Décorrélé            : Event Driven, Merger Arb, L/S, Market Neutral…
  Bloc 3 – Carry / Crédit Structuré   : ABS, CLO, FI Relative Value, Senior Loans…

Shortlist qualitative : 19 fonds pré-sélectionnés (SATELLITE_SHORTLIST) restreignent
  l'univers avant tout filtre quantitatif.  Chaque ticker est recherché dans
  l'ensemble des 3 fichiers STRAT (info + price) pour éviter toute erreur
  de mapping fichier ↔ bloc.

Pipeline de filtrage (toutes métriques calculées sur la fenêtre calib 2019-2020) :
    Niveau Beta – Filtre initial renforcé : rolling beta 3 mois vs Core équipondéré,
                         median(|beta|) <= 35%, q75(|beta|) <= 55%,
                         |beta| <= 35% sur au moins 80% des jours IS
  Niveau 0 – Structurel universel : AUM ≥ 100 M$, premier prix ≤ 01/01/2019,
             filtre devise (optionnel), exclusion de stratégies (optionnel)
  Niveau 1 – Frais, Volatilité & Stale pricing (seuil global 10%, spécifique par ticker)
  Niveau 2 – Qualité quantitative + filtre corrélation IS (corr_core_calib <= 45%)
  Niveau 3 – Comportemental : Skewness, Kurtosis, Concentration
  Filtre pairwise – Cohérence inter-fonds sélectionnés (corr pairwise IS <= 70%)

Score composite décorrélation-first (z-scoré intra-bloc, calib window 2019-2020) :
  -|β_core|(30%) + -corr_core(10%) + Sortino(25%) + ret_rel_covid(15%)
  + -dd_covid(10%) + Skewness(5%) + -Kurtosis(5%)

Sélection finale : 2+3+2 fonds (Bloc1+Bloc2+Bloc3), max 2 par stratégie + filtre pairwise.
Réserves : 2 fonds suivants par bloc exportés dans satellite_reserves.csv.
Export enrichi : satellite_selected_v3.csv avec métriques COVID et corrélation IS.
Anti look-ahead : tous les calculs quantitatifs sur la fenêtre 2019-2020 uniquement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import logging
import os
import unicodedata

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
#  Poids du score composite (modifiables ici)
# ══════════════════════════════════════════════════════════════════════════════

SCORE_WEIGHTS: Dict[str, float] = {
    "neg_abs_beta":   0.30,   # décorrélation : bas beta vs core
    "neg_corr_core":  0.10,   # décorrélation : corrélation IS
    "sortino":        0.25,   # régularité risk-adjusted
    "ret_rel_covid":  0.15,   # résistance crise mars 2020
    "neg_dd_covid":   0.10,   # drawdown pendant COVID (moins négatif = mieux)
    "skew":           0.05,   # légère préférence positive
    "neg_kurtosis":   0.05,   # moins de fat tails = mieux
}

# ── Shortlist qualitative des 19 fonds satellite pré-sélectionnés ─────────────
#    Chaque ticker est recherché dans l'ensemble STRAT1 + STRAT2 + STRAT3.
SATELLITE_SHORTLIST: Dict[str, List[str]] = {
    "Bloc1": [
        "DWMAEIA ID Equity",       # CTA Trend (Dunn)           — dans STRAT2
        "FINVPRI GR Equity",       # Short Vol (FincCam)        — dans STRAT1
    ],
    "Bloc2": [
        "HFHPERC LX Equity",      # Event Driven (Syquant)
        "EXCRISA LX Equity",      # Merger Arb (Exane)
        "PDAIEUR LX Equity",      # Multi-Strat MN (Pictet)
        "HSMSTIC LX Equity",      # Multi-Strat (HSBC)
        "REYLSEB LX Equity",      # Market Neutral (RAM)
        "CARPPFE LX Equity",      # L/S EU (Carmignac)
        "ELEARER LX Equity",      # L/S EU (Eleva)
        "LDAPB2E ID Equity",      # L/S Asia (Dalton)
        "LUMNUDE LX Equity",      # Market Neutral (Man GLG)
        "EXCERFD LX Equity",      # L/S EU value (Exane Ceres)
    ],
    "Bloc3": [
        "AEABSIA ID Equity",      # ABS Senior (Aegon)
        "BNPAOPP LX Equity",      # ABS Diversifié (BNP)
        "CLOHQIA GR Equity",      # CLO Senior (Lupus Alpha)
        "INICLBI GR Equity",      # CLO IG (Infinigon)
        "DISFCPE LX Equity",      # FI Relative Value (Danske)
        "LIOFVS1 ID Equity",      # Credit RV (Lion)
        "MUESEIH ID Equity",      # Senior Loans (Muzinich)
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlocConfig:
    """Paramètres de filtrage et de sélection pour un bloc satellite."""

    nom: str

    # ── Niveau 1 : Frais ──────────────────────────────────────────────────
    expense_max_default: float = 2.0          # % (frais max par défaut)
    expense_max_by_strategy: Dict[str, float] = field(default_factory=dict)

    # ── Niveau 1 : Volatilité (annualisée, calculée sur calib window) ────
    vol_min_default: float = 0.0
    vol_min_by_strategy: Dict[str, float] = field(default_factory=dict)
    vol_max_default: float = 0.30
    vol_max_by_strategy: Dict[str, float] = field(default_factory=dict)

    # ── Niveau 2 : Qualité quantitative (sur calib window) ───────────────
    sharpe_min: float = -0.5
    alpha_min_annual: float = -0.10    # alpha annualisé vs core (plancher)
    drawdown_max: float = -0.80        # max drawdown (valeur négative)

    # ── Niveau 3 : Comportemental ─────────────────────────────────────────
    skew_min: float = -2.5             # skewness sur prix calib
    kurtosis_max: float = 10.0         # kurtosis sur prix calib
    concentration_max: float = 95.0    # % des 10 premières positions (info)

    # ── Filtre beta (override par bloc, -1 = utiliser le global) ─────────
    beta_max_abs_override: float = -1.0       # si > 0, remplace cfg.beta_filter_max_abs
    beta_q75_max_override: float = -1.0       # si > 0, remplace cfg.beta_filter_q75_max
    beta_min_pass_ratio_override: float = -1.0  # si > 0, remplace cfg.beta_filter_min_pass_ratio

    # ── Sélection finale ──────────────────────────────────────────────────
    n_select: int = 3
    max_per_strategy: int = 2


@dataclass
class SatelliteConfig:
    """Configuration globale du pipeline satellite."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # ── Fichiers source (info + prix) ─────────────────────────────────────
    #    Tous les fichiers sont lus et fusionnés ; chaque ticker est trouvé
    #    quel que soit le fichier dans lequel il se trouve.
    info_paths: List[str] = field(default_factory=list)
    price_paths: List[str] = field(default_factory=list)

    # ── Source rendements Core (benchmark alpha & beta) ──────────────────
    core_daily_csv: str = ""           # rempli dans __post_init__
    core_selected_csv: str = ""        # tickers Core retenus (CSV)
    core3_daily_log_returns_csv: str = ""  # rendements log des 3 ETF Core (CSV)

    # ── Fenêtre de calibration (anti look-ahead) ─────────────────────────
    calib_start: str = "2019-01-01"
    calib_end: str = "2020-12-31"

    # ── Niveau 0 – Filtres universels ────────────────────────────────────
    aum_min_m: float = 100.0
    max_start_date: str = "2019-01-01"
    allowed_currencies: List[str] = field(default_factory=lambda: ["Euro"])
    excluded_strategies: List[str] = field(default_factory=list)

    # ── Rolling beta window ───────────────────────────────────────────────
    beta_rolling_days: int = 252

    # ── Filtre beta initial (vs core équipondéré des 3 ETF sélectionnés) ─
    beta_filter_window_days: int = 63
    beta_filter_max_abs: float = 0.35
    beta_filter_min_pass_ratio: float = 0.80
    beta_filter_q75_max: float = 0.55

    # ── Qualité de cotation (stale pricing) ─────────────────────────────
    stale_max_ratio: float = 0.10
    stale_max_ratio_by_ticker: Dict[str, float] = field(
        default_factory=lambda: {
            "LIOFVS1 ID Equity": 0.95,   # valorisation mensuelle → stale ratio très élevé, normal
            "CLOHQIA GR Equity": 0.15,   # CLO : cotation moins fréquente, tolérance élargie
            "INICLBI GR Equity": 0.10,   # CLO IG
        }
    )

    # ── Filtre corrélation IS ─────────────────────────────────────────────
    corr_is_max: float = 0.45

    # ── Filtre cohérence pairwise (post-scoring) ──────────────────────────
    corr_pairwise_is_max: float = 0.70

    # ── Shortlist qualitative ─────────────────────────────────────────────
    use_shortlist: bool = True
    satellite_shortlist: Dict[str, List[str]] = field(
        default_factory=lambda: SATELLITE_SHORTLIST
    )

    # ── Sorties ───────────────────────────────────────────────────────────
    output_selected_csv: str = ""

    # ── Blocs ─────────────────────────────────────────────────────────────
    blocs: Dict[str, BlocConfig] = field(default_factory=dict)

    def __post_init__(self):
        r = self.project_root
        if not self.core_daily_csv:
            self.core_daily_csv = str(r / "outputs" / "core_returns_daily_oos.csv")
        if not self.core_selected_csv:
            self.core_selected_csv = str(r / "outputs" / "core_selected_etfs.csv")
        if not self.core3_daily_log_returns_csv:
            self.core3_daily_log_returns_csv = str(r / "outputs" / "core3_etf_daily_log_returns.csv")
        if not self.output_selected_csv:
            self.output_selected_csv = str(r / "outputs" / "satellite_selected.csv")
        if not self.info_paths:
            self.info_paths = [
                str(r / "data" / "STRAT1_info.xlsx"),
                str(r / "data" / "STRAT2_info.xlsx"),
                str(r / "data" / "STRAT3_info.xlsx"),
            ]
        if not self.price_paths:
            self.price_paths = [
                str(r / "data" / "STRAT1_price.xlsx"),
                str(r / "data" / "STRAT2_price.xlsx"),
                str(r / "data" / "STRAT3_price.xlsx"),
            ]
        if not self.blocs:
            self.blocs = _build_default_blocs()


def _build_default_blocs() -> Dict[str, BlocConfig]:
    """Construit les configurations par défaut des 3 blocs."""
    return {
        "Bloc1": BlocConfig(
            nom="Bloc 1 – Décorrélation / Convexité",
            expense_max_default=2.0,
            vol_min_default=0.0,
            vol_max_default=0.25,
            vol_min_by_strategy={},
            vol_max_by_strategy={},
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.80,
            skew_min=-2.0,
            kurtosis_max=10.0,
            concentration_max=95.0,
            # CTA et Short Vol peuvent avoir un beta épisodique plus élevé
            beta_max_abs_override=0.55,
            beta_q75_max_override=0.75,
            beta_min_pass_ratio_override=0.60,
            n_select=2,
            max_per_strategy=2,
        ),
        "Bloc2": BlocConfig(
            nom="Bloc 2 – Alpha Décorrélé",
            expense_max_default=3.0,
            vol_min_default=0.0,
            vol_max_default=0.20,
            vol_min_by_strategy={},
            vol_max_by_strategy={
                "Neutre au marché":            0.12,
                "CTA/futures gérés":           0.20,
                "Long Short":                  0.15,
                "Equity Hedge":                0.15,
                "Mené par les événements":     0.12,
                "Multi-stratégie":             0.12,
            },
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.50,
            skew_min=-2.0,
            kurtosis_max=10.0,
            concentration_max=95.0,
            n_select=3,
            max_per_strategy=2,
        ),
        "Bloc3": BlocConfig(
            nom="Bloc 3 – Carry / Crédit Structuré",
            expense_max_default=1.8,
            expense_max_by_strategy={
                "Titres adossés à des actifs": 2.0,
                "Prêts bancaires":             2.0,
            },
            vol_min_default=0.0,
            vol_max_default=0.20,
            vol_min_by_strategy={},
            vol_max_by_strategy={
                "Titres adossés à des actifs":     0.10,
                "Prêts bancaires":                 0.10,
                "Obligataire Valeur relative":     0.12,
            },
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.70,
            skew_min=-2.5,
            kurtosis_max=10.0,
            concentration_max=95.0,
            n_select=2,
            max_per_strategy=2,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Lecture des données
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dates(x: pd.Series) -> pd.Series:
    """Parse robuste de dates : datetime natif / string / serial Excel."""
    dt = pd.to_datetime(x, errors="coerce", dayfirst=False)
    if dt.notna().sum() < max(5, int(0.20 * len(x))):
        num = pd.to_numeric(x, errors="coerce")
        mask_valid = num.notna() & (num >= 1) & (num <= 80_000)
        dt2 = pd.Series(pd.NaT, index=x.index)
        if mask_valid.any():
            dt2.loc[mask_valid] = pd.to_datetime(
                num.loc[mask_valid].astype(int),
                errors="coerce", unit="D", origin="1899-12-30"
            )
        dt = dt.fillna(dt2)
    return dt


def lire_prix_wide(path: str) -> pd.DataFrame:
    """
    Lit TOUTES les feuilles d'un fichier de prix satellite.
    Chaque feuille correspond à une stratégie.
    Structure par feuille : ligne 0 = tickers (paires date/prix), lignes 1+ = données.
    Retourne DataFrame wide : index=Date (DatetimeIndex), cols=tickers (toutes feuilles).
    """
    xls = pd.ExcelFile(path)
    seen: set = set()
    all_series: Dict[str, pd.Series] = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        tickers_row = df.iloc[0, :].tolist()
        data = df.iloc[1:]

        for j in range(0, df.shape[1], 2):
            ticker = tickers_row[j] if j < len(tickers_row) else None
            if not isinstance(ticker, str):
                continue
            ticker = ticker.strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)

            price_col = j + 1
            if price_col >= df.shape[1]:
                break

            dates = _parse_dates(data.iloc[:, j])
            prices = pd.to_numeric(data.iloc[:, price_col], errors="coerce")
            valid = dates.notna() & prices.notna()
            if valid.sum() < 5:
                continue

            s = pd.Series(
                prices[valid].values,
                index=pd.DatetimeIndex(dates[valid].values),
                name=ticker,
            ).sort_index()
            s = s[~s.index.duplicated(keep="first")]
            all_series[ticker] = s

    wide = pd.DataFrame(all_series).sort_index()
    wide.index.name = "Date"
    return wide


def lire_info(path: str) -> pd.DataFrame:
    """
    Lit un fichier info satellite. Normalise les colonnes clés.
    Retourne DataFrame indexé sur 'ticker'.
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        cl = col.lower()
        if "total actifs usd" in cl:
            rename_map[col] = "aum_usd_m"
        elif ("ratio des dépenses" in cl or "ratio des depenses" in cl
              or ("expense" in cl and col not in rename_map.values())):
            rename_map[col] = "expense_pct"
        elif "stratégie de fonds" in cl or "strategie de fonds" in cl:
            rename_map[col] = "strategie"
        elif "date de création" in cl or "date de creation" in cl:
            rename_map[col] = "date_creation"
        elif col.strip() == "Dev" or "devise" in cl:
            rename_map[col] = "devise"
        elif "% des 10" in cl or "premières positions" in cl:
            rename_map[col] = "concentration"

    df = df.rename(columns=rename_map)

    for col in ["aum_usd_m", "expense_pct", "concentration"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace("\xa0", "", regex=False)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )

    if "date_creation" in df.columns:
        df["date_creation"] = pd.to_datetime(df["date_creation"], errors="coerce", dayfirst=False)

    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    if "ticker" not in df.columns:
        df["ticker"] = df.iloc[:, 0]
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df.dropna(subset=["ticker"])

    return df.set_index("ticker")


def charger_toutes_les_donnees(cfg: SatelliteConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge et fusionne TOUS les fichiers info et price (STRAT1 + STRAT2 + STRAT3).
    Retourne (all_prices, all_info) couvrant l'ensemble de l'univers.
    Un ticker présent dans plusieurs fichiers est pris depuis le premier fichier trouvé.
    """
    # ── Prix ──────────────────────────────────────────────────────────────
    all_price_series: Dict[str, pd.Series] = {}
    for path in cfg.price_paths:
        print(f"    Lecture prix : {Path(path).name}")
        wide = lire_prix_wide(path)
        for col in wide.columns:
            if col not in all_price_series:
                all_price_series[col] = wide[col].dropna()
    all_prices = pd.DataFrame(all_price_series).sort_index()
    all_prices.index.name = "Date"
    print(f"    → {len(all_prices.columns)} tickers prix chargés au total")

    # ── Info ──────────────────────────────────────────────────────────────
    info_frames = []
    seen_tickers: set = set()
    for path in cfg.info_paths:
        print(f"    Lecture info : {Path(path).name}")
        df = lire_info(path)
        new_tickers = [t for t in df.index if t not in seen_tickers]
        info_frames.append(df.loc[new_tickers])
        seen_tickers.update(new_tickers)
    all_info = pd.concat(info_frames)
    print(f"    → {len(all_info)} tickers info chargés au total")

    return all_prices, all_info


def charger_core_returns(cfg: SatelliteConfig) -> pd.Series:
    """
    Charge les rendements journaliers du Core (log returns) et les convertit
    en rendements simples journaliers.
    """
    df = pd.read_csv(cfg.core_daily_csv, index_col=0, parse_dates=True)
    col = df.columns[0]
    simple = np.exp(df[col]) - 1.0
    simple.name = "core_return"
    return simple.sort_index().dropna()


def charger_core_eqw_returns_from_csv(cfg: SatelliteConfig) -> pd.Series:
    """
    Construit un benchmark Core équipondéré (3 ETF sélectionnés) depuis des CSV.
    """
    selected = pd.read_csv(cfg.core_selected_csv)
    ticker_col = "core_etfs" if "core_etfs" in selected.columns else selected.columns[0]
    tickers = selected[ticker_col].astype(str).str.strip().dropna().tolist()
    if len(tickers) == 0:
        raise ValueError("core_selected_etfs.csv ne contient aucun ticker exploitable.")

    core3_log = pd.read_csv(cfg.core3_daily_log_returns_csv, index_col=0, parse_dates=True).sort_index()
    available = [t for t in tickers if t in core3_log.columns]
    if len(available) < 3:
        raise ValueError(
            "core3_etf_daily_log_returns.csv ne contient pas les 3 ETF sélectionnés. "
            f"Trouvés: {available}"
        )

    core3_log = core3_log[available[:3]].dropna(how="all")
    core3_prices = (100.0 * np.exp(core3_log.cumsum())).ffill()
    core_eqw_price = core3_prices.mean(axis=1).rename("core_eqw_price")
    core_eqw_log = np.log(core_eqw_price).diff().dropna().rename("core_eqw_log_return")
    return core_eqw_log


# ══════════════════════════════════════════════════════════════════════════════
#  Calcul des métriques sur la fenêtre de calibration
# ══════════════════════════════════════════════════════════════════════════════

def _annualized_vol(daily_rets: pd.Series) -> float:
    return float(daily_rets.std() * np.sqrt(252))


def _annualized_sharpe(daily_rets: pd.Series) -> float:
    vol = _annualized_vol(daily_rets)
    if vol < 1e-10:
        return np.nan
    ann_ret = float((1 + daily_rets).prod() ** (252 / len(daily_rets)) - 1)
    return ann_ret / vol


def _max_drawdown(daily_rets: pd.Series) -> float:
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    return float(((cum / peak) - 1).min())


def _ols_alpha_beta(fund_rets: pd.Series, core_rets: pd.Series) -> Tuple[float, float]:
    """OLS : fund = alpha_daily + beta * core + eps → (alpha_annual, beta)."""
    aligned = pd.concat([fund_rets, core_rets], axis=1, sort=True).dropna()
    if len(aligned) < 30:
        return np.nan, np.nan
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    slope, intercept, *_ = stats.linregress(x, y)
    return float(intercept * 252), float(slope)


def calculer_metriques_calib(
    wide_prices: pd.DataFrame,
    core_rets: pd.Series,
    calib_start: str,
    calib_end: str,
) -> pd.DataFrame:
    """
    Calcule pour chaque fonds (sur calib window uniquement) :
    vol, sharpe, sortino, alpha_annual, beta_core, corr_core_calib,
    drawdown, skewness, kurtosis, dd_covid, ret_rel_covid.
    """
    log_rets = np.log(wide_prices).diff().dropna(how="all")
    rets_calib = log_rets.loc[calib_start:calib_end]
    core_calib = core_rets.loc[calib_start:calib_end]

    bday_idx = pd.date_range(calib_start, calib_end, freq="B")

    # Fenêtre COVID (dans la calib window → pas de look-ahead)
    covid_start = "2020-02-01"
    covid_end = "2020-05-31"
    core_covid = core_rets.loc[covid_start:covid_end]

    rows = []
    for ticker in rets_calib.columns:
        s = rets_calib[ticker].dropna()
        if len(s) < 30:
            continue

        p = wide_prices[ticker].loc[calib_start:calib_end].reindex(bday_idx).ffill()
        stale_mask = (p.diff().abs() <= 1e-12) & p.notna() & p.shift(1).notna()
        stale_ratio = float(stale_mask.mean()) if len(stale_mask) else np.nan

        alpha, beta = _ols_alpha_beta(s, core_calib)

        aligned_corr = pd.concat([s, core_calib], axis=1, sort=True).dropna()
        if len(aligned_corr) >= 30:
            corr_core = float(aligned_corr.iloc[:, 0].corr(aligned_corr.iloc[:, 1]))
        else:
            corr_core = np.nan

        ann_ret_s = float((1 + s).prod() ** (252 / len(s)) - 1) if len(s) > 0 else np.nan
        downside = s[s < 0]
        if len(downside) >= 5:
            downside_vol = float(downside.std() * np.sqrt(252))
            sortino = ann_ret_s / downside_vol if downside_vol > 1e-10 else np.nan
        else:
            sortino = np.nan

        s_covid = log_rets[ticker].loc[covid_start:covid_end].dropna()
        if len(s_covid) >= 10:
            dd_covid = _max_drawdown(s_covid)
            aligned_covid = pd.concat([s_covid, core_covid], axis=1, sort=True).dropna()
            if len(aligned_covid) >= 5:
                ret_fund_covid = float((1 + aligned_covid.iloc[:, 0]).prod() - 1)
                ret_core_covid = float((1 + aligned_covid.iloc[:, 1]).prod() - 1)
                ret_rel_covid = ret_fund_covid - ret_core_covid
            else:
                ret_rel_covid = np.nan
        else:
            dd_covid = np.nan
            ret_rel_covid = np.nan

        rows.append(dict(
            ticker=ticker,
            vol_calib=_annualized_vol(s),
            sharpe_calib=_annualized_sharpe(s),
            sortino_calib=sortino,
            alpha_annual=alpha,
            beta_core=beta,
            corr_core_calib=corr_core,
            drawdown_calib=_max_drawdown(s),
            skew_calib=float(stats.skew(s)),
            kurtosis_calib=float(stats.kurtosis(s)),
            n_obs_calib=len(s),
            stale_ratio_calib=stale_ratio,
            dd_covid=dd_covid,
            ret_rel_covid=ret_rel_covid,
        ))

    return pd.DataFrame(rows).set_index("ticker")


def calculer_beta_rolling(
    wide_prices: pd.DataFrame,
    core_rets: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """
    Calcule le beta rolling (fenêtre = window jours) de chaque fonds vs Core.
    """
    fund_rets = np.log(wide_prices).diff().dropna(how="all")
    aligned = pd.concat([fund_rets, core_rets.rename("__core__")], axis=1, sort=True).dropna(how="all")

    fund_cols = aligned.drop(columns=["__core__"])

    betas: Dict[str, pd.Series] = {}
    for ticker in fund_cols.columns:
        s = aligned[[ticker, "__core__"]].dropna()
        if len(s) < window:
            continue
        cov = s[ticker].rolling(window).cov(s["__core__"])
        var = s["__core__"].rolling(window).var()
        betas[ticker] = (cov / var).where(var > 1e-12)

    return pd.DataFrame(betas)


# ══════════════════════════════════════════════════════════════════════════════
#  Niveaux de filtrage
# ══════════════════════════════════════════════════════════════════════════════

def filtrer_niveau0(
    info: pd.DataFrame,
    wide_prices: pd.DataFrame,
    cfg: SatelliteConfig,
) -> List[str]:
    """
    Niveau 0 – Structurel universel :
      - Ticker présent dans info ET dans prices
      - AUM ≥ aum_min_m (USD millions)
      - Premier prix ≤ max_start_date
      - Devise dans allowed_currencies (si liste non vide)
      - Stratégie hors excluded_strategies (si liste non vide)
    """
    valid = []
    max_date = pd.Timestamp(cfg.max_start_date)
    allowed_norm = {
        _normalize_text(c) for c in cfg.allowed_currencies if str(c).strip()
    }

    for ticker in wide_prices.columns:
        if ticker not in info.index:
            continue

        row = info.loc[ticker]

        aum = row.get("aum_usd_m", np.nan)
        if pd.isna(aum) or aum < cfg.aum_min_m:
            continue

        prices = wide_prices[ticker].dropna()
        if len(prices) == 0 or prices.index.min() > max_date:
            continue

        if allowed_norm:
            devise = str(row.get("devise", "")).strip()
            if _normalize_text(devise) not in allowed_norm:
                continue

        if cfg.excluded_strategies:
            strat = str(row.get("strategie", "")).strip()
            if strat in cfg.excluded_strategies:
                continue

        expense = row.get("expense_pct", np.nan)
        # En mode shortlist, les fonds pré-sélectionnés sans expense renseigné passent
        if pd.isna(expense) and not cfg.use_shortlist:
            continue

        valid.append(ticker)
    return valid


def _normalize_text(s: str) -> str:
    """Normalise une étiquette texte pour comparaisons robustes."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()


def filtrer_niveau_beta_initial(
    tickers: List[str],
    beta_rolling: pd.DataFrame,
    cfg: SatelliteConfig,
    bloc_cfg: BlocConfig | None = None,
) -> List[str]:
    """
    Filtre initial beta renforcé : conserve les fonds dont le beta rolling 3M vs core
    respecte simultanément les 3 conditions.
    Accepte des overrides par bloc (via bloc_cfg) pour les stratégies spéciales.
    """
    # Utiliser les overrides du bloc si disponibles, sinon les globaux
    max_abs = cfg.beta_filter_max_abs
    q75_max = cfg.beta_filter_q75_max
    min_pass = cfg.beta_filter_min_pass_ratio
    if bloc_cfg is not None:
        if bloc_cfg.beta_max_abs_override > 0:
            max_abs = bloc_cfg.beta_max_abs_override
        if bloc_cfg.beta_q75_max_override > 0:
            q75_max = bloc_cfg.beta_q75_max_override
        if bloc_cfg.beta_min_pass_ratio_override > 0:
            min_pass = bloc_cfg.beta_min_pass_ratio_override

    valid: List[str] = []
    beta_window = beta_rolling.loc[cfg.calib_start:cfg.calib_end]

    for ticker in tickers:
        if ticker not in beta_window.columns:
            continue
        s = beta_window[ticker].dropna()
        if len(s) == 0:
            continue
        abs_beta = s.abs()
        median_beta = float(abs_beta.median())
        q75_beta = float(abs_beta.quantile(0.75))
        pass_ratio = float((abs_beta <= max_abs).mean())
        if (
            median_beta <= max_abs
            and q75_beta <= q75_max
            and pass_ratio >= min_pass
        ):
            valid.append(ticker)
    return valid


def filtrer_niveau1(
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
    cfg: SatelliteConfig,
) -> List[str]:
    """
    Niveau 1 – Frais, Volatilité & Qualité de cotation (calib window).
    """
    valid = []
    for ticker in tickers:
        row_info = info.loc[ticker] if ticker in info.index else None
        strat = str(row_info.get("strategie", "")) if row_info is not None else ""

        expense_max = bloc_cfg.expense_max_by_strategy.get(strat, bloc_cfg.expense_max_default)
        expense = float(row_info.get("expense_pct", np.nan)) if row_info is not None else np.nan
        if not np.isnan(expense) and expense > expense_max:
            continue

        if ticker not in metrics.index:
            continue
        vol = metrics.at[ticker, "vol_calib"]
        vol_min = bloc_cfg.vol_min_by_strategy.get(strat, bloc_cfg.vol_min_default)
        vol_max = bloc_cfg.vol_max_by_strategy.get(strat, bloc_cfg.vol_max_default)
        if not np.isnan(vol) and vol < vol_min:
            continue
        if not np.isnan(vol) and vol > vol_max:
            continue

        stale = metrics.at[ticker, "stale_ratio_calib"] if "stale_ratio_calib" in metrics.columns else np.nan
        stale_threshold = cfg.stale_max_ratio_by_ticker.get(ticker, cfg.stale_max_ratio)
        if not np.isnan(stale) and stale > stale_threshold:
            continue

        valid.append(ticker)
    return valid


def filtrer_niveau2(
    tickers: List[str],
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
    cfg: SatelliteConfig | None = None,
) -> List[str]:
    """
    Niveau 2 – Qualité quantitative (sur calib window).
    """
    valid = []
    for ticker in tickers:
        if ticker not in metrics.index:
            continue
        row = metrics.loc[ticker]
        if not np.isnan(row["sharpe_calib"]) and row["sharpe_calib"] < bloc_cfg.sharpe_min:
            continue
        if not np.isnan(row["alpha_annual"]) and row["alpha_annual"] < bloc_cfg.alpha_min_annual:
            continue
        if not np.isnan(row["drawdown_calib"]) and row["drawdown_calib"] < bloc_cfg.drawdown_max:
            continue
        if cfg is not None and "corr_core_calib" in metrics.columns:
            corr = row.get("corr_core_calib", np.nan)
            if not np.isnan(corr) and corr > cfg.corr_is_max:
                continue
        valid.append(ticker)
    return valid


def filtrer_niveau3(
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
) -> List[str]:
    """
    Niveau 3 – Comportemental.
    """
    valid = []
    for ticker in tickers:
        if ticker in metrics.index:
            skew = metrics.at[ticker, "skew_calib"]
            kurt = metrics.at[ticker, "kurtosis_calib"]
            if not np.isnan(skew) and skew < bloc_cfg.skew_min:
                continue
            if not np.isnan(kurt) and kurt > bloc_cfg.kurtosis_max:
                continue

        if ticker in info.index and "concentration" in info.columns:
            conc = info.at[ticker, "concentration"]
            if not np.isnan(conc) and conc > bloc_cfg.concentration_max:
                continue

        valid.append(ticker)
    return valid


# ══════════════════════════════════════════════════════════════════════════════
#  Score composite & sélection finale
# ══════════════════════════════════════════════════════════════════════════════

def _zscore_col(series: pd.Series) -> pd.Series:
    """Z-score (NaN remplacés par 0 après normalisation)."""
    mu, sigma = series.mean(), series.std()
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return ((series - mu) / sigma).fillna(0.0)


def scorer(
    tickers: List[str],
    metrics: pd.DataFrame,
    weights: Dict[str, float] = SCORE_WEIGHTS,
) -> pd.Series:
    """
    Score composite intra-bloc (z-scoré) basé sur les métriques calib window.
    """
    df = metrics.loc[[t for t in tickers if t in metrics.index]].copy()
    if df.empty:
        return pd.Series(dtype=float)

    score = (
        weights.get("neg_abs_beta", 0.30)   * _zscore_col(-df["beta_core"].abs())
      + weights.get("neg_corr_core", 0.10)  * _zscore_col(-df["corr_core_calib"])
      + weights.get("sortino", 0.25)        * _zscore_col(df["sortino_calib"])
      + weights.get("ret_rel_covid", 0.15)  * _zscore_col(df["ret_rel_covid"])
      + weights.get("neg_dd_covid", 0.10)   * _zscore_col(-df["dd_covid"])
      + weights.get("skew", 0.05)           * _zscore_col(df["skew_calib"])
      + weights.get("neg_kurtosis", 0.05)   * _zscore_col(-df["kurtosis_calib"])
    )
    return score.sort_values(ascending=False)


def selectionner(
    scores: pd.Series,
    info: pd.DataFrame,
    bloc_cfg: BlocConfig,
) -> List[str]:
    """
    Sélectionne n_select fonds (ordre décroissant des scores).
    Contrainte : max max_per_strategy fonds par stratégie.
    """
    selected: List[str] = []
    strategy_count: Dict[str, int] = {}

    for ticker in scores.index:
        if len(selected) >= bloc_cfg.n_select:
            break
        strat = (
            str(info.at[ticker, "strategie"])
            if (ticker in info.index and "strategie" in info.columns)
            else "Inconnu"
        )
        count = strategy_count.get(strat, 0)
        if count >= bloc_cfg.max_per_strategy:
            continue
        selected.append(ticker)
        strategy_count[strat] = count + 1

    return selected


def filtrer_coherence_pairwise(
    tickers_ranked: List[str],
    wide_prices_calib: pd.DataFrame,
    n_select: int,
    corr_max: float,
) -> Tuple[List[str], List[str]]:
    """
    Filtre de cohérence pairwise (post-scoring, avant sélection finale).
    Retourne (selected, reserves).
    """
    log_rets = np.log(wide_prices_calib).diff().dropna(how="all")

    selected: List[str] = []
    reserves: List[str] = []
    n_reserves = 2

    for ticker in tickers_ranked:
        if ticker not in log_rets.columns:
            continue
        s = log_rets[ticker].dropna()
        if len(s) < 10:
            continue

        ok = True
        for t_sel in selected:
            if t_sel not in log_rets.columns:
                continue
            aligned = pd.concat([s, log_rets[t_sel].dropna()], axis=1, sort=True).dropna()
            if len(aligned) < 10:
                continue
            corr_val = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if abs(corr_val) > corr_max:
                ok = False
                break

        if len(selected) < n_select:
            if ok:
                selected.append(ticker)
        elif len(reserves) < n_reserves:
            if ok:
                reserves.append(ticker)
        else:
            break

    return selected, reserves


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline par bloc + pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def traiter_bloc(
    bloc_name: str,
    bloc_cfg: BlocConfig,
    all_prices: pd.DataFrame,
    all_info: pd.DataFrame,
    core_rets: pd.Series,
    core_eqw_rets: pd.Series,
    cfg: SatelliteConfig,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Traite un bloc complet : shortlist → calibration → filtrage → score → sélection.
    Les données (all_prices, all_info) couvrent l'ensemble STRAT1+2+3 fusionnés.
    """
    print(f"\n{'=' * 60}")
    print(f"  {bloc_cfg.nom}")
    print(f"{'=' * 60}")

    # ── Restreindre aux tickers présents dans prix ET info ────────────────
    common_all = [t for t in all_prices.columns if t in all_info.index]
    prices = all_prices[common_all]
    info = all_info.loc[[t for t in common_all if t in all_info.index]]
    print(f"  [données] {len(common_all)} fonds avec prix ET info (tous fichiers confondus)")

    # ── Filtre shortlist qualitative ──────────────────────────────────────
    if cfg.use_shortlist and bloc_name in cfg.satellite_shortlist:
        shortlist = cfg.satellite_shortlist[bloc_name]
        shortlist_present = [t for t in shortlist if t in prices.columns and t in info.index]
        shortlist_absent = [t for t in shortlist if t not in prices.columns or t not in info.index]
        if shortlist_absent:
            for t in shortlist_absent:
                in_prices = t in all_prices.columns
                in_info = t in all_info.index
                logging.warning(
                    "[%s] Ticker shortlist absent : %s (prix=%s, info=%s)",
                    bloc_name, t, in_prices, in_info,
                )
                print(f"  ⚠ Ticker shortlist absent : {t}  (prix={in_prices}, info={in_info})")
        prices = prices[[t for t in prices.columns if t in shortlist_present]]
        info = info.loc[[t for t in info.index if t in shortlist_present]]
        common = shortlist_present
        print(f"  [shortlist] {len(common)} fonds retenus sur {len(shortlist)} de la shortlist")
    else:
        common = common_all

    print(f"  [métriques] Calcul sur calib {cfg.calib_start} → {cfg.calib_end}...")
    print(f"  [beta init] Rolling {cfg.beta_filter_window_days}j vs Core équipondéré...")
    beta_init = calculer_beta_rolling(prices, core_eqw_rets, cfg.beta_filter_window_days)
    t_beta = filtrer_niveau_beta_initial(common, beta_init, cfg, bloc_cfg)
    # Resolve actual beta thresholds for display
    _beta_max = bloc_cfg.beta_max_abs_override if bloc_cfg.beta_max_abs_override > 0 else cfg.beta_filter_max_abs
    _beta_q75 = bloc_cfg.beta_q75_max_override if bloc_cfg.beta_q75_max_override > 0 else cfg.beta_filter_q75_max
    _beta_pass = bloc_cfg.beta_min_pass_ratio_override if bloc_cfg.beta_min_pass_ratio_override > 0 else cfg.beta_filter_min_pass_ratio
    print(
        f"  [Niv.Beta] median|β|<={_beta_max:.0%} "
        f"q75|β|<={_beta_q75:.0%} "
        f"pass>={_beta_pass:.0%}   {len(t_beta):3d} / {len(common)}"
    )

    if not t_beta:
        print(f"  ⚠  Aucun fonds ne passe le filtre beta initial dans {bloc_name}.")
        return [], pd.DataFrame(), info, pd.DataFrame(), []

    prices = prices[t_beta]
    metrics = calculer_metriques_calib(prices, core_rets, cfg.calib_start, cfg.calib_end)
    print(f"             {len(metrics)} fonds avec données suffisantes")

    t0 = filtrer_niveau0(info, prices, cfg)
    print(f"  [Niv.0] AUM/Date/Devise/Excl.  {len(t0):3d} / {len(common)}")

    t1 = filtrer_niveau1(t0, info, metrics, bloc_cfg, cfg)
    print(
        f"  [Niv.1] Frais/Vol[{bloc_cfg.vol_min_default:.0%},{bloc_cfg.vol_max_default:.0%}]"
        f"/Stale<= {cfg.stale_max_ratio:.0%} "
        f"{len(t1):3d} / {len(t0)}"
    )

    t2 = filtrer_niveau2(t1, metrics, bloc_cfg, cfg)
    print(f"  [Niv.2] Sharpe/Alpha/DD/Corr   {len(t2):3d} / {len(t1)}")

    t3 = filtrer_niveau3(t2, info, metrics, bloc_cfg)
    print(f"  [Niv.3] Skew/Kurt/Conc.        {len(t3):3d} / {len(t2)}")

    if not t3:
        print(f"  ⚠  Aucun fonds ne passe tous les filtres dans {bloc_name}.")
        return [], metrics, info, pd.DataFrame(), []

    scores = scorer(t3, metrics)

    # Ranked list (respectant la contrainte max_per_strategy) étendue pour les réserves
    tickers_ranked_extended = selectionner(
        scores, info,
        BlocConfig(
            nom=bloc_cfg.nom,
            n_select=bloc_cfg.n_select + 2,  # n_select + 2 réserves max
            max_per_strategy=bloc_cfg.max_per_strategy,
        ),
    )

    # Filtre cohérence pairwise → sélection finale + réserves
    prices_calib = prices.loc[cfg.calib_start:cfg.calib_end]
    selected, reserves = filtrer_coherence_pairwise(
        tickers_ranked_extended, prices_calib, bloc_cfg.n_select, cfg.corr_pairwise_is_max
    )

    print(f"\n  [sélection] {bloc_name} – {len(selected)} fonds retenus :")
    for rank, ticker in enumerate(selected, 1):
        strat = info.at[ticker, "strategie"] if (ticker in info.index and "strategie" in info.columns) else "?"
        sc = scores.get(ticker, np.nan)
        beta = metrics.at[ticker, "beta_core"] if ticker in metrics.index else np.nan
        sh = metrics.at[ticker, "sharpe_calib"] if ticker in metrics.index else np.nan
        print(f"    #{rank}  {ticker:<28s}  strat={strat:<28s}  "
              f"score={sc:+.3f}  β={beta:+.3f}  Sharpe={sh:.3f}")
    if reserves:
        print(f"  [réserves] {bloc_name} – {len(reserves)} fonds en réserve :")
        for rank, ticker in enumerate(reserves, 1):
            strat = info.at[ticker, "strategie"] if (ticker in info.index and "strategie" in info.columns) else "?"
            sc = scores.get(ticker, np.nan)
            print(f"    R{rank}  {ticker:<28s}  strat={strat:<28s}  score={sc:+.3f}")

    print(f"\n  [beta rolling {cfg.beta_rolling_days}j]...")
    prices_sel = prices[[t for t in selected if t in prices.columns]]
    beta_roll = calculer_beta_rolling(prices_sel, core_rets, cfg.beta_rolling_days)

    return selected, metrics, info, beta_roll, reserves


def main(cfg: SatelliteConfig | None = None) -> None:
    cfg = cfg or SatelliteConfig()

    env_allowed = os.getenv("SAT_ALLOWED_CURRENCIES", "").strip()
    if env_allowed:
        cfg.allowed_currencies = [c.strip() for c in env_allowed.split(",") if c.strip()]

    print("=" * 60)
    print("  PIPELINE SATELLITE v3 – Shortlist + Filtres IS renforcés")
    print(f"  Fenêtre calib : {cfg.calib_start} → {cfg.calib_end}")
    if cfg.allowed_currencies:
        print(f"  Filtre devise : {cfg.allowed_currencies}")
    else:
        print("  Filtre devise : désactivé")
    if cfg.use_shortlist:
        total_shortlist = sum(len(v) for v in cfg.satellite_shortlist.values())
        print(f"  Shortlist : {total_shortlist} fonds pré-sélectionnés")
    print("=" * 60)

    # ── Chargement centralisé de TOUTES les données ──────────────────────
    print("\n[0] Chargement des données (STRAT1 + STRAT2 + STRAT3)...")
    all_prices, all_info = charger_toutes_les_donnees(cfg)

    print("\n[1] Chargement des rendements Core...")
    core_rets = charger_core_returns(cfg)
    print(f"    {len(core_rets)} obs. daily | "
          f"{core_rets.index.min().date()} → {core_rets.index.max().date()}")

    core_eqw_rets = charger_core_eqw_returns_from_csv(cfg)
    print("    Benchmark filtre beta : Core équipondéré (3 ETF sélectionnés via CSV)")
    print(f"    {len(core_eqw_rets)} obs. daily | "
          f"{core_eqw_rets.index.min().date()} → {core_eqw_rets.index.max().date()}")

    all_selected: Dict[str, List[str]] = {}
    all_reserves: Dict[str, List[str]] = {}
    all_info_bloc: Dict[str, pd.DataFrame] = {}
    all_metrics: Dict[str, pd.DataFrame] = {}
    all_betas: Dict[str, pd.DataFrame] = {}
    all_scores: Dict[str, pd.Series] = {}

    for bloc_name, bloc_cfg in cfg.blocs.items():
        sel, metrics, info, beta_roll, reserves = traiter_bloc(
            bloc_name, bloc_cfg, all_prices, all_info,
            core_rets, core_eqw_rets, cfg
        )
        all_selected[bloc_name] = sel
        all_reserves[bloc_name] = reserves
        all_info_bloc[bloc_name] = info
        all_metrics[bloc_name] = metrics
        all_betas[bloc_name] = beta_roll
        if metrics is not None and not metrics.empty and sel:
            all_scores[bloc_name] = scorer(sel + reserves, metrics)
        else:
            all_scores[bloc_name] = pd.Series(dtype=float)

    # ── Export CSV sélection (compatibilité v2) ──────────────────────────
    print("\n\n[2] Export...")
    rows = []
    for bloc_name, tickers in all_selected.items():
        metrics = all_metrics[bloc_name]
        info = all_info_bloc[bloc_name]
        for ticker in tickers:
            row: Dict = {"bloc": bloc_name, "ticker": ticker}
            if ticker in info.index:
                for col in ["strategie", "devise", "expense_pct", "aum_usd_m"]:
                    row[col] = info.at[ticker, col] if col in info.columns else np.nan
            if ticker in metrics.index:
                for col in ["vol_calib", "sharpe_calib", "alpha_annual", "beta_core",
                        "drawdown_calib", "skew_calib", "kurtosis_calib",
                        "n_obs_calib", "stale_ratio_calib"]:
                    row[col] = metrics.at[ticker, col] if col in metrics.columns else np.nan
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(cfg.output_selected_csv, index=False)
    print(f"    -> {cfg.output_selected_csv}")

    # ── Export satellite_selected_v3.csv ─────────────────────────────────
    out_dir = Path(cfg.output_selected_csv).parent
    v3_path = str(out_dir / "satellite_selected_v3.csv")
    v3_rows = []
    for bloc_name, tickers in all_selected.items():
        metrics = all_metrics[bloc_name]
        info = all_info_bloc[bloc_name]
        scores_bloc = all_scores.get(bloc_name, pd.Series(dtype=float))
        for rank, ticker in enumerate(tickers, 1):
            row_v3: Dict = {
                "ticker": ticker,
                "bloc": bloc_name,
                "rank": rank,
                "score": scores_bloc.get(ticker, np.nan),
            }
            if ticker in info.index:
                row_v3["strategie"] = info.at[ticker, "strategie"] if "strategie" in info.columns else np.nan
            else:
                row_v3["strategie"] = np.nan
            if ticker in metrics.index:
                for col in ["vol_calib", "sharpe_calib", "sortino_calib", "alpha_annual",
                            "beta_core", "corr_core_calib", "dd_covid", "ret_rel_covid",
                            "stale_ratio_calib"]:
                    row_v3[col] = metrics.at[ticker, col] if col in metrics.columns else np.nan
            v3_rows.append(row_v3)
    df_v3 = pd.DataFrame(v3_rows)
    df_v3.to_csv(v3_path, index=False)
    print(f"    -> {v3_path}")

    # ── Export satellite_reserves.csv ─────────────────────────────────────
    reserves_path = str(out_dir / "satellite_reserves.csv")
    reserve_rows = []
    for bloc_name, reserves in all_reserves.items():
        metrics = all_metrics[bloc_name]
        info = all_info_bloc[bloc_name]
        scores_bloc = all_scores.get(bloc_name, pd.Series(dtype=float))
        for rank_r, ticker in enumerate(reserves, 1):
            row_r: Dict = {
                "bloc": bloc_name,
                "rank_reserve": rank_r,
                "ticker": ticker,
                "score": scores_bloc.get(ticker, np.nan),
            }
            row_r["strategie"] = (
                info.at[ticker, "strategie"]
                if (ticker in info.index and "strategie" in info.columns)
                else np.nan
            )
            reserve_rows.append(row_r)
    df_reserves = pd.DataFrame(reserve_rows)
    df_reserves.to_csv(reserves_path, index=False)
    print(f"    -> {reserves_path}")

    # Export beta rolling par bloc
    for bloc_name, beta_roll in all_betas.items():
        if not beta_roll.empty:
            out = str(out_dir / f"satellite_beta_rolling_{bloc_name}.csv")
            beta_roll.to_csv(out)
            print(f"    -> {out}")

    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RÉSUMÉ – Fonds satellite sélectionnés")
    print("=" * 60)
    for bloc_name, tickers in all_selected.items():
        print(f"\n  {bloc_name} ({len(tickers)} fonds) :")
        for t in tickers:
            print(f"    • {t}")
        reserves = all_reserves.get(bloc_name, [])
        if reserves:
            print(f"  Réserves {bloc_name} :")
            for t in reserves:
                print(f"    ◦ {t}")
    total = sum(len(v) for v in all_selected.values())
    print(f"\n  Total : {total} fonds satellite")


if __name__ == "__main__":
    main()