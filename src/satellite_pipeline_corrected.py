"""
Pipeline Satellite v5
=====================

Version corrigée pour un pipeline Core/Satellite cohérent.

Principales corrections :
- benchmark Core cohérent avec la fenêtre de calibration 2019-2020 ;
- combinaison des rendements Core IS + OOS pour les exports de bêta rolling ;
- conventions homogènes : métriques et régressions en rendements simples ;
- benchmark Core équipondéré calculé en rendements simples journaliers ;
- fallback bloc par bloc pour éviter de perdre entièrement un bloc
  quand le Niveau 3 est trop restrictif, en particulier sur le Bloc 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import os
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

from src.risk_free import get_bund_risk_free_daily, sharpe_excess


SCORE_WEIGHTS: Dict[str, float] = {
    "neg_abs_beta": 0.30,
    "neg_corr_core": 0.10,
    "sortino": 0.25,
    "ret_rel_covid": 0.15,
    "neg_dd_covid": 0.10,
    "skew": 0.05,
    "neg_kurtosis": 0.05,
}

SATELLITE_SHORTLIST: Dict[str, List[str]] = {
    "Bloc1": [
        "DWMAEIA ID Equity",
        "FINVPRI GR Equity",
    ],
    "Bloc2": [
        "HFHPERC LX Equity",
        "EXCRISA LX Equity",
        "PDAIEUR LX Equity",
        "HSMSTIC LX Equity",
        "REYLSEB LX Equity",
        "CARPPFE LX Equity",
        "ELEARER LX Equity",
        "LDAPB2E ID Equity",
        "LUMNUDE LX Equity",
        "EXCERFD LX Equity",
    ],
    "Bloc3": [
        "AEABSIA ID Equity",
        "BNPAOPP LX Equity",
        "CLOHQIA GR Equity",
        "INICLBI GR Equity",
        "DISFCPE LX Equity",
        "LIOFVS1 ID Equity",
        "MUESEIH ID Equity",
    ],
}


def _guess_project_root() -> Path:
    """Devine la racine projet à partir de quelques emplacements probables."""
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "outputs").exists() or (candidate / "data").exists():
            return candidate
    return here


@dataclass
class BlocConfig:
    """Paramètres de filtrage et de sélection pour un bloc satellite."""

    nom: str
    expense_max_default: float = 2.0
    expense_max_by_strategy: Dict[str, float] = field(default_factory=dict)
    vol_min_default: float = 0.0
    vol_min_by_strategy: Dict[str, float] = field(default_factory=dict)
    vol_max_default: float = 0.30
    vol_max_by_strategy: Dict[str, float] = field(default_factory=dict)
    sharpe_min: float = -0.5
    alpha_min_annual: float = -0.10
    drawdown_max: float = -0.80
    skew_min: float = -2.5
    kurtosis_max: float = 10.0
    concentration_max: float = 95.0
    beta_max_abs_override: float = -1.0
    beta_q75_max_override: float = -1.0
    beta_min_pass_ratio_override: float = -1.0
    n_select: int = 3
    max_per_strategy: int = 2
    min_select: int = 0
    fallback_skew_min: float | None = None
    fallback_kurtosis_max: float | None = None
    fallback_concentration_max: float | None = None
    fallback_use_level2_if_needed: bool = False


@dataclass
class SatelliteConfig:
    """Configuration globale du pipeline satellite."""

    project_root: Path = field(default_factory=_guess_project_root)

    info_paths: List[str] = field(default_factory=list)
    price_paths: List[str] = field(default_factory=list)

    core_daily_is_csv: str = ""
    core_daily_oos_csv: str = ""
    core_selected_csv: str = ""
    core3_daily_log_returns_csv: str = ""

    calib_start: str = "2019-01-01"
    calib_end: str = "2020-12-31"

    aum_min_m: float = 100.0
    max_start_date: str = "2019-01-01"
    allowed_currencies: List[str] = field(default_factory=lambda: ["Euro"])
    excluded_strategies: List[str] = field(default_factory=list)

    beta_rolling_days: int = 126
    beta_filter_window_days: int = 126
    beta_filter_max_abs: float = 0.20
    beta_filter_min_pass_ratio: float = 0.95
    beta_filter_q75_max: float = 0.30

    stale_max_ratio: float = 0.10
    stale_max_ratio_by_ticker: Dict[str, float] = field(
        default_factory=lambda: {
            "LIOFVS1 ID Equity": 0.95,
            "CLOHQIA GR Equity": 0.15,
            "INICLBI GR Equity": 0.10,
        }
    )

    corr_is_max: float = 0.45
    corr_pairwise_is_max: float = 0.70

    use_shortlist: bool = True
    satellite_shortlist: Dict[str, List[str]] = field(
        default_factory=lambda: SATELLITE_SHORTLIST
    )

    output_selected_csv: str = ""
    blocs: Dict[str, BlocConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        root = self.project_root
        if not self.core_daily_is_csv:
            self.core_daily_is_csv = str(root / "outputs" / "core_returns_daily_is.csv")
        if not self.core_daily_oos_csv:
            self.core_daily_oos_csv = str(root / "outputs" / "core_returns_daily_oos.csv")
        if not self.core_selected_csv:
            self.core_selected_csv = str(root / "outputs" / "core_selected_etfs.csv")
        if not self.core3_daily_log_returns_csv:
            self.core3_daily_log_returns_csv = str(
                root / "outputs" / "core3_etf_daily_log_returns.csv"
            )
        if not self.output_selected_csv:
            self.output_selected_csv = str(root / "outputs" / "satellite_selected.csv")
        if not self.info_paths:
            self.info_paths = [
                str(root / "data" / "STRAT1_info.xlsx"),
                str(root / "data" / "STRAT2_info.xlsx"),
                str(root / "data" / "STRAT3_info.xlsx"),
            ]
        if not self.price_paths:
            self.price_paths = [
                str(root / "data" / "STRAT1_price.xlsx"),
                str(root / "data" / "STRAT2_price.xlsx"),
                str(root / "data" / "STRAT3_price.xlsx"),
            ]
        if not self.blocs:
            self.blocs = _build_default_blocs()


def _build_default_blocs() -> Dict[str, BlocConfig]:
    """Construit les paramètres par défaut des trois blocs satellite."""
    return {
        "Bloc1": BlocConfig(
            nom="Bloc 1 – Décorrélation / Convexité",
            expense_max_default=2.0,
            vol_max_default=0.25,
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.80,
            skew_min=-2.0,
            kurtosis_max=10.0,
            concentration_max=95.0,
            beta_max_abs_override=0.55,
            beta_q75_max_override=0.75,
            beta_min_pass_ratio_override=0.60,
            n_select=2,
            max_per_strategy=2,
            min_select=1,
        ),
        "Bloc2": BlocConfig(
            nom="Bloc 2 – Alpha Décorrélé",
            expense_max_default=3.0,
            vol_max_default=0.20,
            vol_max_by_strategy={
                "Neutre au marché": 0.12,
                "CTA/futures gérés": 0.20,
                "Long Short": 0.15,
                "Equity Hedge": 0.15,
                "Mené par les événements": 0.12,
                "Multi-stratégie": 0.12,
            },
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.50,
            skew_min=-2.0,
            kurtosis_max=10.0,
            concentration_max=95.0,
            n_select=3,
            max_per_strategy=2,
            min_select=1,
        ),
        "Bloc3": BlocConfig(
            nom="Bloc 3 – Carry / Crédit Structuré",
            expense_max_default=1.8,
            expense_max_by_strategy={
                "Titres adossés à des actifs": 2.0,
                "Prêts bancaires": 2.0,
            },
            vol_max_default=0.20,
            vol_max_by_strategy={
                "Titres adossés à des actifs": 0.10,
                "Prêts bancaires": 0.10,
                "Obligataire Valeur relative": 0.12,
            },
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.70,
            skew_min=-3.0,
            kurtosis_max=15.0,
            concentration_max=100.0,
            n_select=2,
            max_per_strategy=2,
            min_select=1,
            fallback_skew_min=-4.0,
            fallback_kurtosis_max=25.0,
            fallback_concentration_max=100.0,
            fallback_use_level2_if_needed=True,
        ),
    }


def _parse_dates(x: pd.Series) -> pd.Series:
    """Parse robuste de dates : datetime natif, texte ou serial Excel."""
    dt = pd.to_datetime(x, errors="coerce", dayfirst=False)
    threshold = max(5, int(0.20 * len(x)))
    if dt.notna().sum() >= threshold:
        return dt

    num = pd.to_numeric(x, errors="coerce")
    valid = num.notna() & (num >= 1) & (num <= 80_000)
    dt2 = pd.Series(pd.NaT, index=x.index)
    if valid.any():
        dt2.loc[valid] = pd.to_datetime(
            num.loc[valid].astype(int),
            errors="coerce",
            unit="D",
            origin="1899-12-30",
        )
    return dt.fillna(dt2)


def _normalize_text(text: str) -> str:
    """Normalise une étiquette texte pour comparaisons robustes."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip().lower()


def _price_to_simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convertit des prix en rendements simples journaliers."""
    rets = prices.sort_index().pct_change(fill_method=None)
    return rets.replace([np.inf, -np.inf], np.nan)


def lire_prix_wide(path: str) -> pd.DataFrame:
    """Lit toutes les feuilles d'un fichier de prix satellite au format wide."""
    xls = pd.ExcelFile(path)
    seen: set[str] = set()
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

            price_col = j + 1
            if price_col >= df.shape[1]:
                continue

            dates = _parse_dates(data.iloc[:, j])
            prices = pd.to_numeric(data.iloc[:, price_col], errors="coerce")
            valid = dates.notna() & prices.notna()
            if valid.sum() < 5:
                continue

            series = pd.Series(
                prices[valid].values,
                index=pd.DatetimeIndex(dates[valid].values),
                name=ticker,
            ).sort_index()
            series = series[~series.index.duplicated(keep="first")]
            all_series[ticker] = series
            seen.add(ticker)

    wide = pd.DataFrame(all_series).sort_index()
    wide.index.name = "Date"
    return wide


def lire_info(path: str) -> pd.DataFrame:
    """Lit un fichier info satellite et normalise les colonnes clés."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        low = col.lower()
        if "total actifs usd" in low:
            rename_map[col] = "aum_usd_m"
        elif "ratio des dépenses" in low or "ratio des depenses" in low or "expense" in low:
            rename_map[col] = "expense_pct"
        elif "stratégie de fonds" in low or "strategie de fonds" in low:
            rename_map[col] = "strategie"
        elif "date de création" in low or "date de creation" in low:
            rename_map[col] = "date_creation"
        elif col.strip() == "Dev" or "devise" in low:
            rename_map[col] = "devise"
        elif "% des 10" in low or "premières positions" in low:
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
        df["date_creation"] = pd.to_datetime(df["date_creation"], errors="coerce")

    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    if "ticker" not in df.columns:
        df["ticker"] = df.iloc[:, 0]

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df.dropna(subset=["ticker"])
    return df.set_index("ticker")


def charger_toutes_les_donnees(cfg: SatelliteConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge et fusionne tous les fichiers info et prix des trois STRAT."""
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

    info_frames = []
    seen_tickers: set[str] = set()
    for path in cfg.info_paths:
        print(f"    Lecture info : {Path(path).name}")
        df = lire_info(path)
        new_tickers = [t for t in df.index if t not in seen_tickers]
        info_frames.append(df.loc[new_tickers])
        seen_tickers.update(new_tickers)

    all_info = pd.concat(info_frames)
    print(f"    → {len(all_info)} tickers info chargés au total")
    return all_prices, all_info


def _read_returns_csv(path: str) -> pd.Series:
    """Lit un CSV de rendements et détecte automatiquement log vs simple."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"CSV vide : {path}")

    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").dropna().sort_index()
    is_log = "log" in col.lower() or "log" in Path(path).stem.lower()
    if is_log:
        series = np.exp(series) - 1.0
    series.name = "core_return"
    return series


def charger_core_returns(cfg: SatelliteConfig) -> pd.Series:
    """Charge et concatène les rendements simples du Core sur IS + OOS."""
    series_list: List[pd.Series] = []
    for path in [cfg.core_daily_is_csv, cfg.core_daily_oos_csv]:
        if Path(path).exists():
            series_list.append(_read_returns_csv(path))

    if not series_list:
        raise FileNotFoundError(
            "Aucun fichier de rendements Core trouvé. "
            "Attendus : core_returns_daily_is.csv et/ou core_returns_daily_oos.csv"
        )

    core = pd.concat(series_list).sort_index()
    core = core[~core.index.duplicated(keep="first")].dropna()
    core.name = "core_return"
    return core


def _extract_core_tickers(selected: pd.DataFrame) -> List[str]:
    """Extrait la liste des tickers Core depuis différents schémas de colonnes."""
    candidates = ["core_etfs", "ticker", "Ticker"]
    for col in candidates:
        if col in selected.columns:
            return selected[col].astype(str).str.strip().dropna().tolist()
    return selected.iloc[:, 0].astype(str).str.strip().dropna().tolist()


def charger_core_eqw_returns_from_csv(cfg: SatelliteConfig) -> pd.Series:
    """Construit un benchmark Core équipondéré journalier en rendements simples."""
    selected = pd.read_csv(cfg.core_selected_csv)
    tickers = _extract_core_tickers(selected)
    if not tickers:
        raise ValueError("core_selected_etfs.csv ne contient aucun ticker exploitable.")

    core3_log = pd.read_csv(
        cfg.core3_daily_log_returns_csv,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    available = [t for t in tickers if t in core3_log.columns]
    if len(available) < 3:
        raise ValueError(
            "core3_etf_daily_log_returns.csv ne contient pas les 3 ETF sélectionnés. "
            f"Trouvés : {available}"
        )

    core3_simple = np.exp(core3_log[available[:3]]) - 1.0
    core_eqw = core3_simple.mean(axis=1).dropna()
    core_eqw.name = "core_eqw_return"
    return core_eqw


def _annualized_return(daily_rets: pd.Series) -> float:
    """Rendement annualisé à partir de rendements simples journaliers."""
    if len(daily_rets) == 0:
        return np.nan
    total = float((1 + daily_rets).prod())
    return float(total ** (252 / len(daily_rets)) - 1)


def _annualized_vol(daily_rets: pd.Series) -> float:
    """Volatilité annualisée à partir de rendements simples journaliers."""
    return float(daily_rets.std() * np.sqrt(252))


def _annualized_sharpe(daily_rets: pd.Series, rf_daily: pd.Series | None = None) -> float:
    """Sharpe annualisé (exces rf si rf_daily est fourni)."""
    if rf_daily is None:
        vol = _annualized_vol(daily_rets)
        if vol < 1e-10:
            return np.nan
        return _annualized_return(daily_rets) / vol

    rf_aligned = rf_daily.reindex(daily_rets.index).ffill().fillna(0.0)
    return sharpe_excess(daily_rets, rf_aligned)


def _max_drawdown(daily_rets: pd.Series) -> float:
    """Maximum drawdown sur une série de rendements simples."""
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    return float(((cum / peak) - 1).min())


def _ols_alpha_beta(fund_rets: pd.Series, core_rets: pd.Series) -> Tuple[float, float]:
    """Régression OLS : fund = alpha_daily + beta * core + eps."""
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
    rf_daily: pd.Series | None = None,
) -> pd.DataFrame:
    """Calcule les métriques IS 2019-2020 pour chaque fonds satellite."""
    fund_rets = _price_to_simple_returns(wide_prices).dropna(how="all")
    rets_calib = fund_rets.loc[calib_start:calib_end]
    core_calib = core_rets.loc[calib_start:calib_end]
    rf_calib = rf_daily.loc[calib_start:calib_end] if rf_daily is not None else None

    if core_calib.empty:
        raise ValueError("La série Core est vide sur la fenêtre de calibration 2019-2020.")

    bday_idx = pd.date_range(calib_start, calib_end, freq="B")
    covid_start, covid_end = "2020-02-01", "2020-05-31"
    core_covid = core_rets.loc[covid_start:covid_end]

    rows = []
    for ticker in rets_calib.columns:
        s = rets_calib[ticker].dropna()
        if len(s) < 30:
            continue

        prices_calib = wide_prices[ticker].loc[calib_start:calib_end]
        prices_b = prices_calib.reindex(bday_idx).ffill(limit=5)
        stale_mask = (prices_b.diff().abs() <= 1e-12) & prices_b.notna() & prices_b.shift(1).notna()
        stale_ratio = float(stale_mask.mean()) if len(stale_mask) else np.nan

        alpha, beta = _ols_alpha_beta(s, core_calib)
        aligned_corr = pd.concat([s, core_calib], axis=1, sort=True).dropna()
        corr_core = float(aligned_corr.iloc[:, 0].corr(aligned_corr.iloc[:, 1])) if len(aligned_corr) >= 30 else np.nan

        ann_ret = _annualized_return(s)
        downside = s[s < 0]
        if len(downside) >= 5:
            downside_vol = float(downside.std() * np.sqrt(252))
            sortino = ann_ret / downside_vol if downside_vol > 1e-10 else np.nan
        else:
            sortino = np.nan

        s_covid = fund_rets[ticker].loc[covid_start:covid_end].dropna()
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

        rows.append(
            {
                "ticker": ticker,
                "vol_calib": _annualized_vol(s),
                "sharpe_calib": _annualized_sharpe(s, rf_calib),
                "sortino_calib": sortino,
                "alpha_annual": alpha,
                "beta_core": beta,
                "corr_core_calib": corr_core,
                "drawdown_calib": _max_drawdown(s),
                "skew_calib": float(stats.skew(s, nan_policy="omit")),
                "kurtosis_calib": float(stats.kurtosis(s, nan_policy="omit")),
                "n_obs_calib": len(s),
                "stale_ratio_calib": stale_ratio,
                "dd_covid": dd_covid,
                "ret_rel_covid": ret_rel_covid,
            }
        )

    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def calculer_beta_rolling(
    wide_prices: pd.DataFrame,
    core_rets: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Calcule le bêta rolling des fonds satellite vs Core."""
    fund_rets = _price_to_simple_returns(wide_prices).dropna(how="all")
    aligned = pd.concat([fund_rets, core_rets.rename("__core__")], axis=1, sort=True)

    betas: Dict[str, pd.Series] = {}
    for ticker in fund_rets.columns:
        pair = aligned[[ticker, "__core__"]].dropna()
        if len(pair) < window:
            continue
        cov = pair[ticker].rolling(window).cov(pair["__core__"])
        var = pair["__core__"].rolling(window).var()
        beta = (cov / var).where(var > 1e-12)
        betas[ticker] = beta

    return pd.DataFrame(betas)


def filtrer_niveau0(
    info: pd.DataFrame,
    wide_prices: pd.DataFrame,
    cfg: SatelliteConfig,
) -> List[str]:
    """Niveau 0 : présence, AUM, historique, devise, stratégie."""
    valid: List[str] = []
    max_date = pd.Timestamp(cfg.max_start_date)
    allowed_norm = {_normalize_text(c) for c in cfg.allowed_currencies if str(c).strip()}

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
        if pd.isna(expense) and not cfg.use_shortlist:
            continue

        valid.append(ticker)

    return valid


def filtrer_niveau_beta_initial(
    tickers: List[str],
    beta_rolling: pd.DataFrame,
    cfg: SatelliteConfig,
    bloc_cfg: BlocConfig | None = None,
) -> List[str]:
    """Filtre initial de bêta rolling 3 mois vs Core équipondéré."""
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

    beta_window = beta_rolling.loc[cfg.calib_start:cfg.calib_end]
    valid: List[str] = []

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

        if median_beta <= max_abs and q75_beta <= q75_max and pass_ratio >= min_pass:
            valid.append(ticker)

    return valid


def filtrer_niveau1(
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
    cfg: SatelliteConfig,
) -> List[str]:
    """Niveau 1 : frais, volatilité et stale pricing."""
    valid: List[str] = []
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

        stale = metrics.at[ticker, "stale_ratio_calib"]
        stale_threshold = cfg.stale_max_ratio_by_ticker.get(ticker, cfg.stale_max_ratio)
        if not np.isnan(stale) and stale > stale_threshold:
            continue

        valid.append(ticker)

    return valid


def filtrer_niveau2(
    tickers: List[str],
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
    cfg: SatelliteConfig,
) -> List[str]:
    """Niveau 2 : qualité quantitative et corrélation IS."""
    valid: List[str] = []
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
        corr = row.get("corr_core_calib", np.nan)
        if not np.isnan(corr) and corr > cfg.corr_is_max:
            continue

        valid.append(ticker)

    return valid


def filtrer_niveau3_custom(
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
    skew_min: float,
    kurtosis_max: float,
    concentration_max: float,
) -> List[str]:
    """Niveau 3 paramétrable : skewness, kurtosis et concentration."""
    valid: List[str] = []
    for ticker in tickers:
        if ticker in metrics.index:
            skew = metrics.at[ticker, "skew_calib"]
            kurt = metrics.at[ticker, "kurtosis_calib"]
            if not np.isnan(skew) and skew < skew_min:
                continue
            if not np.isnan(kurt) and kurt > kurtosis_max:
                continue

        if ticker in info.index and "concentration" in info.columns:
            conc = info.at[ticker, "concentration"]
            if not np.isnan(conc) and conc > concentration_max:
                continue

        valid.append(ticker)

    return valid


def filtrer_niveau3(
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
) -> List[str]:
    """Niveau 3 : skewness, kurtosis et concentration."""
    return filtrer_niveau3_custom(
        tickers=tickers,
        info=info,
        metrics=metrics,
        skew_min=bloc_cfg.skew_min,
        kurtosis_max=bloc_cfg.kurtosis_max,
        concentration_max=bloc_cfg.concentration_max,
    )


def _zscore_col(series: pd.Series) -> pd.Series:
    """Calcule un z-score robuste avec remplacement des NaN par 0."""
    mu, sigma = series.mean(), series.std()
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return ((series - mu) / sigma).fillna(0.0)


def scorer(
    tickers: List[str],
    metrics: pd.DataFrame,
    weights: Dict[str, float] = SCORE_WEIGHTS,
) -> pd.Series:
    """Construit le score composite intra-bloc."""
    df = metrics.loc[[t for t in tickers if t in metrics.index]].copy()
    if df.empty:
        return pd.Series(dtype=float)

    score = (
        weights.get("neg_abs_beta", 0.30) * _zscore_col(-df["beta_core"].abs())
        + weights.get("neg_corr_core", 0.10) * _zscore_col(-df["corr_core_calib"])
        + weights.get("sortino", 0.25) * _zscore_col(df["sortino_calib"])
        + weights.get("ret_rel_covid", 0.15) * _zscore_col(df["ret_rel_covid"])
        + weights.get("neg_dd_covid", 0.10) * _zscore_col(-df["dd_covid"])
        + weights.get("skew", 0.05) * _zscore_col(df["skew_calib"])
        + weights.get("neg_kurtosis", 0.05) * _zscore_col(-df["kurtosis_calib"])
    )
    return score.sort_values(ascending=False)


def selectionner(scores: pd.Series, info: pd.DataFrame, bloc_cfg: BlocConfig) -> List[str]:
    """Sélectionne les meilleurs fonds sous contrainte max par stratégie."""
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
    """Applique le filtre de corrélation pairwise sur la fenêtre IS."""
    fund_rets = _price_to_simple_returns(wide_prices_calib).dropna(how="all")
    selected: List[str] = []
    reserves: List[str] = []
    n_reserves = 2

    for ticker in tickers_ranked:
        if ticker not in fund_rets.columns:
            continue

        s = fund_rets[ticker].dropna()
        if len(s) < 10:
            continue

        ok = True
        for t_sel in selected:
            aligned = pd.concat([s, fund_rets[t_sel].dropna()], axis=1, sort=True).dropna()
            if len(aligned) < 10:
                continue
            corr_val = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if abs(corr_val) > corr_max:
                ok = False
                break

        if len(selected) < n_select and ok:
            selected.append(ticker)
        elif len(reserves) < n_reserves and ok:
            reserves.append(ticker)

        if len(selected) >= n_select and len(reserves) >= n_reserves:
            break

    return selected, reserves


def _print_level3_diagnostics(
    bloc_name: str,
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Affiche pourquoi les candidats restants tombent potentiellement au Niveau 3."""
    if not tickers:
        return
    print(f"\n  [diagnostic {bloc_name} - candidats t2]")
    for ticker in tickers:
        skew = metrics.at[ticker, "skew_calib"] if ticker in metrics.index else np.nan
        kurt = metrics.at[ticker, "kurtosis_calib"] if ticker in metrics.index else np.nan
        conc = (
            info.at[ticker, "concentration"]
            if (ticker in info.index and "concentration" in info.columns)
            else np.nan
        )
        print(
            f"    {ticker:<28s} | skew={skew:>7.3f} | "
            f"kurt={kurt:>7.3f} | concentration={conc}"
        )


def traiter_bloc(
    bloc_name: str,
    bloc_cfg: BlocConfig,
    all_prices: pd.DataFrame,
    all_info: pd.DataFrame,
    core_rets: pd.Series,
    core_eqw_rets: pd.Series,
    rf_daily: pd.Series,
    cfg: SatelliteConfig,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Traite un bloc complet : shortlist, filtres, score, sélection, réserves."""
    print(f"\n{'=' * 60}")
    print(f"  {bloc_cfg.nom}")
    print(f"{'=' * 60}")

    common_all = [t for t in all_prices.columns if t in all_info.index]
    prices = all_prices[common_all]
    info = all_info.loc[[t for t in common_all if t in all_info.index]]
    print(f"  [données] {len(common_all)} fonds avec prix ET info")

    if cfg.use_shortlist and bloc_name in cfg.satellite_shortlist:
        shortlist = cfg.satellite_shortlist[bloc_name]
        shortlist_present = [t for t in shortlist if t in prices.columns and t in info.index]
        shortlist_absent = [t for t in shortlist if t not in prices.columns or t not in info.index]

        for ticker in shortlist_absent:
            in_prices = ticker in all_prices.columns
            in_info = ticker in all_info.index
            logging.warning(
                "[%s] Ticker shortlist absent : %s (prix=%s, info=%s)",
                bloc_name,
                ticker,
                in_prices,
                in_info,
            )
            print(f"  ⚠ Ticker shortlist absent : {ticker} (prix={in_prices}, info={in_info})")

        prices = prices[[t for t in prices.columns if t in shortlist_present]]
        info = info.loc[[t for t in info.index if t in shortlist_present]]
        common = shortlist_present
        print(f"  [shortlist] {len(common)} fonds retenus sur {len(shortlist)}")
    else:
        common = common_all

    if not common:
        print(f"  ⚠ Aucun fonds exploitable dans {bloc_name}.")
        return [], pd.DataFrame(), info, pd.DataFrame(), []

    print(f"  [métriques] Calcul sur {cfg.calib_start} → {cfg.calib_end}")
    print(f"  [beta init] Rolling {cfg.beta_filter_window_days}j vs Core équipondéré")
    beta_init = calculer_beta_rolling(prices, core_eqw_rets, cfg.beta_filter_window_days)
    t_beta = filtrer_niveau_beta_initial(common, beta_init, cfg, bloc_cfg)

    beta_max = (
        bloc_cfg.beta_max_abs_override
        if bloc_cfg.beta_max_abs_override > 0
        else cfg.beta_filter_max_abs
    )
    beta_q75 = (
        bloc_cfg.beta_q75_max_override
        if bloc_cfg.beta_q75_max_override > 0
        else cfg.beta_filter_q75_max
    )
    beta_pass = (
        bloc_cfg.beta_min_pass_ratio_override
        if bloc_cfg.beta_min_pass_ratio_override > 0
        else cfg.beta_filter_min_pass_ratio
    )
    print(
        f"  [Niv.Beta] median|β|<={beta_max:.0%} "
        f"q75|β|<={beta_q75:.0%} "
        f"pass>={beta_pass:.0%}   {len(t_beta):3d} / {len(common)}"
    )

    if not t_beta:
        print(f"  ⚠ Aucun fonds ne passe le filtre beta initial dans {bloc_name}.")
        return [], pd.DataFrame(), info, pd.DataFrame(), []

    prices = prices[t_beta]
    metrics = calculer_metriques_calib(
        prices,
        core_rets,
        cfg.calib_start,
        cfg.calib_end,
        rf_daily=rf_daily,
    )
    print(f"             {len(metrics)} fonds avec données suffisantes")

    t0 = filtrer_niveau0(info, prices, cfg)
    print(f"  [Niv.0] AUM/Date/Devise/Excl.  {len(t0):3d} / {len(common)}")

    t1 = filtrer_niveau1(t0, info, metrics, bloc_cfg, cfg)
    print(
        f"  [Niv.1] Frais/Vol[{bloc_cfg.vol_min_default:.0%},{bloc_cfg.vol_max_default:.0%}]"
        f"/Stale<= {cfg.stale_max_ratio:.0%} {len(t1):3d} / {len(t0)}"
    )

    t2 = filtrer_niveau2(t1, metrics, bloc_cfg, cfg)
    print(f"  [Niv.2] Sharpe/Alpha/DD/Corr   {len(t2):3d} / {len(t1)}")

    t3 = filtrer_niveau3(t2, info, metrics, bloc_cfg)
    print(f"  [Niv.3] Skew/Kurt/Conc.        {len(t3):3d} / {len(t2)}")

    candidate_pool = t3.copy()

    if len(candidate_pool) < bloc_cfg.min_select and len(t2) > 0:
        _print_level3_diagnostics(bloc_name, t2, info, metrics)

        fallback_skew = (
            bloc_cfg.fallback_skew_min
            if bloc_cfg.fallback_skew_min is not None
            else bloc_cfg.skew_min
        )
        fallback_kurt = (
            bloc_cfg.fallback_kurtosis_max
            if bloc_cfg.fallback_kurtosis_max is not None
            else bloc_cfg.kurtosis_max
        )
        fallback_conc = (
            bloc_cfg.fallback_concentration_max
            if bloc_cfg.fallback_concentration_max is not None
            else bloc_cfg.concentration_max
        )

        relaxed_t3 = filtrer_niveau3_custom(
            tickers=t2,
            info=info,
            metrics=metrics,
            skew_min=fallback_skew,
            kurtosis_max=fallback_kurt,
            concentration_max=fallback_conc,
        )

        if len(relaxed_t3) >= bloc_cfg.min_select:
            candidate_pool = relaxed_t3
            print(
                f"  [Fallback Niv.3] seuils assouplis → {len(candidate_pool):3d} / {len(t2)}"
            )
        elif bloc_cfg.fallback_use_level2_if_needed:
            candidate_pool = t2.copy()
            print(
                f"  [Fallback Niv.3] utilisation directe des candidats Niveau 2 "
                f"→ {len(candidate_pool):3d} / {len(t2)}"
            )

    if not candidate_pool:
        print(f"  ⚠ Aucun fonds ne passe tous les filtres dans {bloc_name}.")
        return [], metrics, info, pd.DataFrame(), []

    scores = scorer(candidate_pool, metrics)
    ranked_extended = selectionner(
        scores,
        info,
        BlocConfig(
            nom=bloc_cfg.nom,
            n_select=max(bloc_cfg.n_select + 2, bloc_cfg.min_select + 2),
            max_per_strategy=bloc_cfg.max_per_strategy,
        ),
    )

    prices_calib = prices.loc[cfg.calib_start:cfg.calib_end]
    selected, reserves = filtrer_coherence_pairwise(
        ranked_extended,
        prices_calib,
        bloc_cfg.n_select,
        cfg.corr_pairwise_is_max,
    )

    if len(selected) < bloc_cfg.min_select:
        remaining = [t for t in ranked_extended if t not in selected]
        for ticker in remaining:
            selected.append(ticker)
            if len(selected) >= bloc_cfg.min_select:
                break
        if len(selected) > 0:
            print(
                f"  [Fallback sélection] garantie d'au moins {bloc_cfg.min_select} "
                f"fonds dans {bloc_name}"
            )

    print(f"\n  [sélection] {bloc_name} – {len(selected)} fonds retenus :")
    for rank, ticker in enumerate(selected, 1):
        strat = info.at[ticker, "strategie"] if "strategie" in info.columns else "?"
        sc = scores.get(ticker, np.nan)
        beta = metrics.at[ticker, "beta_core"] if ticker in metrics.index else np.nan
        sh = metrics.at[ticker, "sharpe_calib"] if ticker in metrics.index else np.nan
        print(
            f"    #{rank}  {ticker:<28s}  strat={strat:<28s}  "
            f"score={sc:+.3f}  β={beta:+.3f}  Sharpe={sh:.3f}"
        )

    if reserves:
        print(f"  [réserves] {bloc_name} – {len(reserves)} fonds en réserve :")
        for rank, ticker in enumerate(reserves, 1):
            strat = info.at[ticker, "strategie"] if "strategie" in info.columns else "?"
            sc = scores.get(ticker, np.nan)
            print(f"    R{rank}  {ticker:<28s}  strat={strat:<28s}  score={sc:+.3f}")

    print(f"\n  [beta rolling {cfg.beta_rolling_days}j]...")
    prices_sel = prices[[t for t in selected if t in prices.columns]]
    beta_roll = calculer_beta_rolling(prices_sel, core_rets, cfg.beta_rolling_days)
    return selected, metrics, info, beta_roll, reserves


def _build_export_rows(
    all_selected: Dict[str, List[str]],
    all_info_bloc: Dict[str, pd.DataFrame],
    all_metrics: Dict[str, pd.DataFrame],
) -> List[Dict]:
    """Construit les lignes de l'export satellite_selected.csv."""
    rows: List[Dict] = []
    for bloc_name, tickers in all_selected.items():
        metrics = all_metrics[bloc_name]
        info = all_info_bloc[bloc_name]
        for ticker in tickers:
            row: Dict = {"bloc": bloc_name, "ticker": ticker}
            if ticker in info.index:
                for col in ["strategie", "devise", "expense_pct", "aum_usd_m"]:
                    row[col] = info.at[ticker, col] if col in info.columns else np.nan
            if ticker in metrics.index:
                for col in [
                    "vol_calib",
                    "sharpe_calib",
                    "sortino_calib",
                    "alpha_annual",
                    "beta_core",
                    "corr_core_calib",
                    "drawdown_calib",
                    "skew_calib",
                    "kurtosis_calib",
                    "n_obs_calib",
                    "stale_ratio_calib",
                    "dd_covid",
                    "ret_rel_covid",
                ]:
                    row[col] = metrics.at[ticker, col] if col in metrics.columns else np.nan
            rows.append(row)
    return rows


def main(cfg: SatelliteConfig | None = None) -> None:
    """Lance le pipeline satellite complet."""
    cfg = cfg or SatelliteConfig()

    env_allowed = os.getenv("SAT_ALLOWED_CURRENCIES", "").strip()
    if env_allowed:
        cfg.allowed_currencies = [c.strip() for c in env_allowed.split(",") if c.strip()]

    print("=" * 60)
    print("  PIPELINE SATELLITE v5 – Shortlist + Filtres IS corrigés")
    print(f"  Fenêtre calib : {cfg.calib_start} → {cfg.calib_end}")
    if cfg.allowed_currencies:
        print(f"  Filtre devise : {cfg.allowed_currencies}")
    else:
        print("  Filtre devise : désactivé")
    if cfg.use_shortlist:
        total_shortlist = sum(len(v) for v in cfg.satellite_shortlist.values())
        print(f"  Shortlist : {total_shortlist} fonds pré-sélectionnés")
    print("=" * 60)

    print("\n[0] Chargement des données (STRAT1 + STRAT2 + STRAT3)...")
    all_prices, all_info = charger_toutes_les_donnees(cfg)

    print("\n[1] Chargement des rendements Core...")
    core_rets = charger_core_returns(cfg)
    print(
        f"    Core global : {len(core_rets)} obs. daily | "
        f"{core_rets.index.min().date()} → {core_rets.index.max().date()}"
    )
    core_calib = core_rets.loc[cfg.calib_start:cfg.calib_end]
    print(
        f"    Core calib  : {len(core_calib)} obs. | "
        f"{core_calib.index.min().date()} → {core_calib.index.max().date()}"
    )
    if core_calib.empty:
        raise ValueError("Le Core ne couvre pas la fenêtre de calibration 2019-2020.")

    core_eqw_rets = charger_core_eqw_returns_from_csv(cfg)
    print("    Benchmark filtre beta : Core équipondéré des 3 ETF sélectionnés")
    print(
        f"    Core EQW    : {len(core_eqw_rets)} obs. daily | "
        f"{core_eqw_rets.index.min().date()} → {core_eqw_rets.index.max().date()}"
    )

    rf_daily, rf_source = get_bund_risk_free_daily(core_rets.index)
    print(f"    Risk-free Bund : {rf_source}")

    all_selected: Dict[str, List[str]] = {}
    all_reserves: Dict[str, List[str]] = {}
    all_info_bloc: Dict[str, pd.DataFrame] = {}
    all_metrics: Dict[str, pd.DataFrame] = {}
    all_betas: Dict[str, pd.DataFrame] = {}
    all_scores: Dict[str, pd.Series] = {}

    for bloc_name, bloc_cfg in cfg.blocs.items():
        sel, metrics, info, beta_roll, reserves = traiter_bloc(
            bloc_name,
            bloc_cfg,
            all_prices,
            all_info,
            core_rets,
            core_eqw_rets,
            rf_daily,
            cfg,
        )
        all_selected[bloc_name] = sel
        all_reserves[bloc_name] = reserves
        all_info_bloc[bloc_name] = info
        all_metrics[bloc_name] = metrics
        all_betas[bloc_name] = beta_roll
        if metrics is not None and not metrics.empty:
            all_scores[bloc_name] = scorer(sel + reserves, metrics)
        else:
            all_scores[bloc_name] = pd.Series(dtype=float)

    print("\n[2] Export...")
    out_dir = Path(cfg.output_selected_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(_build_export_rows(all_selected, all_info_bloc, all_metrics))
    df_out.to_csv(cfg.output_selected_csv, index=False)
    print(f"    -> {cfg.output_selected_csv}")

    v3_path = str(out_dir / "satellite_selected_v3.csv")
    v3_rows: List[Dict] = []
    for bloc_name, tickers in all_selected.items():
        metrics = all_metrics[bloc_name]
        info = all_info_bloc[bloc_name]
        scores_bloc = all_scores.get(bloc_name, pd.Series(dtype=float))

        for rank, ticker in enumerate(tickers, 1):
            row: Dict = {
                "ticker": ticker,
                "bloc": bloc_name,
                "rank": rank,
                "score": scores_bloc.get(ticker, np.nan),
                "strategie": info.at[ticker, "strategie"]
                if (ticker in info.index and "strategie" in info.columns)
                else np.nan,
            }
            if ticker in metrics.index:
                for col in [
                    "vol_calib",
                    "sharpe_calib",
                    "sortino_calib",
                    "alpha_annual",
                    "beta_core",
                    "corr_core_calib",
                    "dd_covid",
                    "ret_rel_covid",
                    "stale_ratio_calib",
                ]:
                    row[col] = metrics.at[ticker, col] if col in metrics.columns else np.nan
            v3_rows.append(row)

    df_v3 = pd.DataFrame(v3_rows)
    df_v3.to_csv(v3_path, index=False)
    print(f"    -> {v3_path}")

    reserves_path = str(out_dir / "satellite_reserves.csv")
    reserve_rows: List[Dict] = []
    for bloc_name, reserves in all_reserves.items():
        info = all_info_bloc[bloc_name]
        scores_bloc = all_scores.get(bloc_name, pd.Series(dtype=float))
        for rank_r, ticker in enumerate(reserves, 1):
            reserve_rows.append(
                {
                    "bloc": bloc_name,
                    "rank_reserve": rank_r,
                    "ticker": ticker,
                    "score": scores_bloc.get(ticker, np.nan),
                    "strategie": info.at[ticker, "strategie"]
                    if (ticker in info.index and "strategie" in info.columns)
                    else np.nan,
                }
            )

    df_reserves = pd.DataFrame(reserve_rows)
    df_reserves.to_csv(reserves_path, index=False)
    print(f"    -> {reserves_path}")

    for bloc_name, beta_roll in all_betas.items():
        if beta_roll.empty:
            continue
        out_path = str(out_dir / f"satellite_beta_rolling_{bloc_name}.csv")
        beta_roll.to_csv(out_path)
        print(f"    -> {out_path}")

    print("\n" + "=" * 60)
    print("  RÉSUMÉ – Fonds satellite sélectionnés")
    print("=" * 60)
    for bloc_name, tickers in all_selected.items():
        print(f"\n  {bloc_name} ({len(tickers)} fonds) :")
        for ticker in tickers:
            print(f"    • {ticker}")

        reserves = all_reserves.get(bloc_name, [])
        if reserves:
            print(f"  Réserves {bloc_name} :")
            for ticker in reserves:
                print(f"    ◦ {ticker}")

    total = sum(len(v) for v in all_selected.values())
    print(f"\n  Total : {total} fonds satellite")


if __name__ == "__main__":
    main()
