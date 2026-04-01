"""
Core data loading — lecture Excel, validation structurelle, log-rendements des 3 ETF Core.

Source : data/univers_core_etf_eur_daily_wide.xlsx
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CoreConfig:
    """Paramètres du pipeline Core."""

    project_root: Path = Path(__file__).resolve().parent.parent
    core_excel: Path = project_root / "data" / "univers_core_etf_eur_daily_wide.xlsx"

    # Onglets prix
    sheet_equity_prices: str = "Equity_Wide_Daily_Values"
    sheet_credit_prices: str = "Credit_Wide_Daily_Values"
    sheet_rates_prices: str = "Rates_Wide_Daily_Values"

    # Onglets metadata
    sheet_equity_meta: str = "Equity"
    sheet_credit_meta: str = "Credit"
    sheet_rates_meta: str = "Rates"

    # ETF Core retenus
    selected_equity: str = "XDWD GY Equity"
    selected_rates: str = "EUNH GY Equity"
    selected_credit: str = "XBLC GY Equity"

    # Filtres structurels
    max_start_date: str = "2019-01-01"
    max_avg_gap_days: float = 2.0
    total_fee_budget_bps: float = 80.0
    w_core_mid: float = 0.725
    satellite_expense_bps: float = 60.0
    require_expense_info: bool = False

    # Fenêtres
    warm_up_start: str = "2018-01-01"
    score_start: str = "2019-01-01"
    score_end: str = "2020-12-31"
    oos_start: str = "2021-01-01"
    oos_end: str = "2025-12-31"

    # Backtest
    lookback: int = 252
    rebal_freq: int = 63
    w_min: float = 0.05
    w_max: float = 0.50
    equity_weight_floor: float = 0.30
    rolling_method: str = "risk_parity_tilt"
    equity_weight_ceiling: float = 0.60
    momentum_window: int = 252
    momentum_threshold: float = 0.0
    use_ledoit_wolf: bool = True

    # Sorties
    output_dir_name: str = "outputs"

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.output_dir_name

    @property
    def selected_core_map(self) -> Dict[str, str]:
        return {
            "Equity": self.selected_equity,
            "Rates": self.selected_rates,
            "Credit": self.selected_credit,
        }

    @property
    def max_core_expense_pct(self) -> float:
        w_sat = 1.0 - self.w_core_mid
        max_bps = (
            self.total_fee_budget_bps - w_sat * self.satellite_expense_bps
        ) / self.w_core_mid
        return max_bps / 100.0


# ══════════════════════════════════════════════════════════════════════════════
#  Lecture Excel
# ══════════════════════════════════════════════════════════════════════════════

def _lire_wide_values(path: Path, sheet: str) -> pd.DataFrame:
    """Lit un onglet *_Wide_Daily_Values en format wide."""
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    tickers_row = df.iloc[6].tolist()
    tickers: List[str] = []
    for i, value in enumerate(tickers_row):
        if i == 0:
            continue
        if isinstance(value, str) and value.strip():
            tickers.append(value.strip())
        else:
            tickers.append(f"__col{i}__")

    data = df.iloc[10:].copy()
    data.columns = ["Date"] + tickers[: data.shape[1] - 1]
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).set_index("Date").sort_index()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[[col for col in data.columns if not col.startswith("__col")]]
    data.index.name = "Date"
    return data


def _normalise_header_name(value: object, idx: int) -> str:
    """Normalise les noms de colonnes metadata."""
    text = str(value).strip().lower() if pd.notna(value) else f"col_{idx}"
    if "bloomberg" in text:
        return "ticker"
    if "ter" in text:
        return "ter_pct"
    if "devise" in text:
        return "devise"
    if "nom" in text:
        return "nom"
    if "exposition" in text or "indice" in text:
        return "exposition"
    if "provider" in text:
        return "provider"
    if "isin" in text:
        return "isin"
    if "encours" in text:
        return "encours_eur_m"
    return text


def _lire_metadata(path: Path, sheet: str) -> pd.DataFrame:
    """Lit un onglet metadata et renvoie un DataFrame indexé par ticker."""
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    headers = df.iloc[4].tolist()
    col_map = {i: _normalise_header_name(h, i) for i, h in enumerate(headers)}

    data = df.iloc[5:].copy()
    data.columns = range(len(data.columns))
    data = data.rename(columns=col_map)

    if "ticker" not in data.columns:
        raise ValueError(f"Onglet metadata {sheet} : colonne ticker introuvable")

    data["ticker"] = data["ticker"].astype(str).str.strip()
    data = data.dropna(subset=["ticker"])
    data = data[data["ticker"] != "nan"]
    data = data.set_index("ticker")

    if "ter_pct" in data.columns:
        data["ter_pct"] = pd.to_numeric(data["ter_pct"], errors="coerce")
        valid = data["ter_pct"].dropna()
        if not valid.empty and valid.max() < 0.1:
            data["ter_pct"] = data["ter_pct"] * 100.0

    if "encours_eur_m" in data.columns:
        data["encours_eur_m"] = pd.to_numeric(data["encours_eur_m"], errors="coerce")

    return data


def lire_theme(
    cfg: CoreConfig,
    theme_name: str,
    sheet_prices: str,
    sheet_meta: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lit les prix et metadata d'un thème."""
    if verbose:
        print(f"  Lecture {theme_name}...")
    wide = _lire_wide_values(cfg.core_excel, sheet_prices)
    meta = _lire_metadata(cfg.core_excel, sheet_meta)
    if verbose:
        print(
            f"    -> {wide.shape[1]} ETFs | "
            f"{wide.index.min().date()} à {wide.index.max().date()}"
        )
    return wide, meta


# ══════════════════════════════════════════════════════════════════════════════
#  Validation structurelle
# ══════════════════════════════════════════════════════════════════════════════

def _compute_structure_row(
    ticker: str,
    prices: pd.Series,
    expense_pct: float,
    cfg: CoreConfig,
) -> dict:
    """Construit une ligne de contrôle structurel pour un ETF."""
    clean = prices.dropna()
    if clean.empty:
        raise ValueError(f"{ticker} : aucune donnée de prix exploitable")

    first_date = clean.index.min()
    avg_gap = clean.index.to_series().diff().dropna().dt.days.mean()
    pass_date = first_date <= pd.Timestamp(cfg.max_start_date)
    pass_freq = avg_gap <= cfg.max_avg_gap_days

    if cfg.require_expense_info:
        pass_expense = pd.notna(expense_pct) and expense_pct <= cfg.max_core_expense_pct
    else:
        pass_expense = pd.isna(expense_pct) or expense_pct <= cfg.max_core_expense_pct

    return {
        "ticker": ticker,
        "first_date": first_date.strftime("%Y-%m-%d"),
        "n_obs": int(clean.shape[0]),
        "avg_gap": round(float(avg_gap), 2),
        "expense_pct": expense_pct,
        "pass_date": bool(pass_date),
        "pass_freq": bool(pass_freq),
        "pass_expense": bool(pass_expense),
        "selected": bool(pass_date and pass_freq and pass_expense),
    }


def _validate_selected_ticker(
    theme: str,
    ticker: str,
    wide: pd.DataFrame,
    meta: pd.DataFrame,
    cfg: CoreConfig,
) -> dict:
    """Valide qu'un ETF Core existe et passe les filtres."""
    if ticker not in wide.columns:
        raise ValueError(f"{theme} : ticker {ticker} absent de l'onglet prix")
    if ticker not in meta.index:
        raise ValueError(f"{theme} : ticker {ticker} absent de l'onglet metadata")

    expense = np.nan
    if "ter_pct" in meta.columns:
        value = meta.at[ticker, "ter_pct"]
        if pd.notna(value):
            expense = float(value)

    row = _compute_structure_row(ticker, wide[ticker], expense, cfg)
    if not row["selected"]:
        raise ValueError(
            f"{theme} : {ticker} ne passe pas les filtres "
            f"(date={row['pass_date']}, freq={row['pass_freq']}, "
            f"expense={row['pass_expense']})"
        )
    return row


def _collect_selected_metadata(
    selected_map: Dict[str, str],
    metas: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Assemble les metadata utiles des 3 ETF Core."""
    rows = []
    cols = [
        "nom", "provider", "isin", "devise",
        "ter_pct", "encours_eur_m", "exposition",
    ]
    for theme, ticker in selected_map.items():
        meta = metas[theme]
        row = {"Theme": theme, "Ticker": ticker}
        for col in cols:
            row[col] = (
                meta.at[ticker, col]
                if col in meta.columns and ticker in meta.index
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Construction des log-rendements Core
# ══════════════════════════════════════════════════════════════════════════════

def _build_selected_prices(
    selected_map: Dict[str, str],
    wide_eq: pd.DataFrame,
    wide_rt: pd.DataFrame,
    wide_cr: pd.DataFrame,
    cfg: CoreConfig,
) -> pd.DataFrame:
    """Assemble les 3 ETF Core dans l'ordre Equity / Rates / Credit."""
    prices = pd.concat(
        [
            wide_eq[[selected_map["Equity"]]],
            wide_rt[[selected_map["Rates"]]],
            wide_cr[[selected_map["Credit"]]],
        ],
        axis=1,
    ).sort_index()

    prices = prices.dropna(how="all").ffill()
    prices = prices.loc[cfg.warm_up_start:]
    prices.columns = [
        selected_map["Equity"],
        selected_map["Rates"],
        selected_map["Credit"],
    ]
    return prices


def load_selected_core_log_returns(
    cfg: CoreConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reconstruit les log-rendements des 3 ETF Core depuis l'Excel source."""
    cfg = cfg or CoreConfig()

    if not cfg.core_excel.exists():
        raise FileNotFoundError(f"Fichier introuvable : {cfg.core_excel}")

    wide_eq, meta_eq = lire_theme(
        cfg, "Equity", cfg.sheet_equity_prices, cfg.sheet_equity_meta, verbose=verbose,
    )
    wide_rt, meta_rt = lire_theme(
        cfg, "Rates", cfg.sheet_rates_prices, cfg.sheet_rates_meta, verbose=verbose,
    )
    wide_cr, meta_cr = lire_theme(
        cfg, "Credit", cfg.sheet_credit_prices, cfg.sheet_credit_meta, verbose=verbose,
    )

    selected_map = cfg.selected_core_map
    _validate_selected_ticker("Equity", selected_map["Equity"], wide_eq, meta_eq, cfg)
    _validate_selected_ticker("Rates", selected_map["Rates"], wide_rt, meta_rt, cfg)
    _validate_selected_ticker("Credit", selected_map["Credit"], wide_cr, meta_cr, cfg)

    wide_core = _build_selected_prices(selected_map, wide_eq, wide_rt, wide_cr, cfg)
    core_3_log = np.log(wide_core).diff().dropna()
    core_3_log.index = pd.DatetimeIndex(core_3_log.index).tz_localize(None)
    return core_3_log
