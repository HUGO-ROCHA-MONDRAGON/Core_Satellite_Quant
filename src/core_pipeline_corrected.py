"""
Pipeline Core – version propre et simplifiée pour le projet Core/Satellite.

Logique retenue :
    1) Le Core est figé manuellement sur 3 ETF :
       - Equity : XDWD
       - Rates  : EUNH
       - Credit : D5BG
    2) On vérifie que ces 3 ETF existent bien dans l'Excel et qu'ils passent
       les filtres structurels minimaux.
    3) On construit le Core à partir de ces 3 briques uniquement.
    4) On backteste le portefeuille Core en rolling.
    5) On exporte :
       - les rendements journaliers du Core en simple returns,
       - les versions log,
       - les rendements des 3 ETF,
       - les poids rolling,
       - les fichiers de sélection.

Source de données :
    data/univers_core_etf_eur_daily_wide.xlsx
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
    output_core_daily_csv_name: str = "core_returns_daily_oos.csv"
    output_core_daily_is_csv_name: str = "core_returns_daily_is.csv"
    output_core_daily_log_csv_name: str = "core_returns_daily_oos_log.csv"
    output_core_daily_is_log_csv_name: str = "core_returns_daily_is_log.csv"
    output_selected_core_csv_name: str = "core_selected_etfs.csv"
    output_core_finaux_csv_name: str = "Core_finaux.csv"
    output_core3_simple_csv_name: str = "core3_etf_daily_simple_returns.csv"
    output_core3_log_csv_name: str = "core3_etf_daily_log_returns.csv"
    output_weights_is_csv_name: str = "core_weights_is.csv"
    output_weights_oos_csv_name: str = "core_weights_oos.csv"

    @property
    def output_dir(self) -> Path:
        """Répertoire de sortie."""
        return self.project_root / self.output_dir_name

    @property
    def selected_core_map(self) -> Dict[str, str]:
        """Mapping thème -> ticker retenu."""
        return {
            "Equity": self.selected_equity,
            "Rates": self.selected_rates,
            "Credit": self.selected_credit,
        }

    @property
    def max_core_expense_pct(self) -> float:
        """Frais max (%) autorisés pour un ETF Core."""
        w_sat = 1.0 - self.w_core_mid
        max_bps = (
            self.total_fee_budget_bps - w_sat * self.satellite_expense_bps
        ) / self.w_core_mid
        return max_bps / 100.0


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
        data["encours_eur_m"] = pd.to_numeric(
            data["encours_eur_m"], errors="coerce"
        )

    return data


def lire_theme(
    cfg: CoreConfig,
    theme_name: str,
    sheet_prices: str,
    sheet_meta: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lit les prix et metadata d'un thème."""
    print(f"  Lecture {theme_name}...")
    wide = _lire_wide_values(cfg.core_excel, sheet_prices)
    meta = _lire_metadata(cfg.core_excel, sheet_meta)
    print(
        f"    -> {wide.shape[1]} ETFs | "
        f"{wide.index.min().date()} à {wide.index.max().date()}"
    )
    return wide, meta


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
        "nom",
        "provider",
        "isin",
        "devise",
        "ter_pct",
        "encours_eur_m",
        "exposition",
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


def _risk_parity_weights(cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Poids Risk Parity approchés par inverse-volatilité."""
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / (vols + 1e-12)
    weights = inv_vol / inv_vol.sum()
    weights = np.clip(weights, w_min, w_max)
    weights /= weights.sum()
    return weights


def _ledoit_wolf_cov(window_rets: np.ndarray) -> np.ndarray:
    """Covariance annualisée via Ledoit-Wolf si disponible."""
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(window_rets)
        return lw.covariance_ * 252
    except ImportError:
        return np.cov(window_rets.T) * 252


def _opt_min_var(cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Optimisation Min Variance sous bornes."""
    n_assets = cov.shape[0]
    x0 = np.ones(n_assets) / n_assets

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    bounds = [(w_min, w_max)] * n_assets
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )
    return result.x if result.success else x0


def _opt_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    w_min: float,
    w_max: float,
) -> np.ndarray:
    """Optimisation Max Sharpe (legacy) sous bornes."""
    n_assets = len(mu)
    x0 = np.ones(n_assets) / n_assets

    def neg_sharpe(weights: np.ndarray) -> float:
        vol = float(np.sqrt(weights @ cov @ weights))
        if vol < 1e-12:
            return 1e10
        return -(float(weights @ mu) / vol)

    bounds = [(w_min, w_max)] * n_assets
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )
    return result.x if result.success else x0


def _apply_equity_constraints(
    weights: np.ndarray,
    w_min: float,
    w_max: float,
    equity_floor: float,
) -> np.ndarray:
    """Applique le floor Equity puis renormalise."""
    if equity_floor <= 0 or weights[0] >= equity_floor:
        return weights

    adjusted = weights.copy()
    deficit = equity_floor - adjusted[0]
    adjusted[0] = equity_floor
    others_sum = adjusted[1:].sum()

    if others_sum > 1e-12:
        adjusted[1:] -= deficit * (adjusted[1:] / others_sum)

    adjusted = np.clip(adjusted, w_min, w_max)
    adjusted /= adjusted.sum()
    return adjusted


def _apply_equity_momentum_tilt(
    weights: np.ndarray,
    rets_log: pd.DataFrame,
    start_idx: int,
    w_min: float,
    w_max: float,
    equity_ceiling: float,
    momentum_window: int,
    momentum_threshold: float,
) -> np.ndarray:
    """Surpondère l'Equity si le momentum 12M est positif."""
    if equity_ceiling <= 0:
        return weights

    mom_start = max(0, start_idx - momentum_window)
    equity_rets = rets_log.iloc[mom_start:start_idx].iloc[:, 0]
    momentum = float(equity_rets.sum())

    if momentum <= momentum_threshold or weights[0] >= equity_ceiling:
        return weights

    adjusted = weights.copy()
    excess = equity_ceiling - adjusted[0]
    adjusted[0] = equity_ceiling
    others_sum = adjusted[1:].sum()

    if others_sum > 1e-12:
        adjusted[1:] -= excess * (adjusted[1:] / others_sum)

    adjusted = np.clip(adjusted, w_min, w_max)
    adjusted /= adjusted.sum()
    return adjusted


def _compute_weights(
    window: pd.DataFrame,
    rets_log: pd.DataFrame,
    start_idx: int,
    cfg: CoreConfig,
) -> np.ndarray:
    """Calcule les poids Core à une date de rebalancing."""
    if cfg.use_ledoit_wolf:
        cov = _ledoit_wolf_cov(window.values)
    else:
        cov = window.cov().values * 252

    if cfg.rolling_method == "risk_parity_tilt":
        weights = _risk_parity_weights(cov, cfg.w_min, cfg.w_max)
        weights = _apply_equity_momentum_tilt(
            weights,
            rets_log=rets_log,
            start_idx=start_idx,
            w_min=cfg.w_min,
            w_max=cfg.w_max,
            equity_ceiling=cfg.equity_weight_ceiling,
            momentum_window=cfg.momentum_window,
            momentum_threshold=cfg.momentum_threshold,
        )
    elif cfg.rolling_method == "min_var":
        weights = _opt_min_var(cov, cfg.w_min, cfg.w_max)
    else:
        mu = window.mean().values * 252
        weights = _opt_max_sharpe(mu, cov, cfg.w_min, cfg.w_max)

    return _apply_equity_constraints(
        weights,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
        equity_floor=cfg.equity_weight_floor,
    )


def backtest_rolling(
    prices: pd.DataFrame,
    cfg: CoreConfig,
    start_date: str,
    end_date: str,
    label: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Backtest rolling trimestriel du portefeuille Core."""
    rets_log = np.log(prices).diff().dropna()
    dates = rets_log.index

    portfolio_values: List[float] = []
    portfolio_dates: List[pd.Timestamp] = []
    weight_rows: List[dict] = []

    for start_idx in range(cfg.lookback, len(dates) - cfg.rebal_freq, cfg.rebal_freq):
        window = rets_log.iloc[start_idx - cfg.lookback : start_idx]
        oos = rets_log.iloc[start_idx : start_idx + cfg.rebal_freq]
        weights = _compute_weights(window, rets_log, start_idx, cfg)

        reb_date = oos.index[0]
        weight_rows.append(
            {
                "Date": reb_date,
                f"{prices.columns[0]}_weight": weights[0],
                f"{prices.columns[1]}_weight": weights[1],
                f"{prices.columns[2]}_weight": weights[2],
            }
        )

        portfolio_values.extend((oos.values @ weights).tolist())
        portfolio_dates.extend(oos.index.tolist())

    series = pd.Series(
        portfolio_values,
        index=pd.DatetimeIndex(portfolio_dates),
        name=f"core_log_return_{label.lower()}",
    ).sort_index()

    series = series.loc[start_date:end_date]
    weights_df = pd.DataFrame(weight_rows).set_index("Date").sort_index()
    weights_df = weights_df.loc[start_date:end_date]
    return series, weights_df


def _log_to_simple(
    log_returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Convertit des log-returns en simple returns."""
    return np.exp(log_returns) - 1.0


def _print_perf_summary(log_rets: pd.Series, label: str) -> None:
    """Affiche un résumé de performance."""
    if log_rets.empty:
        print(f"  {label}: aucune donnée")
        return

    cum = np.exp(log_rets.cumsum())
    perf = float(cum.iloc[-1] - 1.0)
    n_years = len(log_rets) / 252
    ann_ret = float((1.0 + perf) ** (1.0 / max(n_years, 0.01)) - 1.0)
    vol = float(log_rets.std() * np.sqrt(252))
    sharpe = ann_ret / vol if vol > 1e-10 else np.nan
    drawdown = float(((cum / cum.cummax()) - 1.0).min())

    print(
        f"  {label}: {log_rets.index.min().date()} → {log_rets.index.max().date()} "
        f"| {len(log_rets)} jours"
    )
    print(f"    Perf cumulée  : {perf:+.2%}")
    print(f"    Ret annualisé : {ann_ret:+.2%}")
    print(f"    Vol annualisée: {vol:.2%}")
    print(f"    Sharpe        : {sharpe:.2f}")
    print(f"    Max Drawdown  : {drawdown:.2%}")


def _export_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Exporte un DataFrame en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def _export_series(series: pd.Series, path: Path) -> None:
    """Exporte une série pandas en CSV."""
    _export_dataframe(series.to_frame(), path)


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


def _print_inter_etf_correlations(
    log_returns: pd.DataFrame,
    start_date: str,
    end_date: str,
    label: str,
) -> None:
    """Affiche les corrélations inter-ETF sur une sous-période."""
    sample = log_returns.loc[start_date:end_date].dropna()
    if len(sample) <= 30:
        print(f"  Corrélations {label}: échantillon insuffisant")
        return

    corr = sample.corr()
    print(f"\n  Corrélations {label} ({start_date} → {end_date}) :")
    for i, name_i in enumerate(corr.columns):
        for j, name_j in enumerate(corr.columns):
            if j > i:
                print(f"    {name_i} vs {name_j} : {corr.iloc[i, j]:.3f}")


def main() -> dict:
    """Exécute le pipeline Core complet. Retourne les DataFrames clés."""
    cfg = CoreConfig()

    if not cfg.core_excel.exists():
        raise FileNotFoundError(f"Fichier introuvable : {cfg.core_excel}")

    print("=" * 68)
    print("  PIPELINE CORE – VERSION PROPRE")
    print(f"  Fichier Excel           : {cfg.core_excel.name}")
    print(f"  ETF Core retenus        : {cfg.selected_core_map}")
    print(f"  IS (calibration)        : {cfg.score_start} → {cfg.score_end}")
    print(f"  OOS (validation)        : {cfg.oos_start} → {cfg.oos_end}")
    print(f"  Méthode rolling         : {cfg.rolling_method}")
    print(f"  Ledoit-Wolf shrinkage   : {cfg.use_ledoit_wolf}")
    print("=" * 68)

    print("\n[1/6] Lecture des données...")
    wide_eq, meta_eq = lire_theme(
        cfg, "Equity", cfg.sheet_equity_prices, cfg.sheet_equity_meta
    )
    wide_rt, meta_rt = lire_theme(
        cfg, "Rates", cfg.sheet_rates_prices, cfg.sheet_rates_meta
    )
    wide_cr, meta_cr = lire_theme(
        cfg, "Credit", cfg.sheet_credit_prices, cfg.sheet_credit_meta
    )

    print("\n[2/6] Validation des 3 ETF Core retenus...")
    selected_map = cfg.selected_core_map
    validation_rows = [
        _validate_selected_ticker(
            "Equity", selected_map["Equity"], wide_eq, meta_eq, cfg
        ),
        _validate_selected_ticker(
            "Rates", selected_map["Rates"], wide_rt, meta_rt, cfg
        ),
        _validate_selected_ticker(
            "Credit", selected_map["Credit"], wide_cr, meta_cr, cfg
        ),
    ]
    validation_df = pd.DataFrame(validation_rows)
    print(validation_df.to_string(index=False))

    print("\n[3/6] Export de la sélection Core...")
    metas = {"Equity": meta_eq, "Rates": meta_rt, "Credit": meta_cr}
    selected_df = _collect_selected_metadata(selected_map, metas)

    print("\n[4/6] Construction des séries des 3 ETF Core...")
    wide_core = _build_selected_prices(selected_map, wide_eq, wide_rt, wide_cr, cfg)
    core_3_log = np.log(wide_core).diff().dropna()
    core_3_simple = _log_to_simple(core_3_log)

    print("\n[5/6] Backtest rolling du portefeuille Core...")
    core_log_is, weights_is = backtest_rolling(
        prices=wide_core,
        cfg=cfg,
        start_date=cfg.score_start,
        end_date=cfg.score_end,
        label="IS",
    )
    core_log_oos, weights_oos = backtest_rolling(
        prices=wide_core,
        cfg=cfg,
        start_date=cfg.oos_start,
        end_date=cfg.oos_end,
        label="OOS",
    )

    _print_perf_summary(core_log_is, "IS")
    _print_perf_summary(core_log_oos, "OOS")

    core_simple_is = _log_to_simple(core_log_is)
    core_simple_oos = _log_to_simple(core_log_oos)

    print("\n[6/6] Corrélations inter-ETF...")
    _print_inter_etf_correlations(core_3_log, cfg.score_start, cfg.score_end, "IS")
    _print_inter_etf_correlations(core_3_log, cfg.oos_start, cfg.oos_end, "OOS")

    print("\nRésumé final")
    print("=" * 68)
    for theme, ticker in selected_map.items():
        print(f"  {theme:10s}: {ticker}")
    print(f"  Budget max frais Core : {cfg.max_core_expense_pct:.2f}%")
    print("=" * 68)

    return {
        "core_finaux": selected_df,
        "core_3_log": core_3_log,
        "core_3_simple": core_3_simple,
    }


if __name__ == "__main__":
    main()
