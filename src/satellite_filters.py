"""
Satellite fund filters — Level 0 (structurel), Level 1 (beta rolling), Level 2 (alpha + expense).

Fusionne les anciens satellite_level0_filter, satellite_level1_beta_filter_final,
et satellite_level2_filter en un seul module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import detect_ticker_col


# ══════════════════════════════════════════════════════════════════════════════
#  Level 0 — filtre structurel (devise + AUM)
# ══════════════════════════════════════════════════════════════════════════════

def _load_satellite_data(data_dir: str = "data") -> pd.DataFrame:
    all_data = []
    for strat_num in [1, 2, 3]:
        filepath = Path(data_dir) / f"STRAT{strat_num}_info.xlsx"
        df = pd.read_excel(filepath, sheet_name=0)
        df["Strat"] = f"STRAT{strat_num}"
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def _normalize_devise_filter(devise: list | str | None, available_values: pd.Series) -> list[str]:
    if devise is None:
        requested = ["Euro"]
    elif isinstance(devise, str):
        requested = [devise]
    else:
        requested = list(devise)

    alias_map = {
        "eur": ["Euro"],
        "euro": ["Euro"],
        "usd": ["Dollar US"],
        "dollar us": ["Dollar US"],
        "gbp": ["Livre sterling", "Pence britannique"],
        "jpy": ["Yen japonais"],
        "chf": ["Franc suisse"],
        "cad": ["Dollar canadien"],
        "aud": ["Dollar australien"],
    }

    available_lookup = {
        str(value).strip().casefold(): str(value).strip()
        for value in available_values.dropna().astype(str)
    }

    normalized = []
    for item in requested:
        key = str(item).strip()
        if not key:
            continue
        lowered = key.casefold()
        candidates = alias_map.get(lowered, [key])
        for candidate in candidates:
            match = available_lookup.get(str(candidate).strip().casefold())
            if match is not None and match not in normalized:
                normalized.append(match)

    return normalized


def filter_satellite_level0(
    data_dir: str = "data",
    devise: list | str | None = None,
    min_aum_usd: float = 50.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline complet de filtrage niveau 0 (devise + AUM)."""
    df = _load_satellite_data(data_dir)
    devise_values = _normalize_devise_filter(devise, df["Dev"])

    if not devise_values:
        raise ValueError(
            "Aucune devise valide après normalisation. "
            "Utiliser par exemple 'eur', 'usd' ou une liste de libellés présents dans la colonne Dev."
        )

    if verbose:
        print("=" * 80)
        print("SATELLITE LEVEL 0 FILTER - DEVISE ET AUM")
        print("=" * 80)
        print(f"\nUnivers initial: {len(df)} fonds")

    mask = df["Dev"].isin(devise_values) & (df["Total actifs USD (M)"] > min_aum_usd)
    df_filtered = df[mask].copy()

    if verbose:
        print(f"\nCritères appliqués:")
        print(f"  - Devise: {devise_values}")
        print(f"  - AUM USD > {min_aum_usd}M")
        print(f"\nFonds restants: {len(df_filtered)}")

    summary = df_filtered.groupby("Strat").size().reset_index(name="Nombre de fonds")

    if verbose:
        print(f"\nComposition par bloc:")
        print(summary.to_string(index=False))
        print(f"\nTotal: {summary['Nombre de fonds'].sum()} fonds")

    return df_filtered, summary


# ══════════════════════════════════════════════════════════════════════════════
#  Level 1 — filtre beta rolling
# ══════════════════════════════════════════════════════════════════════════════

def apply_level1_filter_corrected(
    level0_df: pd.DataFrame,
    prices_aligned: pd.DataFrame,
    core_returns_aligned: pd.Series,
    calib_start: str = "2019-01-01",
    calib_end: str = "2020-12-31",
    rolling_window: int = 126,
    median_beta_max: float = 0.20,
    q75_beta_max: float = 0.30,
    pass_ratio_min: float = 0.95,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtre niveau 1 : beta rolling multi-stat.

    Conditions simultanées sur la fenêtre de calibration :
        1. median(|β|) <= median_beta_max
        2. Q75(|β|)    <= q75_beta_max
        3. ratio(|β| <= median_beta_max) >= pass_ratio_min
    """
    ticker_col = detect_ticker_col(level0_df)

    if verbose:
        print("\n" + "=" * 80)
        print("FILTRE NIVEAU 1 – BETA ROLLING")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Rolling window: {rolling_window} jours")
        print(f"  Conditions: median(|β|)≤{median_beta_max}, Q75(|β|)≤{q75_beta_max}, ratio≥{pass_ratio_min}")
        print(f"  Calib: {calib_start} à {calib_end}")

    # Rendements log journaliers
    fund_rets = np.log(prices_aligned).diff().dropna(how="all")

    aligned_returns = pd.concat(
        [fund_rets, core_returns_aligned.rename("__CORE__")],
        axis=1, sort=True,
    ).dropna(how="all")

    # Rolling beta pour chaque fonds
    betas_rolling = {}
    for ticker in fund_rets.columns:
        if ticker not in aligned_returns.columns:
            continue

        fund_col = aligned_returns[ticker].dropna()
        core_col = aligned_returns["__CORE__"].loc[fund_col.index].dropna()
        both = pd.concat([fund_col, core_col], axis=1, keys=["fund", "core"]).dropna()

        if len(both) < rolling_window + 1:
            continue

        fund_vals = both["fund"].values
        core_vals = both["core"].values
        dates_vals = both.index

        betas_list, betas_dates = [], []
        for i in range(rolling_window, len(fund_vals)):
            fw = fund_vals[i - rolling_window : i]
            cw = core_vals[i - rolling_window : i]
            cov = np.cov(fw, cw)[0, 1]
            var = np.var(cw, ddof=1)
            if var > 1e-12:
                betas_list.append(cov / var)
                betas_dates.append(dates_vals[i])

        if betas_list:
            betas_rolling[ticker] = pd.Series(
                betas_list,
                index=pd.DatetimeIndex(betas_dates),
                name=ticker,
            )

    if verbose:
        print(f"\n  Beta rolling calculés pour {len(betas_rolling)} tickers")

    # Appliquer conditions sur la fenêtre de calibration
    results = []
    passed_tickers = []

    for ticker, beta_series in betas_rolling.items():
        betas_calib = beta_series.loc[calib_start:calib_end]

        if len(betas_calib) < 50:
            strat = "Unknown"
            if ticker in level0_df[ticker_col].values:
                strat = level0_df[level0_df[ticker_col] == ticker]["Strat"].values[0]
            results.append({
                "ticker": ticker, "strat": strat,
                "status": "insufficient_data", "n_obs": len(betas_calib),
                "passed": False, "median_abs_beta": np.nan,
                "q75_abs_beta": np.nan, "pass_ratio": np.nan,
            })
            continue

        abs_beta = betas_calib.abs()
        median_abs_beta = float(abs_beta.median())
        q75_abs_beta = float(abs_beta.quantile(0.75))
        pass_ratio = float((abs_beta <= median_beta_max).mean())

        cond1 = median_abs_beta <= median_beta_max
        cond2 = q75_abs_beta <= q75_beta_max
        cond3 = pass_ratio >= pass_ratio_min
        passed = cond1 and cond2 and cond3

        strat = "Unknown"
        if ticker in level0_df[ticker_col].values:
            strat = level0_df[level0_df[ticker_col] == ticker]["Strat"].values[0]

        results.append({
            "ticker": ticker, "strat": strat,
            "status": "passed" if passed else "failed",
            "n_obs": len(betas_calib), "passed": passed,
            "median_abs_beta": median_abs_beta,
            "q75_abs_beta": q75_abs_beta,
            "pass_ratio": pass_ratio,
            "cond1_median": cond1, "cond2_q75": cond2, "cond3_ratio": cond3,
        })

        if passed:
            passed_tickers.append(ticker)
            if verbose and len(passed_tickers) <= 10:
                print(f"   ✅ {ticker:25s} | med={median_abs_beta:.3f} q75={q75_abs_beta:.3f} ratio={pass_ratio:.1%}")

    results_df = pd.DataFrame(results)
    level1_df = level0_df[level0_df[ticker_col].isin(passed_tickers)].copy()

    if verbose:
        n_passed = len(level1_df)
        print(f"\n" + "-" * 80)
        print(f"RÉSULTATS: {n_passed} / {len(level0_df)} fonds PASSED")
        if n_passed > 0:
            for strat, count in level1_df.groupby("Strat").size().items():
                print(f"  {strat}: {count}")

    return level1_df, results_df


# ══════════════════════════════════════════════════════════════════════════════
#  Level 2 — score Alpha + Expense
# ══════════════════════════════════════════════════════════════════════════════

def _parse_expense_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def _resolve_expense_col(df: pd.DataFrame, expense_col: str) -> str:
    if expense_col in df.columns:
        return expense_col

    normalized_lookup = {
        str(col)
        .strip()
        .lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u"): col
        for col in df.columns
    }

    alias_candidates = [
        expense_col,
        "expense_pct",
        "expense",
        "ratio des dépenses",
        "ratio des depenses",
    ]
    for alias in alias_candidates:
        key = (
            str(alias)
            .strip()
            .lower()
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("à", "a")
            .replace("ù", "u")
        )
        if key in normalized_lookup:
            return normalized_lookup[key]

    raise ValueError(f"Colonne expense introuvable: {expense_col}")


def _safe_zscore(s: pd.Series) -> pd.Series:
    """Z-score pour Level 2 (renvoie NaN pour constante, pas fillna(0))."""
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    std = valid.std(ddof=0)
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - valid.mean()) / std


def _ols_alpha_annual(
    fund_rets: pd.Series,
    core_rets: pd.Series,
    min_obs: int,
) -> Tuple[float, int]:
    aligned = pd.concat([fund_rets, core_rets], axis=1).dropna()
    if len(aligned) < min_obs:
        return np.nan, len(aligned)
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    _slope, intercept, *_ = stats.linregress(x, y)
    return float(intercept * 252), len(aligned)


def apply_level2_alpha_expense(
    level1_df: pd.DataFrame,
    prices_aligned: pd.DataFrame,
    core_returns_aligned: pd.Series,
    calib_start: str = "2019-01-01",
    calib_end: str = "2020-12-31",
    expense_col: str = "Ratio des depenses",
    alpha_weight: float = 0.6,
    expense_weight: float = 0.4,
    min_obs_alpha: int = 60,
    keep_top_per_strat: int = 0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score Level 2 = w_alpha * z(alpha) + w_expense * z(-expense).

    keep_top_per_strat : 0 = garde tout, N>0 = top N par bloc.
    """
    if not np.isclose(alpha_weight + expense_weight, 1.0):
        raise ValueError("alpha_weight + expense_weight doit etre egal a 1.0")

    ticker_col = detect_ticker_col(level1_df)
    if "Strat" not in level1_df.columns:
        raise ValueError("Colonne 'Strat' manquante dans level1_df")
    resolved_expense_col = _resolve_expense_col(level1_df, expense_col)

    tickers = level1_df[ticker_col].astype(str).tolist()
    use_tickers = [t for t in tickers if t in prices_aligned.columns]

    fund_rets = np.log(prices_aligned[use_tickers]).diff()
    fund_rets = fund_rets.loc[calib_start:calib_end]
    core_rets = core_returns_aligned.loc[calib_start:calib_end]

    expense_map = level1_df.set_index(ticker_col)[resolved_expense_col]
    expense_num = _parse_expense_series(expense_map)

    rows = []
    for t in use_tickers:
        alpha_ann, n_obs = _ols_alpha_annual(fund_rets[t], core_rets, min_obs=min_obs_alpha)
        rows.append({
            "ticker": t,
            "alpha_annual": alpha_ann,
            "n_obs_alpha": n_obs,
            "expense_pct": expense_num.get(t, np.nan),
            "Strat": level1_df.loc[level1_df[ticker_col] == t, "Strat"].iloc[0],
        })

    scores_df = pd.DataFrame(rows)
    if len(scores_df) == 0:
        return level1_df.iloc[0:0].copy(), scores_df

    scores_df["z_alpha"] = _safe_zscore(scores_df["alpha_annual"])
    scores_df["z_neg_expense"] = _safe_zscore(-scores_df["expense_pct"])
    scores_df["score_level2"] = (
        alpha_weight * scores_df["z_alpha"] + expense_weight * scores_df["z_neg_expense"]
    )
    scores_df = scores_df.sort_values(
        ["Strat", "score_level2"], ascending=[True, False],
    ).reset_index(drop=True)

    if keep_top_per_strat > 0:
        kept = scores_df.groupby("Strat", group_keys=False).head(keep_top_per_strat).copy()
    else:
        kept = scores_df.copy()

    selected_tickers = set(kept["ticker"].tolist())
    level2_df = level1_df[level1_df[ticker_col].isin(selected_tickers)].copy()

    if verbose:
        print("\n" + "=" * 80)
        print("FILTRE NIVEAU 2 - SCORE ALPHA + EXPENSE")
        print("=" * 80)
        print(f"Periode IS: {calib_start} -> {calib_end}")
        print(f"Poids: alpha={alpha_weight:.2f}, expense={expense_weight:.2f}")
        print(f"Fonds scores: {len(scores_df)}")
        print(f"Fonds retenus niveau 2: {len(level2_df)}")

    return level2_df, scores_df
