"""
Filtre Niveau 2 - Score Alpha + Expense

Objectif:
- Calculer un score par fonds sur la periode IS:
  score = w_alpha * z(alpha_annuel) + w_expense * z(-expense)
- Optionnel: conserver les meilleurs fonds par bloc (Strat)
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats


def _detect_ticker_col(df: pd.DataFrame) -> str:
    for col in ["ticker", "Ticker"]:
        if col in df.columns:
            return col
    raise ValueError("Colonne ticker introuvable (attendu: 'Ticker' ou 'ticker').")


def _parse_expense_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def _safe_zscore(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    std = valid.std(ddof=0)
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    out = (s - valid.mean()) / std
    return out


def _ols_alpha_annual(fund_rets: pd.Series, core_rets: pd.Series, min_obs: int) -> tuple[float, int]:
    aligned = pd.concat([fund_rets, core_rets], axis=1).dropna()
    if len(aligned) < min_obs:
        return np.nan, len(aligned)
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    slope, intercept, *_ = stats.linregress(x, y)
    _ = slope  # beta not used in level2
    alpha_annual = float(intercept * 252)
    return alpha_annual, len(aligned)


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
    Calcule score Level 2 (Alpha + Expense) et filtre optionnellement.

    keep_top_per_strat:
    - 0 => ne filtre pas, garde tous les fonds scores
    - N>0 => conserve top N fonds par bloc (colonne 'Strat')
    """
    if not np.isclose(alpha_weight + expense_weight, 1.0):
        raise ValueError("alpha_weight + expense_weight doit etre egal a 1.0")

    ticker_col = _detect_ticker_col(level1_df)
    if "Strat" not in level1_df.columns:
        raise ValueError("Colonne 'Strat' manquante dans level1_df")
    if expense_col not in level1_df.columns:
        raise ValueError(f"Colonne expense introuvable: {expense_col}")

    tickers = level1_df[ticker_col].astype(str).tolist()
    use_tickers = [t for t in tickers if t in prices_aligned.columns]

    fund_rets = np.log(prices_aligned[use_tickers]).diff()
    fund_rets = fund_rets.loc[calib_start:calib_end]
    core_rets = core_returns_aligned.loc[calib_start:calib_end]

    rows = []
    expense_map = level1_df.set_index(ticker_col)[expense_col]
    expense_num = _parse_expense_series(expense_map)

    for t in use_tickers:
        alpha_ann, n_obs = _ols_alpha_annual(fund_rets[t], core_rets, min_obs=min_obs_alpha)
        rows.append(
            {
                "ticker": t,
                "alpha_annual": alpha_ann,
                "n_obs_alpha": n_obs,
                "expense_pct": expense_num.get(t, np.nan),
                "Strat": level1_df.loc[level1_df[ticker_col] == t, "Strat"].iloc[0],
            }
        )

    scores_df = pd.DataFrame(rows)
    if len(scores_df) == 0:
        return level1_df.iloc[0:0].copy(), scores_df

    scores_df["z_alpha"] = _safe_zscore(scores_df["alpha_annual"])
    scores_df["z_neg_expense"] = _safe_zscore(-scores_df["expense_pct"])
    scores_df["score_level2"] = (
        alpha_weight * scores_df["z_alpha"] + expense_weight * scores_df["z_neg_expense"]
    )

    scores_df = scores_df.sort_values(["Strat", "score_level2"], ascending=[True, False]).reset_index(drop=True)

    if keep_top_per_strat > 0:
        kept = (
            scores_df.groupby("Strat", group_keys=False)
            .head(keep_top_per_strat)
            .copy()
        )
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
