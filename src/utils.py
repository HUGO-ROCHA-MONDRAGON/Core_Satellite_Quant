"""Shared utility functions used across the Core/Satellite pipeline."""

import numpy as np
import pandas as pd


# ── Performance metrics ──────────────────────────────────────────────────────

def ann_return(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan
    return (1.0 + s).prod() ** (252.0 / len(s)) - 1.0


def ann_vol(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan
    return s.std() * np.sqrt(252.0)


def max_dd(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan
    nav = (1.0 + s).cumprod()
    dd = nav / nav.cummax() - 1.0
    return dd.min()


def sharpe0(s, rf_annual=0.02):
    ar = ann_return(s)
    av = ann_vol(s)
    if pd.isna(ar) or pd.isna(av) or av <= 0:
        return np.nan
    return (ar - rf_annual) / av


def calmar(s):
    ar = ann_return(s)
    mdd = max_dd(s)
    if pd.isna(ar) or pd.isna(mdd) or mdd == 0:
        return np.nan
    return ar / abs(mdd)


def sortino0(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    downside = s[s < 0]
    dd = downside.std() * np.sqrt(252.0)
    ar = ann_return(s)
    if pd.isna(ar) or pd.isna(dd) or dd <= 0:
        return np.nan
    return ar / dd


def maxdd_abs(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    nav = (1.0 + s).cumprod()
    dd = nav / nav.cummax() - 1.0
    return abs(dd.min())


def max_drawdown_from_returns(r):
    if r.empty:
        return np.nan
    curve = (1.0 + r).cumprod()
    dd = curve / curve.cummax() - 1.0
    return float(dd.min())


# ── Rolling statistics ───────────────────────────────────────────────────────

def rolling_volatility_ann(returns_series, window):
    return returns_series.rolling(window).std() * np.sqrt(252)


def rolling_zscore(s, z_win, min_p=None):
    if min_p is None:
        min_p = z_win
    mu = s.rolling(z_win, min_periods=min_p).mean()
    sigma = s.rolling(z_win, min_periods=min_p).std()
    return (s - mu) / sigma.replace(0, np.nan)


# ── Regime smoothing ────────────────────────────────────────────────────────

def apply_min_regime_days_causal(regime_series, min_days):
    if regime_series.empty or min_days <= 1:
        return regime_series.copy()

    labels = regime_series.astype(str)
    out = []
    current_regime = labels.iloc[0]
    candidate_regime = None
    candidate_count = 0

    for lbl in labels:
        if lbl == current_regime:
            candidate_regime = None
            candidate_count = 0
            out.append(current_regime)
            continue

        if candidate_regime == lbl:
            candidate_count += 1
        else:
            candidate_regime = lbl
            candidate_count = 1

        if candidate_count >= min_days:
            current_regime = lbl
            candidate_regime = None
            candidate_count = 0

        out.append(current_regime)

    return pd.Series(out, index=regime_series.index, name="Regime")


# ── Index helpers ────────────────────────────────────────────────────────────

def normalize_index(obj):
    obj = obj.copy().sort_index()
    idx = pd.DatetimeIndex(obj.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    obj.index = idx
    return obj


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt_value(value, kind):
    if pd.isna(value):
        return "NaN"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "pct_signed":
        return f"{value:+.2%}"
    if kind == "bps":
        return f"{value:.1f} bps/an"
    if kind == "num":
        return f"{value:.2f}"
    if kind == "int":
        return f"{int(value)}"
    return str(value)


# ── Satellite helpers ────────────────────────────────────────────────────────

def detect_ticker_col(df):
    """Retourne le nom de colonne ticker ('Ticker' ou 'ticker')."""
    for col in ("ticker", "Ticker"):
        if col in df.columns:
            return col
    raise ValueError("Colonne ticker introuvable (attendu: 'Ticker' ou 'ticker').")


def safe_zscore(s):
    """Z-score robuste : renvoie 0.0 si la série est constante / vide."""
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    std = valid.std(ddof=0)
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - valid.mean()) / std).fillna(0.0)
