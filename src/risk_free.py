"""Utilities for risk-free data and excess-return Sharpe calculations."""

from __future__ import annotations

from io import StringIO
from typing import Tuple

import numpy as np
import pandas as pd
import requests


def _annual_yield_to_daily_return(y_ann: pd.Series) -> pd.Series:
    y_ann = pd.to_numeric(y_ann, errors="coerce")
    return (1.0 + y_ann).pow(1.0 / 252.0) - 1.0


def get_bund_risk_free_daily(
    index: pd.DatetimeIndex,
    default_annual: float = 0.02,
    timeout_sec: float = 4.0,
) -> Tuple[pd.Series, str]:
    """
    Return a daily risk-free series aligned to index.

    Priority:
    1) Public API (FRED CSV, Germany LT 10Y yield in % annual)
    2) Constant fallback default_annual
    """
    src = {
        "name": "FRED IRLTLT01DEM156N",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N",
        "date_col": "DATE",
        "value_col": "IRLTLT01DEM156N",
        "is_percent": True,
    }

    try:
        response = requests.get(src["url"], timeout=timeout_sec)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
        if src["date_col"] in raw.columns and src["value_col"] in raw.columns:
            s = raw[[src["date_col"], src["value_col"]]].copy()
            s[src["date_col"]] = pd.to_datetime(s[src["date_col"]], errors="coerce")
            s = s.dropna(subset=[src["date_col"]]).set_index(src["date_col"]).sort_index()
            y = pd.to_numeric(s[src["value_col"]], errors="coerce")
            if src["is_percent"]:
                y = y / 100.0
            rf_daily = _annual_yield_to_daily_return(y).rename("risk_free")
            rf_daily = rf_daily.reindex(index).ffill().fillna(0.0)
            return rf_daily, f"API {src['name']}"
    except Exception:
        pass

    daily = (1.0 + default_annual) ** (1.0 / 252.0) - 1.0
    rf_daily = pd.Series(daily, index=index, name="risk_free")
    return rf_daily, f"Fallback constant {default_annual:.2%} annual"


def sharpe_excess(returns: pd.Series, rf_daily: pd.Series) -> float:
    """Annualized Sharpe using daily excess returns over rf_daily."""
    aligned = pd.concat([returns.rename("ret"), rf_daily.rename("rf")], axis=1).dropna()
    if aligned.empty:
        return np.nan

    vol_ann = float(aligned["ret"].std() * np.sqrt(252.0))
    if vol_ann <= 1e-12:
        return np.nan

    excess_ann = float((aligned["ret"] - aligned["rf"]).mean() * 252.0)
    return excess_ann / vol_ann
