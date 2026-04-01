"""Dynamic satellite allocation by regime + quarterly fund selection integration."""

import numpy as np
import pandas as pd
from pathlib import Path


BLOC_TO_STRAT = {"bloc1": "STRAT1", "bloc2": "STRAT2", "bloc3": "STRAT3"}
STRAT_TO_BLOC = {v: k for k, v in BLOC_TO_STRAT.items()}


def normalize_block_weights(w_dict):
    s = float(sum(w_dict.values()))
    if s <= 0:
        return {k: 0.0 for k in w_dict}
    return {k: float(v) / s for k, v in w_dict.items()}


def canonical_regime(x):
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"stress", "stressed"}:
        return "stress"
    if s in {"normal", "neutre", "neutral", "neutre_market", "neutre_regime"}:
        return "normal"
    if s in {"risk_on", "riskon", "risk_on_market"}:
        return "risk_on"
    return "normal"


def coerce_date_index(df, context):
    date_col = None
    for c in ["Date", "date", "DATE", "datetime", "Datetime", "quarter_date", "Unnamed: 0", "index"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dt = pd.to_datetime(df.index, errors="coerce")

    if dt.isna().all():
        raise ValueError(f"Impossible d'identifier la colonne date dans {context}. Colonnes: {list(df.columns)}")

    return pd.DatetimeIndex(dt).tz_localize(None)


def load_regime_series(df_input):
    """Normalise un DataFrame régime en (DatetimeIndex, colonne 'regime')."""
    if df_input is None or len(df_input) == 0:
        raise ValueError("Le jeu de données régime est vide")

    df = df_input.copy()
    dt = coerce_date_index(df, "regime_core_daily")

    regime_col = None
    for c in ["Regime", "regime", "REGIME"]:
        if c in df.columns:
            regime_col = c
            break
    if regime_col is None:
        for c in df.columns:
            if str(c).lower() in {"market_regime", "state", "signal"}:
                regime_col = c
                break
    if regime_col is None:
        raise ValueError(f"Impossible d'identifier la colonne Regime. Colonnes: {list(df.columns)}")

    out = pd.DataFrame({"date": dt, "regime": df[regime_col].map(canonical_regime)})
    out = out.dropna(subset=["date"]).set_index("date").sort_index()
    return out[["regime"]]


def load_quarterly_selection(df_input):
    """Normalise un DataFrame sélection trimestrielle."""
    if df_input is None or len(df_input) == 0:
        raise ValueError("quarter_selection_df est vide")

    q = df_input.copy()

    if "quarter_date" not in q.columns:
        for c in ["Quarter", "quarter", "date", "Date", "Unnamed: 0"]:
            if c in q.columns:
                q = q.rename(columns={c: "quarter_date"})
                break

    required = {"quarter_date", "Strat", "ticker"}
    miss = required - set(q.columns)
    if miss:
        raise ValueError(f"Colonnes manquantes dans quarter_selection_df: {sorted(miss)}")

    q["quarter_date"] = pd.to_datetime(q["quarter_date"], errors="coerce")
    q = q.dropna(subset=["quarter_date", "Strat", "ticker"]).copy()
    q["Strat"] = q["Strat"].astype(str)
    q["ticker"] = q["ticker"].astype(str)
    q["bloc"] = q["Strat"].map(STRAT_TO_BLOC)
    q = q.dropna(subset=["bloc"]).copy()
    return q.sort_values(["quarter_date", "Strat", "ticker"])


def build_dynamic_satellite_weights(regime_df, quarterly_sel_df, regime_block_weights, missing_block_floor=0.0):
    """
    Build daily ticker weights by combining regime-based block weights
    with the quarterly fund selection.

    Returns: weights_ticker_daily, block_weights_daily, active_funds_daily
    """
    regime_df = regime_df.copy().sort_index()
    q = quarterly_sel_df.copy().sort_values(["quarter_date", "Strat", "ticker"])

    all_dates = regime_df.index.unique().sort_values()
    if len(all_dates) == 0:
        raise ValueError("regime_df ne contient aucune date")

    q_dates = q["quarter_date"].dropna().sort_values().unique()
    if len(q_dates) == 0:
        raise ValueError("quarterly_sel_df ne contient aucune quarter_date valide")

    tickers_all = sorted(q["ticker"].unique().tolist())
    ticker_rows = []
    block_rows = []
    active_rows = []

    for d in all_dates:
        reg = canonical_regime(regime_df.loc[d, "regime"])
        base_w = normalize_block_weights(regime_block_weights.get(reg, regime_block_weights["normal"]))

        eligible_q = q_dates[q_dates <= np.datetime64(d)]
        if len(eligible_q) == 0:
            continue
        qd = pd.Timestamp(eligible_q.max())
        q_active = q[q["quarter_date"] == qd].copy()

        active_by_bloc = {
            b: q_active.loc[q_active["bloc"] == b, "ticker"].drop_duplicates().tolist()
            for b in ["bloc1", "bloc2", "bloc3"]
        }

        adjusted_w = dict(base_w)
        missing_blocs = [b for b, funds in active_by_bloc.items() if len(funds) == 0]
        active_blocs = [b for b, funds in active_by_bloc.items() if len(funds) > 0]

        if len(active_blocs) == 0:
            continue

        if missing_blocs:
            missing_weight = sum(adjusted_w[b] for b in missing_blocs)
            for b in missing_blocs:
                adjusted_w[b] = float(missing_block_floor)
            redistribute = max(0.0, missing_weight - float(missing_block_floor) * len(missing_blocs))
            active_sum = sum(adjusted_w[b] for b in active_blocs)
            if active_sum > 0 and redistribute > 0:
                for b in active_blocs:
                    adjusted_w[b] += redistribute * (adjusted_w[b] / active_sum)

        adjusted_w = normalize_block_weights(adjusted_w)

        row_ticker = {"date": d}
        for t in tickers_all:
            row_ticker[t] = 0.0

        for bloc, funds in active_by_bloc.items():
            wb = adjusted_w.get(bloc, 0.0)
            if len(funds) == 0 or wb <= 0:
                continue
            wt = wb / len(funds)
            strat = BLOC_TO_STRAT[bloc]
            for t in funds:
                row_ticker[t] = row_ticker.get(t, 0.0) + wt
                active_rows.append({
                    "date": d, "quarter_date": qd, "regime": reg, "bloc": bloc,
                    "Strat": strat, "ticker": t, "weight_ticker": wt,
                })

        ticker_rows.append(row_ticker)
        block_rows.append({
            "date": d, "quarter_date": qd, "regime": reg,
            "w_bloc1": adjusted_w.get("bloc1", 0.0),
            "w_bloc2": adjusted_w.get("bloc2", 0.0),
            "w_bloc3": adjusted_w.get("bloc3", 0.0),
        })

    weights_ticker_daily = pd.DataFrame(ticker_rows).set_index("date").sort_index() if ticker_rows else pd.DataFrame()
    block_weights_daily = pd.DataFrame(block_rows).set_index("date").sort_index() if block_rows else pd.DataFrame()
    active_funds_daily = pd.DataFrame(active_rows).sort_values(["date", "bloc", "ticker"]) if active_rows else pd.DataFrame()

    return weights_ticker_daily, block_weights_daily, active_funds_daily


def run_dynamic_allocation(regime_output, quarter_selection_df, regime_block_weights, missing_block_floor=0.0):
    """
    Full dynamic allocation pipeline: load regime + quarterly sel, build weights.

    Returns dict with: weights_ticker_daily, block_weights_daily, active_funds_daily
    """
    regime_daily = load_regime_series(regime_output)
    quarterly_sel = load_quarterly_selection(quarter_selection_df)

    weights_ticker_daily, block_weights_daily, active_funds_daily = build_dynamic_satellite_weights(
        regime_df=regime_daily,
        quarterly_sel_df=quarterly_sel,
        regime_block_weights=regime_block_weights,
        missing_block_floor=missing_block_floor,
    )

    print("Allocation dynamique Satellite calculée")
    print(f"   - Jours calculés: {len(block_weights_daily)}")
    print(f"   - Tickers couverts: {weights_ticker_daily.shape[1] if not weights_ticker_daily.empty else 0}")
    if not block_weights_daily.empty:
        print(f"   - Somme moyenne des poids blocs: {block_weights_daily[['w_bloc1', 'w_bloc2', 'w_bloc3']].sum(axis=1).mean():.4f}")

    return {
        "weights_ticker_daily": weights_ticker_daily,
        "block_weights_daily": block_weights_daily,
        "active_funds_daily": active_funds_daily,
    }
