"""Satellite fund selection pipeline: Level 0/1/2 filtering + rolling annual/quarterly selection."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.satellite_filters import (
    filter_satellite_level0,
    apply_level1_filter_corrected,
    _resolve_expense_col,
    _parse_expense_series,
)
from src.satellite_data_loader import (
    load_all_satellite_prices,
    preprocess_prices,
    align_prices_with_core,
)
from src.utils import ann_return, ann_vol, sortino0, maxdd_abs, detect_ticker_col, safe_zscore


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_strat_group(scores_df):
    if scores_df is None or scores_df.empty or "Strat" not in scores_df.columns:
        return []
    return scores_df.groupby("Strat")


def _extract_tickers_from_price_sheet(price_file, sheet_name):
    try:
        df = pd.read_excel(price_file, sheet_name=sheet_name)
    except Exception:
        return set()
    tickers = set()
    for i in range(0, max(df.shape[1] - 1, 0), 2):
        col_ticker = str(df.columns[i]).strip()
        if "Equity" in col_ticker:
            tickers.add(col_ticker)
    return tickers


def _build_banned_tickers_by_sheet(cfg, data_dir="data"):
    bans = cfg.get("bans", {})
    banned_sheets = bans.get("strat3_banned_sheet_names", [])
    # Individual ticker ban list (overrides sheet-level bans for fine-grained control)
    banned_tickers = set(str(t).strip() for t in bans.get("strat3_banned_tickers", []))
    if banned_sheets:
        price_file = Path(data_dir) / "STRAT3_price.xlsx"
        for sh in banned_sheets:
            banned_tickers |= _extract_tickers_from_price_sheet(price_file, sh)
    return banned_tickers


def _build_pool_from_level1(df_l1, prices_win, cfg):
    """Structural pool filter on L1 survivors: TER hard cap + min price obs. No scoring."""
    ticker_col = detect_ticker_col(df_l1)
    l2_cfg = cfg.get("level2", {})
    max_ter = float(l2_cfg.get("max_ter_pct", 2.0))
    min_obs = int(l2_cfg.get("min_obs_pool", 60))
    expense_col_name = l2_cfg.get("expense_col", "expense_pct")

    try:
        resolved_col = _resolve_expense_col(df_l1, expense_col_name)
        expense_series = _parse_expense_series(df_l1.set_index(ticker_col)[resolved_col])
    except (ValueError, KeyError):
        expense_series = pd.Series(dtype=float)

    rows = []
    for _, row in df_l1.iterrows():
        t = str(row[ticker_col]).strip()
        strat = row["Strat"]

        if t not in prices_win.columns:
            continue
        n_obs = int(prices_win[t].dropna().shape[0])
        if n_obs < min_obs:
            continue

        expense = expense_series.get(t, np.nan)
        if pd.notna(expense) and expense > max_ter:
            continue

        rows.append({"ticker": t, "Strat": strat, "expense_pct": expense, "n_obs_pool": n_obs})

    if not rows:
        return pd.DataFrame(columns=["ticker", "Strat", "expense_pct", "n_obs_pool"])
    return pd.DataFrame(rows)


def _resolve_core_benchmark_returns(core_3_log, params, core_benchmark_returns=None):
    benchmark_cfg = params.get("benchmark", {})
    source = benchmark_cfg.get("source", "equal_weight_core_3")
    returns_type = benchmark_cfg.get("returns_type", "log")

    if source == "equal_weight_core_3":
        core_series = core_3_log.mean(axis=1)
    elif source == "selected_core_portfolio":
        if core_benchmark_returns is None:
            raise ValueError(
                "Le benchmark Core 'selected_core_portfolio' nécessite une série de rendements "
                "passée via core_benchmark_returns."
            )
        if isinstance(core_benchmark_returns, pd.DataFrame):
            if core_benchmark_returns.shape[1] != 1:
                raise ValueError(
                    "core_benchmark_returns doit être une Series ou un DataFrame à une seule colonne."
                )
            core_series = core_benchmark_returns.iloc[:, 0]
        else:
            core_series = pd.Series(core_benchmark_returns)
    else:
        raise ValueError(
            "Source de benchmark Core inconnue: "
            f"{source}. Utiliser 'equal_weight_core_3' ou 'selected_core_portfolio'."
        )

    core_series = pd.to_numeric(core_series, errors="coerce").dropna().copy()
    core_series.index = pd.to_datetime(core_series.index)
    core_series = core_series.sort_index()

    if returns_type == "simple":
        core_series = np.log1p(core_series)
    elif returns_type != "log":
        raise ValueError(
            f"returns_type inconnu: {returns_type}. Utiliser 'log' ou 'simple'."
        )

    return core_series.rename("core_benchmark"), source


# ── Scoring ──────────────────────────────────────────────────────────────────

def _is_fund_dead(px_series, window_end, min_recency_pct=0.20):
    """Return True if the fund has no valid price in the last min_recency_pct of the window."""
    if px_series.empty:
        return True
    last_valid = px_series.last_valid_index()
    if last_valid is None:
        return True
    n = len(px_series)
    recency_cutoff = px_series.index[max(0, int(n * (1.0 - min_recency_pct)))]
    return last_valid < recency_cutoff


def _ewm_ann_return(rets, halflife):
    """EWM-weighted annualized return."""
    w = _ewm_weights(len(rets), halflife)
    return float(np.sum(w * rets.values)) * 252


def _ewm_ann_vol(rets, halflife):
    """EWM-weighted annualized volatility."""
    w = _ewm_weights(len(rets), halflife)
    mu = float(np.sum(w * rets.values))
    var = float(np.sum(w * (rets.values - mu) ** 2))
    return np.sqrt(max(var, 0.0)) * np.sqrt(252)


def _ewm_weights(n, halflife):
    """Normalized exponential weights, most recent = highest."""
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    raw = np.array([(1.0 - alpha) ** i for i in range(n - 1, -1, -1)])
    return raw / raw.sum()


def _score_pool_by_bloc_formulas(pool_scores, prices_win, core_win, cfg):
    if pool_scores.empty:
        return pool_scores

    out = pool_scores.copy()
    if "ticker" not in out.columns:
        try:
            tcol = detect_ticker_col(out)
        except ValueError:
            return out
        out["ticker"] = out[tcol].astype(str)
    else:
        out["ticker"] = out["ticker"].astype(str)

    if "Strat" not in out.columns:
        return out

    core_s = pd.to_numeric(core_win, errors="coerce").dropna()
    if core_s.empty:
        return out

    stress_q = float(cfg["level2"].get("stress_quantile", 0.20))
    stress_days = core_s[core_s <= core_s.quantile(stress_q)].index
    min_obs = int(cfg["level2"].get("min_obs_score", cfg["level2"].get("min_obs_alpha", 60)))
    halflife = int(cfg["level2"].get("score_halflife_days", 63))  # ~3 mois par défaut

    window_end = prices_win.index[-1] if len(prices_win) > 0 else None

    metrics_rows = []
    dead_tickers = set()
    for t in out["ticker"].dropna().unique():
        if t not in prices_win.columns:
            continue

        px_raw = pd.to_numeric(prices_win[t], errors="coerce")

        # --- Dead fund detection: no valid price in last 20% of window ---
        if _is_fund_dead(px_raw, window_end):
            dead_tickers.add(t)
            continue

        px = px_raw.dropna()
        fund_ret = np.log(px).diff().dropna()
        aligned = pd.concat([fund_ret.rename("fund"), core_s.rename("core")], axis=1).dropna()
        if len(aligned) < min_obs:
            continue

        f = aligned["fund"]
        c = aligned["core"]

        # --- EWM-weighted metrics (recent observations count more) ---
        ret_ann = _ewm_ann_return(f, halflife)
        vol = _ewm_ann_vol(f, halflife)
        core_ret_ann = _ewm_ann_return(c, halflife)

        core_var = c.var()
        beta = f.cov(c) / core_var if pd.notna(core_var) and core_var > 0 else np.nan

        beta_stress = np.nan
        stress_aligned = aligned.loc[aligned.index.intersection(stress_days)]
        if len(stress_aligned) >= 2:
            core_var_stress = stress_aligned["core"].var()
            if pd.notna(core_var_stress) and core_var_stress > 0:
                beta_stress = stress_aligned["fund"].cov(stress_aligned["core"]) / core_var_stress

        beta_bloc1 = beta_stress if pd.notna(beta_stress) else beta
        corr = f.corr(c)
        sharpe = ret_ann / vol if pd.notna(ret_ann) and pd.notna(vol) and vol > 0 else np.nan
        sortino = sortino0(f)
        maxdd = maxdd_abs(f)
        skewness = f.skew()
        kurtosis = f.kurt()

        alpha = ret_ann - beta * core_ret_ann if pd.notna(ret_ann) and pd.notna(beta) and pd.notna(core_ret_ann) else np.nan

        # Information ratio = annualized active return / tracking error
        active_rets = f - c
        te = active_rets.std() * np.sqrt(252)
        active_ann = active_rets.mean() * 252
        info_ratio = active_ann / te if pd.notna(te) and te > 1e-12 else np.nan

        rs = aligned.loc[aligned.index.intersection(stress_days), "fund"]
        return_stress = ann_return(rs) if len(rs) > 0 else np.nan

        metrics_rows.append({
            "ticker": t, "beta": beta, "beta_stress": beta_stress, "beta_bloc1": beta_bloc1,
            "corr": corr, "return_stress": return_stress, "skewness": skewness, "maxdd_abs": maxdd,
            "alpha_annual": alpha, "sharpe": sharpe, "sortino": sortino, "vol": vol,
            "ret_ann": ret_ann, "kurtosis": kurtosis, "info_ratio": info_ratio,
        })

    if not metrics_rows:
        return out

    m = pd.DataFrame(metrics_rows)
    out = out.merge(m, on="ticker", how="left", suffixes=("", "_new"))

    if "alpha_annual_new" in out.columns:
        out["alpha_annual"] = out["alpha_annual_new"].combine_first(pd.to_numeric(out.get("alpha_annual"), errors="coerce"))
    if "beta_new" in out.columns:
        out["beta"] = out["beta_new"].combine_first(pd.to_numeric(out.get("beta"), errors="coerce"))
    if "corr_new" in out.columns:
        out["corr"] = out["corr_new"].combine_first(pd.to_numeric(out.get("corr"), errors="coerce"))

    parts = []
    for strat, g in out.groupby("Strat", dropna=False):
        g = g.copy()
        su = str(strat).upper()

        z_beta_b1 = safe_zscore(g["beta_bloc1"].abs())
        z_corr_abs = safe_zscore(g["corr"].abs())
        z_stress = safe_zscore(g["return_stress"])
        z_skew = safe_zscore(g["skewness"])
        z_mdd = safe_zscore(g["maxdd_abs"])
        z_alpha = safe_zscore(g["alpha_annual"])
        z_so = safe_zscore(g["sortino"])
        z_ret = safe_zscore(g["ret_ann"])
        z_kurt = safe_zscore(g["kurtosis"])
        z_info = safe_zscore(g["info_ratio"])

        if su == "STRAT1":
            # Couverture / Tail Risk : decorrelation + stress protection
            g["score_custom"] = (
                -0.35 * z_beta_b1
                - 0.25 * z_corr_abs
                + 0.20 * z_stress
                + 0.10 * z_so
                - 0.10 * z_mdd
            )
        elif su == "STRAT2":
            # Alpha / Valeur Relative : alpha + info ratio + risk-adjusted
            g["score_custom"] = (
                0.40 * z_alpha
                + 0.25 * z_info
                + 0.20 * z_so
                - 0.10 * z_corr_abs
                - 0.05 * z_mdd
            )
        elif su == "STRAT3":
            # Alternatives / Cat Bonds : carry + decorrelation + tail profile
            g["score_custom"] = (
                0.30 * z_ret
                + 0.25 * z_alpha
                - 0.20 * z_corr_abs
                + 0.15 * z_skew
                - 0.10 * z_kurt
            )
        else:
            g["score_custom"] = np.nan

        if "score_level2" in g.columns:
            g["score_level2"] = g["score_custom"].combine_first(pd.to_numeric(g["score_level2"], errors="coerce"))
        else:
            g["score_level2"] = g["score_custom"]
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=[c for c in out.columns if c.endswith("_new")])

    # --- Dead funds → score = -1000 (ensures natural replacement) ---
    if dead_tickers:
        dead_mask = out["ticker"].isin(dead_tickers)
        out.loc[dead_mask, "score_level2"] = -1000.0
        out.loc[dead_mask, "score_custom"] = -1000.0
    return out


# ── Selection helpers ────────────────────────────────────────────────────────

def _pick_ranked_for_strat(g, prices_window, params_final):
    p = params_final
    g2 = g.sort_values("score_level2", ascending=False).reset_index(drop=True)
    if g2.empty:
        return []

    picks = [{
        "ticker": str(g2.iloc[0]["ticker"]),
        "rank_in_strat": 1,
        "score_level2": float(g2.iloc[0]["score_level2"]),
        "selected_reason": "top1",
        "pair_corr": np.nan,
        "score_gap_vs_top1": 0.0,
    }]

    if p["max_per_strat"] < 2 or len(g2) < 2:
        return picks

    top1 = g2.iloc[0]
    top2 = g2.iloc[1]
    t1, t2 = str(top1["ticker"]), str(top2["ticker"])

    pair = prices_window[[t1, t2]].copy() if (t1 in prices_window.columns and t2 in prices_window.columns) else pd.DataFrame()
    pair_rets = np.log(pair).diff().dropna() if not pair.empty else pd.DataFrame()
    pair_corr = float(pair_rets[t1].corr(pair_rets[t2])) if len(pair_rets) > 10 else np.nan
    gap = float(top1["score_level2"] - top2["score_level2"])

    cond_corr = np.isnan(pair_corr) or abs(pair_corr) <= p["corr_max_for_second"]
    cond_gap = gap <= p["score_gap_for_second"]

    if cond_corr and cond_gap:
        picks.append({
            "ticker": t2, "rank_in_strat": 2, "score_level2": float(top2["score_level2"]),
            "selected_reason": "top2_diversified", "pair_corr": pair_corr, "score_gap_vs_top1": gap,
        })
    return picks


def _build_annual_review_dates(prices_idx, cfg):
    start = pd.Timestamp(cfg["rolling"]["oos_start"]).normalize()
    end = pd.Timestamp(cfg["rolling"]["oos_end"]).normalize()
    month = int(cfg["rolling"]["annual_review_month"])
    day = int(cfg["rolling"]["annual_review_day"])

    year_candidates = list(range(start.year, end.year + 1))
    review_dates = []
    for y in year_candidates:
        target = pd.Timestamp(year=y, month=month, day=day)
        valid = prices_idx[prices_idx >= target]
        if len(valid) == 0:
            continue
        d = pd.Timestamp(valid.min()).normalize()
        if d <= end:
            review_dates.append(d)

    return sorted(pd.unique(pd.DatetimeIndex(review_dates)))


def _annual_pool_at_date(review_date, df_level0, prices_all_aligned, core_all_aligned, cfg):
    lookback_years = int(cfg["rolling"]["lookback_years_for_score"])
    calib_end = pd.Timestamp(review_date) - pd.Timedelta(days=1)
    calib_start = calib_end - pd.DateOffset(years=lookback_years) + pd.Timedelta(days=1)

    prices_win = prices_all_aligned.loc[calib_start:calib_end].copy()
    core_win = core_all_aligned.loc[calib_start:calib_end].copy()

    if len(prices_win) < max(80, int(cfg["level1"]["rolling_window"])) or len(core_win) < 80:
        return pd.DataFrame(), pd.DataFrame(), calib_start, calib_end

    l1 = cfg["level1"]
    df_l1, res_l1 = apply_level1_filter_corrected(
        df_level0, prices_win, core_win,
        calib_start=calib_start.strftime("%Y-%m-%d"),
        calib_end=calib_end.strftime("%Y-%m-%d"),
        rolling_window=l1["rolling_window"],
        median_beta_max=l1["median_beta_max"],
        q75_beta_max=l1["q75_beta_max"],
        pass_ratio_min=l1["pass_ratio_min"],
        verbose=False,
    )

    if df_l1.empty:
        if not res_l1.empty:
            res_l1 = res_l1.copy()
            res_l1["review_date"] = pd.Timestamp(review_date)
            res_l1["calib_start"] = calib_start
            res_l1["calib_end"] = calib_end
        return pd.DataFrame(), res_l1, calib_start, calib_end

    # Structural pool filter (TER cap + min obs) — replaces L2 scoring as gating
    pool_df = _build_pool_from_level1(df_l1, prices_win, cfg)
    if pool_df.empty:
        return pd.DataFrame(), pd.DataFrame(), calib_start, calib_end

    # Compute bloc scores for display/tracking (pool is NOT gated by score)
    scored = _score_pool_by_bloc_formulas(pool_df.copy(), prices_win, core_win, cfg)

    scored["review_date"] = pd.Timestamp(review_date)
    scored["calib_start"] = calib_start
    scored["calib_end"] = calib_end

    return scored, scored, calib_start, calib_end


def _quarterly_review_dates(prices_idx, cfg):
    start = pd.Timestamp(cfg["rolling"]["oos_start"]).normalize()
    end = pd.Timestamp(cfg["rolling"]["oos_end"]).normalize()

    # Support review_freq_months (integer) or quarterly_freq (pandas offset string)
    freq_months = cfg["rolling"].get("review_freq_months")
    if freq_months is not None:
        freq = f"{int(freq_months)}MS"
    else:
        freq = cfg["rolling"]["quarterly_freq"]

    raw_q = pd.date_range(start=start, end=end, freq=freq)

    q_dates = []
    for d in raw_q:
        valid = prices_idx[prices_idx >= d]
        if len(valid) == 0:
            continue
        qd = pd.Timestamp(valid.min()).normalize()
        if qd <= end:
            q_dates.append(qd)
    return sorted(pd.unique(pd.DatetimeIndex(q_dates)))


def _rescore_pool_quarterly(pool_scores, prices_all_aligned, core_all_aligned, qdate, cfg):
    if pool_scores.empty:
        return pool_scores
    lookback_years = int(cfg["rolling"].get("lookback_years_for_score", 1))
    win_end = pd.Timestamp(qdate) - pd.Timedelta(days=1)
    win_start = win_end - pd.DateOffset(years=lookback_years) + pd.Timedelta(days=1)
    prices_win = prices_all_aligned.loc[win_start:win_end]
    core_win = core_all_aligned.loc[win_start:win_end]
    if len(prices_win) < 50 or len(core_win) < 50:
        return pool_scores
    return _score_pool_by_bloc_formulas(pool_scores.copy(), prices_win, core_win, cfg)


def _pick_from_pool_for_quarter(pool_scores, prices_all_aligned, core_all_aligned, qdate, cfg):
    p = cfg["final"]
    win_end = pd.Timestamp(qdate) - pd.Timedelta(days=1)
    win_start = win_end - pd.DateOffset(years=cfg["rolling"]["lookback_years_for_score"]) + pd.Timedelta(days=1)
    prices_win = prices_all_aligned.loc[win_start:win_end]

    # Always rescore with fresh lookback — this is the sole scoring for selection
    pool_eff = _rescore_pool_quarterly(pool_scores.copy(), prices_all_aligned, core_all_aligned, qdate, cfg)

    rows = []
    # Build a simple (strat, ticker) -> score dict for fast lookup in switching logic
    fresh_scores_dict = {}  # (strat_str, ticker_str) -> float score
    for strat, g in _safe_strat_group(pool_eff):
        picks = _pick_ranked_for_strat(g, prices_win, p)
        for r in picks:
            rows.append({
                "quarter_date": pd.Timestamp(qdate), "Strat": strat,
                "ticker": r["ticker"], "rank_in_strat": r["rank_in_strat"],
                "score_level2": r["score_level2"], "selected_reason": r["selected_reason"],
                "pair_corr": r["pair_corr"], "score_gap_vs_top1": r["score_gap_vs_top1"],
            })
        # Capture ALL fund scores (not just picked ones) for held-fund freshness lookup
        tc = detect_ticker_col(g) if not g.empty else None
        if tc is not None:
            for _, row in g.iterrows():
                s = float(row.get("score_level2", np.nan))
                if not np.isnan(s):
                    fresh_scores_dict[(str(strat), str(row[tc]))] = s
    # Return top picks AND score dict (avoids DataFrame key/type issues in switching logic)
    return pd.DataFrame(rows), fresh_scores_dict


def _apply_quarterly_switching(all_quarter_candidates, cfg, pool_scores_by_quarter=None):
    """pool_scores_by_quarter: dict[pd.Timestamp -> dict[(strat_str, ticker_str) -> float]]"""
    if all_quarter_candidates.empty:
        return all_quarter_candidates

    out_rows = []
    switch_buffer = float(cfg["final"]["switch_score_buffer"])

    for strat, g in all_quarter_candidates.groupby("Strat"):
        g = g.sort_values(["quarter_date", "rank_in_strat"]).copy()
        held_by_rank = {}

        for qd, gq in g.groupby("quarter_date"):
            # Normalize to pd.Timestamp to guarantee dict key match
            qd_ts = pd.Timestamp(qd)
            q_rows = []
            for rank in sorted(gq["rank_in_strat"].unique()):
                cand = gq[gq["rank_in_strat"] == rank].sort_values("score_level2", ascending=False).iloc[0].to_dict()
                held = held_by_rank.get(rank)

                if held is None:
                    chosen = cand
                    chosen["switch_flag"] = 1
                    chosen["switch_reason"] = "init"
                else:
                    held_ticker = held["ticker"]
                    cand_ticker = cand["ticker"]
                    held_score = float(held.get("score_level2", np.nan))
                    cand_score = float(cand.get("score_level2", np.nan))

                    # Look up held fund's CURRENT score from fresh pool rescore dict
                    # Uses simple (strat, ticker) tuple key — avoids DataFrame type issues
                    fresh_held_score = held_score  # fallback to stale if not found
                    if pool_scores_by_quarter is not None and qd_ts in pool_scores_by_quarter:
                        score_map = pool_scores_by_quarter[qd_ts]
                        fresh_held_score = score_map.get((str(strat), str(held_ticker)), held_score)

                    if held_ticker == cand_ticker:
                        chosen = cand
                        chosen["switch_flag"] = 0
                        chosen["switch_reason"] = "same_ticker"
                    elif cand_score >= fresh_held_score + switch_buffer:
                        chosen = cand
                        chosen["switch_flag"] = 1
                        chosen["switch_reason"] = "better_score"
                    else:
                        chosen = held.copy()
                        chosen["quarter_date"] = qd
                        chosen["score_level2"] = fresh_held_score  # update with fresh score
                        chosen["switch_flag"] = 0
                        chosen["switch_reason"] = "hold_buffer"

                q_rows.append(chosen)

            used = set()
            fixed_rows = []
            for row in sorted(q_rows, key=lambda x: x["rank_in_strat"]):
                if row["ticker"] in used:
                    alt = gq[(gq["rank_in_strat"] == row["rank_in_strat"]) & (~gq["ticker"].isin(list(used)))]
                    if alt.empty:
                        continue
                    row = alt.sort_values("score_level2", ascending=False).iloc[0].to_dict()
                    row["switch_flag"] = 1
                    row["switch_reason"] = "deduplicate"
                used.add(row["ticker"])
                held_by_rank[row["rank_in_strat"]] = row.copy()
                fixed_rows.append(row)

            out_rows.extend(fixed_rows)

    return pd.DataFrame(out_rows).sort_values(["quarter_date", "Strat", "rank_in_strat"]).reset_index(drop=True)


# ── Main runner ──────────────────────────────────────────────────────────────

def run_satellite_pipeline(core_3_log, params, core_benchmark_returns=None):
    """
    Run the full satellite pipeline: Level0 filter → annual pool → quarterly selection.

    Returns dict with keys:
        df_satellite_level0, annual_pool_scores, quarter_selection_df,
        satellite_final_selection, results_level2, df_satellite_level2,
        df_satellite_prices_all, df_satellite_prices_aligned, core_returns_aligned
    """
    # Level 0 filter
    print("\n[Etape 3 glissante] Niveau 0 - filtre structurel")
    df_satellite_level0, summary_level0 = filter_satellite_level0(
        data_dir="data",
        devise=params["level0"]["devise"],
        min_aum_usd=params["level0"]["min_aum_usd_m"],
        verbose=False,
    )
    print(f"N0 initial: {len(df_satellite_level0)} fonds")

    # Ban STRAT3 sheets
    banned_tickers_strat3 = _build_banned_tickers_by_sheet(params, data_dir="data")
    try:
        ticker_col_level0 = detect_ticker_col(df_satellite_level0)
    except ValueError:
        ticker_col_level0 = None
    if ticker_col_level0 is not None and banned_tickers_strat3:
        n_before_ban = len(df_satellite_level0)
        mask_keep = ~(
            (df_satellite_level0["Strat"] == "STRAT3")
            & (df_satellite_level0[ticker_col_level0].astype(str).isin(banned_tickers_strat3))
        )
        df_satellite_level0 = df_satellite_level0.loc[mask_keep].copy()
        print(f"N0 après bannissement feuilles STRAT3: {len(df_satellite_level0)} (retirés: {n_before_ban - len(df_satellite_level0)})")

    # Load prices + benchmark
    print("\n[Etape 3 glissante] Chargement univers prix + benchmark core")
    core_returns_benchmark, benchmark_source = _resolve_core_benchmark_returns(
        core_3_log,
        params,
        core_benchmark_returns=core_benchmark_returns,
    )
    print(f"Benchmark Core utilisé: {benchmark_source}")

    df_satellite_prices_all = load_all_satellite_prices(data_dir="data")
    global_start = (pd.Timestamp(params["rolling"]["oos_start"]) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    global_end = params["rolling"]["oos_end"]

    df_satellite_prices_global, valid_tickers_prices = preprocess_prices(
        df_satellite_prices_all,
        start_date=global_start,
        end_date=global_end,
        ffill_limit=params["data"]["ffill_limit"],
        min_obs=params["data"]["min_obs_prices"],
    )

    df_satellite_prices_aligned, core_returns_aligned = align_prices_with_core(
        df_satellite_prices_global,
        core_returns_benchmark,
        ffill_limit=params["data"]["ffill_limit"],
    )

    # Annual review
    all_idx = df_satellite_prices_aligned.index.sort_values()
    annual_dates = _build_annual_review_dates(all_idx, params)
    quarter_dates = _quarterly_review_dates(all_idx, params)

    # Label dynamique selon la fréquence de revue
    _freq_k = params["rolling"].get("review_freq_months")
    _freq_labels = {1: "mensuel", 2: "bimestriel", 3: "trimestriel", 6: "semestriel", 12: "annuel"}
    _freq_label = _freq_labels.get(_freq_k, f"{_freq_k}-mensuel") if _freq_k else "trimestriel"

    print(f"Revue annuelle: {len(annual_dates)} dates | Revue {_freq_label}: {len(quarter_dates)} dates")

    print("\n[Etape 3 glissante] Construction du pool annuel")
    annual_pool_rows = []
    annual_top_rows = []

    for i, review_date in enumerate(annual_dates):
        df_pool, res_scores, calib_start, calib_end = _annual_pool_at_date(
            review_date, df_satellite_level0, df_satellite_prices_aligned, core_returns_aligned, params
        )
        if not res_scores.empty and "score_level2" in res_scores.columns:
            annual_pool_rows.append(res_scores.copy())

        next_review = annual_dates[i + 1] if i + 1 < len(annual_dates) else pd.Timestamp(params["rolling"]["oos_end"]) + pd.Timedelta(days=1)
        active_to = next_review - pd.Timedelta(days=1)

        if not df_pool.empty:
            top = df_pool.copy()
            ticker_col = "Ticker" if "Ticker" in top.columns else ("ticker" if "ticker" in top.columns else None)
            if ticker_col is None:
                continue
            top["ticker"] = top[ticker_col].astype(str)
            top["review_date"] = pd.Timestamp(review_date)
            top["active_from"] = pd.Timestamp(review_date)
            top["active_to"] = pd.Timestamp(active_to)
            top["calib_start"] = pd.Timestamp(calib_start)
            top["calib_end"] = pd.Timestamp(calib_end)
            annual_top_rows.append(top)

    annual_pool_scores = pd.concat(annual_pool_rows, ignore_index=True) if annual_pool_rows else pd.DataFrame()
    annual_pool_top = pd.concat(annual_top_rows, ignore_index=True) if annual_top_rows else pd.DataFrame()
    print(f"Pools annuels construits: {annual_pool_top['review_date'].nunique() if not annual_pool_top.empty else 0}")

    # Periodic selection with switching
    print(f"\n[Etape 3 glissante] Selection {_freq_label} avec switching")
    quarter_candidate_rows = []
    pool_scores_by_quarter = {}  # qd -> full rescored pool (for fresh held-fund score lookup)

    for qd in quarter_dates:
        if annual_pool_top.empty or annual_pool_scores.empty or "review_date" not in annual_pool_scores.columns:
            continue
        review_candidates = sorted(pd.to_datetime(annual_pool_top["review_date"].dropna().unique()))
        eligible_reviews = [d for d in review_candidates if d <= qd]
        if not eligible_reviews:
            continue
        active_review = max(eligible_reviews)
        pool_q = annual_pool_scores[annual_pool_scores["review_date"] == pd.Timestamp(active_review)].copy()
        if pool_q.empty:
            continue
        q_candidates, fresh_scores_dict = _pick_from_pool_for_quarter(pool_q, df_satellite_prices_aligned, core_returns_aligned, qd, params)
        pool_scores_by_quarter[pd.Timestamp(qd)] = fresh_scores_dict
        if not q_candidates.empty:
            q_candidates["active_review_date"] = pd.Timestamp(active_review)
            quarter_candidate_rows.append(q_candidates)

    quarter_candidates = pd.concat(quarter_candidate_rows, ignore_index=True) if quarter_candidate_rows else pd.DataFrame()
    # pool_scores_by_quarter: dict[pd.Timestamp -> dict[(strat,ticker)->score]]
    quarter_selection_df = _apply_quarterly_switching(quarter_candidates, params, pool_scores_by_quarter=pool_scores_by_quarter) if not quarter_candidates.empty else pd.DataFrame()

    # Final selection
    if not quarter_selection_df.empty:
        latest_q = quarter_selection_df["quarter_date"].max()
        final_selection_df = quarter_selection_df[quarter_selection_df["quarter_date"] == latest_q].copy()
    else:
        latest_q = pd.NaT
        final_selection_df = pd.DataFrame()

    meta_source = annual_pool_top.copy() if not annual_pool_top.empty else pd.DataFrame()
    if not final_selection_df.empty and not meta_source.empty:
        cols_meta = [c for c in ["ticker", "Strat", "Nom", "Dev", "Ratio des dépenses", "Total actifs USD (M)"] if c in meta_source.columns]
        satellite_final_selection = final_selection_df.merge(
            meta_source[cols_meta].drop_duplicates(subset=["ticker"]),
            on=["ticker", "Strat"] if "Strat" in cols_meta else ["ticker"],
            how="left",
        )
    else:
        satellite_final_selection = final_selection_df.copy()

    results_level2 = annual_pool_scores.copy()
    df_satellite_level2 = annual_pool_top.copy()

    print(f"\n[Etape 3 glissante] Termine")
    print(f"  - N0 initial: {len(df_satellite_level0)}")
    print(f"  - Reviews annuelles: {len(annual_dates)}")
    print(f"  - Periodes evaluees ({_freq_label}): {len(quarter_dates)}")
    print(f"  - Lignes pool annuel (scores): {len(annual_pool_scores)}")
    print(f"  - Lignes selection {_freq_label}: {len(quarter_selection_df)}")
    if pd.notna(latest_q):
        print(f"  - Derniere selection active ({latest_q.date()}): {len(final_selection_df)} lignes")
    else:
        print("  - Aucune selection finale produite")

    return {
        "df_satellite_level0": df_satellite_level0,
        "annual_pool_scores": annual_pool_scores,
        "annual_pool_top": annual_pool_top,
        "quarter_selection_df": quarter_selection_df,
        "satellite_final_selection": satellite_final_selection,
        "results_level2": results_level2,
        "df_satellite_level2": df_satellite_level2,
        "df_satellite_prices_all": df_satellite_prices_all,
        "df_satellite_prices_aligned": df_satellite_prices_aligned,
        "core_returns_aligned": core_returns_aligned,
    }


# ── Display helpers ──────────────────────────────────────────────────────────

def display_pool_scores(annual_pool_scores):
    """Display annual pool score summary (last rows)."""
    from IPython.display import display as _display
    import pandas as pd

    if annual_pool_scores is None or not isinstance(annual_pool_scores, pd.DataFrame) or annual_pool_scores.empty:
        print('Aucun score de pool annuel disponible.')
        return

    cols = [c for c in ['review_date', 'Strat', 'ticker', 'score_level2', 'alpha_annual', 'expense_pct'] if c in annual_pool_scores.columns]
    if cols:
        sort_by = [c for c in ['review_date', 'Strat', 'score_level2'] if c in cols]
        df_show = annual_pool_scores[cols].sort_values(sort_by, ascending=[True, True, False] if len(sort_by) == 3 else [True] * len(sort_by))
        print(f'\nPool annuel : {len(annual_pool_scores)} lignes')
        _display(df_show.tail(30))
    else:
        _display(annual_pool_scores.tail(30))


def display_satellite_selection(results_level2, quarter_selection_df, satellite_final_selection, review_freq_months=3):
    """Display L2 shortlist, full periodic selection history, and final snapshot."""
    from IPython.display import display as _display
    import pandas as pd

    # Label dynamique selon la fréquence
    _freq_labels = {1: "mensuel", 2: "bimestriel", 3: "trimestriel", 6: "semestriel", 12: "annuel"}
    _freq_label = _freq_labels.get(review_freq_months, f"{review_freq_months}-mensuel")
    # Offset pandas pour groupby (ex: 1 mois = 'M', 3 mois = 'Q')
    _period_freq = 'M' if review_freq_months == 1 else ('Q' if review_freq_months == 3 else f"{review_freq_months}ME")

    # 1) Score niveau 2
    lvl2 = results_level2.copy() if isinstance(results_level2, pd.DataFrame) and not results_level2.empty else None
    if lvl2 is not None:
        print(f'\n📊 Niveau 2 scores: {len(lvl2)} lignes')
        cols = [c for c in ['ticker', 'Strat', 'alpha_annual', 'expense_pct', 'score_level2'] if c in lvl2.columns]
        if cols:
            sort_cols = [c for c in ['Strat', 'score_level2'] if c in lvl2.columns]
            if sort_cols:
                _display(lvl2[cols].sort_values(sort_cols, ascending=[True, False]).head(30))
            else:
                _display(lvl2[cols].head(30))

    # 2) Historique complet
    if not isinstance(quarter_selection_df, pd.DataFrame) or quarter_selection_df.empty:
        print('Historique des sélections indisponible.')
        return

    sel_hist = quarter_selection_df.copy()
    if 'Ticker' in sel_hist.columns and 'ticker' not in sel_hist.columns:
        sel_hist = sel_hist.rename(columns={'Ticker': 'ticker'})
    sel_hist['quarter_date'] = pd.to_datetime(sel_hist['quarter_date'], errors='coerce')
    sel_hist = sel_hist.dropna(subset=['quarter_date']).copy()
    sel_hist = sel_hist.sort_values(
        by=[c for c in ['quarter_date', 'Strat', 'rank_in_strat', 'ticker'] if c in sel_hist.columns]
    ).reset_index(drop=True)

    hist_cols = [c for c in [
        'quarter_date', 'Strat', 'ticker', 'rank_in_strat', 'score_level2', 'selected_reason',
        'switch_flag', 'same_ticker', 'pair_corr', 'score_gap_vs_top1', 'Nom',
        'Dev', 'Ratio des dépenses', 'Total actifs USD (M)'
    ] if c in sel_hist.columns]

    # dynamique: compte les périodes selon la fréquence
    n_periods = sel_hist['quarter_date'].dt.to_period('M').nunique() if review_freq_months == 1 else sel_hist['quarter_date'].dt.to_period('Q').nunique()
    period_noun = "période" + ("s" if n_periods > 1 else "")
    print(
        f"\n\U0001f5c2\ufe0f Historique complet des sélections {_freq_label}s: "
        f"{n_periods} {period_noun} | {len(sel_hist)} lignes"
    )
    _display(sel_hist[hist_cols] if hist_cols else sel_hist)

    if 'switch_flag' in sel_hist.columns:
        agg_dict = {'n_fonds': ('ticker', 'count'), 'n_switches': ('switch_flag', 'sum')}
        if 'score_level2' in sel_hist.columns:
            agg_dict['score_moyen'] = ('score_level2', 'mean')
        switch_summary = (
            sel_hist.assign(periode=sel_hist['quarter_date'].dt.to_period(_period_freq).astype(str))
            .groupby('periode', as_index=False).agg(**agg_dict)
        )
        print(f'\nRésumé {_freq_label} des sélections:')
        if 'score_moyen' in switch_summary.columns:
            _display(switch_summary.style.format({'score_moyen': '{:.3f}'}))
        else:
            _display(switch_summary)

    # 3) Snapshot final
    final_df = satellite_final_selection if isinstance(satellite_final_selection, pd.DataFrame) and not satellite_final_selection.empty else None
    if final_df is not None:
        print(f"\n✅ Snapshot final le plus récent: {len(final_df)} lignes")
        cols_f = [c for c in [
            'Strat', 'ticker', 'rank_in_strat', 'score_level2', 'selected_reason',
            'pair_corr', 'score_gap_vs_top1', 'Nom', 'Dev', 'Ratio des dépenses', 'Total actifs USD (M)'
        ] if c in final_df.columns]
        if cols_f:
            sort_f = [c for c in ['Strat', 'rank_in_strat'] if c in final_df.columns]
            _display(final_df[cols_f].sort_values(sort_f) if sort_f else final_df[cols_f])
    else:
        print('Sélection finale introuvable.')


# ── Holdings timeline display ────────────────────────────────────────────────

def display_holdings_timeline(quarter_selection_df, figsize=(17, 7)):
    """
    Gantt-style chart: shows exactly which fund is held at each period, per strat and per rank slot.
    Switch events are highlighted with a red dashed line.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba

    if quarter_selection_df is None or quarter_selection_df.empty:
        print("Aucune sélection disponible.")
        return None

    df = quarter_selection_df.copy()
    df["quarter_date"] = pd.to_datetime(df["quarter_date"])

    all_tickers = sorted(df["ticker"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(all_tickers), 1))
    ticker_color = {t: cmap(i) for i, t in enumerate(all_tickers)}

    def short(t):
        return t.split(" ")[0]

    strats = sorted(df["Strat"].unique())
    ranks = sorted(df["rank_in_strat"].unique())

    fig, axes = plt.subplots(len(strats), 1, figsize=figsize, sharex=True)
    if len(strats) == 1:
        axes = [axes]

    all_dates = sorted(df["quarter_date"].unique())
    freq_months = int(round((all_dates[1] - all_dates[0]).days / 30)) if len(all_dates) > 1 else 1

    for ax, strat in zip(axes, strats):
        sub = df[df["Strat"] == strat].copy().sort_values(["quarter_date", "rank_in_strat"])
        strat_ranks = sorted(sub["rank_in_strat"].unique())
        y_pos = {r: i for i, r in enumerate(reversed(strat_ranks))}

        for rank in strat_ranks:
            r_df = sub[sub["rank_in_strat"] == rank].sort_values("quarter_date")
            if r_df.empty:
                continue
            # Build consecutive holding blocks
            groups, cur = [], None
            for _, row in r_df.iterrows():
                if cur is None or row["ticker"] != cur["ticker"]:
                    if cur:
                        groups.append(cur)
                    cur = {"ticker": row["ticker"], "start": row["quarter_date"],
                           "end": row["quarter_date"]}
                else:
                    cur["end"] = row["quarter_date"]
            if cur:
                groups.append(cur)

            y = y_pos[rank]
            for grp in groups:
                end_draw = grp["end"] + pd.DateOffset(months=freq_months)
                width_days = (end_draw - grp["start"]).days
                color = ticker_color[grp["ticker"]]
                ax.barh(y, width_days, left=grp["start"], height=0.65,
                        color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
                mid_date = grp["start"] + (end_draw - grp["start"]) / 2
                ax.text(mid_date, y, short(grp["ticker"]),
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white")

        # Switch markers (red dashed lines)
        switches = sub[sub["switch_flag"] == 1]
        switch_dates = switches["quarter_date"].unique()
        for sd in switch_dates:
            ax.axvline(sd, color="crimson", linewidth=1.5, linestyle="--", alpha=0.75)

        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels([f"Rang {r}" for r in reversed(strat_ranks)], fontsize=9)
        ax.set_title(f"  {strat}", fontweight="bold", loc="left", pad=3, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2, linestyle=":")

    # X-axis formatting
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    n_months = (all_dates[-1] - all_dates[0]).days // 30 if len(all_dates) > 1 else 12
    interval = max(1, n_months // 20)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    # Legend
    handles = [mpatches.Patch(color=ticker_color[t], label=short(t)) for t in all_tickers]
    switch_handle = plt.Line2D([0], [0], color="crimson", linestyle="--", linewidth=1.5, label="Switch")
    fig.legend(handles=handles + [switch_handle], loc="lower center",
               ncol=min(10, len(all_tickers) + 1), fontsize=8,
               title="Fonds détenus", title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("📋 Holdings Satellite — Timeline des fonds détenus", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def display_holdings_table(quarter_selection_df, review_freq_months=1):
    """
    Pivot table: rows = periods, columns = (STRAT, Rang), values = fund held.
    Switches are highlighted in bold / yellow background via pandas Styler.
    """
    from IPython.display import display as _display

    if quarter_selection_df is None or quarter_selection_df.empty:
        print("Aucune sélection disponible.")
        return

    df = quarter_selection_df.copy()
    df["quarter_date"] = pd.to_datetime(df["quarter_date"])

    _freq_labels = {1: "mensuel", 2: "bimestriel", 3: "trimestriel", 6: "semestriel", 12: "annuel"}
    _freq_label = _freq_labels.get(review_freq_months, f"{review_freq_months}-mensuel")

    # Short display name
    df["fond"] = df["ticker"].str.split(" ").str[0]

    # Pivot: period x (Strat, Rang)
    pivot = df.pivot_table(
        index="quarter_date", columns=["Strat", "rank_in_strat"],
        values="fond", aggfunc="first"
    )
    pivot.index = pivot.index.strftime("%Y-%m")
    pivot.columns = [f"{s} R{r}" for s, r in pivot.columns]

    # Switch flags for highlighting
    df["col_key"] = df.apply(lambda r: f"{r['Strat']} R{r['rank_in_strat']}", axis=1)
    df["period"] = df["quarter_date"].dt.strftime("%Y-%m")
    switch_cells = set(
        zip(df.loc[df["switch_flag"] == 1, "period"],
            df.loc[df["switch_flag"] == 1, "col_key"])
    )

    def highlight_switches(df_styled):
        styles = pd.DataFrame("", index=df_styled.index, columns=df_styled.columns)
        for (period, col) in switch_cells:
            if period in df_styled.index and col in df_styled.columns:
                styles.loc[period, col] = "background-color: #fff3cd; font-weight: bold;"
        return styles

    n_switches = df["switch_flag"].sum()
    n_periods = pivot.shape[0]
    print(f"\n📋 Table des fonds détenus — révision {_freq_label} | {n_periods} périodes | {n_switches} switches (↓ cellules en jaune)")
    _display(
        pivot.style
        .apply(highlight_switches, axis=None)
        .set_properties(**{"font-size": "11px", "text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("font-size", "10px"), ("text-align", "center")]}])
    )
