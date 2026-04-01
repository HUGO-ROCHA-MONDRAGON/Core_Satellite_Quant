"""Satellite fund selection pipeline: Level 0/1/2 filtering + rolling annual/quarterly selection."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.satellite_filters import (
    filter_satellite_level0,
    apply_level1_filter_corrected,
    apply_level2_alpha_expense,
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
    banned_sheets = cfg.get("bans", {}).get("strat3_banned_sheet_names", [])
    if not banned_sheets:
        return set()
    price_file = Path(data_dir) / "STRAT3_price.xlsx"
    banned_tickers = set()
    for sh in banned_sheets:
        banned_tickers |= _extract_tickers_from_price_sheet(price_file, sh)
    return banned_tickers


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
    min_obs = int(cfg["level2"].get("min_obs_alpha", 60))

    metrics_rows = []
    for t in out["ticker"].dropna().unique():
        if t not in prices_win.columns:
            continue
        px = pd.to_numeric(prices_win[t], errors="coerce").dropna()
        fund_ret = np.log(px).diff().dropna()
        aligned = pd.concat([fund_ret.rename("fund"), core_s.rename("core")], axis=1).dropna()
        if len(aligned) < min_obs:
            continue

        core_var = aligned["core"].var()
        beta = aligned["fund"].cov(aligned["core"]) / core_var if pd.notna(core_var) and core_var > 0 else np.nan

        beta_stress = np.nan
        stress_aligned = aligned.loc[aligned.index.intersection(stress_days)]
        if len(stress_aligned) >= 2:
            core_var_stress = stress_aligned["core"].var()
            if pd.notna(core_var_stress) and core_var_stress > 0:
                beta_stress = stress_aligned["fund"].cov(stress_aligned["core"]) / core_var_stress

        beta_bloc1 = beta_stress if pd.notna(beta_stress) else beta
        corr = aligned["fund"].corr(aligned["core"])
        ret_ann = ann_return(aligned["fund"])
        vol = ann_vol(aligned["fund"])
        sharpe = ret_ann / vol if pd.notna(ret_ann) and pd.notna(vol) and vol > 0 else np.nan
        sortino = sortino0(aligned["fund"])
        maxdd = maxdd_abs(aligned["fund"])
        skewness = aligned["fund"].skew()
        kurtosis = aligned["fund"].kurt()

        core_ret_ann = ann_return(aligned["core"])
        alpha = ret_ann - beta * core_ret_ann if pd.notna(ret_ann) and pd.notna(beta) and pd.notna(core_ret_ann) else np.nan

        rs = aligned.loc[aligned.index.intersection(stress_days), "fund"]
        return_stress = ann_return(rs) if len(rs) > 0 else np.nan

        metrics_rows.append({
            "ticker": t, "beta": beta, "beta_stress": beta_stress, "beta_bloc1": beta_bloc1,
            "corr": corr, "return_stress": return_stress, "skewness": skewness, "maxdd_abs": maxdd,
            "alpha_annual": alpha, "sharpe": sharpe, "sortino": sortino, "vol": vol,
            "ret_ann": ret_ann, "kurtosis": kurtosis,
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
        z_corr = safe_zscore(g["corr"])
        z_corr_abs = safe_zscore(g["corr"].abs())
        z_stress = safe_zscore(g["return_stress"])
        z_skew = safe_zscore(g["skewness"])
        z_mdd = safe_zscore(g["maxdd_abs"])
        z_alpha = safe_zscore(g["alpha_annual"])
        z_expense = safe_zscore(-pd.to_numeric(g.get("expense_pct"), errors="coerce"))
        z_sh = safe_zscore(g["sharpe"])
        z_so = safe_zscore(g["sortino"])
        z_vol = safe_zscore(g["vol"])
        z_ret = safe_zscore(g["ret_ann"])
        z_kurt = safe_zscore(g["kurtosis"])

        if su == "STRAT1":
            g["score_custom"] = -0.40 * z_beta_b1 - 0.20 * z_corr_abs + 0.20 * z_stress + 0.15 * z_sh - 0.15 * z_mdd
        elif su == "STRAT2":
            g["score_custom"] = 0.60 * z_alpha + 0.30 * z_expense + 0.10 * z_sh - 0.05 * z_corr_abs
        elif su == "STRAT3":
            g["score_custom"] = 0.55 * z_alpha + 0.35 * z_expense + 0.10 * z_ret - 0.05 * z_kurt
        else:
            g["score_custom"] = np.nan

        g["score_level2"] = g["score_custom"].combine_first(pd.to_numeric(g.get("score_level2"), errors="coerce"))
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=[c for c in out.columns if c.endswith("_new")])
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

    l2 = cfg["level2"]
    df_l2, res_l2 = apply_level2_alpha_expense(
        df_l1, prices_win, core_win,
        calib_start=calib_start.strftime("%Y-%m-%d"),
        calib_end=calib_end.strftime("%Y-%m-%d"),
        expense_col=l2["expense_col"],
        alpha_weight=l2["alpha_weight"],
        expense_weight=l2["expense_weight"],
        min_obs_alpha=l2["min_obs_alpha"],
        keep_top_per_strat=l2["keep_top_per_strat"],
        verbose=False,
    )

    if res_l2.empty:
        return pd.DataFrame(), pd.DataFrame(), calib_start, calib_end

    scored = _score_pool_by_bloc_formulas(res_l2.copy(), prices_win, core_win, cfg)
    keep_n = int(l2.get("keep_top_per_strat", 7))
    df_l2_custom = (
        scored.sort_values(["Strat", "score_level2"], ascending=[True, False])
        .groupby("Strat", group_keys=False)
        .head(keep_n)
        .copy()
    )

    for df_ in (scored, df_l2_custom):
        df_["review_date"] = pd.Timestamp(review_date)
        df_["calib_start"] = calib_start
        df_["calib_end"] = calib_end

    return df_l2_custom, scored, calib_start, calib_end


def _quarterly_review_dates(prices_idx, cfg):
    start = pd.Timestamp(cfg["rolling"]["oos_start"]).normalize()
    end = pd.Timestamp(cfg["rolling"]["oos_end"]).normalize()
    raw_q = pd.date_range(start=start, end=end, freq=cfg["rolling"]["quarterly_freq"])

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

    pool_eff = pool_scores.copy()
    if bool(cfg["rolling"].get("quarterly_rescore_pool_1y", False)):
        pool_eff = _rescore_pool_quarterly(pool_eff, prices_all_aligned, core_all_aligned, qdate, cfg)

    rows = []
    for strat, g in _safe_strat_group(pool_eff):
        picks = _pick_ranked_for_strat(g, prices_win, p)
        for r in picks:
            rows.append({
                "quarter_date": pd.Timestamp(qdate), "Strat": strat,
                "ticker": r["ticker"], "rank_in_strat": r["rank_in_strat"],
                "score_level2": r["score_level2"], "selected_reason": r["selected_reason"],
                "pair_corr": r["pair_corr"], "score_gap_vs_top1": r["score_gap_vs_top1"],
            })
    return pd.DataFrame(rows)


def _apply_quarterly_switching(all_quarter_candidates, cfg):
    if all_quarter_candidates.empty:
        return all_quarter_candidates

    out_rows = []
    switch_buffer = float(cfg["final"]["switch_score_buffer"])

    for strat, g in all_quarter_candidates.groupby("Strat"):
        g = g.sort_values(["quarter_date", "rank_in_strat"]).copy()
        held_by_rank = {}

        for qd, gq in g.groupby("quarter_date"):
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

                    if held_ticker == cand_ticker:
                        chosen = cand
                        chosen["switch_flag"] = 0
                        chosen["switch_reason"] = "same_ticker"
                    elif cand_score >= held_score + switch_buffer:
                        chosen = cand
                        chosen["switch_flag"] = 1
                        chosen["switch_reason"] = "better_score"
                    else:
                        chosen = held.copy()
                        chosen["quarter_date"] = qd
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
    print(f"Revue annuelle: {len(annual_dates)} dates | Revue trimestrielle: {len(quarter_dates)} dates")

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

    # Quarterly selection with switching
    print("\n[Etape 3 glissante] Selection trimestrielle avec switching")
    quarter_candidate_rows = []

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
        q_candidates = _pick_from_pool_for_quarter(pool_q, df_satellite_prices_aligned, core_returns_aligned, qd, params)
        if not q_candidates.empty:
            q_candidates["active_review_date"] = pd.Timestamp(active_review)
            quarter_candidate_rows.append(q_candidates)

    quarter_candidates = pd.concat(quarter_candidate_rows, ignore_index=True) if quarter_candidate_rows else pd.DataFrame()
    quarter_selection_df = _apply_quarterly_switching(quarter_candidates, params) if not quarter_candidates.empty else pd.DataFrame()

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
    print(f"  - Trimestres evalues: {len(quarter_dates)}")
    print(f"  - Lignes pool annuel (scores): {len(annual_pool_scores)}")
    print(f"  - Lignes selection trimestrielle: {len(quarter_selection_df)}")
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


def display_satellite_selection(results_level2, quarter_selection_df, satellite_final_selection):
    """Display L2 shortlist, full quarterly history, and final snapshot."""
    from IPython.display import display as _display
    import pandas as pd

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

    print(
        f"\n🗂️ Historique complet des sélections trimestrielles: "
        f"{sel_hist['quarter_date'].dt.to_period('Q').nunique()} trimestres | {len(sel_hist)} lignes"
    )
    _display(sel_hist[hist_cols] if hist_cols else sel_hist)

    if 'switch_flag' in sel_hist.columns:
        agg_dict = {'n_fonds': ('ticker', 'count'), 'n_switches': ('switch_flag', 'sum')}
        if 'score_level2' in sel_hist.columns:
            agg_dict['score_moyen'] = ('score_level2', 'mean')
        switch_summary = (
            sel_hist.assign(quarter=sel_hist['quarter_date'].dt.to_period('Q').astype(str))
            .groupby('quarter', as_index=False).agg(**agg_dict)
        )
        print('\nRésumé trimestriel des sélections:')
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
