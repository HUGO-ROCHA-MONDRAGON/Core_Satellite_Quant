"""Core vs Satellite comparison: returns construction, full analytics, and dashboard."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from src.utils import ann_return, ann_vol, max_dd, sharpe0, calmar, normalize_index


def _to_naive_ts(value):
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts


def _safe_ratio(num, den):
    return float(num) / float(den) if den else np.nan


def _build_coverage_tables(core_ret_full_valid, common_dates, comparable_dates, missing_sat_active):
    core_days_full = int(len(core_ret_full_valid))
    common_days = int(len(common_dates))
    comparable_days = int(len(comparable_dates))
    no_satellite_live_days = max(0, core_days_full - common_days)
    missing_active_days = int(missing_sat_active.sum())

    coverage_summary = pd.DataFrame(
        {
            "core_days_full": [core_days_full],
            "common_days": [common_days],
            "comparable_days": [comparable_days],
            "coverage_vs_core": [_safe_ratio(comparable_days, core_days_full)],
            "coverage_vs_common": [_safe_ratio(comparable_days, common_days)],
            "core_no_satellite_live_days": [no_satellite_live_days],
            "sat_missing_active_days": [missing_active_days],
        },
        index=["Global"],
    )

    core_year = core_ret_full_valid.groupby(core_ret_full_valid.index.year).size()
    common_year = pd.Series(1, index=common_dates).groupby(pd.Index(common_dates).year).sum()
    comp_year = pd.Series(1, index=comparable_dates).groupby(pd.Index(comparable_dates).year).sum()
    missing_year = missing_sat_active.astype(int).groupby(missing_sat_active.index.year).sum()

    all_years = sorted(set(core_year.index) | set(common_year.index) | set(comp_year.index))
    if len(all_years) == 0:
        coverage_by_year = pd.DataFrame(
            columns=[
                "core_days_full",
                "common_days",
                "comparable_days",
                "coverage_vs_core",
                "coverage_vs_common",
                "core_no_satellite_live_days",
                "sat_missing_active_days",
            ]
        )
        return coverage_summary, coverage_by_year

    coverage_by_year = pd.DataFrame(index=all_years)
    coverage_by_year["core_days_full"] = core_year.reindex(all_years, fill_value=0).astype(int)
    coverage_by_year["common_days"] = common_year.reindex(all_years, fill_value=0).astype(int)
    coverage_by_year["comparable_days"] = comp_year.reindex(all_years, fill_value=0).astype(int)
    coverage_by_year["core_no_satellite_live_days"] = (
        coverage_by_year["core_days_full"] - coverage_by_year["common_days"]
    ).clip(lower=0).astype(int)
    coverage_by_year["sat_missing_active_days"] = (
        missing_year.reindex(all_years, fill_value=0).astype(int)
    )
    coverage_by_year["coverage_vs_core"] = coverage_by_year.apply(
        lambda x: _safe_ratio(x["comparable_days"], x["core_days_full"]), axis=1
    )
    coverage_by_year["coverage_vs_common"] = coverage_by_year.apply(
        lambda x: _safe_ratio(x["comparable_days"], x["common_days"]), axis=1
    )

    return coverage_summary, coverage_by_year


# ── Build ret_cmp (cell 28 data prep) ───────────────────────────────────────

def build_core_vs_satellite_returns(
    portfolio_df,
    weights_ticker_daily,
    sat_prices,
    analysis_start,
    analysis_end,
    core_returns=None,
    core_3_log=None,
    strict_core=True,
    allow_fallback=True,
    coverage_warning_threshold=0.80,
):
    """
    Build aligned daily returns DataFrame with 'core' and 'satellite' columns.

    Core source priority:
      1) portfolio_df['portfolio_value']
      2) core_returns (if passed)
      3) core_3_log fallback (equal-weight mean)

    Returns (ret_cmp, core_source_name).
    """
    analysis_start = _to_naive_ts(analysis_start)
    analysis_end = _to_naive_ts(analysis_end)

    core_ret_all = None
    core_source = None

    # Source 1: portfolio_df
    if (
        portfolio_df is not None
        and isinstance(portfolio_df, pd.DataFrame)
        and "portfolio_value" in portfolio_df.columns
    ):
        core_nav = normalize_index(portfolio_df["portfolio_value"])
        core_ret_all = (
            core_nav.pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .rename("core")
        )
        core_source = "portfolio_df.portfolio_value"

    # Source 2: core_returns
    if (
        core_ret_all is None
        and core_returns is not None
        and isinstance(core_returns, pd.Series)
        and not core_returns.empty
    ):
        core_ret_all = normalize_index(core_returns).replace([np.inf, -np.inf], np.nan).rename("core")
        core_source = "core_returns"

    if core_ret_all is None and strict_core:
        raise RuntimeError("Source Core non trouvée.")

    # Source 3: fallback
    if core_ret_all is None and allow_fallback and core_3_log is not None:
        core_log = core_3_log.copy().sort_index().mean(axis=1)
        core_log = normalize_index(core_log)
        core_ret_all = np.expm1(core_log).rename("core")
        core_source = "fallback core_3_log.mean"

    if core_ret_all is None:
        raise RuntimeError("Aucune source Core exploitable.")

    core_ret_full = core_ret_all.loc[
        (core_ret_all.index >= analysis_start) & (core_ret_all.index <= analysis_end)
    ].rename("core")
    core_ret_full_valid = core_ret_full.dropna()
    if core_ret_full_valid.empty:
        raise RuntimeError("Aucun rendement Core valide sur la fenêtre demandée.")

    # Satellite returns
    w_ticker = normalize_index(weights_ticker_daily)
    prices_all = normalize_index(sat_prices)
    sat_rets_all = prices_all.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    common_dates = w_ticker.index.intersection(sat_rets_all.index).intersection(core_ret_all.index)
    common_dates = common_dates[(common_dates >= analysis_start) & (common_dates <= analysis_end)]
    common_dates = common_dates.sort_values()
    common_tickers = [c for c in w_ticker.columns if c in sat_rets_all.columns]

    if len(common_dates) == 0 or len(common_tickers) == 0:
        raise ValueError("No common dates/tickers.")

    w = w_ticker.loc[common_dates, common_tickers].fillna(0.0)
    r_sat = sat_rets_all.loc[common_dates, common_tickers]
    w_exec = w.shift(1).fillna(0.0)

    # Avoid silently imputing missing fund returns with 0 when there is active exposure.
    missing_sat_weight = w_exec.where(r_sat.isna(), 0.0).sum(axis=1)
    missing_sat_active = missing_sat_weight.abs() > 1e-12

    sat_ret_legacy = (w_exec * r_sat.fillna(0.0)).sum(axis=1).rename("satellite_legacy_fill0")
    sat_ret = sat_ret_legacy.mask(missing_sat_active).rename("satellite")
    core_ret_common = core_ret_all.loc[common_dates].rename("core")

    # Keep full Core history for reporting, and build an explicit comparable subset for C/S stats.
    full_index = core_ret_full_valid.index.sort_values()
    panel_full = pd.DataFrame(index=full_index)
    panel_full["core_standalone"] = core_ret_full_valid.reindex(full_index)
    panel_full["core_aligned"] = np.nan
    panel_full["satellite"] = np.nan
    panel_full["satellite_legacy_fill0"] = np.nan
    panel_full["missing_sat_active"] = False

    idx_common_in_full = full_index.intersection(common_dates)
    panel_full.loc[idx_common_in_full, "core_aligned"] = core_ret_common.reindex(idx_common_in_full)
    panel_full.loc[idx_common_in_full, "satellite"] = sat_ret.reindex(idx_common_in_full)
    panel_full.loc[idx_common_in_full, "satellite_legacy_fill0"] = sat_ret_legacy.reindex(idx_common_in_full)
    panel_full.loc[idx_common_in_full, "missing_sat_active"] = (
        missing_sat_active.reindex(idx_common_in_full).fillna(False).astype(bool)
    )
    panel_full["comparable_flag"] = panel_full["core_aligned"].notna() & panel_full["satellite"].notna()
    panel_full["reason_non_comparable"] = ""
    panel_full.loc[
        ~panel_full.index.isin(common_dates),
        "reason_non_comparable",
    ] = "satellite_not_live_or_no_common_data"
    panel_full.loc[
        panel_full.index.isin(common_dates) & panel_full["satellite"].isna(),
        "reason_non_comparable",
    ] = "satellite_missing_with_active_exposure"

    ret_cmp = (
        panel_full.loc[panel_full["comparable_flag"], ["core_aligned", "satellite"]]
        .rename(columns={"core_aligned": "core"})
        .dropna(how="any")
    )
    if ret_cmp.empty:
        raise RuntimeError(
            "Aucun jour comparable Core/Satellite après filtrage strict des données manquantes."
        )

    coverage_summary, coverage_by_year = _build_coverage_tables(
        core_ret_full_valid=core_ret_full_valid,
        common_dates=common_dates,
        comparable_dates=ret_cmp.index,
        missing_sat_active=missing_sat_active,
    )
    cov_global = coverage_summary.loc["Global"]

    ret_cmp.attrs["core_source"] = core_source
    ret_cmp.attrs["core_ret_full"] = core_ret_full_valid.copy()
    ret_cmp.attrs["ret_panel_full"] = panel_full.copy()
    ret_cmp.attrs["coverage_summary"] = coverage_summary.copy()
    ret_cmp.attrs["coverage_by_year"] = coverage_by_year.copy()
    ret_cmp.attrs["coverage_warning_threshold"] = float(coverage_warning_threshold)

    print(f"Source Core : {core_source}")
    print(
        "Core standalone : "
        f"{core_ret_full_valid.index.min().date()} → {core_ret_full_valid.index.max().date()} "
        f"({int(cov_global['core_days_full'])} obs)"
    )
    print(
        "Comparatif aligné : "
        f"{ret_cmp.index.min().date()} → {ret_cmp.index.max().date()} "
        f"({int(cov_global['comparable_days'])} obs)"
    )
    print(
        "Couverture (comparables / Core full) : "
        f"{cov_global['coverage_vs_core']:.1%} "
        f"({int(cov_global['comparable_days'])}/{int(cov_global['core_days_full'])})"
    )
    n_core_no_sat_live = int(cov_global["core_no_satellite_live_days"])
    n_missing_days = int(cov_global["sat_missing_active_days"])
    if n_core_no_sat_live > 0 or n_missing_days > 0:
        print(
            "Jours non comparables (décomposition) : "
            f"Core sans Satellite live/no common={n_core_no_sat_live}, "
            f"Satellite active mais retour manquant={n_missing_days}"
        )
    if (
        not np.isnan(cov_global["coverage_vs_core"])
        and cov_global["coverage_vs_core"] < coverage_warning_threshold
    ):
        print(
            "WARNING: couverture faible "
            f"(< {coverage_warning_threshold:.0%}) ; interpréter les comparaisons avec prudence."
        )

    return ret_cmp, core_source


# ── Full analysis + dashboard (cell 28) ─────────────────────────────────────

def analyze_core_vs_satellite(ret_cmp, analysis_start, analysis_end, outdir="outputs"):
    """
    Complete Core vs Satellite comparison with two explicit views:
      1) Core standalone (full history)
      2) Core vs Satellite aligned (common sample)

    Returns dict with metrics DataFrames and key statistics.
    """
    analysis_start = _to_naive_ts(analysis_start)
    analysis_end = _to_naive_ts(analysis_end)

    ret_cmp_attrs = dict(getattr(ret_cmp, "attrs", {}))
    ret_panel_full = ret_cmp_attrs.get("ret_panel_full")
    ret_cmp = normalize_index(ret_cmp[["core", "satellite"]])
    ret_cmp = (
        ret_cmp.loc[(ret_cmp.index >= analysis_start) & (ret_cmp.index <= analysis_end)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )
    if ret_cmp.empty:
        raise RuntimeError("Fenêtre Core/Satellite comparables vide après nettoyage.")

    core_source = ret_cmp_attrs.get("core_source", "unknown")
    coverage_warning_threshold = float(ret_cmp_attrs.get("coverage_warning_threshold", 0.80))
    portfolio_input_mode = str(ret_cmp_attrs.get("portfolio_input_mode", "strict_comparable")).strip().lower()
    sat_missing_filled_attr = int(ret_cmp_attrs.get("sat_missing_filled", 0))
    legacy_partial_fill_days_attr = ret_cmp_attrs.get("legacy_partial_fill_days", np.nan)
    try:
        legacy_partial_fill_days_attr = float(legacy_partial_fill_days_attr)
    except (TypeError, ValueError):
        legacy_partial_fill_days_attr = np.nan
    legacy_partial_fill_days_disp = (
        int(legacy_partial_fill_days_attr) if np.isfinite(legacy_partial_fill_days_attr) else 0
    )
    is_investable_mode = portfolio_input_mode == "investable_stale_satellite"
    is_legacy_global_mode = portfolio_input_mode == "global_fill0_legacy"

    if is_investable_mode:
        core_label = "Core (investable)"
        sat_label = "Satellite (investable)"
        annual_suffix = "(investissable)"
        fig_title = "Core vs Satellite (vue investissable — satellite NA imputés à 0)"
    elif is_legacy_global_mode:
        core_label = "Core (global fill0)"
        sat_label = "Satellite (global fill0)"
        annual_suffix = "(global fill0)"
        fig_title = "Core vs Satellite (vue globale legacy — NA ticker Satellite imputés à 0)"
    else:
        core_label = "Core (aligned)"
        sat_label = "Satellite (aligned)"
        annual_suffix = "(aligné)"
        fig_title = "Core vs Satellite (échantillon strict aligné) — Performance, Risk, Alpha & Coverage"

    if isinstance(ret_panel_full, pd.DataFrame) and not ret_panel_full.empty:
        ret_panel_full = normalize_index(ret_panel_full)
        ret_panel_full = ret_panel_full.loc[
            (ret_panel_full.index >= analysis_start) & (ret_panel_full.index <= analysis_end)
        ].copy()
    else:
        ret_panel_full = pd.DataFrame(index=ret_cmp.index)
        ret_panel_full["core_standalone"] = ret_cmp["core"]
        ret_panel_full["core_aligned"] = ret_cmp["core"]
        ret_panel_full["satellite"] = ret_cmp["satellite"]
        ret_panel_full["comparable_flag"] = True
        ret_panel_full["reason_non_comparable"] = ""

    core_ret_full = ret_cmp_attrs.get("core_ret_full")
    if not isinstance(core_ret_full, pd.Series) or core_ret_full.empty:
        core_ret_full = ret_panel_full["core_standalone"].dropna().rename("core")
    else:
        core_ret_full = normalize_index(core_ret_full).rename("core")
        core_ret_full = (
            core_ret_full.loc[(core_ret_full.index >= analysis_start) & (core_ret_full.index <= analysis_end)]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if core_ret_full.empty:
            core_ret_full = ret_panel_full["core_standalone"].dropna().rename("core")

    coverage_summary = ret_cmp_attrs.get("coverage_summary")
    coverage_by_year = ret_cmp_attrs.get("coverage_by_year")
    if not isinstance(coverage_summary, pd.DataFrame) or coverage_summary.empty:
        dummy_missing = pd.Series(False, index=ret_cmp.index)
        coverage_summary, coverage_by_year = _build_coverage_tables(
            core_ret_full_valid=core_ret_full,
            common_dates=ret_cmp.index,
            comparable_dates=ret_cmp.index,
            missing_sat_active=dummy_missing,
        )
    elif not isinstance(coverage_by_year, pd.DataFrame):
        coverage_by_year = pd.DataFrame()

    if "core_no_satellite_live_days" not in coverage_summary.columns:
        coverage_summary["core_no_satellite_live_days"] = (
            coverage_summary["core_days_full"] - coverage_summary["common_days"]
        ).clip(lower=0)
    if "sat_missing_active_days" not in coverage_summary.columns:
        coverage_summary["sat_missing_active_days"] = (
            coverage_summary["common_days"] - coverage_summary["comparable_days"]
        ).clip(lower=0)
    if not coverage_by_year.empty:
        if "core_no_satellite_live_days" not in coverage_by_year.columns:
            coverage_by_year["core_no_satellite_live_days"] = (
                coverage_by_year["core_days_full"] - coverage_by_year["common_days"]
            ).clip(lower=0)
        if "sat_missing_active_days" not in coverage_by_year.columns:
            coverage_by_year["sat_missing_active_days"] = (
                coverage_by_year["common_days"] - coverage_by_year["comparable_days"]
            ).clip(lower=0)

    OUT = Path(outdir)
    OUT.mkdir(exist_ok=True)
    ret_cmp.to_csv(OUT / "core_vs_satellite_daily_returns.csv", index=True)
    core_ret_full.to_frame("core").to_csv(OUT / "core_standalone_daily_returns.csv", index=True)
    ret_panel_full.to_csv(OUT / "core_vs_satellite_panel_full.csv", index=True)
    coverage_summary.to_csv(OUT / "core_vs_satellite_coverage_summary.csv", index=True)
    coverage_by_year.to_csv(OUT / "core_vs_satellite_coverage_by_year.csv", index=True)

    # View 1: Core standalone metrics (full available history)
    core_standalone_metrics = pd.DataFrame(
        {
            "Cumul total": [(1.0 + core_ret_full).prod() - 1.0],
            "Ann Return": [ann_return(core_ret_full)],
            "Ann Vol": [ann_vol(core_ret_full)],
            "Sharpe (rf=2%)": [sharpe0(core_ret_full)],
            "Max DD": [max_dd(core_ret_full)],
            "Calmar": [calmar(core_ret_full)],
        },
        index=["Core standalone"],
    )
    core_standalone_annual = core_ret_full.groupby(core_ret_full.index.year).apply(
        lambda x: (1.0 + x).prod() - 1.0
    ).rename("Core standalone")
    core_standalone_annual.to_frame().to_csv(OUT / "core_standalone_annual_returns.csv", index=True)

    # View 2: Aligned comparative metrics
    beta_static = (
        ret_cmp["satellite"].cov(ret_cmp["core"]) / ret_cmp["core"].var()
        if ret_cmp["core"].var() != 0
        else np.nan
    )
    corr_static = ret_cmp["satellite"].corr(ret_cmp["core"])
    tracking_error = (ret_cmp["satellite"] - ret_cmp["core"]).std() * np.sqrt(252.0)

    alpha_daily = ret_cmp["satellite"] - (beta_static * ret_cmp["core"])
    alpha_daily.name = "alpha_daily"

    # Rolling (2 ans, min_periods=63)
    win = 504
    min_p = 63
    beta_roll = (
        ret_cmp["satellite"].rolling(win, min_periods=min_p).cov(ret_cmp["core"])
        / ret_cmp["core"].rolling(win, min_periods=min_p).var().replace(0, np.nan)
    )
    corr_roll = ret_cmp["satellite"].rolling(win, min_periods=min_p).corr(ret_cmp["core"])
    vol_roll = ret_cmp.rolling(win, min_periods=min_p).std() * np.sqrt(252.0)
    alpha_rolling = (
        ret_cmp["satellite"].rolling(win, min_periods=min_p).mean()
        - (beta_roll * ret_cmp["core"].rolling(win, min_periods=min_p).mean())
    )

    alpha_annual = ret_cmp.groupby(ret_cmp.index.year).apply(
        lambda x: ann_return(x["satellite"]) - beta_static * ann_return(x["core"])
    )
    alpha_ann_total = ann_return(alpha_daily)

    sat_total_ret = ann_return(ret_cmp["satellite"])
    core_total_ret = ann_return(ret_cmp["core"])
    return_diff = sat_total_ret - core_total_ret

    metrics_cmp = pd.DataFrame(
        {
            "Ann Return": [core_total_ret, sat_total_ret],
            "Ann Vol": [ann_vol(ret_cmp["core"]), ann_vol(ret_cmp["satellite"])],
            "Sharpe (rf=2%)": [sharpe0(ret_cmp["core"]), sharpe0(ret_cmp["satellite"])],
            "Max DD": [max_dd(ret_cmp["core"]), max_dd(ret_cmp["satellite"])],
            "Calmar": [calmar(ret_cmp["core"]), calmar(ret_cmp["satellite"])],
            "Return Diff.": [np.nan, return_diff],
            "Alpha (CAPM)": [np.nan, alpha_ann_total],
        },
        index=[core_label, sat_label],
    )

    cov_global = coverage_summary.loc["Global"]
    summary_cross = pd.DataFrame(
        {
            "Metric": [
                "Static beta",
                "Static correlation",
                "Tracking error ann.",
                "Simple Outperformance",
                "Alpha annualized (CAPM)",
                "Coverage vs Core full",
                "Comparable days",
                "Core full days",
                "Core-only days (satellite not live/no common)",
                "Satellite missing with active exposure days",
            ],
            "Value": [
                beta_static,
                corr_static,
                tracking_error,
                return_diff,
                alpha_ann_total,
                cov_global["coverage_vs_core"],
                cov_global["comparable_days"],
                cov_global["core_days_full"],
                cov_global["core_no_satellite_live_days"],
                cov_global["sat_missing_active_days"],
            ],
        }
    )
    if is_investable_mode:
        summary_cross = pd.concat(
            [
                summary_cross,
                pd.DataFrame(
                    {
                        "Metric": ["Satellite NA imputés à 0 (jours)"],
                        "Value": [sat_missing_filled_attr],
                    }
                ),
            ],
            ignore_index=True,
        )
    elif is_legacy_global_mode and np.isfinite(legacy_partial_fill_days_attr):
        summary_cross = pd.concat(
            [
                summary_cross,
                pd.DataFrame(
                    {
                        "Metric": ["Jours avec NA Satellite actif (imputation partielle legacy)"],
                        "Value": [legacy_partial_fill_days_attr],
                    }
                ),
            ],
            ignore_index=True,
        )

    annual_cmp = ret_cmp.groupby(ret_cmp.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    nav_cmp = (1.0 + ret_cmp).cumprod()
    dd_cmp = nav_cmp / nav_cmp.cummax() - 1.0
    analysis_start_actual = ret_cmp.index.min()

    # Display
    print("\nVue 1 — Core standalone (historique complet)")
    print(
        f"Source: {core_source} | Fenêtre: {core_ret_full.index.min().date()} → "
        f"{core_ret_full.index.max().date()} ({len(core_ret_full)} obs)"
    )
    display(
        core_standalone_metrics.style.format(
            {
                "Cumul total": "{:+.2%}",
                "Ann Return": "{:+.2%}",
                "Ann Vol": "{:.2%}",
                "Sharpe (rf=2%)": "{:.2f}",
                "Max DD": "{:.2%}",
                "Calmar": "{:.2f}",
            }
        )
    )
    print("\nRendements annuels (Core standalone):")
    display(core_standalone_annual.to_frame().style.format("{:+.2%}"))

    print(f"\nVue 2 — Comparatif Core vs Satellite {annual_suffix}")
    display(
        metrics_cmp.style.format(
            {
                "Ann Return": "{:+.2%}",
                "Ann Vol": "{:.2%}",
                "Sharpe (rf=2%)": "{:.2f}",
                "Max DD": "{:.2%}",
                "Calmar": "{:.2f}",
                "Return Diff.": "{:+.2%}",
                "Alpha (CAPM)": "{:+.2%}",
            }
        )
    )
    print("\nCross metrics:")
    display(summary_cross.style.format({"Value": "{:.4f}"}))
    print(f"\nRendements annuels {annual_suffix}:")
    display(annual_cmp.style.format("{:+.2%}"))

    print("\nCouverture de comparaison:")
    display(
        coverage_summary.style.format(
            {
                "core_days_full": "{:.0f}",
                "common_days": "{:.0f}",
                "comparable_days": "{:.0f}",
                "coverage_vs_core": "{:.1%}",
                "coverage_vs_common": "{:.1%}",
                "core_no_satellite_live_days": "{:.0f}",
                "sat_missing_active_days": "{:.0f}",
            }
        )
    )
    if not coverage_by_year.empty:
        print("\nCouverture annuelle:")
        display(
            coverage_by_year.style.format(
                {
                    "core_days_full": "{:.0f}",
                    "common_days": "{:.0f}",
                    "comparable_days": "{:.0f}",
                    "coverage_vs_core": "{:.1%}",
                    "coverage_vs_common": "{:.1%}",
                    "core_no_satellite_live_days": "{:.0f}",
                    "sat_missing_active_days": "{:.0f}",
                }
            )
        )

    low_cov_years = (
        coverage_by_year[coverage_by_year["coverage_vs_core"] < coverage_warning_threshold].index.tolist()
        if "coverage_vs_core" in coverage_by_year.columns
        else []
    )
    if (
        not np.isnan(cov_global["coverage_vs_core"])
        and cov_global["coverage_vs_core"] < coverage_warning_threshold
    ):
        print(
            "\nWARNING: couverture globale faible "
            f"({cov_global['coverage_vs_core']:.1%} < {coverage_warning_threshold:.0%})."
        )
    if len(low_cov_years) > 0:
        print(
            "WARNING: années avec couverture faible "
            f"(< {coverage_warning_threshold:.0%}) : {', '.join(map(str, low_cov_years))}"
        )

    # Dashboard (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)

    axes[0, 0].plot(nav_cmp.index, nav_cmp["core"], lw=2.0, label=core_label, color="#1f77b4")
    axes[0, 0].plot(
        nav_cmp.index, nav_cmp["satellite"], lw=2.0, label=sat_label, color="#ff7f0e"
    )
    axes[0, 0].set_title(f"NAV cumulée {annual_suffix}")
    axes[0, 0].set_ylabel("Base 1")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(dd_cmp.index, dd_cmp["core"] * 100, lw=1.7, label="Core", color="#1f77b4")
    axes[0, 1].plot(dd_cmp.index, dd_cmp["satellite"] * 100, lw=1.7, label="Satellite", color="#ff7f0e")
    axes[0, 1].set_title(f"Drawdown {annual_suffix}")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(vol_roll.index, vol_roll["core"] * 100, lw=1.7, label="Core", color="#1f77b4")
    axes[0, 2].plot(vol_roll.index, vol_roll["satellite"] * 100, lw=1.7, label="Satellite", color="#ff7f0e")
    axes[0, 2].set_title(f"Vol rolling {win}j annualisée")
    axes[0, 2].set_ylabel("%")
    axes[0, 2].grid(True, alpha=0.25)
    axes[0, 2].legend(frameon=False)

    axes[0, 3].axis("off")
    status_label = "OK" if cov_global["coverage_vs_core"] >= coverage_warning_threshold else "ALERTE"
    if is_investable_mode:
        coverage_text = (
            f"Mode : investissable\n"
            f"Source Core : {core_source}\n"
            f"Période : {ret_cmp.index.min().date()} → {ret_cmp.index.max().date()}\n"
            f"Obs utilisées : {len(ret_cmp):.0f}\n"
            f"Satellite NA imputés à 0 : {sat_missing_filled_attr:.0f} jours"
        )
    elif is_legacy_global_mode:
        coverage_text = (
            f"Mode : global fill0 (legacy)\n"
            f"Source Core : {core_source}\n"
            f"Période : {ret_cmp.index.min().date()} → {ret_cmp.index.max().date()}\n"
            f"Obs utilisées : {len(ret_cmp):.0f}\n"
            f"NA Satellite actifs (imputés partiellement ticker=0) : {legacy_partial_fill_days_disp:.0f} jours"
        )
    else:
        coverage_text = (
            f"Mode : strict comparable\n"
            f"Source Core : {core_source}\n"
            f"Core standalone : {len(core_ret_full):.0f} jours\n"
            f"Jours comparables : {cov_global['comparable_days']:.0f}/{cov_global['core_days_full']:.0f}\n"
            f"Couverture : {cov_global['coverage_vs_core']:.1%}\n"
            f"Core sans Satellite live/no common : {cov_global['core_no_satellite_live_days']:.0f}\n"
            f"Jours exclus (sat NA + expo) : {cov_global['sat_missing_active_days']:.0f}\n"
            f"Seuil qualité : {coverage_warning_threshold:.0%} ({status_label})"
        )
    axes[0, 3].text(0.02, 0.98, coverage_text, va="top", ha="left", fontsize=10)
    axes[0, 3].set_title("Convention de calcul")

    axes[1, 0].plot(beta_roll.index, beta_roll, lw=1.8, color="#2ca02c", label="Beta rolling")
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    if beta_roll.notna().any():
        axes[1, 0].axhline(
            beta_roll.mean(), color="#d62728", lw=1.2, ls="--", label=f"Mean {beta_roll.mean():.2f}"
        )
    axes[1, 0].set_title(f"Beta rolling {win}j")
    axes[1, 0].set_ylabel("Beta")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(corr_roll.index, corr_roll, lw=1.8, color="#9467bd")
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].set_title(f"Corrélation rolling {win}j")
    axes[1, 1].set_ylabel("Corr")
    axes[1, 1].grid(True, alpha=0.25)

    axes[1, 2].plot(alpha_rolling.index, alpha_rolling * 100, lw=1.8, color="#d62728", label="Alpha rolling")
    axes[1, 2].axhline(0.0, color="black", lw=0.8)
    if alpha_rolling.notna().any():
        axes[1, 2].axhline(
            alpha_rolling.mean() * 100,
            color="#2ca02c",
            lw=1.2,
            ls="--",
            label=f"Mean {alpha_rolling.mean() * 100:.2f}%",
        )
    axes[1, 2].set_title(f"Alpha rolling {win}j")
    axes[1, 2].set_ylabel("Alpha (%)")
    axes[1, 2].grid(True, alpha=0.25)
    axes[1, 2].legend(frameon=False)

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
        ax.set_xlim(analysis_start_actual, analysis_end)

    if len(annual_cmp) > 0:
        x = np.arange(len(annual_cmp.index))
        bw = 0.25
        axes[1, 3].bar(
            x - bw,
            annual_cmp["core"] * 100,
            width=bw,
            label=core_label,
            color="#1f77b4",
            alpha=0.9,
        )
        axes[1, 3].bar(
            x,
            annual_cmp["satellite"] * 100,
            width=bw,
            label=sat_label,
            color="#ff7f0e",
            alpha=0.9,
        )
        axes[1, 3].bar(
            x + bw, alpha_annual * 100, width=bw, label="Alpha", color="#d62728", alpha=0.9
        )
        axes[1, 3].set_xticks(x)
        axes[1, 3].set_xticklabels(annual_cmp.index.astype(str), rotation=0)
    axes[1, 3].axhline(0.0, color="black", lw=0.8)
    axes[1, 3].set_title(f"Rendements + Alpha annuels {annual_suffix}")
    axes[1, 3].set_ylabel("%")
    axes[1, 3].grid(True, axis="y", alpha=0.25)
    if len(annual_cmp) > 0:
        axes[1, 3].legend(frameon=False, fontsize=8)

    fig.suptitle(fig_title, fontsize=15, y=1.02)
    plt.show()

    return {
        "core_standalone_metrics": core_standalone_metrics,
        "core_standalone_annual": core_standalone_annual,
        "metrics_cmp": metrics_cmp,
        "summary_cross": summary_cross,
        "coverage_summary": coverage_summary,
        "coverage_by_year": coverage_by_year,
        "annual_cmp": annual_cmp,
        "alpha_annual": alpha_annual,
        "beta_static": beta_static,
        "corr_static": corr_static,
        "tracking_error": tracking_error,
        "alpha_ann_total": alpha_ann_total,
        "return_diff": return_diff,
        "nav_cmp": nav_cmp,
        "ret_cmp": ret_cmp,
        "ret_panel_full": ret_panel_full,
        "core_ret_full": core_ret_full,
        "portfolio_input_mode": portfolio_input_mode,
        "sat_missing_filled": sat_missing_filled_attr,
    }


def display_core_vs_satellite_summary(cvs_results, coverage_warning_threshold=0.80):
    """Print a concise execution summary: standalone Core, aligned comparison, and coverage."""
    core_standalone_metrics = cvs_results.get("core_standalone_metrics")
    metrics_cmp = cvs_results.get("metrics_cmp")
    coverage_summary = cvs_results.get("coverage_summary")
    coverage_by_year = cvs_results.get("coverage_by_year")

    print("\n" + "=" * 78)
    print("SYNTHÈSE EXÉCUTIVE — CORE/SATELLITE")
    print("=" * 78)

    if isinstance(core_standalone_metrics, pd.DataFrame) and not core_standalone_metrics.empty:
        row = core_standalone_metrics.loc["Core standalone"]
        print(
            "Core standalone (historique complet) : "
            f"AnnRet={row['Ann Return']:+.2%} | Vol={row['Ann Vol']:.2%} | "
            f"Sharpe={row['Sharpe (rf=2%)']:.2f} | MaxDD={row['Max DD']:.2%}"
        )

    if isinstance(metrics_cmp, pd.DataFrame) and not metrics_cmp.empty:
        core_idx = next((idx for idx in metrics_cmp.index if str(idx).lower().startswith("core")), None)
        sat_idx = next((idx for idx in metrics_cmp.index if str(idx).lower().startswith("satellite")), None)
        if core_idx is not None and sat_idx is not None:
            core_al = metrics_cmp.loc[core_idx]
            sat_al = metrics_cmp.loc[sat_idx]
        else:
            core_al = metrics_cmp.iloc[0]
            sat_al = metrics_cmp.iloc[min(1, len(metrics_cmp) - 1)]
        print(
            "Comparatif Core/Satellite (échantillon courant) : "
            f"Core={core_al['Ann Return']:+.2%} | Satellite={sat_al['Ann Return']:+.2%} | "
            f"Diff={sat_al['Return Diff.']:+.2%}"
        )

    if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
        cov = coverage_summary.loc["Global"]
        cov_vs_core = float(cov["coverage_vs_core"])
        if "core_no_satellite_live_days" in cov.index:
            core_no_live = int(cov["core_no_satellite_live_days"])
        elif {"core_days_full", "common_days"}.issubset(set(cov.index)):
            core_no_live = int(max(0, cov["core_days_full"] - cov["common_days"]))
        else:
            core_no_live = 0
        if "sat_missing_active_days" in cov.index:
            sat_missing = int(cov["sat_missing_active_days"])
        elif {"common_days", "comparable_days"}.issubset(set(cov.index)):
            sat_missing = int(max(0, cov["common_days"] - cov["comparable_days"]))
        else:
            sat_missing = 0
        print(
            "Couverture globale : "
            f"{cov_vs_core:.1%} ({int(cov['comparable_days'])}/{int(cov['core_days_full'])} jours)"
        )
        print(
            "Décomposition des jours non comparables : "
            f"Core sans Satellite live/no common={core_no_live}, "
            f"Satellite active mais retour manquant={sat_missing}"
        )
        if cov_vs_core < coverage_warning_threshold:
            print(
                f"ALERTE: couverture < {coverage_warning_threshold:.0%} "
                "(comparaison moins robuste)."
            )

    if isinstance(coverage_by_year, pd.DataFrame) and not coverage_by_year.empty:
        weak_years = coverage_by_year[
            coverage_by_year["coverage_vs_core"] < float(coverage_warning_threshold)
        ]
        if len(weak_years) > 0:
            years = ", ".join(map(str, weak_years.index.tolist()))
            print(f"Années à couverture faible: {years}")

    print("=" * 78)


def build_portfolio_input_from_core_sat(
    cvs_results,
    mode="investable_stale_satellite",
    analysis_start=None,
    analysis_end=None,
    verbose=True,
):
    """
    Build portfolio-level input returns ('core', 'satellite') from Core/Satellite analysis outputs.

    Modes
    -----
    - "investable_stale_satellite":
        Keep all days where Core is available on common market dates and set missing
        satellite returns to 0.0 (stale NAV convention for portfolio accounting).
    - "strict_comparable":
        Keep only days where both Core and Satellite returns are available.
    - "global_fill0_legacy":
        Legacy global convention (version initiale): satellite return is computed with
        ticker-level NA filled to 0.0, then all Core-aligned dates are kept.
    """
    panel_full = cvs_results.get("ret_panel_full")
    if not isinstance(panel_full, pd.DataFrame) or panel_full.empty:
        raise RuntimeError("ret_panel_full absent/vide dans cvs_results.")

    panel_full = normalize_index(panel_full.copy())
    if analysis_start is not None:
        panel_full = panel_full.loc[panel_full.index >= _to_naive_ts(analysis_start)]
    if analysis_end is not None:
        panel_full = panel_full.loc[panel_full.index <= _to_naive_ts(analysis_end)]

    if "core_aligned" in panel_full.columns and panel_full["core_aligned"].notna().any():
        core_series = panel_full["core_aligned"].copy()
        core_source = "core_aligned_on_common_dates"
    elif "core_standalone" in panel_full.columns:
        core_series = panel_full["core_standalone"].copy()
        core_source = "core_standalone_fallback"
    else:
        raise RuntimeError("Colonnes Core absentes dans ret_panel_full.")

    if "satellite" not in panel_full.columns:
        raise RuntimeError("Colonne 'satellite' absente dans ret_panel_full.")

    mode_norm_raw = str(mode).strip().lower()
    if mode_norm_raw in {"global_fill0_legacy", "legacy_partial_fill0", "global_fill0"}:
        mode_norm = "global_fill0_legacy"
    elif mode_norm_raw in {"investable_stale_satellite", "strict_comparable"}:
        mode_norm = mode_norm_raw
    else:
        raise ValueError(
            f"mode inconnu: {mode}. Utiliser 'investable_stale_satellite', "
            "'strict_comparable' ou 'global_fill0_legacy'."
        )

    sat_series = panel_full["satellite"]
    if mode_norm == "global_fill0_legacy":
        if "satellite_legacy_fill0" in panel_full.columns:
            sat_series = panel_full["satellite_legacy_fill0"]
        else:
            # Backward compatibility if ret_panel_full was created before this column existed.
            sat_series = panel_full["satellite"].fillna(0.0)

    ret = pd.DataFrame({"core": core_series, "satellite": sat_series})
    ret = ret.dropna(subset=["core"]).sort_index()
    if ret.empty:
        raise RuntimeError("Série portefeuille vide après filtrage Core.")

    sat_missing_before = int(ret["satellite"].isna().sum())
    if mode_norm == "investable_stale_satellite":
        ret["satellite"] = ret["satellite"].fillna(0.0)
        sat_missing_filled = sat_missing_before
    elif mode_norm == "strict_comparable":
        ret = ret.dropna(how="any")
        sat_missing_filled = 0
    else:  # mode_norm == "global_fill0_legacy"
        ret["satellite"] = ret["satellite"].fillna(0.0)
        sat_missing_filled = sat_missing_before

    if ret.empty:
        raise RuntimeError("Série portefeuille vide après application du mode.")

    legacy_partial_fill_days = 0
    if mode_norm == "global_fill0_legacy":
        if "missing_sat_active" in panel_full.columns:
            legacy_partial_fill_days = int(
                panel_full.loc[ret.index, "missing_sat_active"].fillna(False).astype(bool).sum()
            )
        else:
            legacy_partial_fill_days = sat_missing_before

    ret.attrs["portfolio_input_mode"] = mode_norm
    ret.attrs["core_source_for_portfolio"] = core_source
    ret.attrs["sat_missing_before_policy"] = sat_missing_before
    ret.attrs["sat_missing_filled"] = sat_missing_filled
    ret.attrs["legacy_partial_fill_days"] = legacy_partial_fill_days

    if verbose:
        print(
            "[Portfolio input mode] "
            f"{mode_norm} | obs={len(ret):,} | core_source={core_source} "
            f"| sat_missing_before={sat_missing_before:,} | sat_missing_filled={sat_missing_filled:,}"
        )
        if mode_norm == "global_fill0_legacy":
            print(
                "Convention legacy: "
                f"jours avec NA satellite actif imputés partiellement (ticker=0) = {legacy_partial_fill_days:,}"
            )
        print(f"Période portefeuille: {ret.index.min().date()} → {ret.index.max().date()}")

    return ret


# ── Alpha interpretation (cell 29) ──────────────────────────────────────────

def display_alpha_interpretation(cvs_results):
    """Print clarification of alpha metrics for intuitive interpretation."""
    ret_cmp = cvs_results["ret_cmp"]
    beta_static = cvs_results["beta_static"]
    alpha_ann_total = cvs_results["alpha_ann_total"]
    tracking_error = cvs_results["tracking_error"]
    portfolio_input_mode = str(cvs_results.get("portfolio_input_mode", "strict_comparable")).strip().lower()

    if portfolio_input_mode == "investable_stale_satellite":
        core_label = "Core (investable)"
    elif portfolio_input_mode == "global_fill0_legacy":
        core_label = "Core (global fill0)"
    else:
        core_label = "Core (aligné)"

    sat_total = ann_return(ret_cmp["satellite"])
    core_total = ann_return(ret_cmp["core"])
    simple_outperformance = sat_total - core_total
    sat_sharpe = sharpe0(ret_cmp["satellite"])
    core_sharpe = sharpe0(ret_cmp["core"])
    sharpe_diff = sat_sharpe - core_sharpe
    info_ratio = alpha_ann_total / tracking_error if tracking_error > 0 else np.nan

    print("\n" + "=" * 70)
    print("INTERPRÉTATION DES MÉTRIQUES D'ALPHA")
    print("=" * 70)
    print(f"\n1) CAPM Alpha : Beta = {beta_static:.4f}")
    print("\n2) Simple Outperformance :")
    print(
        f"   Satellite : {sat_total:+.2%} | {core_label} : {core_total:+.2%} "
        f"| Diff : {simple_outperformance:+.2%}"
    )
    print(f"   → {'UNDERPERFORMANCE' if simple_outperformance < 0 else 'OUTPERFORMANCE'}")
    print(f"\n3) Sharpe delta : Sat={sat_sharpe:+.2f} | Core={core_sharpe:+.2f} | Diff={sharpe_diff:+.2f}")
    print(f"\n4) Information Ratio : {info_ratio:+.2f} (TE={tracking_error:.2%}, Alpha={alpha_ann_total:+.2%})")

    coverage_summary = cvs_results.get("coverage_summary")
    if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
        cov = coverage_summary.loc["Global"]
        if "core_no_satellite_live_days" in cov.index:
            core_no_live = int(cov["core_no_satellite_live_days"])
        elif {"core_days_full", "common_days"}.issubset(set(cov.index)):
            core_no_live = int(max(0, cov["core_days_full"] - cov["common_days"]))
        else:
            core_no_live = 0
        if "sat_missing_active_days" in cov.index:
            sat_missing = int(cov["sat_missing_active_days"])
        elif {"common_days", "comparable_days"}.issubset(set(cov.index)):
            sat_missing = int(max(0, cov["common_days"] - cov["comparable_days"]))
        else:
            sat_missing = 0
        print(
            f"\n5) Couverture : {cov['coverage_vs_core']:.1%} "
            f"({cov['comparable_days']:.0f}/{cov['core_days_full']:.0f} jours comparables)"
        )
        print(
            "   Décomposition non comparable : "
            f"Core sans Satellite live/no common={core_no_live}, "
            f"satellite active mais retour manquant={sat_missing}"
        )

    print("=" * 70)
