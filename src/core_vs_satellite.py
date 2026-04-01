"""Core vs Satellite comparison: returns construction, full analytics, and dashboard."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from src.utils import ann_return, ann_vol, max_dd, sharpe0, calmar, normalize_index


# ── Build ret_cmp (cell 28 data prep) ───────────────────────────────────────

def build_core_vs_satellite_returns(portfolio_df, weights_ticker_daily, sat_prices,
                                    analysis_start, analysis_end,
                                    core_returns=None, core_3_log=None,
                                    strict_core=True, allow_fallback=True):
    """
    Build aligned daily returns DataFrame with 'core' and 'satellite' columns.

    Core source priority:
      1) portfolio_df['portfolio_value']
      2) core_returns (if passed)
      3) core_3_log fallback (equal-weight mean)

    Returns (ret_cmp, core_source_name).
    """
    core_ret_all = None
    core_source = None

    # Source 1: portfolio_df
    if portfolio_df is not None and isinstance(portfolio_df, pd.DataFrame) and 'portfolio_value' in portfolio_df.columns:
        core_nav = portfolio_df['portfolio_value'].copy().sort_index()
        core_nav.index = pd.DatetimeIndex(core_nav.index).tz_localize(None)
        core_ret_all = core_nav.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0).rename('core')
        core_source = 'portfolio_df.portfolio_value'

    # Source 2: core_returns
    if core_ret_all is None and core_returns is not None and isinstance(core_returns, pd.Series) and not core_returns.empty:
        core_ret_all = core_returns.copy().sort_index()
        core_ret_all.index = pd.DatetimeIndex(core_ret_all.index).tz_localize(None)
        core_ret_all = core_ret_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename('core')
        core_source = 'core_returns'

    if core_ret_all is None and strict_core:
        raise RuntimeError("Source Core non trouvée.")

    # Source 3: fallback
    if core_ret_all is None and allow_fallback and core_3_log is not None:
        core_log = core_3_log.copy().sort_index().mean(axis=1)
        core_log.index = pd.DatetimeIndex(core_log.index).tz_localize(None)
        core_ret_all = np.expm1(core_log).rename('core')
        core_source = 'fallback core_3_log.mean'

    if core_ret_all is None:
        raise RuntimeError('Aucune source Core exploitable.')

    # Satellite returns
    w_ticker = weights_ticker_daily.copy().sort_index()
    prices_all = sat_prices.copy().sort_index()
    w_ticker.index = pd.DatetimeIndex(w_ticker.index).tz_localize(None)
    prices_all.index = pd.DatetimeIndex(prices_all.index).tz_localize(None)

    sat_rets_all = prices_all.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    common_dates = w_ticker.index.intersection(sat_rets_all.index).intersection(core_ret_all.index)
    common_dates = common_dates[(common_dates >= analysis_start) & (common_dates <= analysis_end)]
    common_tickers = [c for c in w_ticker.columns if c in sat_rets_all.columns]

    if len(common_dates) == 0 or len(common_tickers) == 0:
        raise ValueError('No common dates/tickers.')

    w = w_ticker.loc[common_dates, common_tickers].fillna(0.0)
    r_sat = sat_rets_all.loc[common_dates, common_tickers].fillna(0.0)
    w_exec = w.shift(1).fillna(0.0)
    sat_ret = (w_exec * r_sat).sum(axis=1).rename('satellite')
    core_ret = core_ret_all.loc[common_dates].fillna(0.0).rename('core')

    ret_cmp = pd.concat([core_ret, sat_ret], axis=1).dropna(how='any')

    print(f'Source Core : {core_source}')
    print(f"Fenêtre : {ret_cmp.index.min().date()} → {ret_cmp.index.max().date()} ({len(ret_cmp)} obs)")

    return ret_cmp, core_source


# ── Full analysis + dashboard (cell 28) ─────────────────────────────────────

def analyze_core_vs_satellite(ret_cmp, analysis_start, analysis_end, outdir='outputs'):
    """
    Complete Core vs Satellite comparison: metrics, rolling stats, and 8-panel dashboard.

    Returns dict with metrics DataFrames and key statistics.
    """
    OUT = Path(outdir)
    OUT.mkdir(exist_ok=True)
    ret_cmp.to_csv(OUT / 'core_vs_satellite_daily_returns.csv', index=True)

    # Static metrics
    beta_static = (ret_cmp['satellite'].cov(ret_cmp['core']) / ret_cmp['core'].var()
                   if ret_cmp['core'].var() != 0 else np.nan)
    corr_static = ret_cmp['satellite'].corr(ret_cmp['core'])
    tracking_error = (ret_cmp['satellite'] - ret_cmp['core']).std() * np.sqrt(252.0)

    alpha_daily = ret_cmp['satellite'] - (beta_static * ret_cmp['core'])
    alpha_daily.name = 'alpha_daily'

    # Rolling
    win = 126
    beta_roll = (ret_cmp['satellite'].rolling(win).cov(ret_cmp['core'])
                 / ret_cmp['core'].rolling(win).var().replace(0, np.nan))
    corr_roll = ret_cmp['satellite'].rolling(win).corr(ret_cmp['core'])
    vol_roll = ret_cmp.rolling(win).std() * np.sqrt(252.0)
    alpha_rolling = (ret_cmp['satellite'].rolling(win).mean()
                     - (beta_roll * ret_cmp['core'].rolling(win).mean()))

    alpha_annual = ret_cmp.groupby(ret_cmp.index.year).apply(
        lambda x: ann_return(x['satellite']) - beta_static * ann_return(x['core'])
    )
    alpha_ann_total = ann_return(alpha_daily)

    sat_total_ret = ann_return(ret_cmp['satellite'])
    core_total_ret = ann_return(ret_cmp['core'])
    return_diff = sat_total_ret - core_total_ret

    metrics_cmp = pd.DataFrame({
        'Ann Return': [core_total_ret, sat_total_ret],
        'Ann Vol': [ann_vol(ret_cmp['core']), ann_vol(ret_cmp['satellite'])],
        'Sharpe (rf=0)': [sharpe0(ret_cmp['core']), sharpe0(ret_cmp['satellite'])],
        'Max DD': [max_dd(ret_cmp['core']), max_dd(ret_cmp['satellite'])],
        'Calmar': [calmar(ret_cmp['core']), calmar(ret_cmp['satellite'])],
        'Return Diff.': [np.nan, return_diff],
        'Alpha (CAPM)': [np.nan, alpha_ann_total],
    }, index=['Core', 'Satellite'])

    summary_cross = pd.DataFrame({
        'Metric': ['Static beta', 'Static correlation', 'Tracking error ann.',
                   'Simple Outperformance', 'Alpha annualized (CAPM)'],
        'Value': [beta_static, corr_static, tracking_error, return_diff, alpha_ann_total],
    })

    annual_cmp = ret_cmp.groupby(ret_cmp.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    nav_cmp = (1.0 + ret_cmp).cumprod()
    dd_cmp = nav_cmp / nav_cmp.cummax() - 1.0
    analysis_start_actual = ret_cmp.index.min()

    # Display
    print('\nComparative metrics: Core vs Satellite')
    display(metrics_cmp.style.format({
        'Ann Return': '{:+.2%}', 'Ann Vol': '{:.2%}', 'Sharpe (rf=0)': '{:.2f}',
        'Max DD': '{:.2%}', 'Calmar': '{:.2f}', 'Return Diff.': '{:+.2%}',
        'Alpha (CAPM)': '{:+.2%}',
    }))
    print('\nCross metrics:')
    display(summary_cross.style.format({'Value': '{:.4f}'}))
    print('\nAnnual returns:')
    display(annual_cmp.style.format('{:+.2%}'))

    # Dashboard (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)

    axes[0, 0].plot(nav_cmp.index, nav_cmp['core'], lw=2.0, label='Core', color='#1f77b4')
    axes[0, 0].plot(nav_cmp.index, nav_cmp['satellite'], lw=2.0, label='Satellite', color='#ff7f0e')
    axes[0, 0].set_title('NAV cumulée')
    axes[0, 0].set_ylabel('Base 1')
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(dd_cmp.index, dd_cmp['core'] * 100, lw=1.7, label='Core', color='#1f77b4')
    axes[0, 1].plot(dd_cmp.index, dd_cmp['satellite'] * 100, lw=1.7, label='Satellite', color='#ff7f0e')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(vol_roll.index, vol_roll['core'] * 100, lw=1.7, label='Core', color='#1f77b4')
    axes[0, 2].plot(vol_roll.index, vol_roll['satellite'] * 100, lw=1.7, label='Satellite', color='#ff7f0e')
    axes[0, 2].set_title(f'Vol rolling {win}j annualisée')
    axes[0, 2].set_ylabel('%')
    axes[0, 2].grid(True, alpha=0.25)
    axes[0, 2].legend(frameon=False)

    axes[0, 3].axis('off')

    axes[1, 0].plot(beta_roll.index, beta_roll, lw=1.8, color='#2ca02c', label='Beta rolling')
    axes[1, 0].axhline(0.0, color='black', lw=0.8)
    if beta_roll.notna().any():
        axes[1, 0].axhline(beta_roll.mean(), color='#d62728', lw=1.2, ls='--',
                           label=f"Mean {beta_roll.mean():.2f}")
    axes[1, 0].set_title(f'Beta rolling {win}j')
    axes[1, 0].set_ylabel('Beta')
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(corr_roll.index, corr_roll, lw=1.8, color='#9467bd')
    axes[1, 1].axhline(0.0, color='black', lw=0.8)
    axes[1, 1].set_title(f'Corrélation rolling {win}j')
    axes[1, 1].set_ylabel('Corr')
    axes[1, 1].grid(True, alpha=0.25)

    axes[1, 2].plot(alpha_rolling.index, alpha_rolling * 100, lw=1.8, color='#d62728')
    axes[1, 2].axhline(0.0, color='black', lw=0.8)
    if alpha_rolling.notna().any():
        axes[1, 2].axhline(alpha_rolling.mean() * 100, color='#2ca02c', lw=1.2, ls='--',
                           label=f"Mean {alpha_rolling.mean() * 100:.2f}%")
    axes[1, 2].set_title(f'Alpha rolling {win}j')
    axes[1, 2].set_ylabel('Alpha (%)')
    axes[1, 2].grid(True, alpha=0.25)
    axes[1, 2].legend(frameon=False)

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
        ax.set_xlim(analysis_start_actual, analysis_end)

    if len(annual_cmp) > 0:
        x = np.arange(len(annual_cmp.index))
        bw = 0.25
        axes[1, 3].bar(x - bw, annual_cmp['core'] * 100, width=bw, label='Core',
                       color='#1f77b4', alpha=0.9)
        axes[1, 3].bar(x, annual_cmp['satellite'] * 100, width=bw, label='Satellite',
                       color='#ff7f0e', alpha=0.9)
        axes[1, 3].bar(x + bw, alpha_annual * 100, width=bw, label='Alpha',
                       color='#d62728', alpha=0.9)
        axes[1, 3].set_xticks(x)
        axes[1, 3].set_xticklabels(annual_cmp.index.astype(str), rotation=0)
    axes[1, 3].axhline(0.0, color='black', lw=0.8)
    axes[1, 3].set_title('Rendements + Alpha annuels')
    axes[1, 3].set_ylabel('%')
    axes[1, 3].grid(True, axis='y', alpha=0.25)
    axes[1, 3].legend(frameon=False, fontsize=8)

    fig.suptitle('Core vs Satellite — Performance, Risk & Alpha Dashboard', fontsize=15, y=1.02)
    plt.show()

    return {
        'metrics_cmp': metrics_cmp, 'summary_cross': summary_cross,
        'annual_cmp': annual_cmp, 'alpha_annual': alpha_annual,
        'beta_static': beta_static, 'corr_static': corr_static,
        'tracking_error': tracking_error, 'alpha_ann_total': alpha_ann_total,
        'return_diff': return_diff, 'nav_cmp': nav_cmp,
        'ret_cmp': ret_cmp,
    }


# ── Alpha interpretation (cell 29) ──────────────────────────────────────────

def display_alpha_interpretation(cvs_results):
    """Print clarification of alpha metrics for intuitive interpretation."""
    ret_cmp = cvs_results['ret_cmp']
    beta_static = cvs_results['beta_static']
    alpha_ann_total = cvs_results['alpha_ann_total']
    tracking_error = cvs_results['tracking_error']

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
    print(f"\n2) Simple Outperformance :")
    print(f"   Satellite : {sat_total:+.2%} | Core : {core_total:+.2%} | Diff : {simple_outperformance:+.2%}")
    print(f"   → {'UNDERPERFORMANCE' if simple_outperformance < 0 else 'OUTPERFORMANCE'}")
    print(f"\n3) Sharpe delta : Sat={sat_sharpe:+.2f} | Core={core_sharpe:+.2f} | Diff={sharpe_diff:+.2f}")
    print(f"\n4) Information Ratio : {info_ratio:+.2f} (TE={tracking_error:.2%}, Alpha={alpha_ann_total:+.2%})")
    print("=" * 70)
