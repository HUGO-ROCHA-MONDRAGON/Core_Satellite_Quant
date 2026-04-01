"""Final Core/Satellite portfolio: dynamic allocation, metrics, fees, and dashboard."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from src.utils import ann_return, ann_vol, max_dd, sharpe0, calmar, normalize_index, fmt_value
from src.core_satellite_allocator import allocate_core_satellite_dynamic


def run_final_portfolio_analysis(ret_cmp, allocator_params, analysis_start, analysis_end,
                                 ter_wavg_daily=None, core_ter_bps=9.0,
                                 weighted_core_ter_bps=None,
                                 outdir='outputs'):
    """
    Run dynamic Core/Satellite allocation on real returns, compute full
    metrics (including fees), and produce 6-panel dashboard.

    Parameters
    ----------
    ret_cmp : DataFrame with 'core' and 'satellite' columns
    allocator_params : dict with keys for allocate_core_satellite_dynamic
    analysis_start, analysis_end : Timestamps
    ter_wavg_daily : Series of satellite TER in bps (optional)
    core_ter_bps : float, estimated Core TER in bps/year
    outdir : str, output directory

    Returns dict with all results.
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    fig_dir = outdir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Prepare returns
    real_returns_input = normalize_index(ret_cmp[['core', 'satellite']])
    real_returns_input = real_returns_input.loc[
        (real_returns_input.index >= analysis_start) & (real_returns_input.index <= analysis_end)
    ].replace([np.inf, -np.inf], np.nan).dropna(how='any')

    if real_returns_input.empty:
        raise RuntimeError('Fenêtre de rendements réels vide après nettoyage.')

    lookback = allocator_params.get('lookback', 63)
    vol_core_in = real_returns_input['core'].rolling(lookback, min_periods=lookback).std() * np.sqrt(252)
    vol_sat_in = real_returns_input['satellite'].rolling(lookback, min_periods=lookback).std() * np.sqrt(252)

    print('\n' + '=' * 72)
    print("ALLOCATION DYNAMIQUE CORE/SATELLITE — SÉRIES RÉELLES")
    print('=' * 72)
    print(f"Période : {real_returns_input.index.min().date()} → {real_returns_input.index.max().date()}")
    print(f"Nb observations : {len(real_returns_input):,}")
    print(f"Vol Core 63j (moy): {vol_core_in.mean():.2%} | Vol Sat 63j (moy): {vol_sat_in.mean():.2%}")

    # Dynamic allocation
    weights_dyn_real, port_ret_real, port_vol_roll_real, rebalance_log_real = (
        allocate_core_satellite_dynamic(
            returns=real_returns_input,
            core_col='core',
            sat_col='satellite',
            **allocator_params,
        )
    )

    weights_dyn_real = normalize_index(weights_dyn_real)
    port_ret_real = normalize_index(port_ret_real).rename('portfolio_return')
    port_vol_roll_real = normalize_index(port_vol_roll_real).rename('portfolio_vol_rolling')
    rebalance_log_real = normalize_index(rebalance_log_real)

    sat_levels = np.sort(weights_dyn_real['w_sat'].dropna().unique())
    print(f'Niveaux w_sat observés : {[f"{x:.0%}" for x in sat_levels]}')
    print(f"Décisions : {rebalance_log_real['decision'].value_counts(dropna=False).to_dict()}")

    # Exports
    real_returns_input.to_csv(outdir / 'core_sat_dynamic_input_returns_real.csv')
    weights_dyn_real.to_csv(outdir / 'core_sat_dynamic_weights_real.csv')
    port_ret_real.to_frame('portfolio_return').to_csv(outdir / 'core_sat_dynamic_returns_real.csv')
    port_vol_roll_real.to_frame('portfolio_vol_rolling').to_csv(
        outdir / 'core_sat_dynamic_vol_rolling_real.csv')
    rebalance_log_real.to_csv(outdir / 'core_sat_dynamic_rebalance_log_real.csv')

    # Metrics
    global_ret = port_ret_real.copy().rename('global')
    core_ret = real_returns_input['core'].reindex(global_ret.index).dropna().rename('core')
    sat_ret = real_returns_input['satellite'].reindex(global_ret.index).dropna().rename('satellite')
    common_perf_dates = global_ret.index.intersection(core_ret.index).intersection(sat_ret.index)

    global_ret = global_ret.loc[common_perf_dates]
    core_ret = core_ret.loc[common_perf_dates]
    sat_ret = sat_ret.loc[common_perf_dates]
    weights_dyn_real = weights_dyn_real.loc[common_perf_dates]
    port_vol_roll_real = port_vol_roll_real.loc[common_perf_dates]

    comparison_metrics = pd.DataFrame({
        'Cumul total': [(1.0 + global_ret).prod() - 1.0, (1.0 + core_ret).prod() - 1.0,
                        (1.0 + sat_ret).prod() - 1.0],
        'Ret ann.': [ann_return(global_ret), ann_return(core_ret), ann_return(sat_ret)],
        'Vol ann.': [ann_vol(global_ret), ann_vol(core_ret), ann_vol(sat_ret)],
        'Sharpe (rf=0)': [sharpe0(global_ret), sharpe0(core_ret), sharpe0(sat_ret)],
        'Max DD': [max_dd(global_ret), max_dd(core_ret), max_dd(sat_ret)],
        'Calmar': [calmar(global_ret), calmar(core_ret), calmar(sat_ret)],
    }, index=['Portefeuille global', 'Core choisi', 'Satellite dynamique'])

    beta_global = global_ret.cov(core_ret) / core_ret.var() if core_ret.var() != 0 else np.nan
    corr_global = global_ret.corr(core_ret)
    tracking_error_global = (global_ret - core_ret).std() * np.sqrt(252.0)

    annual_returns_table = pd.concat([
        global_ret.groupby(global_ret.index.year).apply(lambda x: (1.0 + x).prod() - 1.0).rename('Global'),
        core_ret.groupby(core_ret.index.year).apply(lambda x: (1.0 + x).prod() - 1.0).rename('Core'),
        sat_ret.groupby(sat_ret.index.year).apply(lambda x: (1.0 + x).prod() - 1.0).rename('Satellite'),
    ], axis=1)

    # Fees
    core_ter_bps_est = float(weighted_core_ter_bps) if weighted_core_ter_bps is not None else float(core_ter_bps)

    if ter_wavg_daily is not None and isinstance(ter_wavg_daily, pd.Series) and not ter_wavg_daily.empty:
        sat_ter_bps_daily = normalize_index(ter_wavg_daily).reindex(common_perf_dates).ffill().bfill()
    else:
        sat_ter_bps_daily = pd.Series(0.0, index=common_perf_dates)

    portfolio_ter_bps_daily = (
        weights_dyn_real['w_core'] * core_ter_bps_est
        + weights_dyn_real['w_sat'] * sat_ter_bps_daily
    ).rename('portfolio_ter_bps_annualized')
    portfolio_ter_daily_drag = (portfolio_ter_bps_daily / 10000.0 / 252.0).rename('portfolio_ter_daily_drag')

    global_ret_net_est = (global_ret - portfolio_ter_daily_drag).rename('portfolio_return_net_est')
    gross_nav = (1.0 + global_ret).cumprod().rename('portfolio_nav_gross')
    net_nav_est = (1.0 + global_ret_net_est).cumprod().rename('portfolio_nav_net_est')
    fee_drag_nav = (gross_nav - net_nav_est).rename('fee_drag_nav')

    alloc_turnover_daily = (
        0.5 * weights_dyn_real.diff().abs().sum(axis=1)
    ).fillna(0.0).rename('allocation_turnover_daily')
    monthly_fee_bps = portfolio_ter_bps_daily.resample('ME').mean()
    monthly_turnover = alloc_turnover_daily.resample('ME').sum()

    annual_fee_table = pd.concat([
        portfolio_ter_bps_daily.resample('YE').mean().rename('TER global moyen (bps/an)'),
        alloc_turnover_daily.resample('YE').sum().rename('Turnover allocation cumulé'),
    ], axis=1)
    annual_fee_table.index = annual_fee_table.index.year

    # Fee detail export
    fee_detail_df = pd.concat([
        global_ret.rename('portfolio_return_gross'), global_ret_net_est,
        gross_nav, net_nav_est, fee_drag_nav,
        portfolio_ter_bps_daily, portfolio_ter_daily_drag, alloc_turnover_daily,
    ], axis=1)
    fee_detail_df.to_csv(outdir / 'core_sat_dynamic_fee_estimates.csv')
    comparison_metrics.to_csv(outdir / 'core_sat_dynamic_metrics.csv')
    annual_returns_table.to_csv(outdir / 'core_sat_dynamic_annual_returns.csv')
    annual_fee_table.to_csv(outdir / 'core_sat_dynamic_fee_summary.csv')

    # Diagnostics table
    global_diagnostics = pd.DataFrame([
        {'Indicateur': 'Beta statique global vs Core', 'Valeur': beta_global, 'Type': 'num'},
        {'Indicateur': 'Corrélation globale vs Core', 'Valeur': corr_global, 'Type': 'num'},
        {'Indicateur': 'Tracking error annualisée', 'Valeur': tracking_error_global, 'Type': 'pct'},
        {'Indicateur': 'Poids satellite moyen', 'Valeur': weights_dyn_real['w_sat'].mean(), 'Type': 'pct'},
        {'Indicateur': 'Poids satellite min', 'Valeur': weights_dyn_real['w_sat'].min(), 'Type': 'pct'},
        {'Indicateur': 'Poids satellite max', 'Valeur': weights_dyn_real['w_sat'].max(), 'Type': 'pct'},
        {'Indicateur': 'TER Core utilisé', 'Valeur': core_ter_bps_est, 'Type': 'bps'},
        {'Indicateur': 'TER satellite moyen', 'Valeur': sat_ter_bps_daily.mean(), 'Type': 'bps'},
        {'Indicateur': 'TER global moyen estimé', 'Valeur': portfolio_ter_bps_daily.mean(), 'Type': 'bps'},
        {'Indicateur': 'Drag TER cumulé estimé',
         'Valeur': gross_nav.iloc[-1] - net_nav_est.iloc[-1], 'Type': 'pct'},
        {'Indicateur': 'Nb rebalancements effectifs',
         'Valeur': (rebalance_log_real['decision'] == 'rebalance').sum(), 'Type': 'int'},
        {'Indicateur': 'Turnover allocation cumulé',
         'Valeur': alloc_turnover_daily.sum(), 'Type': 'num'},
    ])
    diag_display = global_diagnostics.copy()
    diag_display['Valeur'] = [fmt_value(v, t) for v, t in zip(diag_display['Valeur'], diag_display['Type'])]
    diag_display = diag_display[['Indicateur', 'Valeur']]

    # Display
    print('\nMétriques comparatives :')
    display(comparison_metrics.style.format({
        'Cumul total': '{:+.2%}', 'Ret ann.': '{:+.2%}', 'Vol ann.': '{:.2%}',
        'Sharpe (rf=0)': '{:.2f}', 'Max DD': '{:.2%}', 'Calmar': '{:.2f}',
    }))
    print('\nIndicateurs portefeuille global :')
    display(diag_display)
    print('\nRendements annuels :')
    display(annual_returns_table.style.format('{:+.2%}'))
    print('\nFrais / turnover annuels :')
    display(annual_fee_table.style.format({
        'TER global moyen (bps/an)': '{:.1f}', 'Turnover allocation cumulé': '{:.2f}',
    }))

    # Dashboard (3x2)
    nav_compare = pd.concat([
        gross_nav.rename('Global brut'), net_nav_est.rename('Global net estimé'),
        ((1.0 + core_ret).cumprod()).rename('Core'),
        ((1.0 + sat_ret).cumprod()).rename('Satellite'),
    ], axis=1).dropna(how='all')

    dd_compare = nav_compare.div(nav_compare.cummax()).sub(1.0)
    vol_compare = pd.concat([
        global_ret.rolling(63).std() * np.sqrt(252),
        core_ret.rolling(63).std() * np.sqrt(252),
        sat_ret.rolling(63).std() * np.sqrt(252),
    ], axis=1)
    vol_compare.columns = ['Global', 'Core', 'Satellite']

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)

    # NAV
    axes[0, 0].plot(nav_compare.index, nav_compare['Global brut'], lw=2.2, color='#1f77b4',
                    label='Global brut')
    axes[0, 0].plot(nav_compare.index, nav_compare['Global net estimé'], lw=2.0, color='#0f766e',
                    ls='--', label='Global net estimé')
    axes[0, 0].plot(nav_compare.index, nav_compare['Core'], lw=1.7, color='#ff7f0e', label='Core')
    axes[0, 0].plot(nav_compare.index, nav_compare['Satellite'], lw=1.7, color='#6f42c1',
                    label='Satellite')
    axes[0, 0].set_title('NAV cumulée')
    axes[0, 0].set_ylabel('Base 1')
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)

    # Drawdown
    axes[0, 1].plot(dd_compare.index, dd_compare['Global brut'] * 100, lw=1.8, color='#1f77b4',
                    label='Global brut')
    axes[0, 1].plot(dd_compare.index, dd_compare['Core'] * 100, lw=1.5, color='#ff7f0e', label='Core')
    axes[0, 1].plot(dd_compare.index, dd_compare['Satellite'] * 100, lw=1.5, color='#6f42c1',
                    label='Satellite')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False)

    # Vol rolling
    axes[1, 0].plot(vol_compare.index, vol_compare['Global'] * 100, lw=2.0, color='#1f77b4',
                    label='Global')
    axes[1, 0].plot(vol_compare.index, vol_compare['Core'] * 100, lw=1.5, color='#ff7f0e',
                    label='Core')
    axes[1, 0].plot(vol_compare.index, vol_compare['Satellite'] * 100, lw=1.5, color='#6f42c1',
                    label='Satellite')
    axes[1, 0].axhline(10.0, color='black', ls='--', lw=1.0, label='Cible 10%')
    axes[1, 0].set_title('Volatilité rolling 63j annualisée')
    axes[1, 0].set_ylabel('%')
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    # Dynamic weights
    axes[1, 1].step(weights_dyn_real.index, weights_dyn_real['w_sat'], where='post',
                    label='w_sat', color='#1f77b4')
    axes[1, 1].step(weights_dyn_real.index, weights_dyn_real['w_core'], where='post',
                    label='w_core', color='#ff7f0e')
    axes[1, 1].set_title('Poids dynamiques Core / Satellite')
    axes[1, 1].set_ylabel('Poids')
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(frameon=False)

    # Fees + turnover
    axes[2, 0].plot(monthly_fee_bps.index, monthly_fee_bps, lw=2.0, color='#0f766e',
                    label='TER global moyen mensuel')
    axes[2, 0].axhline(portfolio_ter_bps_daily.mean(), color='#0f766e', ls='--', lw=1.0,
                       label=f"Moyenne {portfolio_ter_bps_daily.mean():.1f} bps")
    axes[2, 0].set_title('Frais estimés du portefeuille global')
    axes[2, 0].set_ylabel('bps/an')
    axes[2, 0].grid(True, alpha=0.25)
    ax_fee = axes[2, 0].twinx()
    ax_fee.plot(monthly_turnover.index, monthly_turnover, lw=1.3, color='#d62728', alpha=0.75,
                label='Turnover allocation mensuel')
    ax_fee.set_ylabel('Turnover')
    lines1, labels1 = axes[2, 0].get_legend_handles_labels()
    lines2, labels2 = ax_fee.get_legend_handles_labels()
    axes[2, 0].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper left')

    # Annual returns
    annual_plot = annual_returns_table.fillna(0.0)
    if len(annual_plot) > 0:
        x = np.arange(len(annual_plot.index))
        bw = 0.25
        axes[2, 1].bar(x - bw, annual_plot['Global'] * 100, width=bw, color='#1f77b4', alpha=0.9,
                       label='Global')
        axes[2, 1].bar(x, annual_plot['Core'] * 100, width=bw, color='#ff7f0e', alpha=0.9,
                       label='Core')
        axes[2, 1].bar(x + bw, annual_plot['Satellite'] * 100, width=bw, color='#6f42c1', alpha=0.9,
                       label='Satellite')
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels(annual_plot.index.astype(str))
    axes[2, 1].axhline(0.0, color='black', lw=0.8)
    axes[2, 1].set_title('Rendements annuels')
    axes[2, 1].set_ylabel('%')
    axes[2, 1].grid(True, axis='y', alpha=0.25)
    axes[2, 1].legend(frameon=False)

    fig.suptitle('Portefeuille global Core/Satellite — Performance, allocation et frais',
                 fontsize=15, y=1.02)
    fig_path = fig_dir / 'core_sat_dynamic_global_dashboard.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Figure sauvegardée : {fig_path}')

    return {
        'weights_dyn_real': weights_dyn_real,
        'port_ret_real': port_ret_real,
        'port_vol_roll_real': port_vol_roll_real,
        'rebalance_log_real': rebalance_log_real,
        'comparison_metrics': comparison_metrics,
        'annual_returns_table': annual_returns_table,
        'annual_fee_table': annual_fee_table,
        'global_diagnostics': global_diagnostics,
        'ter_wavg_daily_used': sat_ter_bps_daily,
    }
