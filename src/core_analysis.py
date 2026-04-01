"""Core portfolio simulation, rebalancing analysis, and visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from src.core_rebalancing import CoreRebalancer
from src.utils import rolling_volatility_ann


# ── Simulation (cells 9+10) ─────────────────────────────────────────────────

def run_core_simulation(strategies, core_3_log, core_df, CORE_TICKERS, config_core):
    """
    Configure and run the Core rebalancing simulation.

    Combines: strategy selection, TER loading, returns prep, rebalancing,
    transaction costs.

    Returns dict with all results needed by downstream analysis.
    """
    selected_strategy = config_core['selected_strategy']
    tickers = list(CORE_TICKERS)

    # Weights from frontier
    core_weights_array = strategies[selected_strategy]["w"]
    core_weights = dict(zip(tickers, core_weights_array))

    # TER
    ter_col = 'ter_pct' if 'ter_pct' in core_df.columns else None
    ticker_col = 'Ticker' if 'Ticker' in core_df.columns else None
    if ter_col is None or ticker_col is None:
        raise ValueError("Colonnes 'Ticker' et 'ter_pct' introuvables dans core_df.")

    core_ter_map_pct = (
        core_df[[ticker_col, ter_col]]
        .dropna(subset=[ticker_col])
        .set_index(ticker_col)[ter_col]
        .to_dict()
    )
    for t in tickers:
        core_ter_map_pct.setdefault(t, 0.0)

    weighted_core_ter_pct = sum(core_weights[t] * core_ter_map_pct[t] for t in tickers)
    weighted_core_ter_bps = weighted_core_ter_pct * 100

    # Prepare returns
    daily_returns_df = core_3_log.copy()
    daily_returns_df.index = pd.DatetimeIndex(daily_returns_df.index).tz_localize(None)
    daily_returns_oos = daily_returns_df.loc[
        config_core['sim_start_date']:config_core['sim_end_date']
    ].copy()
    daily_simple_returns = np.expm1(daily_returns_oos)

    # Net TER returns
    ter_annual_decimal = pd.Series({t: core_ter_map_pct.get(t, 0.0) / 100.0 for t in tickers})
    ter_daily_drag = (1.0 + ter_annual_decimal) ** (1.0 / 252.0) - 1.0

    daily_returns_for_sim = daily_simple_returns.copy()
    for t in tickers:
        daily_returns_for_sim[t] = (1.0 + daily_simple_returns[t]) / (1.0 + ter_daily_drag[t]) - 1.0

    # Rebalancer
    rebalancer = CoreRebalancer(
        core_weights=core_weights,
        daily_returns=daily_returns_for_sim,
        rebalance_dates=config_core['rebalance_frequency'],
        start_value=config_core['start_value'],
    )
    portfolio_df, weights_df, rebalance_df = rebalancer.compute_portfolio_evolution()
    portfolio_df['portfolio_value_gross'] = portfolio_df['portfolio_value']

    # Transaction costs
    tx_bps_side = float(config_core.get('transaction_cost_bps_per_side', 0.0))
    tx_fixed_bps = float(config_core.get('transaction_cost_fixed_bps', 0.0))
    tx_rows = []

    if rebalance_df is not None and len(rebalance_df) > 0:
        gross_values = portfolio_df['portfolio_value_gross'].values
        dates = portfolio_df.index
        tx_cost_rates = np.zeros(len(portfolio_df), dtype=float)
        tx_cost_values = np.zeros(len(portfolio_df), dtype=float)
        net_values = gross_values.copy()

        rebalance_lookup = {}
        for _, row in rebalance_df.iterrows():
            dt = pd.Timestamp(row['date']) if 'date' in row else pd.Timestamp(row.name)
            turnover = float(row.get('turnover', 0.0)) if hasattr(row, 'get') else 0.0
            event_cost_pct = turnover * 2.0 * tx_bps_side / 10000.0 + tx_fixed_bps / 10000.0
            rebalance_lookup[dt] = max(event_cost_pct, 0.0)

        for i, d in enumerate(dates):
            val_before_cost = gross_values[i]
            event_cost_pct = rebalance_lookup.get(pd.Timestamp(d), 0.0)
            event_cost_value = val_before_cost * event_cost_pct
            net_values[i] = val_before_cost - event_cost_value
            tx_cost_values[i] = event_cost_value
            tx_cost_rates[i] = event_cost_pct

        portfolio_df['tx_cost_pct'] = tx_cost_rates
        portfolio_df['tx_cost_value'] = tx_cost_values
        portfolio_df['portfolio_value_net'] = net_values
        portfolio_df['portfolio_value'] = portfolio_df['portfolio_value_net']

        tx_rows = [
            {'date': d, 'tx_cost_pct': tx_cost_rates[i],
             'tx_cost_value': tx_cost_values[i], 'portfolio_value_net': net_values[i]}
            for i, d in enumerate(dates) if tx_cost_rates[i] > 0
        ]
    else:
        portfolio_df['tx_cost_pct'] = 0.0
        portfolio_df['tx_cost_value'] = 0.0
        portfolio_df['portfolio_value_net'] = portfolio_df['portfolio_value_gross']

    transaction_costs_df = pd.DataFrame(tx_rows) if tx_rows else pd.DataFrame(
        columns=['date', 'tx_cost_pct', 'tx_cost_value', 'portfolio_value_net']
    )

    freq_labels = {
        'daily': 'quotidien', 'weekly': 'hebdomadaire', 'monthly': 'mensuel',
        'quarterly': 'trimestriel', 'annual': 'annuel',
    }
    freq_display = freq_labels.get(config_core['rebalance_frequency'], 'personnalisé')

    # Print summary
    print("=" * 70)
    print(f"SIMULATION CORE — {selected_strategy} ({freq_display})")
    print("=" * 70)
    for t, w in core_weights.items():
        print(f"  {t:<20} : {w:.2%} | TER: {core_ter_map_pct[t]:.2f}%")
    print(f"  TER pondéré : {weighted_core_ter_pct:.3f}% ({weighted_core_ter_bps:.1f} bps)")
    print(f"Période : {daily_returns_for_sim.index.min().date()} → "
          f"{daily_returns_for_sim.index.max().date()} ({len(daily_returns_for_sim)} jours)")
    print(f"Valeur initiale : {config_core['start_value']:.2f}")
    print(f"Valeur finale brute : {portfolio_df['portfolio_value_gross'].iloc[-1]:.2f}")
    print(f"Valeur finale nette : {portfolio_df['portfolio_value'].iloc[-1]:.2f}")
    print(f"Rendement total net : "
          f"{(portfolio_df['portfolio_value'].iloc[-1] / config_core['start_value'] - 1):.2%}")
    print(f"Nb rebalancings : {len(rebalance_df) if rebalance_df is not None else 0}")

    return {
        'selected_strategy': selected_strategy,
        'core_weights': core_weights,
        'core_weights_array': core_weights_array,
        'core_ter_map_pct': core_ter_map_pct,
        'weighted_core_ter_pct': weighted_core_ter_pct,
        'weighted_core_ter_bps': weighted_core_ter_bps,
        'daily_returns_df': daily_returns_df,
        'daily_simple_returns': daily_simple_returns,
        'daily_returns_for_sim': daily_returns_for_sim,
        'ter_daily_drag': ter_daily_drag,
        'rebalancer': rebalancer,
        'portfolio_df': portfolio_df,
        'weights_df': weights_df,
        'rebalance_df': rebalance_df,
        'transaction_costs_df': transaction_costs_df,
        'tickers': tickers,
        'freq_display': freq_display,
        'config_core': config_core,
    }


# ── Visualization: portfolio weights (cell 11) ──────────────────────────────

def plot_portfolio_weights(sim):
    """4-subplot dashboard: NAV, weight drift, composition, annual drift."""
    portfolio_df = sim['portfolio_df']
    weights_df = sim['weights_df']
    core_weights = sim['core_weights']
    tickers = sim['tickers']
    freq_display = sim['freq_display']
    config_core = sim['config_core']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # NAV
    ax = axes[0, 0]
    ax.plot(portfolio_df.index, portfolio_df['portfolio_value'], linewidth=2, color='#1f77b4')
    ax.axhline(config_core['start_value'], color='gray', linestyle='--', alpha=0.5, label='Valeur initiale')
    ax.set_title(f"Évolution du portefeuille Core (rebalancing {freq_display})",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Valeur')
    ax.grid(alpha=0.3)
    ax.legend()

    # Drift
    ax = axes[0, 1]
    for ticker in tickers:
        drift = weights_df[ticker] - core_weights[ticker]
        ax.plot(weights_df.index, drift, label=ticker, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Dérive des poids (réels − cibles)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dérive (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Stacked composition
    ax = axes[1, 0]
    ax.stackplot(weights_df.index,
                 weights_df[tickers[0]], weights_df[tickers[1]], weights_df[tickers[2]],
                 labels=tickers, alpha=0.8)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title('Composition du portefeuille', fontsize=11, fontweight='bold')
    ax.set_ylabel('Poids cumulés')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(alpha=0.3, axis='y')

    # Annual drift
    ax = axes[1, 1]
    wdf = weights_df.copy()
    wdf['year'] = wdf.index.year
    annual_drift = []
    for year in sorted(wdf['year'].unique()):
        year_data = wdf[wdf['year'] == year]
        max_drift = 0
        for ticker in tickers:
            drift = max(abs(year_data[ticker].max() - core_weights[ticker]),
                        abs(year_data[ticker].min() - core_weights[ticker]))
            max_drift = max(max_drift, drift)
        annual_drift.append({'year': year, 'max_drift': max_drift})
    adf = pd.DataFrame(annual_drift)
    ax.bar(adf['year'].astype(str), adf['max_drift'] * 100, color='#ff7f0e', alpha=0.7)
    ax.set_title('Dérive maximale annuelle des poids', fontsize=11, fontweight='bold')
    ax.set_ylabel('Max Dérive (%)')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    return fig


# ── Rebalancing summary (cell 12) ───────────────────────────────────────────

def display_rebalancing_summary(sim):
    """Display rebalancing summary table and rebalance event log."""
    rebalancer = sim['rebalancer']
    weights_df = sim['weights_df']
    rebalance_df = sim['rebalance_df']
    freq_display = sim['freq_display']
    tickers = sim['tickers']

    print("\n" + "=" * 80)
    print(f"RÉSUMÉ DU REBALANCING ({freq_display.upper()})")
    print("=" * 80)

    summary_table = rebalancer.summary_table(weights_df)

    summary_display = pd.DataFrame()
    for year in summary_table.index:
        row_data = []
        for ticker in tickers:
            target = summary_table.loc[year, f"{ticker}_target"]
            drift = summary_table.loc[year, f"{ticker}_drift"]
            max_dev = summary_table.loc[year, f"{ticker}_max_dev"]
            row_data.append({
                'année': year,
                'ticker': ticker.replace(' GY Equity', ''),
                'poids_cible': f"{target:.2%}",
                'dérive_max': f"{drift:.2%}",
                'écart_max_cible': f"{max_dev:.2%}"
            })
        summary_display = pd.concat([summary_display, pd.DataFrame(row_data)], ignore_index=True)

    display(summary_display.style
            .hide(axis="index")
            .set_table_styles([{'selector': 'th', 'props': [('border', '1px solid black')]}]))

    if rebalance_df is not None and len(rebalance_df) > 0:
        print(f"\nDates de rebalancing ({freq_display}) : {len(rebalance_df)} opérations")
        for _, row in rebalance_df.iterrows():
            print(f"   {row['date'].date()} : Portefeuille = {row['portfolio_value_before']:.2f}")
    else:
        print(f"\nAucun rebalancing {freq_display} dans cette période")


# ── Tracking error (cell 13) ────────────────────────────────────────────────

def plot_tracking_error(sim):
    """Compute tracking error and plot comparison with/without rebalancing."""
    rebalancer = sim['rebalancer']
    portfolio_df = sim['portfolio_df']
    freq_display = sim['freq_display']

    print("\n" + "=" * 80)
    print("TRACKING ERROR & IMPACT DU REBALANCING")
    print("=" * 80)

    te_results = rebalancer.compute_tracking_error(portfolio_df)

    print(f"\nTracking Error (annualisée) : {te_results['tracking_error']:.3%}")
    print(f"   Valeur finale (avec rebalancing {freq_display})  : "
          f"{te_results['final_value_rebalanced']:>8.2f}")
    print(f"   Valeur finale (portefeuille statique)          : "
          f"{te_results['final_value_static']:>8.2f}")
    print(f"   Impact net (rebalancing)                       : "
          f"{te_results['cumulative_impact']:>7.2f}%")

    if 'portfolio_value_gross' in portfolio_df.columns:
        gross_final = float(portfolio_df['portfolio_value_gross'].iloc[-1])
        net_final = float(portfolio_df['portfolio_value'].iloc[-1])
        tx_drag_pct = (net_final / gross_final - 1.0) * 100 if gross_final > 0 else 0.0
        print(f"\n   Valeur finale brute (avant tx) : {gross_final:.2f}")
        print(f"   Valeur finale nette (après tx) : {net_final:.2f}")
        print(f"   Drag transaction costs         : {tx_drag_pct:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(portfolio_df.index, te_results['port_with_rebal'],
            label=f'Avec rebalancing ({freq_display})', linewidth=2.5, color='#1f77b4')
    ax.plot(portfolio_df.index, te_results['port_static'],
            label='Sans rebalancing (poids fixes)', linewidth=2.5, color='#ff7f0e', linestyle='--')
    if 'portfolio_value_gross' in portfolio_df.columns:
        ax.plot(portfolio_df.index, portfolio_df['portfolio_value_gross'],
                label='Avec rebalancing (avant tx)', linewidth=1.8, color='#9467bd', alpha=0.8, linestyle=':')
    ax.set_title('Portefeuille avec vs sans rebalancing', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valeur')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    cumulative_diff = np.cumsum(te_results['daily_differential_returns'])
    ax.plot(portfolio_df.index[1:], cumulative_diff * 100, linewidth=2, color='#2ca02c')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(portfolio_df.index[1:], 0, cumulative_diff * 100,
                    where=(cumulative_diff >= 0), alpha=0.3, color='green', label='Bénéfice')
    ax.fill_between(portfolio_df.index[1:], 0, cumulative_diff * 100,
                    where=(cumulative_diff < 0), alpha=0.3, color='red', label='Coût')
    ax.set_title(f'Rendement différentiel cumulé (Rebalancing {freq_display} vs statique)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Rendement différentiel (%)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return te_results


# ── Volatility & costs (cell 14) ────────────────────────────────────────────

def plot_volatility_and_costs(sim):
    """Rolling volatility analysis + annual cost breakdown (TER + tx)."""
    portfolio_df = sim['portfolio_df']
    weights_df = sim['weights_df']
    core_weights = sim['core_weights']
    core_ter_map_pct = sim['core_ter_map_pct']
    tickers = sim['tickers']
    config_core = sim['config_core']
    freq_display = sim['freq_display']
    daily_simple_returns = sim['daily_simple_returns']
    daily_returns_for_sim = sim['daily_returns_for_sim']

    print("\n" + "=" * 80)
    print("VOLATILITÉ DU CORE & COÛT TOTAL")
    print("=" * 80)

    # A) Rolling volatility
    vol_window_map = {'6m': 126, '1y': 252}
    vol_window_choice = config_core.get('vol_window', '6m')
    if vol_window_choice not in vol_window_map:
        vol_window_choice = '6m'
    vol_window = vol_window_map[vol_window_choice]
    window_label = '6 mois (126j)' if vol_window_choice == '6m' else '1 an (252j)'

    core_weights_array = np.array([core_weights[t] for t in tickers])
    core_returns = (daily_returns_for_sim * core_weights_array).sum(axis=1)
    vol_rolling = rolling_volatility_ann(core_returns, window=vol_window)
    vol_6m = rolling_volatility_ann(core_returns, window=126)
    vol_1y = rolling_volatility_ann(core_returns, window=252)

    print(f"\nVolatilité rolling Core ({window_label}, annualisée)")
    print(f"   Moyenne : {vol_rolling.mean():.2%} | Médiane : {vol_rolling.median():.2%}")
    print(f"   Min     : {vol_rolling.min():.2%}  ({vol_rolling.idxmin().date()})")
    print(f"   Max     : {vol_rolling.max():.2%}  ({vol_rolling.idxmax().date()})")
    print(f"\n   6m — moy: {vol_6m.mean():.2%} | 1y — moy: {vol_1y.mean():.2%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
    ax1.plot(vol_rolling.index, vol_rolling, lw=2.3, color='#1f77b4',
             label=f'Vol rolling {window_label}')
    ax1.plot(vol_1y.index, vol_1y, lw=1.2, color='#7f7f7f', ls='--', alpha=0.8,
             label='Vol rolling 1 an (réf.)')
    ax1.axhline(vol_rolling.mean(), color='#ff7f0e', ls='--', lw=1.5,
                label=f'Moyenne : {vol_rolling.mean():.2%}')
    ax1.fill_between(vol_rolling.index, vol_rolling.min(), vol_rolling.max(),
                     alpha=0.1, color='#1f77b4')
    ax1.set_title(f'Volatilité du Core — Rolling {window_label} (annualisée)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Volatilité annualisée')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.hist(vol_rolling.dropna(), bins=35, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.axvline(vol_rolling.mean(), color='#ff7f0e', ls='--', lw=2, label='Moyenne')
    ax2.axvline(vol_rolling.median(), color='#2ca02c', ls='--', lw=2, label='Médiane')
    ax2.set_title(f'Distribution de la volatilité rolling ({window_label})',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Volatilité annualisée')
    ax2.set_ylabel('Fréquence')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # B) Annual costs (TER + TX)
    print(f"\nCOÛTS PAR ANNÉE (NET TER + TX) — Rebalancing {freq_display}")

    tx_cost_series = portfolio_df.get('tx_cost_pct', pd.Series(0.0, index=portfolio_df.index))

    ter_annual_decimal_fb = pd.Series({t: core_ter_map_pct.get(t, 0.0) / 100.0 for t in tickers})
    ter_drag_by_ticker = (1.0 + ter_annual_decimal_fb) ** (1.0 / 252.0) - 1.0

    weights_for_ter = weights_df[tickers].reindex(portfolio_df.index).ffill().fillna(0.0)
    daily_ter_drag = (weights_for_ter * ter_drag_by_ticker.reindex(tickers).values).sum(axis=1).clip(lower=0.0)

    annual_cost_rows = []
    for y in sorted(portfolio_df.index.year.unique()):
        y_mask = portfolio_df.index.year == y
        ter_y = daily_ter_drag.loc[y_mask].values
        tx_y = tx_cost_series.loc[y_mask].values
        ter_cost_pct = float(1.0 - np.prod(1.0 / (1.0 + ter_y))) if len(ter_y) > 0 else 0.0
        tx_cost_pct_y = float(1.0 - np.prod(1.0 - tx_y)) if len(tx_y) > 0 else 0.0
        total_cost_pct = float(1.0 - (1.0 - ter_cost_pct) * (1.0 - tx_cost_pct_y))
        annual_cost_rows.append({
            'year': y,
            'TER_bps': ter_cost_pct * 10000.0,
            'TX_bps': tx_cost_pct_y * 10000.0,
            'Total_bps': total_cost_pct * 10000.0,
            'nb_rebal_tx': int((tx_cost_series.loc[y_mask] > 0).sum()),
        })

    annual_cost_df = pd.DataFrame(annual_cost_rows).set_index('year')
    display(annual_cost_df.style.format({
        'TER_bps': '{:.2f}', 'TX_bps': '{:.2f}', 'Total_bps': '{:.2f}', 'nb_rebal_tx': '{:.0f}',
    }))

    # Annual cost bar chart
    x = np.arange(len(annual_cost_df))
    w = 0.25
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, annual_cost_df['TER_bps'], width=w, label='TER', color='#1f77b4', alpha=0.85)
    ax.bar(x, annual_cost_df['TX_bps'], width=w, label='Transaction', color='#d62728', alpha=0.85)
    ax.bar(x + w, annual_cost_df['Total_bps'], width=w, label='Total', color='#2ca02c', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(annual_cost_df.index.astype(str))
    ax.set_title('Coût annuel en bps (TER, TX, Total)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coût (bps)')
    ax.grid(alpha=0.25, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/10c_annual_costs_histogram.png', dpi=160, bbox_inches='tight')
    plt.show()
