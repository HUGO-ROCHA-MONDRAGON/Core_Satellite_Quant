"""Satellite dynamic performance analysis, turnover diagnostics, and cache validation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from src.utils import ann_return, ann_vol


# ── Validation / cache (cell 21) ────────────────────────────────────────────

def validate_satellite_cache(regime_output, quarter_selection_df, df_satellite_prices_all,
                             data_start, data_end):
    """
    Validate upstream inputs and build shared satellite price cache.

    Returns SATELLITE_PRICES_SHARED DataFrame.
    """
    if regime_output is None or not isinstance(regime_output, pd.DataFrame) or regime_output.empty:
        raise RuntimeError('regime_output indisponible.')
    if quarter_selection_df is None or not isinstance(quarter_selection_df, pd.DataFrame) or quarter_selection_df.empty:
        raise RuntimeError('quarter_selection_df indisponible.')
    if df_satellite_prices_all is None or not isinstance(df_satellite_prices_all, pd.DataFrame) or df_satellite_prices_all.empty:
        raise RuntimeError('df_satellite_prices_all indisponible.')

    prices = df_satellite_prices_all.copy().sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices = prices.loc[(prices.index >= data_start) & (prices.index <= data_end)].copy()

    if prices.empty:
        raise RuntimeError('La fenêtre de prix satellite partagée est vide.')

    print(f'regime_output : {len(regime_output)} lignes')
    print(f'quarter_selection_df : {len(quarter_selection_df)} lignes')
    print(f'Prix satellite : {prices.index.min().date()} → {prices.index.max().date()} '
          f'| {len(prices.columns)} tickers')
    return prices


# ── Satellite dynamic performance (cell 24) ─────────────────────────────────

def analyze_satellite_dynamic(weights_ticker_daily, block_weights_daily, active_funds_daily,
                              portfolio_df, sat_prices, analysis_start, analysis_end):
    """
    Full satellite dynamic performance analysis: metrics, NAV, drawdown,
    rolling vol, bloc contributions, beta vs Core.

    Returns dict with key series and DataFrames.
    """
    w_ticker = weights_ticker_daily.copy().sort_index()
    w_block = block_weights_daily.copy().sort_index()
    active_funds = active_funds_daily.copy().sort_values(['date', 'bloc', 'ticker'])
    core_nav = portfolio_df['portfolio_value'].copy().sort_index()

    w_ticker.index = pd.DatetimeIndex(w_ticker.index).tz_localize(None)
    w_block.index = pd.DatetimeIndex(w_block.index).tz_localize(None)
    core_nav.index = pd.DatetimeIndex(core_nav.index).tz_localize(None)

    prices_all = sat_prices.copy().sort_index()
    prices_all.index = pd.DatetimeIndex(prices_all.index).tz_localize(None)

    rets_all = prices_all.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how='all')
    core_ret_all = core_nav.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    common_dates = w_ticker.index.intersection(rets_all.index).intersection(core_ret_all.index)
    common_dates = common_dates[(common_dates >= analysis_start) & (common_dates <= analysis_end)]
    common_tickers = [c for c in w_ticker.columns if c in rets_all.columns]

    if len(common_dates) == 0 or len(common_tickers) == 0:
        raise ValueError("Aucune intersection dates/tickers.")

    w = w_ticker.loc[common_dates, common_tickers].fillna(0.0)
    r = rets_all.loc[common_dates, common_tickers].fillna(0.0)
    core_ret = core_ret_all.loc[common_dates].fillna(0.0)

    w_exec = w.shift(1).fillna(0.0)
    sat_ret = (w_exec * r).sum(axis=1)
    sat_nav = (1.0 + sat_ret).cumprod()
    sat_dd = sat_nav / sat_nav.cummax() - 1.0

    # Bloc contributions
    active = active_funds.copy()
    active['date'] = pd.to_datetime(active['date'], errors='coerce')
    active = active.dropna(subset=['date', 'bloc', 'ticker', 'weight_ticker']).copy()
    active = active[active['date'].isin(common_dates) & active['ticker'].isin(common_tickers)].copy()

    ret_long = r.stack().rename('ret_ticker').reset_index()
    ret_long.columns = ['date', 'ticker', 'ret_ticker']
    m = active.merge(ret_long, on=['date', 'ticker'], how='left')
    m['ret_ticker'] = m['ret_ticker'].fillna(0.0)
    m['contrib_bloc'] = m['weight_ticker'] * m['ret_ticker']
    bloc_contrib = m.groupby(['date', 'bloc'], as_index=False)['contrib_bloc'].sum()
    bloc_contrib_piv = bloc_contrib.pivot(index='date', columns='bloc', values='contrib_bloc').fillna(0.0)
    for b in ['bloc1', 'bloc2', 'bloc3']:
        if b not in bloc_contrib_piv.columns:
            bloc_contrib_piv[b] = 0.0
    bloc_contrib_piv = bloc_contrib_piv[['bloc1', 'bloc2', 'bloc3']].sort_index()
    bloc_cum_contrib = bloc_contrib_piv.cumsum()

    # Metrics
    sat_metrics = pd.DataFrame({
        'Ret ann.': [ann_return(sat_ret)],
        'Vol ann.': [ann_vol(sat_ret)],
        'Sharpe (rf=0)': [(ann_return(sat_ret) / ann_vol(sat_ret))
                          if pd.notna(ann_vol(sat_ret)) and ann_vol(sat_ret) > 0 else np.nan],
        'Max DD': [sat_dd.min()],
    }, index=['Satellite dynamique'])

    ann_sat = sat_ret.groupby(sat_ret.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    ann_bloc_contrib = bloc_contrib_piv.groupby(bloc_contrib_piv.index.year).sum()

    beta_window = 126
    sat_beta_rolling = (
        sat_ret.rolling(beta_window).cov(core_ret)
        / core_ret.rolling(beta_window).var().replace(0, np.nan)
    )

    # Display
    print('Métriques Satellite dynamique :')
    display(sat_metrics.style.format({
        'Ret ann.': '{:+.2%}', 'Vol ann.': '{:.2%}',
        'Sharpe (rf=0)': '{:.2f}', 'Max DD': '{:.2%}',
    }))
    print('\nReturns annuels Satellite :')
    display(ann_sat.to_frame('ret_ann').style.format('{:+.2%}'))
    print('\nContribution annuelle par bloc :')
    display(ann_bloc_contrib.style.format('{:+.2%}'))
    print(f'\nFenêtre : {common_dates.min().date()} → {common_dates.max().date()} ({len(common_dates)} obs)')
    print(f'Beta rolling moyen: {sat_beta_rolling.mean():.3f} | '
          f'min: {sat_beta_rolling.min():.3f} | max: {sat_beta_rolling.max():.3f}')

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(sat_nav.index, sat_nav, lw=2.2, color='#1f77b4')
    axes[0, 0].set_title('Satellite dynamique — NAV cumulée')
    axes[0, 0].set_ylabel('NAV (base 1)')
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].fill_between(sat_dd.index, sat_dd.values * 100, 0, color='#d62728', alpha=0.35)
    axes[0, 1].plot(sat_dd.index, sat_dd.values * 100, lw=1.5, color='#d62728')
    axes[0, 1].set_title('Satellite dynamique — Drawdown')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(alpha=0.25)

    roll_vol = sat_ret.rolling(63).std() * np.sqrt(252)
    axes[1, 0].plot(roll_vol.index, roll_vol * 100, lw=1.8, color='#2ca02c')
    axes[1, 0].set_title('Satellite dynamique — Vol rolling 63j')
    axes[1, 0].set_ylabel('Vol annualisée (%)')
    axes[1, 0].grid(alpha=0.25)

    for b, c in [('bloc1', '#1f77b4'), ('bloc2', '#ff7f0e'), ('bloc3', '#2ca02c')]:
        axes[1, 1].plot(bloc_cum_contrib.index, bloc_cum_contrib[b] * 100, lw=1.8, label=b, color=c)
    axes[1, 1].axhline(0, color='black', lw=0.8)
    axes[1, 1].set_title('Contribution cumulée au rendement satellite')
    axes[1, 1].set_ylabel('Contribution cumulée (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Annual returns bar
    fig2, ax = plt.subplots(figsize=(10, 4.5))
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in ann_sat.values]
    ax.bar(ann_sat.index.astype(str), ann_sat.values * 100, color=colors, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('Satellite dynamique — Returns annuels')
    ax.set_ylabel('Return annuel (%)')
    ax.grid(alpha=0.25, axis='y')
    plt.tight_layout()
    plt.show()

    # Beta rolling
    fig3, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sat_beta_rolling.index, sat_beta_rolling, color='#6f42c1', lw=1.8,
            label=f'Beta rolling {beta_window}j')
    ax.axhline(0, color='black', lw=0.8)
    if sat_beta_rolling.notna().any():
        ax.axhline(sat_beta_rolling.mean(), color='#ff7f0e', lw=1.2, ls='--',
                   label=f"Moyenne: {sat_beta_rolling.mean():.2f}")
    ax.set_title('Satellite dynamique — Beta rolling vs portefeuille Core choisi')
    ax.set_ylabel('Beta')
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        'sat_ret': sat_ret, 'sat_nav': sat_nav, 'sat_dd': sat_dd,
        'sat_metrics': sat_metrics, 'ann_sat': ann_sat,
        'bloc_contrib_piv': bloc_contrib_piv, 'ann_bloc_contrib': ann_bloc_contrib,
        'sat_beta_rolling': sat_beta_rolling, 'core_ret': core_ret,
        'common_dates': common_dates,
    }


# ── Turnover & switching diagnostics (cell 26) ──────────────────────────────

def analyze_satellite_turnover(weights_ticker_daily, active_funds_daily, quarter_selection_df):
    """
    Satellite turnover analysis: daily turnover, quarterly switching,
    annual aggregates, and diagnostic charts.

    Returns dict with turnover series and summaries.
    """
    w_ticker = weights_ticker_daily.copy().sort_index().fillna(0.0)
    active_daily = active_funds_daily.copy().rename(columns={"Strat": "strat"})
    sel_q = quarter_selection_df.copy().rename(columns={"Strat": "strat"})

    # Daily turnover
    turnover_target_daily = 0.5 * (w_ticker.diff().abs().sum(axis=1)).fillna(0.0)
    turnover_target_daily.name = "turnover_target_daily"

    w_exec = w_ticker.shift(1).fillna(0.0)
    turnover_exec_daily = 0.5 * (w_exec.diff().abs().sum(axis=1)).fillna(0.0)
    turnover_exec_daily.name = "turnover_exec_daily"

    # Quarter change dates
    sort_cols = [c for c in ["date", "strat"] if c in active_daily.columns]
    if sort_cols:
        active_daily = active_daily.sort_values(sort_cols)
    quarter_key_by_day = active_daily.groupby("date")["quarter_date"].first().sort_index()
    quarter_change_dates = quarter_key_by_day.index[
        quarter_key_by_day.ne(quarter_key_by_day.shift(1)).fillna(False)
    ]
    quarter_change_dates = pd.DatetimeIndex(
        [d for d in quarter_change_dates if d in turnover_target_daily.index]
    )

    # Quarterly turnover
    rows = []
    for qd in quarter_change_dates:
        qd = pd.Timestamp(qd)
        t_target = float(turnover_target_daily.loc[qd]) if qd in turnover_target_daily.index else np.nan
        pos = turnover_exec_daily.index.searchsorted(qd, side="right")
        if pos < len(turnover_exec_daily.index):
            exec_date = turnover_exec_daily.index[pos]
            t_exec = float(turnover_exec_daily.iloc[pos])
        else:
            exec_date = pd.NaT
            t_exec = np.nan
        rows.append({
            "quarter_change_date": qd, "quarter": str(qd.to_period("Q")),
            "year": int(qd.year), "turnover_target_on_change": t_target,
            "exec_date": exec_date, "turnover_exec_next_day": t_exec,
        })
    turnover_quarterly = pd.DataFrame(rows)
    if not turnover_quarterly.empty:
        turnover_quarterly = turnover_quarterly.sort_values("quarter_change_date").reset_index(drop=True)

    # Switches
    if "quarter_date" in sel_q.columns:
        sel_q["quarter_date"] = pd.to_datetime(sel_q["quarter_date"], errors="coerce")
    date_col = "quarter_date" if "quarter_date" in sel_q.columns else None
    strat_col = "strat" if "strat" in sel_q.columns else None

    if date_col and strat_col and "switch_flag" in sel_q.columns:
        sw = sel_q.copy()
        sw["switch_flag"] = pd.to_numeric(sw["switch_flag"], errors="coerce").fillna(0).astype(int)
        sw = sw.dropna(subset=[date_col])
        sw["quarter"] = sw[date_col].dt.to_period("Q").astype(str)
        switches_by_quarter = (
            sw.groupby(["quarter", strat_col], dropna=False)["switch_flag"].sum()
            .reset_index().rename(columns={strat_col: "strat", "switch_flag": "n_switches"})
        )
        switches_total_quarter = (
            sw.groupby("quarter", dropna=False)["switch_flag"].sum().reset_index(name="n_switches_total")
        )
    else:
        switches_by_quarter = pd.DataFrame(columns=["quarter", "strat", "n_switches"])
        switches_total_quarter = pd.DataFrame(columns=["quarter", "n_switches_total"])

    # Annual aggregates
    turnover_annual = (
        pd.concat([turnover_target_daily, turnover_exec_daily], axis=1)
        .assign(year=lambda x: x.index.year)
        .groupby("year")
        .agg(
            turnover_target_sum=("turnover_target_daily", "sum"),
            turnover_exec_sum=("turnover_exec_daily", "sum"),
            turnover_target_mean=("turnover_target_daily", "mean"),
            turnover_exec_mean=("turnover_exec_daily", "mean"),
        )
    )

    # Merge turnover with switches
    if not switches_total_quarter.empty and not turnover_quarterly.empty:
        turnover_switch_quarter = turnover_quarterly.merge(
            switches_total_quarter, on="quarter", how="left"
        )
        turnover_switch_quarter["n_switches_total"] = (
            turnover_switch_quarter["n_switches_total"].fillna(0).astype(int)
        )
    else:
        turnover_switch_quarter = turnover_quarterly.copy()
        turnover_switch_quarter["n_switches_total"] = 0

    # ----- Figures -----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    axes[0, 0].plot(turnover_exec_daily.index, turnover_exec_daily.values, lw=1.0, alpha=0.65,
                    label="Daily turnover (exec)")
    axes[0, 0].plot(turnover_exec_daily.index,
                    turnover_exec_daily.rolling(21, min_periods=5).mean().values,
                    lw=1.8, label="21d moving avg")
    axes[0, 0].set_title("Satellite Turnover Quotidien (exécuté)")
    axes[0, 0].set_ylabel("Turnover (0-1)")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)

    if not turnover_quarterly.empty:
        x = np.arange(len(turnover_quarterly))
        bw = 0.4
        axes[0, 1].bar(x - bw / 2, turnover_quarterly["turnover_target_on_change"],
                       width=bw, alpha=0.85, label="Target (change day)")
        axes[0, 1].bar(x + bw / 2, turnover_quarterly["turnover_exec_next_day"],
                       width=bw, alpha=0.85, label="Executed (next day)")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(turnover_quarterly["quarter"], rotation=90)
        axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].set_title("Turnover autour des changements trimestriels")
    axes[0, 1].set_ylabel("Turnover (0-1)")
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    if not turnover_annual.empty:
        x = np.arange(len(turnover_annual))
        bw = 0.4
        axes[1, 0].bar(x - bw / 2, turnover_annual["turnover_target_sum"],
                       width=bw, alpha=0.85, label="Target")
        axes[1, 0].bar(x + bw / 2, turnover_annual["turnover_exec_sum"],
                       width=bw, alpha=0.85, label="Executed")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(turnover_annual.index.astype(str))
        axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_title("Turnover annuel cumulé")
    axes[1, 0].set_ylabel("Somme annuelle")
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    if not switches_by_quarter.empty:
        sw_pivot = switches_by_quarter.pivot(
            index="quarter", columns="strat", values="n_switches"
        ).fillna(0)
        sw_pivot.plot(kind="bar", stacked=True, ax=axes[1, 1], alpha=0.9)
        axes[1, 1].legend(frameon=False, fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No switch data", ha="center", va="center")
    axes[1, 1].set_title("Switches par trimestre (par STRAT)")
    axes[1, 1].set_ylabel("# Switches")
    axes[1, 1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Satellite Dynamic — Turnover & Switching Diagnostics", fontsize=14, y=1.02)
    plt.show()

    # Scatter: turnover vs switches
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if not turnover_switch_quarter.empty:
        ax2.scatter(turnover_switch_quarter["n_switches_total"],
                    turnover_switch_quarter["turnover_exec_next_day"], alpha=0.8)
        for _, row in turnover_switch_quarter.iterrows():
            ax2.annotate(str(row["quarter"]),
                         (row["n_switches_total"], row["turnover_exec_next_day"]),
                         fontsize=7, alpha=0.8)
    ax2.set_xlabel("# switches (quarter)")
    ax2.set_ylabel("Executed turnover (next day)")
    ax2.set_title("Executed Turnover vs Number of Switches (Quarterly)")
    ax2.grid(True, alpha=0.25)
    plt.show()

    # Console summary
    print("\n=== Turnover Summary ===")
    print(f"Mean daily turnover (target): {turnover_target_daily.mean():.4f}")
    print(f"Mean daily turnover (executed): {turnover_exec_daily.mean():.4f}")
    print(f"Median daily turnover (executed): {turnover_exec_daily.median():.4f}")
    if not turnover_quarterly.empty:
        print(f"Mean turnover on quarter change (target): "
              f"{turnover_quarterly['turnover_target_on_change'].mean():.4f}")
        print(f"Mean turnover next day (executed): "
              f"{turnover_quarterly['turnover_exec_next_day'].mean():.4f}")

    return {
        'turnover_target_daily': turnover_target_daily,
        'turnover_exec_daily': turnover_exec_daily,
        'turnover_quarterly': turnover_quarterly,
        'turnover_annual': turnover_annual,
        'switches_by_quarter': switches_by_quarter,
        'turnover_switch_quarter': turnover_switch_quarter,
    }
