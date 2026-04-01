"""Core ETF selection: load blocks, rebase, enhanced tables, plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.core_data import CoreConfig, lire_theme


# ── Helpers ──────────────────────────────────────────────────────────────────

def first_valid_date(series):
    clean = series.dropna()
    if clean.empty:
        return pd.NaT
    return pd.Timestamp(clean.index.min())


def load_block(cfg, theme_name, sheet_prices, sheet_meta, start_date):
    prices, meta = lire_theme(cfg, theme_name, sheet_prices, sheet_meta, verbose=False)
    prices = prices.copy().sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    first_dates = prices.apply(first_valid_date, axis=0)
    eligible = first_dates[first_dates.notna() & (first_dates <= start_date)].index.tolist()
    prices = prices[eligible].loc[prices.index >= start_date].dropna(axis=1, how="all")
    meta = meta.reindex(prices.columns).copy()
    first_dates = first_dates.reindex(prices.columns)
    return prices, meta, first_dates


def robust_base(series, window=5):
    clean = series.dropna()
    if clean.empty:
        return np.nan
    baseline = clean.iloc[:window].median()
    if pd.isna(baseline) or baseline <= 0:
        positive = clean[clean > 0]
        if positive.empty:
            return np.nan
        baseline = positive.iloc[:window].median()
    return float(baseline)


def rebase_to_100(prices):
    rebased = prices.copy()
    for col in rebased.columns:
        base = robust_base(rebased[col])
        if pd.isna(base) or base <= 0:
            rebased[col] = np.nan
            continue
        rebased[col] = rebased[col] / base * 100.0
    return rebased


def enhanced_table(prices, meta, first_dates):
    """Vol annualisée + return annualisé + returns par année 2019-2025."""
    returns = prices.pct_change(fill_method=None)
    vol_ann = returns.std(skipna=True) * np.sqrt(252.0)
    n_obs = returns.notna().sum()

    total_days = returns.notna().sum()
    cum_ret = (1 + returns).prod() - 1
    years_frac = total_days / 252.0
    ann_ret = (1 + cum_ret) ** (1 / years_frac.replace(0, np.nan)) - 1

    meta_cols = [c for c in ["nom", "exposition", "ter_pct", "devise", "provider"] if c in meta.columns]
    table = meta.reindex(vol_ann.index)[meta_cols].copy() if meta_cols else pd.DataFrame(index=vol_ann.index)
    table.insert(0, "ticker", vol_ann.index)
    table["first_valid_date"] = pd.to_datetime(first_dates.reindex(vol_ann.index)).dt.strftime("%Y-%m-%d").values
    table["n_obs_returns"] = n_obs.values
    table["vol_annualisee"] = vol_ann.values
    table["ret_annualise"] = ann_ret.values

    for year in range(2019, 2026):
        mask = returns.index.year == year
        yr_ret = returns.loc[mask]
        if yr_ret.empty:
            table[f"ret_{year}"] = np.nan
        else:
            table[f"ret_{year}"] = ((1 + yr_ret).prod() - 1).values

    table = table.sort_values("vol_annualisee", ascending=False, na_position="last").reset_index(drop=True)
    return table


# ── Main runner ──────────────────────────────────────────────────────────────

def load_all_core_blocks(project_root, core_tickers, start_date):
    """
    Load all Core ETF blocks, compute rebased prices & enhanced tables.

    Returns
    -------
    block_data : dict  {block_name: {prices, rebased, table, first_dates, color, meta}}
    cfg : CoreConfig
    excel_path : Path
    """
    excel_candidates = [
        project_root / "data" / "univers_core_etf_eur_daily_wide.xlsx",
        project_root / "univers_core_etf_eur_daily_wide_VF.xlsx",
    ]
    excel_path = next((p for p in excel_candidates if p.exists()), None)
    if excel_path is None:
        raise FileNotFoundError("Fichier Core introuvable dans data/ ou à la racine du projet.")

    cfg = CoreConfig(core_excel=excel_path)

    BLOCKS = {
        "Equity": {
            "sheet_prices": cfg.sheet_equity_prices,
            "sheet_meta": cfg.sheet_equity_meta,
            "color": "#1f77b4",
        },
        "Rates": {
            "sheet_prices": cfg.sheet_rates_prices,
            "sheet_meta": cfg.sheet_rates_meta,
            "color": "#ff7f0e",
        },
        "Credit": {
            "sheet_prices": cfg.sheet_credit_prices,
            "sheet_meta": cfg.sheet_credit_meta,
            "color": "#2ca02c",
        },
    }

    block_data = {}
    for block_name, params in BLOCKS.items():
        prices, meta, fd = load_block(cfg, block_name, params["sheet_prices"], params["sheet_meta"], start_date)
        if prices.empty:
            raise ValueError(f"Aucun indice du bloc {block_name} ne commence au plus tard le {start_date.date()}.")
        block_data[block_name] = {
            "prices": prices,
            "rebased": rebase_to_100(prices),
            "table": enhanced_table(prices, meta, fd),
            "first_dates": fd,
            "color": params["color"],
            "meta": meta,
        }

    return block_data, cfg, excel_path


def build_core_variables(block_data, core_equity, core_rates, core_credit, core_tickers):
    """
    Produce core_df, core_3_log, core_3_simple from block_data.

    Returns
    -------
    core_df : pd.DataFrame   metadata of the 3 selected ETFs
    core_3_log : pd.DataFrame  log returns
    core_3_simple : pd.DataFrame  simple returns
    """
    _core_map = {"Equity": core_equity, "Rates": core_rates, "Credit": core_credit}
    _rows = []
    for theme, ticker in _core_map.items():
        meta = block_data[theme]["meta"]
        r = meta.loc[ticker].to_dict() if ticker in meta.index else {}
        r["Theme"] = theme
        r["Ticker"] = ticker
        _rows.append(r)
    core_df = pd.DataFrame(_rows)
    for col in ["nom", "provider", "isin", "devise", "ter_pct", "encours_eur_m", "exposition"]:
        if col not in core_df.columns:
            core_df[col] = np.nan

    _all_prices = pd.concat([block_data[t]["prices"] for t in ["Equity", "Rates", "Credit"]], axis=1)
    _sel_prices = _all_prices[core_tickers].dropna()
    core_3_simple = _sel_prices.pct_change(fill_method=None).dropna()
    core_3_log = np.log1p(core_3_simple)

    return core_df, core_3_log, core_3_simple


def display_core_blocks(block_data, core_tickers, excel_path, start_date):
    """Print block info and display enhanced tables per block."""
    from IPython.display import display as _display

    print(f'Fichier source : {excel_path}')
    print(f"Fenêtre d'analyse : {start_date.date()} → aujourd'hui")
    print(f"Contrainte : première date valide ≤ {start_date.date()}")
    for bname, bdata in block_data.items():
        n = bdata['prices'].shape[1]
        d0 = bdata['prices'].index.min().date()
        d1 = bdata['prices'].index.max().date()
        print(f'  {bname}: {n} ETF retenus | {d0} → {d1}')
    print(f'\nETF Core sélectionnés : {", ".join(core_tickers)}')

    fmt_pct = {c: '{:.2%}' for c in ['vol_annualisee', 'ret_annualise'] + [f'ret_{y}' for y in range(2019, 2026)]}
    fmt_num = {'ter_pct': '{:.2f}'}
    fmt_all = {**fmt_pct, **fmt_num}

    for block_name, data in block_data.items():
        print('=' * 130)
        print(f'Tableau — bloc {block_name} (depuis 2019) : volatilité + returns annualisés + returns par année')
        print('=' * 130)
        _display(data['table'].style.format(fmt_all))


def plot_core_blocks(block_data, core_tickers, fig_dir=None):
    """Plot 3 subplots with all ETFs per block, highlight selected Core ETFs."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 18), sharex=True, constrained_layout=True)
    for ax, (block_name, data) in zip(axes, block_data.items()):
        rebased = data["rebased"]
        for ticker in rebased.columns:
            s = rebased[ticker].dropna()
            if len(s) == 0:
                continue
            lw = 2.2 if ticker in core_tickers else 1.0
            alpha = 1.0 if ticker in core_tickers else 0.7
            ax.plot(s.index, s.values, linewidth=lw, alpha=alpha, label=ticker)
        ax.set_title(f"Bloc Core {block_name} — tous les ETF depuis 2019", fontsize=13, fontweight="bold")
        ax.set_ylabel("Base 100")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False, ncol=1)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Core — évolution de tous les ETF par bloc depuis 2019", fontsize=16, y=1.01)

    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(fig_dir) / "core_blocs_tous_etf_depuis_2019.png", dpi=160, bbox_inches="tight")

    return fig
