"""Satellite fee analysis: TER loading, weighted TER computation, fee charts."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path


def load_info(path):
    """Load a single STRAT info Excel file with ticker, TER and name."""
    df = pd.read_excel(path)[["Ticker", "Ratio des dépenses", "Nom"]].copy()
    df.columns = ["ticker", "ter_pct", "nom"]
    df["ter_pct"] = pd.to_numeric(df["ter_pct"], errors="coerce")
    return df


def load_fund_info(data_dir="data"):
    """Load and merge fund info from STRAT1/2/3_info.xlsx files."""
    def _load_info(path):
        return load_info(path)

    info_all = pd.concat([
        _load_info(Path(data_dir) / "STRAT1_info.xlsx"),
        _load_info(Path(data_dir) / "STRAT2_info.xlsx"),
        _load_info(Path(data_dir) / "STRAT3_info.xlsx"),
    ], ignore_index=True).drop_duplicates("ticker")
    info_all["ter_dec"] = info_all["ter_pct"] / 100.0
    return info_all


def short_name(row):
    nom = str(row.get("nom", "")).strip()
    if nom and nom.lower() not in {"nan", "none", ""}:
        return nom[:30]
    return row["ticker"]


def analyze_satellite_fees(weights_ticker_daily, quarter_selection_df, data_dir="data",
                           outdir="outputs", budget_bps=80):
    """
    Full satellite fee analysis.

    Returns dict with: info_all, sel_info, summary, ter_wavg_daily,
    ter_wavg_monthly, strat_ter_monthly, contrib_df, avg_contrib, STRAT_TICKERS
    """
    info_all = load_fund_info(data_dir)

    sel = quarter_selection_df.copy()
    sel = sel.rename(columns={"Strat": "strat", "Ticker": "ticker"}) if "Strat" in sel.columns else sel
    if "strat" not in sel.columns and "Strat" in sel.columns:
        sel = sel.rename(columns={"Strat": "strat"})
    sel["quarter_date"] = pd.to_datetime(sel["quarter_date"])
    sel_info = sel.merge(info_all[["ticker", "ter_pct", "ter_dec", "nom"]], on="ticker", how="left")
    sel_info["label"] = sel_info.apply(short_name, axis=1)

    # Impute missing TER
    strat_col = "strat" if "strat" in sel_info.columns else "Strat"
    for strat, grp in sel_info.groupby(strat_col):
        med = grp["ter_pct"].median(skipna=True)
        mask = sel_info[strat_col] == strat
        sel_info.loc[mask, "ter_pct"] = sel_info.loc[mask, "ter_pct"].fillna(med)
        sel_info.loc[mask, "ter_dec"] = sel_info.loc[mask, "ter_dec"].fillna(med / 100.0)

    summary = (
        sel_info.groupby(["ticker", "nom", strat_col])
        .agg(ter_pct=("ter_pct", "first"), n_quarters=("quarter_date", "count"))
        .reset_index()
        .sort_values([strat_col, "ter_pct"], ascending=[True, False])
    )

    # Weighted TER daily
    w = weights_ticker_daily.copy().sort_index()
    ter_map = info_all.set_index("ticker")["ter_dec"].to_dict()
    tickers_w = w.columns.tolist()
    ter_vec = pd.Series({t: ter_map.get(t, np.nan) for t in tickers_w})
    med_global = ter_vec.median(skipna=True)
    ter_vec = ter_vec.fillna(med_global)

    ter_wavg_daily = (w * ter_vec.values).sum(axis=1) * 10000
    ter_wavg_monthly = ter_wavg_daily.resample("ME").mean()

    # Contributions
    contrib_df = w * ter_vec.values * 10000
    avg_contrib = contrib_df.mean().sort_values(ascending=False)
    avg_contrib = avg_contrib[avg_contrib > 0]

    # Per-strat TER
    STRAT_TICKERS = {}
    for strat in ["STRAT1", "STRAT2", "STRAT3"]:
        tickers_strat = sel_info[sel_info[strat_col] == strat]["ticker"].unique().tolist()
        STRAT_TICKERS[strat] = [t for t in tickers_strat if t in w.columns]

    strat_ter_monthly = {}
    for strat, tickers_s in STRAT_TICKERS.items():
        if not tickers_s:
            continue
        strat_ter_monthly[strat] = (w[tickers_s] * ter_vec[tickers_s].values * 10000).sum(axis=1).resample("ME").mean()

    return {
        "info_all": info_all,
        "sel_info": sel_info,
        "summary": summary,
        "ter_wavg_daily": ter_wavg_daily,
        "ter_wavg_monthly": ter_wavg_monthly,
        "strat_ter_monthly": strat_ter_monthly,
        "contrib_df": contrib_df,
        "avg_contrib": avg_contrib,
        "STRAT_TICKERS": STRAT_TICKERS,
        "ter_vec": ter_vec,
    }


def plot_satellite_fees(summary, ter_wavg_monthly, ter_wavg_daily_mean, strat_ter_monthly,
                        avg_contrib, sel_info, info_all, budget_bps=80, fig_dir=None):
    """Plot 4-panel satellite fee dashboard. Returns fig."""
    PALETTE = {"STRAT1": "#3B82F6", "STRAT2": "#F59E0B", "STRAT3": "#10B981"}
    BAR_COLOR = "#6366F1"

    strat_col = "strat" if "strat" in sel_info.columns else "Strat"

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Analyse des Frais — Poche Satellite", fontsize=15, fontweight="bold", y=1.01)

    # TER per fund
    ax = axes[0, 0]
    _s = summary.sort_values("ter_pct", ascending=True).reset_index(drop=True)
    colors_bar = [PALETTE.get(s, "#94A3B8") for s in _s[strat_col]]
    bars = ax.barh(_s["ticker"], _s["ter_pct"], color=colors_bar, edgecolor="white", height=0.7)
    for bar, val in zip(bars, _s["ter_pct"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", ha="left", fontsize=8)
    ax.set_xlabel("TER (% par an)")
    ax.set_title("TER par fonds sélectionné", fontweight="bold")
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)

    # Dynamic weighted TER
    ax = axes[0, 1]
    ax.fill_between(ter_wavg_monthly.index, ter_wavg_monthly.values, alpha=0.25, color=BAR_COLOR)
    ax.plot(ter_wavg_monthly.index, ter_wavg_monthly.values, color=BAR_COLOR, lw=2, label="TER pondéré satellite (bps/an)")
    ax.axhline(ter_wavg_daily_mean, color=BAR_COLOR, lw=1.2, ls="--", label=f"Moyenne : {ter_wavg_daily_mean:.1f} bps")
    ax.axhline(budget_bps, color="crimson", lw=1.2, ls=":", label=f"Budget total : {budget_bps} bps")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.set_ylabel("bps / an")
    ax.set_title("TER pondéré dynamique de la poche Satellite", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Stacked strat contribution
    ax = axes[1, 0]
    strat_keys = [k for k in ["STRAT1", "STRAT2", "STRAT3"] if k in strat_ter_monthly]
    if strat_keys:
        strat_stack = pd.DataFrame(strat_ter_monthly)[strat_keys].fillna(0)
        bottom = np.zeros(len(strat_stack))
        for strat in strat_keys:
            ax.bar(strat_stack.index, strat_stack[strat], bottom=bottom, color=PALETTE[strat], label=strat, width=20)
            bottom += strat_stack[strat].values
    ax.set_ylabel("bps / an")
    ax.set_title("Contribution TER par Bloc (empilé)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Top 10 fund contributions
    ax = axes[1, 1]
    label_map = info_all.set_index("ticker")["nom"].to_dict()

    def get_label(t):
        n = str(label_map.get(t, t)).strip()
        return n[:25] if n and n.lower() not in {"nan", "none", ""} else t

    top_contrib = avg_contrib.head(10)
    labels_top = [get_label(t) for t in top_contrib.index]
    colors_top = []
    for t in top_contrib.index:
        strat_of_t = sel_info[sel_info["ticker"] == t][strat_col].values
        s = strat_of_t[0] if len(strat_of_t) > 0 else "STRAT1"
        colors_top.append(PALETTE.get(s, "#94A3B8"))
    bars2 = ax.barh(labels_top[::-1], top_contrib.values[::-1], color=colors_top[::-1], edgecolor="white", height=0.7)
    for bar, val in zip(bars2, top_contrib.values[::-1]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.1f} bps", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Contribution moyenne (bps / an)")
    ax.set_title("Contribution TER moyenne par fonds", fontweight="bold")
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    if fig_dir is not None:
        fig_path = Path(fig_dir) / "satellite_fees_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {fig_path}")

    return fig


# ── All-in-one runner (replaces cell 30 inline code) ─────────────────────────

def run_and_display_satellite_fees(weights_ticker_daily, quarter_selection_df,
                                   budget_bps=80, outdir='outputs'):
    """
    Run full satellite fee analysis, display results, plot dashboard, export CSVs.

    Returns dict with all fee results (including ter_wavg_daily for downstream use).
    """
    from IPython.display import display as _display

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    fig_dir = outdir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fee = analyze_satellite_fees(weights_ticker_daily, quarter_selection_df,
                                 data_dir='data', outdir=str(outdir), budget_bps=budget_bps)

    summary = fee['summary']
    sel_info = fee['sel_info']
    ter_wavg_daily = fee['ter_wavg_daily']
    ter_wavg_monthly = fee['ter_wavg_monthly']
    strat_ter_monthly = fee['strat_ter_monthly']
    avg_contrib = fee['avg_contrib']
    info_all = fee['info_all']
    ter_vec = fee['ter_vec']
    w = weights_ticker_daily.copy().sort_index()

    # Summary table
    strat_col = 'strat' if 'strat' in summary.columns else 'Strat'
    print("═" * 65)
    print("FONDS SATELLITE SÉLECTIONNÉS — TER (Ratio des dépenses)")
    print("═" * 65)
    show_cols = [c for c in [strat_col, 'ticker', 'nom', 'ter_pct', 'n_quarters'] if c in summary.columns]
    print(summary[show_cols].to_string(index=False))

    print(f"\nTER pondéré satellite moyen (toute la période) : {ter_wavg_daily.mean():.1f} bps/an")
    print(f"Min : {ter_wavg_daily.min():.1f} bps  |  Max : {ter_wavg_daily.max():.1f} bps")

    # Plot
    fig = plot_satellite_fees(
        summary, ter_wavg_monthly, ter_wavg_daily.mean(),
        strat_ter_monthly, avg_contrib, sel_info, info_all,
        budget_bps=budget_bps, fig_dir=str(fig_dir),
    )
    plt.show()

    # Budget check
    w_sat_total = w.sum(axis=1).mean()
    w_core_total = 1.0 - w_sat_total
    ter_core_bps = 9.0
    ter_sat_bps = ter_wavg_daily.mean()
    ter_total_bps = ter_core_bps * w_core_total + ter_sat_bps * w_sat_total

    print("\n" + "═" * 65)
    print("SYNTHÈSE DES FRAIS SATELLITE")
    print("═" * 65)
    print(f"  Poids moyen satellite dans portefeuille : {w_sat_total*100:.2f}%")
    print(f"  TER pondéré satellite moyen             : {ter_sat_bps:.1f} bps/an")
    print(f"  TER Core (estimation)                   : {ter_core_bps:.1f} bps/an")
    print(f"  TER total portefeuille (estimé)          : {ter_total_bps:.1f} bps/an")
    print(f"  Budget frais                             : {budget_bps} bps/an")
    budget_ok = ter_total_bps <= budget_bps
    print(f"  → Budget {'RESPECTÉ ✓' if budget_ok else 'DÉPASSÉ ✗'}")
    print("═" * 65)

    # Quarterly table
    q_dates = pd.to_datetime(sel_info['quarter_date'].unique())
    rows = []
    for qd in sorted(q_dates):
        q_end = qd + pd.offsets.QuarterEnd(0)
        q_mask = (w.index >= qd) & (w.index <= q_end)
        if not q_mask.any():
            continue
        w_q = w.loc[q_mask].mean()
        ter_q = (w_q * ter_vec).sum() * 10_000
        rows.append({'quarter': str(qd.to_period('Q')), 'ter_wavg_bps': round(ter_q, 1),
                     'n_fonds_actifs': int((w_q > 0.001).sum())})
    df_qter = pd.DataFrame(rows)
    print("\nTER pondéré satellite par trimestre:")
    print(df_qter.to_string(index=False))

    # Exports
    df_qter.to_csv(outdir / 'satellite_fees_quarterly.csv', index=False)
    sel_info.to_csv(outdir / 'satellite_fees_detail.csv', index=False)
    print(f"\nCSV sauvegardés : satellite_fees_quarterly.csv, satellite_fees_detail.csv")

    fee['ter_total_bps'] = ter_total_bps
    fee['budget_ok'] = budget_ok
    return fee
