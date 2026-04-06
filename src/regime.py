"""Regime detection & grid optimization for the Core portfolio."""

from itertools import product
import time

import numpy as np
import pandas as pd

from src.utils import rolling_zscore, apply_min_regime_days_causal, max_drawdown_from_returns


# ── Regime construction ──────────────────────────────────────────────────────

def build_regime(daily_returns_df, core_tickers, core_equity, core_rates,
                 core_weights, core_ter_map_pct, regime_cfg):
    """
    Build regime indicator from Core returns.

    Returns
    -------
    base_full : pd.DataFrame  full IS+OOS data with indicators & regime
    base_df : pd.DataFrame    OOS-only slice
    regime_output : pd.DataFrame  exported regime series
    regime_stats_df : pd.DataFrame  performance stats per regime
    coherence_df : pd.DataFrame  economic coherence check
    first_oos : pd.Timestamp
    switches_raw : int
    switches_smooth : int
    q_low_is : float
    q_high_is : float
    """
    required_cols = core_tickers
    full_log = daily_returns_df[required_cols].copy().sort_index()
    full_simple = np.expm1(full_log)

    ter_annual_decimal = pd.Series({t: core_ter_map_pct.get(t, 0.0) / 100.0 for t in required_cols})
    ter_daily_drag_full = (1.0 + ter_annual_decimal) ** (1.0 / 252.0) - 1.0

    full_net = full_simple.copy()
    for t in required_cols:
        full_net[t] = (1.0 + full_simple[t]) / (1.0 + ter_daily_drag_full[t]) - 1.0

    core_weights_series = pd.Series(core_weights)
    core_ret_full = (full_net[required_cols] * core_weights_series.reindex(required_cols).values).sum(axis=1)
    core_ret_full = core_ret_full.rename("core_ret")

    equity_ret_full = full_net[core_equity].rename("equity_ret")
    bonds_ret_full = full_net[core_rates].rename("bonds_ret")

    base_full = pd.concat([core_ret_full, equity_ret_full, bonds_ret_full], axis=1).dropna()
    base_full = base_full.loc[regime_cfg["is_start"]:].copy()

    if base_full.empty:
        raise ValueError("Base regime vide apres filtrage des dates.")

    # Indicators
    win = int(regime_cfg["indicator_window_days"])
    base_full["vol_63"] = base_full["core_ret"].rolling(win, min_periods=win).std() * np.sqrt(252)
    base_full["corr_eq_bond_63"] = base_full["equity_ret"].rolling(win, min_periods=win).corr(base_full["bonds_ret"])
    base_full["mom_63"] = (1.0 + base_full["core_ret"]).rolling(win, min_periods=win).apply(np.prod, raw=True) - 1.0

    # Drawdown indicator
    nav = (1.0 + base_full["core_ret"]).cumprod()
    rolling_peak = nav.rolling(win, min_periods=win).max()
    base_full["dd_63"] = nav / rolling_peak - 1.0

    # Z-scores
    z_win = int(regime_cfg["zscore_window_days"])
    min_p = int(regime_cfg["zscore_min_periods"])
    base_full["vol_z"] = rolling_zscore(base_full["vol_63"], z_win=z_win, min_p=min_p)
    base_full["corr_z"] = rolling_zscore(base_full["corr_eq_bond_63"], z_win=z_win, min_p=min_p)
    base_full["mom_z"] = rolling_zscore(base_full["mom_63"], z_win=z_win, min_p=min_p)
    base_full["dd_z"] = rolling_zscore(base_full["dd_63"], z_win=z_win, min_p=min_p)

    # Weighted score causal (component weights from regime_cfg)
    w_vol = float(regime_cfg.get("w_vol", 0.45))
    w_corr = float(regime_cfg.get("w_corr", 0.15))
    w_mom = float(regime_cfg.get("w_mom", 0.40))
    w_dd = float(regime_cfg.get("w_dd", 0.20))
    base_full["RegimeScore_raw"] = (
        w_vol * base_full["vol_z"]
        + w_corr * base_full["corr_z"]
        - w_mom * base_full["mom_z"]
        - w_dd * base_full["dd_z"]
    )
    base_full["RegimeScore"] = base_full["RegimeScore_raw"].shift(1)

    # Rolling thresholds
    low_q = float(regime_cfg["threshold_low_q"])
    high_q = float(regime_cfg["threshold_high_q"])
    thr_win = int(regime_cfg["rolling_threshold_days"])

    is_mask = (base_full.index >= pd.Timestamp(regime_cfg["is_start"])) & (base_full.index <= pd.Timestamp(regime_cfg["is_end"]))
    is_scores = base_full.loc[is_mask, "RegimeScore_raw"].dropna()
    if is_scores.empty:
        raise ValueError("Impossible de calibrer les seuils IS: RegimeScore indisponible sur IS.")

    q_low_is = float(is_scores.quantile(low_q))
    q_high_is = float(is_scores.quantile(high_q))

    base_full["q_low_roll"] = base_full["RegimeScore_raw"].rolling(thr_win, min_periods=1).quantile(low_q).shift(1)
    base_full["q_high_roll"] = base_full["RegimeScore_raw"].rolling(thr_win, min_periods=1).quantile(high_q).shift(1)
    base_full["q_low_roll"] = base_full["q_low_roll"].fillna(q_low_is)
    base_full["q_high_roll"] = base_full["q_high_roll"].fillna(q_high_is)

    # Regime classification
    base_full["Regime_raw"] = "Neutre"
    base_full.loc[base_full["RegimeScore"] <= base_full["q_low_roll"], "Regime_raw"] = "Risk-on"
    base_full.loc[base_full["RegimeScore"] >= base_full["q_high_roll"], "Regime_raw"] = "Stress"

    # Causal smoothing
    regime_before_smooth = base_full["Regime_raw"].copy()
    base_full["Regime"] = apply_min_regime_days_causal(
        base_full["Regime_raw"], min_days=int(regime_cfg["min_regime_days"])
    )

    switches_raw = int((regime_before_smooth != regime_before_smooth.shift(1)).sum() - 1)
    switches_smooth = int((base_full["Regime"] != base_full["Regime"].shift(1)).sum() - 1)

    # OOS slice
    base_df = base_full.loc[regime_cfg["oos_start"]:].copy()
    if base_df.empty:
        raise ValueError("Aucune donnee regime sur la periode OOS demandee.")

    first_oos = base_df.index.min()

    # Regime output
    regime_output = base_df[["RegimeScore", "Regime", "q_low_roll", "q_high_roll", "vol_63", "corr_eq_bond_63", "mom_63", "dd_63"]].copy()

    # Stats per regime
    stats_rows = []
    for regime in ["Stress", "Neutre", "Risk-on"]:
        r = base_df.loc[base_df["Regime"] == regime, "core_ret"].dropna()
        n = len(r)
        if n == 0:
            stats_rows.append({"Regime": regime, "N_obs": 0, "Perf_ann": np.nan, "Vol_ann": np.nan, "MaxDD": np.nan})
            continue
        perf_ann = float((1.0 + r).prod() ** (252.0 / n) - 1.0)
        vol_ann = float(r.std() * np.sqrt(252.0))
        mdd = max_drawdown_from_returns(r)
        stats_rows.append({"Regime": regime, "N_obs": n, "Perf_ann": perf_ann, "Vol_ann": vol_ann, "MaxDD": mdd})

    regime_stats_df = pd.DataFrame(stats_rows).set_index("Regime")

    # Coherence
    coherence_df = base_df.groupby("Regime")[["vol_63", "mom_63", "corr_eq_bond_63", "dd_63"]].mean()
    coherence_df = coherence_df.reindex(["Stress", "Neutre", "Risk-on"])

    return {
        "base_full": base_full,
        "base_df": base_df,
        "regime_output": regime_output,
        "regime_stats_df": regime_stats_df,
        "coherence_df": coherence_df,
        "first_oos": first_oos,
        "switches_raw": switches_raw,
        "switches_smooth": switches_smooth,
        "q_low_is": q_low_is,
        "q_high_is": q_high_is,
    }


def plot_regime(base_df, regime_cfg):
    """Plot regime score + discrete regime."""
    low_q = float(regime_cfg["threshold_low_q"])
    high_q = float(regime_cfg["threshold_high_q"])

    regime_num_map = {"Risk-on": -1, "Neutre": 0, "Stress": 1}
    regime_num = base_df["Regime"].map(regime_num_map)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    ax.plot(base_df.index, base_df["RegimeScore"], color="#1f77b4", linewidth=1.7, label="RegimeScore (ex-ante)")
    ax.plot(base_df.index, base_df["q_low_roll"], color="#2ca02c", linestyle="--", linewidth=1.2,
            label=f"Seuil bas rolling ({int(low_q * 100)}%)")
    ax.plot(base_df.index, base_df["q_high_roll"], color="#d62728", linestyle="--", linewidth=1.2,
            label=f"Seuil haut rolling ({int(high_q * 100)}%)")
    ax.set_title("RegimeScore Core (seuils rolling causaux calibres sur IS)")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.step(base_df.index, regime_num, where="post", color="#9467bd", linewidth=1.6)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Risk-on", "Neutre", "Stress"])
    ax.set_title(f"Regime discret lisse (causal, min {int(regime_cfg['min_regime_days'])} jours)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig


# ── Grid optimization ────────────────────────────────────────────────────────

TRADING_DAYS_PER_MONTH = 21


def months_to_days(m):
    return int(m * TRADING_DAYS_PER_MONTH)


def evaluate_period_quality(df, period_mask):
    sub = df.loc[period_mask, ["Regime", "core_ret"]].dropna().copy()
    if len(sub) < 180:
        return {"ok": False, "score": -1e9}

    counts = sub["Regime"].value_counts(normalize=True)
    target = 1.0 / 3.0

    g = sub.groupby("Regime")["core_ret"]
    ret_ann = g.apply(lambda x: (1.0 + x).prod() ** (252.0 / len(x)) - 1.0 if len(x) > 0 else np.nan)
    vol_ann = g.std() * np.sqrt(252.0)

    rr = {k: float(ret_ann.get(k, np.nan)) for k in ["Risk-on", "Neutre", "Stress"]}
    vv = {k: float(vol_ann.get(k, np.nan)) for k in ["Risk-on", "Neutre", "Stress"]}

    if any(np.isnan(v) for v in rr.values()) or any(np.isnan(v) for v in vv.values()):
        return {"ok": False, "score": -1e9}

    ret_order_ok = rr["Risk-on"] > rr["Neutre"] > rr["Stress"]
    vol_order_ok = vv["Stress"] > vv["Neutre"] > vv["Risk-on"]

    base_std = float(sub["core_ret"].std())
    if base_std <= 0 or np.isnan(base_std):
        return {"ok": False, "score": -1e9}

    ret_gap = (rr["Risk-on"] - rr["Stress"]) / (base_std * np.sqrt(252.0))
    vol_gap = (vv["Stress"] - vv["Risk-on"]) / (base_std * np.sqrt(252.0))

    abs_thr = float(sub["core_ret"].abs().quantile(0.80))
    hv = sub[sub["core_ret"].abs() >= abs_thr]
    stress_capture = float((hv["Regime"] == "Stress").mean()) if len(hv) > 0 else 0.0

    up_thr = float(sub["core_ret"].quantile(0.80))
    hu = sub[sub["core_ret"] >= up_thr]
    riskon_capture = float((hu["Regime"] == "Risk-on").mean()) if len(hu) > 0 else 0.0

    balance_penalty = sum((float(counts.get(r, 0.0)) - target) ** 2 for r in ["Stress", "Neutre", "Risk-on"])

    switches = int((sub["Regime"] != sub["Regime"].shift(1)).sum() - 1)
    years = max((sub.index.max() - sub.index.min()).days / 365.25, 0.5)
    switches_per_year = switches / years

    score = 0.0
    score += 2.0 if ret_order_ok else -2.0
    score += 2.0 if vol_order_ok else -2.0
    score += float(np.clip(ret_gap, -3.0, 3.0))
    score += float(np.clip(vol_gap, -3.0, 3.0))
    score += 1.5 * stress_capture
    score += 1.5 * riskon_capture
    score -= 2.0 * balance_penalty
    score -= 0.08 * abs(switches_per_year - 10.0)

    return {
        "ok": True,
        "score": score,
        "ret_order_ok": ret_order_ok,
        "vol_order_ok": vol_order_ok,
        "ret_gap": ret_gap,
        "vol_gap": vol_gap,
        "stress_capture": stress_capture,
        "riskon_capture": riskon_capture,
        "switches_per_year": switches_per_year,
        "pct_stress": float(counts.get("Stress", 0.0)),
        "pct_neutre": float(counts.get("Neutre", 0.0)),
        "pct_riskon": float(counts.get("Risk-on", 0.0)),
    }


def optimize_regime_cfg(daily_returns_df, core_tickers, core_equity, core_rates,
                        core_weights, core_ter_map_pct, regime_cfg):
    """
    Grid-search over regime parameters. Returns (res_df, best_regime_cfg, best_row).
    """
    t0 = time.perf_counter()

    fixed_is_start = regime_cfg["is_start"]
    fixed_is_end = regime_cfg["is_end"]
    fixed_oos_start = regime_cfg["oos_start"]

    required_cols = core_tickers
    full_log = daily_returns_df[required_cols].copy().sort_index()
    full_simple = np.expm1(full_log)
    ter_annual_decimal = pd.Series({t: core_ter_map_pct.get(t, 0.0) / 100.0 for t in required_cols})
    ter_daily_drag_full = (1.0 + ter_annual_decimal) ** (1.0 / 252.0) - 1.0

    full_net = full_simple.copy()
    for t in required_cols:
        full_net[t] = (1.0 + full_simple[t]) / (1.0 + ter_daily_drag_full[t]) - 1.0

    core_weights_series = pd.Series(core_weights)
    core_ret_full = (full_net[required_cols] * core_weights_series.reindex(required_cols).values).sum(axis=1).rename("core_ret")
    equity_ret_full = full_net[core_equity].rename("equity_ret")
    bonds_ret_full = full_net[core_rates].rename("bonds_ret")

    base_template = pd.concat([core_ret_full, equity_ret_full, bonds_ret_full], axis=1).dropna()
    base_template = base_template.loc[fixed_is_start:].copy()

    is_mask_global = (base_template.index >= pd.Timestamp(fixed_is_start)) & (base_template.index <= pd.Timestamp(fixed_is_end))
    oos_mask_global = base_template.index >= pd.Timestamp(fixed_oos_start)

    # Grid
    indicator_window_months_grid = [2, 3, 4]
    zscore_window_months_grid = [9, 12, 15, 18, 21, 24]
    zscore_min_frac_grid = [0.45, 0.50, 0.60]
    rolling_threshold_months_grid = [18, 24, 30]
    quantile_pairs_grid = [(0.25, 0.75), (0.30, 0.70), (0.35, 0.65), (0.25, 0.80)]
    min_regime_days_grid = [7, 10, 14]
    score_weights_grid = [
        (0.45, 0.15, 0.40, 0.20),  # default: vol-led, low corr
        (0.35, 0.20, 0.35, 0.20),  # balanced
        (0.40, 0.10, 0.30, 0.30),  # drawdown-led
    ]

    combos = list(product(
        indicator_window_months_grid,
        zscore_window_months_grid,
        zscore_min_frac_grid,
        rolling_threshold_months_grid,
        quantile_pairs_grid,
        min_regime_days_grid,
        score_weights_grid,
    ))

    results = []

    for ind_m, z_m, z_frac, thr_m, q_pair, min_days, sw in combos:
        low_q, high_q = q_pair
        w_vol, w_corr, w_mom, w_dd = sw
        ind_win = months_to_days(ind_m)
        z_win = months_to_days(z_m)
        thr_win = months_to_days(thr_m)
        min_p = max(30, int(round(z_win * z_frac)))
        min_p = min(min_p, z_win)

        df = base_template.copy()

        df["vol_63"] = df["core_ret"].rolling(ind_win, min_periods=ind_win).std() * np.sqrt(252)
        df["corr_eq_bond_63"] = df["equity_ret"].rolling(ind_win, min_periods=ind_win).corr(df["bonds_ret"])
        df["mom_63"] = (1.0 + df["core_ret"]).rolling(ind_win, min_periods=ind_win).apply(np.prod, raw=True) - 1.0

        nav = (1.0 + df["core_ret"]).cumprod()
        rolling_peak = nav.rolling(ind_win, min_periods=ind_win).max()
        df["dd_63"] = nav / rolling_peak - 1.0

        df["vol_z"] = rolling_zscore(df["vol_63"], z_win=z_win, min_p=min_p)
        df["corr_z"] = rolling_zscore(df["corr_eq_bond_63"], z_win=z_win, min_p=min_p)
        df["mom_z"] = rolling_zscore(df["mom_63"], z_win=z_win, min_p=min_p)
        df["dd_z"] = rolling_zscore(df["dd_63"], z_win=z_win, min_p=min_p)

        df["RegimeScore_raw"] = (
            w_vol * df["vol_z"]
            + w_corr * df["corr_z"]
            - w_mom * df["mom_z"]
            - w_dd * df["dd_z"]
        )
        df["RegimeScore"] = df["RegimeScore_raw"].shift(1)

        is_scores = df.loc[is_mask_global, "RegimeScore_raw"].dropna()
        if len(is_scores) < 120:
            continue

        q_low_is = float(is_scores.quantile(low_q))
        q_high_is = float(is_scores.quantile(high_q))

        df["q_low_roll"] = df["RegimeScore_raw"].rolling(thr_win, min_periods=1).quantile(low_q).shift(1).fillna(q_low_is)
        df["q_high_roll"] = df["RegimeScore_raw"].rolling(thr_win, min_periods=1).quantile(high_q).shift(1).fillna(q_high_is)

        df["Regime_raw"] = "Neutre"
        df.loc[df["RegimeScore"] <= df["q_low_roll"], "Regime_raw"] = "Risk-on"
        df.loc[df["RegimeScore"] >= df["q_high_roll"], "Regime_raw"] = "Stress"
        df["Regime"] = apply_min_regime_days_causal(df["Regime_raw"], min_days=min_days)

        met_is = evaluate_period_quality(df, is_mask_global)
        met_oos = evaluate_period_quality(df, oos_mask_global)
        if not met_is["ok"] or not met_oos["ok"]:
            continue

        global_score = 0.35 * met_is["score"] + 0.65 * met_oos["score"]

        results.append({
            "indicator_window_months": ind_m,
            "zscore_window_months": z_m,
            "rolling_threshold_months": thr_m,
            "indicator_window_days": ind_win,
            "zscore_window_days": z_win,
            "zscore_min_periods": min_p,
            "rolling_threshold_days": thr_win,
            "threshold_low_q": low_q,
            "threshold_high_q": high_q,
            "min_regime_days": min_days,
            "w_vol": w_vol,
            "w_corr": w_corr,
            "w_mom": w_mom,
            "w_dd": w_dd,
            "score_is": met_is["score"],
            "score_oos": met_oos["score"],
            "score_global": global_score,
            "ret_order_oos": met_oos["ret_order_ok"],
            "vol_order_oos": met_oos["vol_order_ok"],
            "stress_capture_oos": met_oos["stress_capture"],
            "riskon_capture_oos": met_oos["riskon_capture"],
            "switches_per_year_oos": met_oos["switches_per_year"],
            "pct_stress_oos": met_oos["pct_stress"],
            "pct_neutre_oos": met_oos["pct_neutre"],
            "pct_riskon_oos": met_oos["pct_riskon"],
        })

    if not results:
        raise RuntimeError("Aucune combinaison valide. Elargir la grille ou verifier les donnees.")

    res_df = pd.DataFrame(results).sort_values("score_global", ascending=False).reset_index(drop=True)
    best = res_df.iloc[0]

    best_regime_cfg = {
        "indicator_window_days": int(best["indicator_window_days"]),
        "zscore_window_days": int(best["zscore_window_days"]),
        "zscore_min_periods": int(best["zscore_min_periods"]),
        "rolling_threshold_days": int(best["rolling_threshold_days"]),
        "threshold_low_q": float(best["threshold_low_q"]),
        "threshold_high_q": float(best["threshold_high_q"]),
        "is_start": fixed_is_start,
        "is_end": fixed_is_end,
        "oos_start": fixed_oos_start,
        "min_regime_days": int(best["min_regime_days"]),
        "w_vol": float(best["w_vol"]),
        "w_corr": float(best["w_corr"]),
        "w_mom": float(best["w_mom"]),
        "w_dd": float(best["w_dd"]),
    }

    elapsed = time.perf_counter() - t0
    print(f"Optimisation terminée: {len(combos)} combinaisons en {elapsed:.2f}s ({len(combos) / elapsed:.1f}/s)")

    return res_df, best_regime_cfg, best


# Need matplotlib import for plot_regime
import matplotlib.pyplot as plt


# ── Display helpers ──────────────────────────────────────────────────────────

def display_regime_results(regime_result, regime_cfg):
    """Display full regime analysis: period info, thresholds, regime distribution, stats, coherence."""
    from IPython.display import display as _display

    base_full = regime_result['base_full']
    base_df = regime_result['base_df']
    regime_output = regime_result['regime_output']
    regime_stats_df = regime_result['regime_stats_df']
    coherence_df = regime_result['coherence_df']
    first_oos = regime_result['first_oos']
    q_low_is = regime_result['q_low_is']
    q_high_is = regime_result['q_high_is']

    low_q = float(regime_cfg['threshold_low_q'])
    high_q = float(regime_cfg['threshold_high_q'])

    w_vol = regime_cfg.get('w_vol', 1.0)
    w_corr = regime_cfg.get('w_corr', 1.0)
    w_mom = regime_cfg.get('w_mom', 1.0)
    w_dd = regime_cfg.get('w_dd', 0.0)

    print("\n" + "=" * 80)
    print("REGIMES DE MARCHE CORE (Vol / Corr / Momentum / Drawdown)")
    print("=" * 80)
    print(
        f"Parametres regime: win={regime_cfg['indicator_window_days']}j, "
        f"q=({low_q:.0%},{high_q:.0%}), "
        f"roll_thr={regime_cfg['rolling_threshold_days']}j, min_regime={regime_cfg['min_regime_days']}j"
    )
    print(
        f"Poids score: w_vol={w_vol:.2f}, w_corr={w_corr:.2f}, "
        f"w_mom={w_mom:.2f}, w_dd={w_dd:.2f}"
    )
    print(f"\nPeriode utilisee (IS+OOS): {base_full.index.min().date()} -> {base_full.index.max().date()} ({len(base_full)} obs)")
    print(f"Calibration IS ({regime_cfg['is_start']} -> {regime_cfg['is_end']})")
    print(f"Seuils IS fixes: q{int(low_q*100)}={q_low_is:.3f} | q{int(high_q*100)}={q_high_is:.3f}")
    print(f"Seuils rolling fin periode OOS: q_low={base_df['q_low_roll'].iloc[-1]:.3f} | q_high={base_df['q_high_roll'].iloc[-1]:.3f}")
    print(f"Switches regime (avant/apres lissage): {regime_result['switches_raw']} -> {regime_result['switches_smooth']}")
    print("Repartition des regimes lisses (OOS):")
    _display((base_df['Regime'].value_counts(normalize=True) * 100).rename('pct').to_frame().style.format('{:.1f}%'))

    print(f"\nDebut OOS: {first_oos.date()} | Score={base_df.loc[first_oos, 'RegimeScore']:.3f} | "
          f"q_low={base_df.loc[first_oos, 'q_low_roll']:.3f} | q_high={base_df.loc[first_oos, 'q_high_roll']:.3f} | "
          f"Regime={base_df.loc[first_oos, 'Regime']}")
    _display(regime_output.tail(10))

    print("\nStats descriptives Core par regime :")
    _display(regime_stats_df.style.format({
        'N_obs': '{:.0f}', 'Perf_ann': '{:+.2%}', 'Vol_ann': '{:.2%}', 'MaxDD': '{:.2%}',
    }))

    print("\nVerification coherence economique (moyennes indicateurs) :")
    fmt_coherence = {
        'vol_63': '{:.2%}', 'mom_63': '{:+.2%}', 'corr_eq_bond_63': '{:+.3f}',
    }
    if 'dd_63' in coherence_df.columns:
        fmt_coherence['dd_63'] = '{:+.2%}'
    _display(coherence_df.style.format(fmt_coherence))
    print("Attendu: Stress = vol haute & momentum negatif & drawdown negatif; Risk-on = vol basse & momentum positif & drawdown ~0.")


def display_regime_optimization(res_df, best_regime_cfg, best):
    """Display optimization results: top 10, best config details, OOS quality metrics."""
    from IPython.display import display as _display

    print("\n" + "=" * 90)
    print("OPTIMISATION REGIME_CFG (qualitative, sans forward-looking)")
    print("=" * 90)

    cols_show = [
        'indicator_window_months', 'zscore_window_months', 'rolling_threshold_months',
        'zscore_min_periods', 'threshold_low_q', 'threshold_high_q', 'min_regime_days',
        'w_vol', 'w_corr', 'w_mom', 'w_dd',
        'score_is', 'score_oos', 'score_global', 'ret_order_oos', 'vol_order_oos',
        'stress_capture_oos', 'riskon_capture_oos', 'switches_per_year_oos'
    ]
    print("\nTop 10 configurations (score_global):")
    _display(res_df[cols_show].head(10))

    print("\nMeilleure configuration recommandee (IS/OOS fixes):")
    print(
        f"  indicator_window: {int(best['indicator_window_months'])} mois ({int(best['indicator_window_days'])}j)\n"
        f"  zscore_window: {int(best['zscore_window_months'])} mois ({int(best['zscore_window_days'])}j)\n"
        f"  rolling_threshold: {int(best['rolling_threshold_months'])} mois ({int(best['rolling_threshold_days'])}j)"
    )
    for k, v in best_regime_cfg.items():
        if k in {'indicator_window_days', 'zscore_window_days', 'rolling_threshold_days'}:
            continue
        print(f"  {k}: {v}")

    print("\nQualite OOS de la meilleure config:")
    print(f"  score_oos           : {best['score_oos']:.3f}")
    print(f"  ret_order_oos       : {bool(best['ret_order_oos'])}")
    print(f"  vol_order_oos       : {bool(best['vol_order_oos'])}")
    print(f"  stress_capture_oos  : {best['stress_capture_oos']:.1%}")
    print(f"  riskon_capture_oos  : {best['riskon_capture_oos']:.1%}")
    print(f"  switches/year OOS   : {best['switches_per_year_oos']:.1f}")
    print(
        "  repartition OOS     : "
        f"Stress={best['pct_stress_oos']:.1%}, "
        f"Neutre={best['pct_neutre_oos']:.1%}, "
        f"Risk-on={best['pct_riskon_oos']:.1%}"
    )
