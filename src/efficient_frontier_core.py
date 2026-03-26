"""
Comparaison des stratégies Core via frontière efficiente (données daily).

Entrée :
- outputs/core3_etf_daily_log_returns.csv  (log-rendements journaliers des 3 ETF)

Fenêtre IS  : 2019-01-01 → 2020-12-31  (calibration, sans look-ahead)
Fenêtre OOS : 2021-01-01 → 2025-12-31  (évaluation)

Stratégies comparées (statiques) :
    1. Max Sharpe  – poids optimisés (Max Sharpe) sur IS, appliqués fixement sur OOS
    2. Min Variance – poids Min Variance sur IS, fixes sur OOS
    3. Equal Weight – 1/3 chacun (benchmark naïf)
    4. Risk Parity  – poids ∝ 1/vol_IS, normalisés
    5. Efficient Vol X% – portefeuilles efficients sous contrainte de vol cible (10% à 20%)

Sorties :
- outputs/figures/06_efficient_frontier_core.png    (nuage + stratégies statiques)
- outputs/figures/07_core_strategies_oos_perf.png   (perfs OOS cumulées)
- outputs/core_portfolio_comparison.csv             (métriques IS + OOS)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import minimize

from src.risk_free import get_bund_risk_free_daily

project_root = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FrontierConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    input_csv:    Path = project_root / "outputs" / "core3_etf_daily_log_returns.csv"
    fig_dir:      Path = project_root / "outputs" / "figures"
    out_csv:      Path = project_root / "outputs" / "core_portfolio_comparison.csv"

    calib_start: str = "2019-01-01"
    calib_end:   str = "2020-12-31"
    oos_start:   str = "2021-01-01"
    oos_end:     str = "2025-12-31"

    n_sim:       int   = 15_000
    rf:          float = 0.0
    w_min:       float = 0.05
    w_max:       float = 0.90

    target_vols: Tuple[float, ...] = tuple(np.round(np.arange(0.10, 0.201, 0.01), 2))
    dpi:          int   = 160


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers statistiques (daily)
# ══════════════════════════════════════════════════════════════════════════════

def _stats_daily(log_rets: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """μ et Σ annualisés à partir de log-rendements journaliers."""
    mu  = log_rets.mean().values * 252
    cov = log_rets.cov().values  * 252
    return mu, cov


def _port_stats(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf_ann: float = 0.0,
) -> Tuple[float, float, float]:
    """Rendement (converti en simple), vol et Sharpe (log-exces rf) annualisés."""
    r_log = float(w @ mu)
    r = float(np.expm1(r_log))
    v = float(np.sqrt(w @ cov @ w))
    rf_log_ann = float(np.log1p(rf_ann)) if rf_ann > -0.999999 else np.nan
    s = (r_log - rf_log_ann) / v if v > 1e-10 else np.nan
    return r, v, s


def _batch_stats(
    W: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf_ann: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ret_simple_ann, vol_ann, sharpe_log_exces_rf) pour une matrice N×k de poids."""
    rets_log = W @ mu
    rets   = np.expm1(rets_log)
    vols   = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    rf_log_ann = float(np.log1p(rf_ann)) if rf_ann > -0.999999 else np.nan
    sharpe = np.where(vols > 1e-10, (rets_log - rf_log_ann) / vols, np.nan)
    return rets, vols, sharpe


def _sim_portfolios(k: int, n: int, seed: int = 42) -> np.ndarray:
    """Dirichlet random portfolios."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(k), size=n)


def _opt_max_ret(mu: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Maximize expected annual log return under weight constraints."""
    n = len(mu)
    w0 = np.ones(n) / n
    res = minimize(
        lambda w: -(w @ mu),
        w0,
        method="SLSQP",
        bounds=[(w_min, w_max)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    if not res.success:
        return w0
    w = np.maximum(res.x, 0)
    return w / w.sum()


def _opt_min_ret(mu: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Minimize expected annual log return under weight constraints."""
    n = len(mu)
    w0 = np.ones(n) / n
    res = minimize(
        lambda w: (w @ mu),
        w0,
        method="SLSQP",
        bounds=[(w_min, w_max)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    if not res.success:
        return w0
    w = np.maximum(res.x, 0)
    return w / w.sum()


def _compute_constrained_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    w_min: float,
    w_max: float,
    n_points: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient frontier under constraints using QP slices.

    For each target annual log return, solve:
      min_w  w' Σ w
      s.t.   1'w = 1
             μ'w = target
             w_min <= w_i <= w_max
    """
    n = len(mu)
    w_min_ret = _opt_min_ret(mu, w_min, w_max)
    w_max_ret = _opt_max_ret(mu, w_min, w_max)
    r_min = float(w_min_ret @ mu)
    r_max = float(w_max_ret @ mu)

    targets = np.linspace(r_min, r_max, n_points)
    w0 = np.ones(n) / n
    vols: List[float] = []
    rets_simple: List[float] = []

    for t in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, tt=t: float(w @ mu - tt)},
        ]
        res = minimize(
            lambda w: float(w @ cov @ w),
            w0,
            method="SLSQP",
            bounds=[(w_min, w_max)] * n,
            constraints=constraints,
        )
        if not res.success:
            continue
        w = np.maximum(res.x, 0)
        if w.sum() <= 1e-12:
            continue
        w /= w.sum()
        v = float(np.sqrt(w @ cov @ w))
        r_log = float(w @ mu)
        vols.append(v)
        rets_simple.append(float(np.expm1(r_log)))
        w0 = w

    if not vols:
        return np.array([]), np.array([])

    vols_arr = np.array(vols)
    rets_arr = np.array(rets_simple)
    order = np.argsort(vols_arr)
    return vols_arr[order], rets_arr[order]


# ══════════════════════════════════════════════════════════════════════════════
#  Optimiseurs
# ══════════════════════════════════════════════════════════════════════════════

def _opt_max_sharpe(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float, rf_ann: float = 0.0) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n
    rf_log_ann = float(np.log1p(rf_ann)) if rf_ann > -0.999999 else 0.0
    def neg_sr(w):
        v = np.sqrt(w @ cov @ w)
        return -((w @ mu) - rf_log_ann) / v if v > 1e-12 else 1e10
    res = minimize(neg_sr, w0, method="SLSQP",
                   bounds=[(w_min, w_max)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return np.maximum(res.x, 0) / np.maximum(res.x, 0).sum() if res.success else w0


def _opt_min_var(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    n = len(mu)
    w0 = np.ones(n) / n
    def port_var(w):
        return w @ cov @ w
    res = minimize(port_var, w0, method="SLSQP",
                   bounds=[(w_min, w_max)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return np.maximum(res.x, 0) / np.maximum(res.x, 0).sum() if res.success else w0


def _opt_target_vol(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float, target_vol: float) -> np.ndarray:
    """Portefeuille efficient: max rendement sous contrainte de vol <= target_vol."""
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_ret(w: np.ndarray) -> float:
        return -float(w @ mu)

    constraints = [
        {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        {"type": "ineq", "fun": lambda w: float(target_vol**2 - (w @ cov @ w))},
    ]

    res = minimize(
        neg_ret,
        w0,
        method="SLSQP",
        bounds=[(w_min, w_max)] * n,
        constraints=constraints,
    )
    if not res.success:
        return _opt_min_var(mu, cov, w_min, w_max)
    w = np.maximum(res.x, 0)
    return w / w.sum() if w.sum() > 1e-12 else _opt_min_var(mu, cov, w_min, w_max)


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    """Poids inversement proportionnels à la vol individuelle."""
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / (vols + 1e-12)
    return inv_vol / inv_vol.sum()


def _risk_parity_weights(cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """
    Poids Risk Parity (∝ 1/vol individuelle), clipés dans [w_min, w_max].
    Pas de dépendance à μ → robuste OOS.
    """
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / (vols + 1e-12)
    w = inv_vol / inv_vol.sum()
    w = np.clip(w, w_min, w_max)
    w /= w.sum()
    return w


def _ledoit_wolf_cov(window_rets: np.ndarray) -> np.ndarray:
    """
    Estime la matrice de covariance annualisée par Ledoit-Wolf shrinkage.
    Fallback vers la covariance empirique si sklearn indisponible.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(window_rets)
        return lw.covariance_ * 252
    except ImportError:
        return np.cov(window_rets.T) * 252


# ══════════════════════════════════════════════════════════════════════════════
#  Backtest OOS
# ══════════════════════════════════════════════════════════════════════════════

def _backtest_fixed(
    log_rets_oos: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Log-rendements journaliers OOS avec poids fixes."""
    r_simple = np.expm1(log_rets_oos.values)
    port_simple = r_simple @ weights
    port_log = np.log1p(port_simple)
    return pd.Series(port_log, index=log_rets_oos.index, name="log_ret")


def _backtest_rolling(
    log_rets: pd.DataFrame,
    w_min: float, w_max: float,
    lookback: int, rebal: int,
    oos_start: str, oos_end: str,
    rf_daily: pd.Series | None = None,
    equity_floor: float = 0.0,
    equity_ceiling: float = 0.0,
    momentum_window: int = 252,
    momentum_threshold: float = 0.0,
    rolling_method: str = "max_sharpe",
    use_ledoit_wolf: bool = False,
) -> pd.Series:
    """
    Backtest rolling générique.

    rolling_method :
      - "max_sharpe"        : Max Sharpe avec μ historique (legacy).
      - "risk_parity_tilt"  : Risk Parity (1/vol) + tilt Equity momentum.
      - "min_var"           : Min Variance sous bornes.

    Si use_ledoit_wolf=True, la covariance est estimée par Ledoit-Wolf shrinkage.
    """
    dates = log_rets.index
    port_rets: List[float] = []
    port_dates: List[pd.Timestamp] = []
    for start in range(lookback, len(dates), rebal):
        window = log_rets.iloc[start - lookback:start]
        oos    = log_rets.iloc[start:start + rebal]
        if oos.empty:
            continue

        # ── Estimation de la covariance ───────────────────────────────────
        if use_ledoit_wolf:
            cov = _ledoit_wolf_cov(window.values)
        else:
            cov = window.cov().values * 252

        # ── Calcul des poids selon la méthode choisie ─────────────────────
        if rolling_method == "risk_parity_tilt":
            w = _risk_parity_weights(cov, w_min, w_max)
            mom_start = max(0, start - momentum_window)
            equity_rets = log_rets.iloc[mom_start:start].iloc[:, 0]
            equity_momentum = float(equity_rets.sum())
            if equity_ceiling > 0 and equity_momentum > momentum_threshold and w[0] < equity_ceiling:
                tilt_target = equity_ceiling
                excess = tilt_target - w[0]
                w[0] = tilt_target
                others_sum = w[1:].sum()
                if others_sum > 1e-12:
                    w[1:] -= excess * (w[1:] / others_sum)
                w = np.clip(w, w_min, w_max)
                w /= w.sum()
        elif rolling_method == "min_var":
            n = cov.shape[0]
            w0 = np.ones(n) / n
            res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                           bounds=[(w_min, w_max)] * n,
                           constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
            w = np.maximum(res.x, 0) / np.maximum(res.x, 0).sum() if res.success else w0
        else:  # "max_sharpe" (legacy)
            mu = window.mean().values * 252
            rf_ann_win = 0.0
            if rf_daily is not None:
                rf_win = rf_daily.reindex(window.index).ffill().fillna(0.0)
                rf_ann_win = float(rf_win.mean() * 252) if len(rf_win) else 0.0
            w  = _opt_max_sharpe(mu, cov, w_min, w_max, rf_ann=rf_ann_win)

        # ── Appliquer le floor Equity si nécessaire ───────────────────────
        if equity_floor > 0 and w[0] < equity_floor:
            deficit = equity_floor - w[0]
            w[0] = equity_floor
            others_sum = w[1:].sum()
            if others_sum > 1e-12:
                w[1:] -= deficit * (w[1:] / others_sum)
            w = np.clip(w, w_min, w_max)
            w /= w.sum()

        r_oos_simple = np.expm1(oos.values) @ w
        r_oos = np.log1p(r_oos_simple)
        port_rets.extend(r_oos.tolist())
        port_dates.extend(oos.index.tolist())
    s = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates)).sort_index()
    return s.loc[oos_start:oos_end].rename("log_ret")


def _perf_metrics(log_ret: pd.Series, rf_daily: pd.Series | None = None) -> Dict:
    """Métriques annualisées à partir de log-rendements journaliers."""
    n = len(log_ret)
    ann_log_ret = float(log_ret.mean() * 252) if n > 0 else np.nan
    ann_ret = float(np.expm1(ann_log_ret)) if n > 0 else np.nan
    ann_vol = float(log_ret.std() * np.sqrt(252))
    if rf_daily is None:
        sharpe = ann_log_ret / ann_vol if ann_vol > 1e-10 else np.nan
    else:
        rf_aligned = rf_daily.reindex(log_ret.index).ffill().fillna(0.0)
        rf_log = np.log1p(rf_aligned)
        excess_ann = float((log_ret - rf_log).mean() * 252) if n > 0 else np.nan
        sharpe = excess_ann / ann_vol if ann_vol > 1e-10 else np.nan
    cum     = np.exp(log_ret.cumsum())
    mdd     = float(((cum / cum.cummax()) - 1).min())
    return {"ret_ann": ann_ret, "vol_ann": ann_vol, "sharpe": sharpe, "mdd": mdd}


# ══════════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Max Sharpe":              "#e6194b",
    "Min Variance":            "#3cb44b",
    "Equal Weight":            "#4363d8",
    "Risk Parity":             "#f58231",
}


def _build_target_style_maps(target_vols: Tuple[float, ...]) -> Tuple[Dict[str, str], Dict[str, str]]:
    target_names = [f"Efficient Vol {int(round(v * 100))}%" for v in target_vols]
    target_colors = plt.cm.plasma(np.linspace(0.15, 0.90, len(target_names)))
    color_map = {
        name: mcolors.to_hex(col)
        for name, col in zip(target_names, target_colors)
    }
    marker_cycle = ["s", "h", "X", "v", "<", ">", "d", "p", "8", "o", "*"]
    marker_map = {
        name: marker_cycle[i % len(marker_cycle)]
        for i, name in enumerate(target_names)
    }
    return color_map, marker_map


def _plot_frontier(
    sim_rets: np.ndarray, sim_vols: np.ndarray, sim_sharpe: np.ndarray,
    strategies: Dict,
    tickers: List[str],
    target_vols: Tuple[float, ...],
    fig_dir: Path, dpi: int,
    sharpe_label: str = "Sharpe (IS)",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(sim_vols, sim_rets, c=sim_sharpe, cmap="viridis",
                    s=5, alpha=0.3, label="_nolegend_")
    plt.colorbar(sc, ax=ax, label=sharpe_label)

    target_colors, target_markers = _build_target_style_maps(target_vols)
    colors = dict(COLORS)
    colors.update(target_colors)
    markers = {"Max Sharpe": "*", "Min Variance": "D", "Equal Weight": "P", "Risk Parity": "^"}
    markers.update(target_markers)

    for name, d in strategies.items():
        if "w" not in d:
            continue
        w = d["w"]
        ax.scatter(d["vol_is"], d["ret_is"], marker=markers.get(name, "o"),
                   s=160, zorder=5, color=colors.get(name, "grey"),
                   label=f"{name}  Sharpe(exces rf)={d['sharpe_is']:.2f}")
        for i, t in enumerate(tickers):
            ax.annotate(f"{w[i]:.0%}", (d["vol_is"], d["ret_is"]),
                        textcoords="offset points", xytext=(6, -4*(i-1)),
                        fontsize=7, color=colors.get(name, "grey"))

    ax.set_xlabel("Volatilité annualisée (IS 2019-2020)")
    ax.set_ylabel("Rendement annualisé (IS 2019-2020)")
    ax.set_title("Frontière efficiente – 3 ETF Core\n(calibration 2019-2020, daily)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "06_efficient_frontier_core.png", dpi=dpi)
    plt.close()


def _plot_oos_perf(strategies: Dict, fig_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Cumulative OOS ────────────────────────────────────────────────────────
    ax = axes[0]
    for name, d in strategies.items():
        if "oos_ret" not in d:
            continue
        cum = 100 * (1 + d["oos_ret"]).cumprod()
        ax.plot(cum.index, cum.values, label=name, color=COLORS.get(name, "grey"))
    ax.set_title("Performance cumulée OOS 2021-2025 (base 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Métriques OOS bar chart ───────────────────────────────────────────────
    ax2 = axes[1]
    names  = list(strategies.keys())
    sharpes = [strategies[n].get("oos_sharpe", 0) for n in names]
    bars = ax2.bar(range(len(names)), sharpes,
                   color=[COLORS.get(n, "grey") for n in names])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax2.set_title("Sharpe OOS (exces rf) 2021-2025 par stratégie")
    ax2.set_ylabel("Sharpe (exces rf)")
    ax2.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "07_core_strategies_oos_perf.png", dpi=dpi)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> Tuple[Dict, pd.DataFrame]:
    cfg = FrontierConfig()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  COMPARAISON DES STRATÉGIES CORE (Markowitz daily)")
    print(f"  IS   : {cfg.calib_start} → {cfg.calib_end}")
    print(f"  OOS  : {cfg.oos_start} → {cfg.oos_end}")
    print("=" * 60)

    # ── Chargement ────────────────────────────────────────────────────────────
    print("\n[1] Chargement des log-rendements journaliers des 3 ETF...")
    df = pd.read_csv(cfg.input_csv, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df.sort_index()
    tickers = list(df.columns)
    print(f"  ETFs : {tickers}")
    print(f"  Période : {df.index.min().date()} → {df.index.max().date()}")

    rf_daily_all, rf_source = get_bund_risk_free_daily(df.index)
    rf_is = rf_daily_all.reindex(df.loc[cfg.calib_start:cfg.calib_end].index).ffill().fillna(0.0)
    rf_oos = rf_daily_all.reindex(df.loc[cfg.oos_start:cfg.oos_end].index).ffill().fillna(0.0)
    rf_ann_is = float(rf_is.mean() * 252) if len(rf_is) else 0.0
    print(f"  Risk-free Bund : {rf_source}")

    # ── Fenêtres ──────────────────────────────────────────────────────────────
    is_df  = df.loc[cfg.calib_start:cfg.calib_end].dropna()
    oos_df = df.loc[cfg.oos_start:cfg.oos_end].dropna()
    print(f"  IS  : {len(is_df)} obs | OOS : {len(oos_df)} obs")

    mu_is, cov_is = _stats_daily(is_df)

    # ── Portefeuilles simulés sur IS ──────────────────────────────────────────
    print("\n[2] Simulation de la frontière (IS)...")
    W_sim  = _sim_portfolios(len(tickers), cfg.n_sim)
    s_rets, s_vols, s_sharpes = _batch_stats(W_sim, mu_is, cov_is, rf_ann=rf_ann_is)

    # ── Stratégies statiques ─────────────────────────────────────────────────
    print("\n[3] Optimisation des stratégies statiques sur IS...")
    strat_configs = {
        "Max Sharpe":    _opt_max_sharpe(mu_is, cov_is, cfg.w_min, cfg.w_max, rf_ann=rf_ann_is),
        "Min Variance":  _opt_min_var(mu_is, cov_is, cfg.w_min, cfg.w_max),
        "Equal Weight":  np.ones(len(tickers)) / len(tickers),
        "Risk Parity":   _risk_parity_weights(cov_is, cfg.w_min, cfg.w_max),
    }
    for v in cfg.target_vols:
        strat_configs[f"Efficient Vol {int(round(v * 100))}%"] = _opt_target_vol(
            mu_is, cov_is, cfg.w_min, cfg.w_max, float(v)
        )

    target_colors, _ = _build_target_style_maps(cfg.target_vols)
    COLORS.update(target_colors)

    strategies: Dict = {}
    for name, w in strat_configs.items():
        r_is, v_is, s_is = _port_stats(w, mu_is, cov_is, rf_ann=rf_ann_is)
        oos_ret = _backtest_fixed(oos_df, w)
        is_ret = _backtest_fixed(is_df, w)
        m_is = _perf_metrics(is_ret, rf_is)
        m_oos   = _perf_metrics(oos_ret, rf_oos)
        strategies[name] = {
            "w":          w,
            "ret_is":     m_is["ret_ann"],
            "vol_is":     m_is["vol_ann"],
            "sharpe_is":  m_is["sharpe"],
            "oos_ret":    oos_ret,
            "oos_ret_ann": m_oos["ret_ann"], "oos_vol": m_oos["vol_ann"],
            "oos_sharpe":  m_oos["sharpe"],  "oos_mdd": m_oos["mdd"],
        }
        print(f"  {name:<22s}  w=[{', '.join(f'{x:.1%}' for x in w)}]  "
              f"IS Sharpe(exces rf)={m_is['sharpe']:.2f}  OOS Sharpe(exces rf)={m_oos['sharpe']:.2f}")

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TABLEAU COMPARATIF")
    print("=" * 60)
    rows = []
    for name, d in strategies.items():
        poids_str = f"[{', '.join(f'{x:.1%}' for x in d['w'])}]" if "w" in d else "rolling"
        row = {
            "strategie":    name,
            "poids":        poids_str,
            "IS_ret_ann":   d.get("ret_is", np.nan),
            "IS_vol_ann":   d.get("vol_is", np.nan),
            "IS_sharpe":    d.get("sharpe_is", np.nan),
            "OOS_ret_ann":  d["oos_ret_ann"],
            "OOS_vol_ann":  d["oos_vol"],
            "OOS_sharpe":   d["oos_sharpe"],
            "OOS_mdd":      d["oos_mdd"],
        }
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index("strategie")

    print(f"\n{'Stratégie':<26} {'IS Ret':>8} {'IS Vol':>8} {'IS Sh':>7} "
          f"{'OOS Ret':>8} {'OOS Vol':>8} {'OOS Sh':>7} {'OOS MDD':>9}")
    print("-" * 85)
    for name, row in comp_df.iterrows():
        print(f"{name:<26} {row['IS_ret_ann']:>7.1%} {row['IS_vol_ann']:>8.1%} "
              f"{row['IS_sharpe']:>7.2f} {row['OOS_ret_ann']:>8.1%} "
              f"{row['OOS_vol_ann']:>8.1%} {row['OOS_sharpe']:>7.2f} {row['OOS_mdd']:>8.1%}")

    best = comp_df["OOS_sharpe"].idxmax()
    print(f"\n  ★  Meilleure stratégie OOS (Sharpe exces rf) : {best}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    comp_df.to_csv(cfg.out_csv)
    print(f"\n  -> {cfg.out_csv}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[5] Génération des figures...")
    _plot_frontier(
        s_rets,
        s_vols,
        s_sharpes,
        strategies,
        tickers,
        cfg.target_vols,
        cfg.fig_dir,
        cfg.dpi,
        sharpe_label="Sharpe IS (exces rf Bund)",
    )
    print(f"  -> {cfg.fig_dir / '06_efficient_frontier_core.png'}")

    _plot_oos_perf(strategies, cfg.fig_dir, cfg.dpi)
    print(f"  -> {cfg.fig_dir / '07_core_strategies_oos_perf.png'}")

    print("\n  ✓  Frontière efficiente terminée.")
    return strategies, comp_df


if __name__ == "__main__":
    main()
