"""
fond_construction.py – Construction et backtest du fonds Core-Satellite

Calibration : 2019-01-01 → 2020-12-31
Backtest OOS: 2021-01-01 → 2025-12-31

Objectifs :
  - Frais totaux ≤ 80 bps/an
  - Volatilité totale cible 8-12 %
  - Poche Core 70-75 %  (rebalancement trimestriel sans réoptimisation)
  - Poche Satellite 25-30 %  (poids equal-weight 1/n)
  - Satellite : alpha > 0 vs Core, beta rolling 3 M ≈ 0
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── import lire_prix_wide depuis satellite_pipeline (même package) ────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from satellite_pipeline import lire_prix_wide  # noqa: E402

project_root = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FondConfig:
    # ── Fenêtres temporelles ──────────────────────────────────────────────────
    calib_start:    str = "2019-01-01"
    calib_end:      str = "2020-12-31"
    backtest_start: str = "2021-01-01"
    backtest_end:   str = "2025-12-31"

    # ── Allocation cible ──────────────────────────────────────────────────────
    w_core_min: float = 0.70
    w_core_max: float = 0.75
    w_sat_abs_max: float = 0.30

    # ── Volatilité totale annualisée cible ──────────────────────────────────
    vol_target_min: float = 0.08
    vol_target_max: float = 0.12
    vol_target_mid: float = 0.10

    # ── Pas de levier ─────────────────────────────────────────────────────────
    # Exposition brute = 100 % (w_core + w_sat = 1.0), aucun scaling.
    portfolio_scale: float = 1.00

    # ── Frais (bps/an) ────────────────────────────────────────────────────────
    fees_bps_max:         float = 80.0
    fees_core_bps:        float = 23.0    # ≈ égal-pondéré DJSC(35)+CBE3(15)+IEAC(20)
    fees_sat_default_bps: float = 200.0   # défaut conservateur pour fees non renseignées

    # ── Beta max poche satellite vs Core ──────────────────────────────────────
    beta_sat_max: float = 0.25

    # ── Au moins ce poids (de la poche sat) pour chaque bloc ─────────────────
    min_weight_per_bloc: float = 0.05

    # ── Nombre minimum d'observations requis dans chaque fenêtre ─────────────
    min_calib_obs:    int = 60
    min_backtest_obs: int = 100

    # ── Rebalancement trimestriel (~63 jours ouvrés) ──────────────────────────
    rebal_freq_days: int = 63

    # ── Fichiers d'entrée ────────────────────────────────────────────────────
    core_daily_csv:       str = str(project_root / "outputs" / "core_returns_daily_oos.csv")
    satellite_selected_csv: str = str(project_root / "outputs" / "satellite_selected.csv")
    price_paths: List[str] = field(default_factory=lambda: [
        str(project_root / "data" / "STRAT1_price.xlsx"),
        str(project_root / "data" / "STRAT2_price.xlsx"),
        str(project_root / "data" / "STRAT3_price.xlsx"),
    ])

    # ── Fichiers de sortie ───────────────────────────────────────────────────
    output_returns_csv:     str = str(project_root / "outputs" / "fond_returns_daily.csv")
    output_weights_csv:     str = str(project_root / "outputs" / "fond_weights.csv")
    output_metrics_csv:     str = str(project_root / "outputs" / "fond_metrics.csv")
    output_beta_rolling_csv: str = str(project_root / "outputs" / "fond_beta_rolling.csv")
    output_annual_csv:      str = str(project_root / "outputs" / "fond_annual_perf.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement des données
# ══════════════════════════════════════════════════════════════════════════════

def charger_core_rets(cfg: FondConfig) -> pd.Series:
    """Lit les log-rendements journaliers du Core et les convertit en simples."""
    df = pd.read_csv(cfg.core_daily_csv, index_col=0, parse_dates=True)
    s = np.exp(df.iloc[:, 0]) - 1.0
    s.name = "core"
    return s.sort_index().dropna()


def charger_prix_satellite(tickers: List[str], cfg: FondConfig) -> pd.DataFrame:
    """Charge les prix journaliers des fonds satellites sélectionnés."""
    frames: List[pd.DataFrame] = []
    for path in cfg.price_paths:
        wide = lire_prix_wide(path)
        cols = [t for t in tickers if t in wide.columns]
        if cols:
            frames.append(wide[cols])

    if not frames:
        raise ValueError("Aucun ticker satellite trouvé dans les fichiers de prix.")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"  ⚠  Tickers absents des fichiers prix : {missing}")

    found = [t for t in tickers if t in prices.columns]
    return prices[found]


# ══════════════════════════════════════════════════════════════════════════════
#  Poids satellite : equal-weight (1/n)
# ══════════════════════════════════════════════════════════════════════════════

def poids_satellite_equal_weight(
    sat_rets_calib: pd.DataFrame,
    core_rets_calib: pd.Series,
    cfg: FondConfig,
    blocs_dict: Dict[str, List[str]] | None = None,
) -> pd.Series:
    """
    Allocation equal-weight (1/n) de la poche satellite.

    Seuls les fonds ayant suffisamment d'observations dans la fenêtre
    de calibration sont retenus. On vérifie que chaque bloc est représenté.
    """
    combined = pd.concat(
        [core_rets_calib.rename("core"), sat_rets_calib], axis=1
    ).loc[cfg.calib_start:cfg.calib_end].dropna(subset=["core"])

    min_obs = max(60, int(0.30 * len(combined)))
    valid = [
        t for t in sat_rets_calib.columns
        if t in combined.columns and combined[t].dropna().shape[0] >= min_obs
    ]
    if not valid:
        raise ValueError("Aucun fonds satellite avec données suffisantes sur la calib window.")

    # Vérification : au moins 1 fonds par bloc
    if blocs_dict:
        for bloc_name, bloc_tickers in blocs_dict.items():
            if not any(t in valid for t in bloc_tickers):
                print(f"  ⚠  Bloc '{bloc_name}' : aucun fonds retenu")

    n = len(valid)
    w = np.ones(n) / n
    print(f"  Fonds retenus : {len(valid)} / {len(sat_rets_calib.columns)}")
    print(f"  Poids par fonds (equal-weight) : {1/n:.1%}")
    return pd.Series(w, index=valid, name="weight")


# ══════════════════════════════════════════════════════════════════════════════
#  Calibration de l'allocation Core / Satellite
# ══════════════════════════════════════════════════════════════════════════════

def calibrer_allocation(
    core_rets_calib: pd.Series,
    sat_pocket_calib: pd.Series,   # rendements de la poche satellite (θ déjà appliqués)
    cfg: FondConfig,
) -> Tuple[float, float]:
    """
    Trouve w_core ∈ [w_core_min, w_core_max] qui maintient la volatilité totale
    dans [vol_target_min, vol_target_max].
    Retourne (w_core, w_sat).
    """
    aligned = pd.concat(
        [core_rets_calib.rename("core"), sat_pocket_calib.rename("sat")], axis=1
    ).dropna()

    vc  = float(aligned["core"].var())
    vs  = float(aligned["sat"].var())
    ccs = float(aligned.cov().at["core", "sat"])

    def portfolio_vol(w_c: float) -> float:
        w_s = 1.0 - w_c
        return np.sqrt(252.0 * (w_c**2 * vc + w_s**2 * vs + 2.0 * w_c * w_s * ccs))

    candidates = np.linspace(cfg.w_core_min, cfg.w_core_max, 500)
    vols = np.array([portfolio_vol(w) for w in candidates])

    feasible_mask = (vols >= cfg.vol_target_min) & (vols <= cfg.vol_target_max)

    if feasible_mask.any():
        w_mid = (cfg.w_core_min + cfg.w_core_max) / 2.0
        dists = np.abs(candidates[feasible_mask] - w_mid)
        w_opt = float(candidates[feasible_mask][np.argmin(dists)])
    else:
        w_opt = float(candidates[np.argmin(np.abs(vols - cfg.vol_target_mid))])
        v_at_opt = portfolio_vol(w_opt)
        print(f"  ⚠  Vol cible [{cfg.vol_target_min:.0%}, {cfg.vol_target_max:.0%}] non atteignable "
              f"dans w_core ∈ [{cfg.w_core_min:.0%}, {cfg.w_core_max:.0%}]. "
              f"→ w_core = {w_opt:.1%}, vol ≈ {v_at_opt:.1%}")

    return w_opt, 1.0 - w_opt


# ══════════════════════════════════════════════════════════════════════════════
#  Backtest journalier avec rebalancement trimestriel
# ══════════════════════════════════════════════════════════════════════════════

def backtest(
    core_rets: pd.Series,       # rendements simples journaliers, plein historique
    sat_prices: pd.DataFrame,   # prix journaliers, plein historique
    sat_weights: pd.Series,     # θ optimisés (somme = 1)
    w_core: float,
    w_sat: float,
    cfg: FondConfig,
) -> pd.DataFrame:
    """
    Backtest journalier buy-and-hold entre les rebalancements (tous les ~63j).
    Poche Core    : utilise directement core_rets (Max-Sharpe rolling de core_pipeline).
    Poche Satellite: poids θ fixes, remis en place chaque trimestre.
    Retourne DataFrame ['core_ret', 'sat_pocket_ret', 'portfolio_ret'].
    """
    tickers  = sat_weights.index.tolist()
    gross_target = w_core + w_sat

    # Rendements simples satellite (forward-fill les NAV manquants)
    sat_rets_full = sat_prices[tickers].ffill().pct_change()

    # Fenêtre backtest
    cr = core_rets.loc[cfg.backtest_start:cfg.backtest_end]
    sr = sat_rets_full.loc[cfg.backtest_start:cfg.backtest_end]

    dates = cr.index.intersection(sr.index)
    cr = cr.loc[dates]
    sr = sr.loc[dates].fillna(0.0)

    theta = sat_weights.reindex(tickers).fillna(0.0).values   # shape (n,)
    n     = len(tickers)
    N     = len(dates)

    # Poids courants absolus dans le portefeuille total
    w_c = w_core
    w_s = theta * w_sat    # shape (n,)

    port_rets  = np.empty(N)
    sat_p_rets = np.empty(N)
    last_rebal = 0

    for idx in range(N):
        r_c = float(cr.iloc[idx])
        r_s = sr.iloc[idx].values            # shape (n,)

        # ── Rendement journalier ──────────────────────────────────────────────
        port_rets[idx] = w_c * r_c + w_s @ r_s
        denom = w_s.sum()
        sat_p_rets[idx] = (w_s / denom) @ r_s if denom > 1e-10 else 0.0

        # ── Drift des poids overnight (buy & hold) ────────────────────────────
        w_c_new = w_c * (1.0 + r_c)
        w_s_new = w_s * (1.0 + r_s)
        tot = w_c_new + w_s_new.sum()
        if tot > 1e-10:
            # Preserve the intended gross exposure (possibly > 100%).
            w_c = gross_target * (w_c_new / tot)
            w_s = gross_target * (w_s_new / tot)

        # ── Rebalancement trimestriel (remise aux poids cibles) ───────────────
        if (idx - last_rebal) >= cfg.rebal_freq_days and idx < N - 1:
            w_c = w_core
            w_s = theta * w_sat
            last_rebal = idx

    return pd.DataFrame(
        {"core_ret": cr.values, "sat_pocket_ret": sat_p_rets, "portfolio_ret": port_rets},
        index=dates,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Métriques de performance
# ══════════════════════════════════════════════════════════════════════════════

def _ols(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """OLS y = α + β·x → (alpha_daily, beta)."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(b[0]), float(b[1])


def beta_rolling(y: pd.Series, x: pd.Series, window: int = 63) -> pd.Series:
    """Rolling OLS beta de y sur x (fenêtre par défaut = 63j ≈ 3 mois)."""
    cov_ = y.rolling(window).cov(x)
    var_ = x.rolling(window).var()
    return (cov_ / var_).rename("beta_rolling_63j")


def calculer_metriques(
    bt_df:       pd.DataFrame,   # output de backtest()
    sat_weights: pd.Series,      # θ optimisés
    fees_bps:    pd.Series,      # frais en bps par ticker
    w_core:      float,
    w_sat:       float,
    cfg:         FondConfig,
) -> Dict:
    """Calcul complet des métriques sur la période de backtest."""
    r_p = bt_df["portfolio_ret"]
    r_c = bt_df["core_ret"]
    r_s = bt_df["sat_pocket_ret"]
    # ── Volatilité & rendement annualisés ─────────────────────────────────────
    vol_p = float(r_p.std() * np.sqrt(252))
    vol_c = float(r_c.std() * np.sqrt(252))
    vol_s = float(r_s.std() * np.sqrt(252))

    n_days = len(r_p)
    ann_p = float((1.0 + r_p).prod() ** (252.0 / n_days) - 1.0)
    ann_c = float((1.0 + r_c).prod() ** (252.0 / len(r_c)) - 1.0)
    ann_s = float((1.0 + r_s).prod() ** (252.0 / len(r_s)) - 1.0)

    sharpe_p = ann_p / vol_p if vol_p > 1e-6 else np.nan
    sharpe_c = ann_c / vol_c if vol_c > 1e-6 else np.nan

    alpha_p_daily, beta_p = _ols(r_p.values, r_c.values)
    alpha_p_ann = float((1.0 + alpha_p_daily) ** 252 - 1.0)

    # ── Alpha & Beta de la poche satellite seule ──────────────────────────────
    alpha_s_daily, beta_s_static = _ols(r_s.values, r_c.values)
    alpha_s_ann = float((1.0 + alpha_s_daily) ** 252 - 1.0)

    # ── Max Drawdown du portefeuille total ────────────────────────────────────
    cum_wealth = (1.0 + r_p).cumprod()
    peak       = cum_wealth.cummax()
    mdd        = float(((cum_wealth / peak) - 1.0).min())

    # ── Beta rolling 63j de la poche satellite vs Core ────────────────────────
    rb = beta_rolling(r_s, r_c, window=63)
    beta_sat_roll_mean = float(rb.dropna().mean())
    beta_sat_roll_std  = float(rb.dropna().std())

    # ── Frais estimés ─────────────────────────────────────────────────────────
    fees_arr        = fees_bps.reindex(sat_weights.index).fillna(cfg.fees_sat_default_bps).values
    fees_sat_wavg   = float(sat_weights.values @ fees_arr)           # bps poche sat
    fees_total_bps  = w_core * cfg.fees_core_bps + w_sat * fees_sat_wavg

    # ── Performance annuelle ──────────────────────────────────────────────────
    annual_df = pd.DataFrame({
        "portfolio": (1.0 + r_p).resample("YE").prod() - 1.0,
        "core":      (1.0 + r_c).resample("YE").prod() - 1.0,
        "satellite": (1.0 + r_s).resample("YE").prod() - 1.0,
    })
    annual_df.index = annual_df.index.year

    # ── Corrélation OOS ─────────────────────────────────────────────────────
    corr_core_sat = float(r_c.corr(r_s))

    return {
        # ── Portefeuille total ────────────────────────────────────────────────
        "vol_portfolio_ann":     vol_p,
        "ret_ann_portfolio":     ann_p,
        "sharpe_portfolio":      sharpe_p,
        "max_drawdown":          mdd,
        "alpha_portfolio_ann":   alpha_p_ann,
        "beta_portfolio":        beta_p,
        # ── Poche Core ────────────────────────────────────────────────────────
        "vol_core_ann":          vol_c,
        "ret_ann_core":          ann_c,
        "sharpe_core":           sharpe_c,
        # ── Poche Satellite ───────────────────────────────────────────────────
        "vol_satellite_ann":     vol_s,
        "ret_ann_satellite":     ann_s,
        "alpha_satellite_ann":   alpha_s_ann,
        "beta_satellite_static": beta_s_static,
        "beta_sat_rolling_mean": beta_sat_roll_mean,
        "beta_sat_rolling_std":  beta_sat_roll_std,
        "corr_core_satellite":   corr_core_sat,
        # ── Allocation ────────────────────────────────────────────────────────
        "w_core": w_core,
        "w_sat":  w_sat,
        # ── Frais ─────────────────────────────────────────────────────────────
        "fees_core_contrib_bps":  w_core * cfg.fees_core_bps,
        "fees_sat_wavg_bps":      fees_sat_wavg,
        "fees_sat_contrib_bps":   w_sat * fees_sat_wavg,
        "fees_total_bps":         fees_total_bps,
        "fees_ok":                fees_total_bps <= cfg.fees_bps_max,
        # ── Objets annexes (non exportés en scalaire) ─────────────────────────
        "_annual_df":              annual_df,
        "_beta_rolling_series":    rb,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = FondConfig()

    print("=" * 65)
    print("  CONSTRUCTION DU FONDS CORE-SATELLITE")
    print(f"  Calibration : {cfg.calib_start} → {cfg.calib_end}")
    print(f"  Backtest    : {cfg.backtest_start} → {cfg.backtest_end}")
    print("=" * 65)

    # ── [1] Chargement des données ────────────────────────────────────────────
    print("\n[1] Chargement des données...")
    core_rets = charger_core_rets(cfg)
    print(f"    Core  : {len(core_rets)} obs. | "
          f"{core_rets.index.min().date()} → {core_rets.index.max().date()}")

    sat_info     = pd.read_csv(cfg.satellite_selected_csv)
    tickers_sat  = sat_info["ticker"].tolist()

    # expense_pct en % (ex. 0.95 = 0.95 %) → conversion en bps
    fees_bps = (
        sat_info.set_index("ticker")["expense_pct"] * 100.0
    ).fillna(cfg.fees_sat_default_bps)

    sat_prices = charger_prix_satellite(tickers_sat, cfg)
    print(f"    Satellite : {sat_prices.shape[1]} fonds | "
          f"{sat_prices.index.min().date()} → {sat_prices.index.max().date()}")

    # Disponibilité par fonds dans les deux périodes (calib ET backtest)
    print("    Couverture par période :")
    tickers_with_backtest: List[str] = []
    for t in tickers_sat:
        if t in sat_prices.columns:
            sub_calib = sat_prices[t].loc[cfg.calib_start:cfg.calib_end].dropna()
            sub_oos   = sat_prices[t].loc[cfg.backtest_start:cfg.backtest_end].dropna()
            ok_calib  = len(sub_calib) >= cfg.min_calib_obs
            ok_oos    = len(sub_oos)   >= cfg.min_backtest_obs
            flag = "✓" if (ok_calib and ok_oos) else (
                "⚠ exclu (calib)" if not ok_calib else "⚠ exclu (backtest)"
            )
            print(f"      {t:<30s}  calib={len(sub_calib):3d} obs  oos={len(sub_oos):4d} obs  {flag}")
            if ok_calib and ok_oos:
                tickers_with_backtest.append(t)

    excluded = [t for t in tickers_sat if t not in tickers_with_backtest]
    if excluded:
        print(f"\n  → Fonds exclus (données insuffisantes sur calib ou backtest) : {excluded}")

    # ── [2] Rendements simples satellite ──────────────────────────────────────
    sat_rets_full  = sat_prices.ffill().pct_change()
    # Restreindre à ceux qui ont des données backtest
    sat_rets_calib = sat_rets_full[tickers_with_backtest].loc[cfg.calib_start:cfg.calib_end]
    core_calib     = core_rets.loc[cfg.calib_start:cfg.calib_end]

    # ── [3] Blocs pour contrainte min-par-bloc ────────────────────────────────
    blocs_dict: Dict[str, List[str]] = {}
    for _, row in sat_info[sat_info["ticker"].isin(tickers_with_backtest)].iterrows():
        b = str(row.get("bloc", "Inconnu"))
        blocs_dict.setdefault(b, []).append(row["ticker"])
    print(f"\n  Blocs ({len(blocs_dict)}) :", {b: len(v) for b, v in blocs_dict.items()})

    # ── [4] Allocation equal-weight des fonds satellite ──────────────────────
    print("\n[2] Allocation equal-weight des poids satellite...")
    sat_weights = poids_satellite_equal_weight(sat_rets_calib, core_calib, cfg, blocs_dict)

    print(f"\n  Poids satellite (equal-weight 1/n) :")
    for ticker, w in sat_weights.sort_values(ascending=False).items():
        bloc = sat_info.loc[sat_info["ticker"] == ticker, "bloc"].values
        bloc_str = bloc[0] if len(bloc) else "?"
        strat = sat_info.loc[sat_info["ticker"] == ticker, "strategie"].values
        strat_str = strat[0] if len(strat) else "?"
        print(f"    {ticker:<30s}  {w:6.1%}   [{bloc_str} – {strat_str}]")

    # ── [4] Rendements de la poche satellite sur calib ────────────────────────
    aligned_calib = pd.concat(
        [core_calib.rename("core"),
         sat_rets_calib[sat_weights.index].fillna(0.0)], axis=1
    ).dropna(subset=["core"])

    sat_pocket_calib = pd.Series(
        aligned_calib[sat_weights.index].values @ sat_weights.values,
        index=aligned_calib.index,
        name="sat_pocket",
    )
    core_calib_aligned = aligned_calib["core"]

    # Vérification beta + corrélation calib
    alpha_s_cal, beta_s_cal = _ols(sat_pocket_calib.values, core_calib_aligned.values)
    corr_calib = float(core_calib_aligned.corr(sat_pocket_calib))
    print(f"\n  Vérification calib 2019-2020 :")
    print(f"    β satellite vs Core = {beta_s_cal:+.3f}  (cible |β| ≤ {cfg.beta_sat_max:.2f})")
    print(f"    α satellite (ann.)  = {(1+alpha_s_cal)**252-1:+.2%}")
    print(f"    ρ(Core, Satellite)  = {corr_calib:+.3f}  (décorrélation structurelle)")

    # ── [5] Calibration de l'allocation w_core / w_sat ───────────────────────
    print("\n[3] Calibration de l'allocation Core / Satellite...")
    w_core, w_sat = calibrer_allocation(core_calib_aligned, sat_pocket_calib, cfg)

    # Vol réalisée sur calib
    vc  = float(core_calib_aligned.var())
    vs  = float(sat_pocket_calib.var())
    ccs = float(core_calib_aligned.cov(sat_pocket_calib))
    vol_total_calib = np.sqrt(252.0 * (
        w_core**2 * vc + w_sat**2 * vs + 2.0 * w_core * w_sat * ccs
    ))

    print(f"  w_core = {w_core:.1%}  |  w_sat = {w_sat:.1%}")
    print(f"  Vol calib → Core: {np.sqrt(252*vc):.1%}  "
          f"| Satellite: {np.sqrt(252*vs):.1%}  "
          f"| Total: {vol_total_calib:.1%}  "
          f"(cible [{cfg.vol_target_min:.0%}, {cfg.vol_target_max:.0%}])")

    # Pas de levier : exposition brute = 100 %
    scale_total = cfg.portfolio_scale   # = 1.0
    w_core_eff = w_core
    w_sat_eff  = w_sat

    print(f"  Exposition brute      Core {w_core_eff:.1%} | Satellite {w_sat_eff:.1%} "
          f"| Total {w_core_eff + w_sat_eff:.1%}  (pas de levier)")
    print(f"  Vol calib : {vol_total_calib:.1%}")

    # ── [6] Frais estimés ─────────────────────────────────────────────────────
    fees_arr_opt    = fees_bps.reindex(sat_weights.index).fillna(cfg.fees_sat_default_bps).values
    fees_sat_wavg   = float(sat_weights.values @ fees_arr_opt)
    fees_total_bps  = w_core_eff * cfg.fees_core_bps + w_sat_eff * fees_sat_wavg
    fees_ok         = fees_total_bps <= cfg.fees_bps_max

    print(f"\n[4] Frais estimés :")
    print(f"  Core      : {w_core_eff:.1%} × {cfg.fees_core_bps:.0f} bps "
          f"= {w_core_eff*cfg.fees_core_bps:.1f} bps")
    print(f"  Satellite : {w_sat_eff:.1%} × {fees_sat_wavg:.0f} bps "
          f"= {w_sat_eff*fees_sat_wavg:.1f} bps")
    print(f"  TOTAL     : {fees_total_bps:.1f} bps (budget {cfg.fees_bps_max:.0f} bps) "
          f"{'✓' if fees_ok else '⚠  DÉPASSÉ'}")

    # ── [7] Backtest OOS 2021-2025 ────────────────────────────────────────────
    print("\n[5] Backtest OOS 2021-2025...")
    bt_df = backtest(core_rets, sat_prices, sat_weights, w_core_eff, w_sat_eff, cfg)
    print(f"  {len(bt_df)} observations | "
          f"{bt_df.index.min().date()} → {bt_df.index.max().date()}")

    # ── [8] Métriques complètes ───────────────────────────────────────────────
    print("\n[6] Calcul des métriques de performance...")
    metrics = calculer_metriques(bt_df, sat_weights, fees_bps, w_core_eff, w_sat_eff, cfg)

    # ── Affichage ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RÉSULTATS – BACKTEST 2021-2025")
    print("=" * 65)

    print(f"\n  PORTEFEUILLE TOTAL")
    print(f"    Allocation           Core {metrics['w_core']:.1%}  |  Satellite {metrics['w_sat']:.1%}")
    print(f"    Rendement annualisé  {metrics['ret_ann_portfolio']:+.2%}")
    print(f"    Volatilité           {metrics['vol_portfolio_ann']:.2%}  "
          f"(cible [{cfg.vol_target_min:.0%}, {cfg.vol_target_max:.0%}])  "
          f"{'✓' if cfg.vol_target_min <= metrics['vol_portfolio_ann'] <= cfg.vol_target_max else '⚠'}")
    print(f"    Sharpe               {metrics['sharpe_portfolio']:.3f}")
    print(f"    Max Drawdown         {metrics['max_drawdown']:.2%}")

    print(f"\n  ALPHA / BETA vs CORE")
    print(f"    Alpha (portefeuille) {metrics['alpha_portfolio_ann']:+.2%}")
    print(f"    Beta  (portefeuille) {metrics['beta_portfolio']:+.3f}")

    print(f"\n  POCHE CORE (Max-Sharpe rolling)")
    print(f"    Rendement annualisé  {metrics['ret_ann_core']:+.2%}")
    print(f"    Volatilité           {metrics['vol_core_ann']:.2%}")
    print(f"    Sharpe               {metrics['sharpe_core']:.3f}")

    print(f"\n  POCHE SATELLITE")
    print(f"    Rendement annualisé  {metrics['ret_ann_satellite']:+.2%}")
    print(f"    Volatilité           {metrics['vol_satellite_ann']:.2%}")
    print(f"    Alpha vs Core (ann.) {metrics['alpha_satellite_ann']:+.2%}")
    print(f"    Beta statique        {metrics['beta_satellite_static']:+.3f}")
    print(f"    Beta rolling 3M – μ  {metrics['beta_sat_rolling_mean']:+.3f}  "
          f"(cible ≈ 0)")
    print(f"    Beta rolling 3M – σ  {metrics['beta_sat_rolling_std']:.3f}  "
          f"(stabilité : plus faible = mieux)")

    # Corrélation réalisée OOS entre Core et Satellite
    corr_oos = float(bt_df["core_ret"].corr(bt_df["sat_pocket_ret"]))
    print(f"\n  DÉCORRÉLATION STRUCTURELLE (OOS)")
    print(f"    ρ(Core, Satellite)   {corr_oos:+.3f}  "
          f"({'✓ décorrélé' if abs(corr_oos) < 0.30 else '⚠ corrélation élevée'})")

    print(f"\n  FRAIS")
    print(f"    Total estimé         {metrics['fees_total_bps']:.1f} bps "
          f"(budget {cfg.fees_bps_max:.0f} bps) "
          f"{'✓' if metrics['fees_ok'] else '⚠  DÉPASSÉ'}")

    print(f"\n  PERFORMANCE ANNUELLE :")
    annual_df = metrics["_annual_df"]
    header = f"    {'Année':>6}  {'Portefeuille':>13}  {'Core':>8}  {'Satellite':>10}  {'Excès':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for year, row in annual_df.iterrows():
        excess_y = row["portfolio"] - row["core"]
        print(f"    {year:>6}  {row['portfolio']:>+13.2%}  "
              f"{row['core']:>+8.2%}  {row['satellite']:>+10.2%}  {excess_y:>+8.2%}")

    # ── [9] Export des résultats ──────────────────────────────────────────────
    print("\n[7] Export des résultats...")

    bt_df.to_csv(cfg.output_returns_csv)
    print(f"  -> {cfg.output_returns_csv}")

    # Rendements individuels des fonds satellite sur la période OOS
    sat_rets_oos = sat_prices[sat_weights.index].ffill().pct_change()
    sat_rets_oos = sat_rets_oos.loc[cfg.backtest_start:cfg.backtest_end].fillna(0.0)
    sat_rets_oos.to_csv(str(project_root / "outputs" / "satellite_individual_returns.csv"))
    print(f"  -> {project_root / 'outputs' / 'satellite_individual_returns.csv'}")

    # Poids satellites + allocation globale
    weights_out = sat_weights.to_frame("theta_satellite")
    weights_out["absolute_weight"] = sat_weights * w_sat
    weights_out["w_core"] = w_core_eff
    weights_out["w_sat"]  = w_sat_eff
    weights_out["portfolio_scale"] = scale_total
    weights_out.to_csv(cfg.output_weights_csv)
    print(f"  -> {cfg.output_weights_csv}")

    # Métriques scalaires
    scalar_metrics = {k: (float(v) if isinstance(v, bool) else v)
                      for k, v in metrics.items() if not k.startswith("_")}
    pd.DataFrame([scalar_metrics]).T.rename(columns={0: "valeur"}).to_csv(cfg.output_metrics_csv)
    print(f"  -> {cfg.output_metrics_csv}")

    annual_df.to_csv(cfg.output_annual_csv)
    print(f"  -> {cfg.output_annual_csv}")

    metrics["_beta_rolling_series"].to_frame().to_csv(cfg.output_beta_rolling_csv)
    print(f"  -> {cfg.output_beta_rolling_csv}")

    print("\n  ✓  Fonds Core-Satellite construit avec succès.")


if __name__ == "__main__":
    main()
