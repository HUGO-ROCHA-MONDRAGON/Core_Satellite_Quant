"""
fond_construction.py – Construction et backtest du fonds Core-Satellite

Version alignée avec :
- core_pipeline_corrected.py
- satellite_pipeline_corrected.py

Choix principal :
- allocation satellite en equal-weight par défaut
- pas de levier
- Core lu en simple returns
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from satellite_pipeline_corrected import lire_prix_wide  # type: ignore
except ImportError:
    from satellite_pipeline import lire_prix_wide  # type: ignore

from risk_free import get_bund_risk_free_daily, sharpe_excess

project_root = Path(__file__).resolve().parent.parent


@dataclass
class FondConfig:
    """Paramètres de construction et de backtest du fonds Core-Satellite."""

    calib_start: str = "2019-01-01"
    calib_end: str = "2020-12-31"
    backtest_start: str = "2021-01-01"
    backtest_end: str = "2025-12-31"

    w_core_min: float = 0.70
    w_core_max: float = 1.00
    w_sat_abs_max: float = 0.30

    vol_target_min: float = 0.08
    vol_target_max: float = 0.12
    vol_target_mid: float = 0.12
    strict_vol_target: bool = False

    portfolio_scale: float = 1.00

    fees_bps_max: float = 80.0
    fees_core_bps: float = 23.0
    fees_sat_default_bps: float = 200.0
    fees_core_fallback_bps_per_etf: float = 25.0

    beta_sat_max: float = 0.25
    min_weight_per_bloc: float = 0.05

    min_calib_obs: int = 60
    min_backtest_obs: int = 100
    rebal_freq_days: int = 63

    # Modes possibles : "equal_weight" | "beta_inverse" | "score_prop" | "min_corr"
    satellite_alloc_mode: str = "beta_inverse"

    core_daily_csv: str = str(project_root / "outputs" / "core_returns_daily_oos.csv")
    core_daily_is_csv: str = str(project_root / "outputs" / "core_returns_daily_is.csv")
    core_selected_csv: str = str(project_root / "outputs" / "core_selected_etfs.csv")
    core3_etf_log_csv: str = str(project_root / "outputs" / "core3_etf_daily_log_returns.csv")
    core3_etf_simple_csv: str = str(project_root / "outputs" / "core3_etf_daily_simple_returns.csv")
    core_meta_excel: str = str(project_root / "data" / "univers_core_etf_eur_daily_wide.xlsx")
    core_meta_sheets: tuple = ("Equity", "Rates", "Credit")

    satellite_selected_csv: str = str(project_root / "outputs" / "satellite_selected.csv")
    satellite_selected_v3_csv: str = str(project_root / "outputs" / "satellite_selected_v3.csv")
    price_paths: List[str] = field(default_factory=lambda: [
        str(project_root / "data" / "STRAT1_price.xlsx"),
        str(project_root / "data" / "STRAT2_price.xlsx"),
        str(project_root / "data" / "STRAT3_price.xlsx"),
    ])

    output_returns_csv: str = str(project_root / "outputs" / "fond_returns_daily.csv")
    output_weights_csv: str = str(project_root / "outputs" / "fond_weights.csv")
    output_metrics_csv: str = str(project_root / "outputs" / "fond_metrics.csv")
    output_constraints_csv: str = str(project_root / "outputs" / "fond_constraints_check.csv")
    output_beta_rolling_csv: str = str(project_root / "outputs" / "fond_beta_rolling.csv")
    output_annual_csv: str = str(project_root / "outputs" / "fond_annual_perf.csv")


def _read_core_return_series(path: str) -> pd.Series:
    """Lit un CSV de rendements Core et détecte simple vs log."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"CSV vide : {path}")

    col = df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna().sort_index()
    is_log = "log" in col.lower() or "log" in Path(path).stem.lower()
    if is_log:
        s = np.exp(s) - 1.0
    s.name = "core"
    return s


def charger_core_rets(cfg: FondConfig) -> pd.Series:
    """Charge les rendements journaliers du Core sur IS + OOS."""
    parts: List[pd.Series] = []

    if Path(cfg.core_daily_is_csv).exists():
        parts.append(_read_core_return_series(cfg.core_daily_is_csv))
    if Path(cfg.core_daily_csv).exists():
        parts.append(_read_core_return_series(cfg.core_daily_csv))

    if parts:
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="first")].dropna()
        s.name = "core"
        return s

    simple_path = Path(cfg.core3_etf_simple_csv)
    if simple_path.exists():
        print("  ℹ  Fallback : reconstruction Core depuis core3_etf_daily_simple_returns.csv")
        df_simple = pd.read_csv(simple_path, index_col=0, parse_dates=True).sort_index()
        s = df_simple.mean(axis=1).dropna()
        s.name = "core"
        return s

    log_path = Path(cfg.core3_etf_log_csv)
    if log_path.exists():
        print("  ℹ  Fallback : reconstruction Core depuis core3_etf_daily_log_returns.csv")
        df_log = pd.read_csv(log_path, index_col=0, parse_dates=True).sort_index()
        df_simple = np.exp(df_log) - 1.0
        s = df_simple.mean(axis=1).dropna()
        s.name = "core"
        return s

    raise FileNotFoundError("Aucun fichier Core (IS, OOS ou ETF) trouvé.")


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

    prices = pd.concat(frames, axis=1, sort=False).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"  ⚠  Tickers absents des fichiers prix : {missing}")

    found = [t for t in tickers if t in prices.columns]
    return prices[found]


def estimer_frais_core_bps(cfg: FondConfig) -> float:
    """Estime les frais Core (bps/an) à partir des ETF sélectionnés."""
    try:
        sel = pd.read_csv(cfg.core_selected_csv)
        ticker_col = "core_etfs" if "core_etfs" in sel.columns else sel.columns[0]
        tickers = sel[ticker_col].astype(str).str.strip().dropna().tolist()
        if not tickers:
            return cfg.fees_core_bps

        exp_map: Dict[str, float] = {}
        for sheet in cfg.core_meta_sheets:
            try:
                df = pd.read_excel(cfg.core_meta_excel, sheet_name=sheet, header=None)
                headers = df.iloc[4].tolist()
                data = df.iloc[5:].copy()
                data.columns = range(len(data.columns))

                ticker_idx = None
                ter_idx = None
                for i, h in enumerate(headers):
                    hs = str(h).strip().lower() if pd.notna(h) else ""
                    if "bloomberg" in hs:
                        ticker_idx = i
                    elif "ter" in hs:
                        ter_idx = i

                if ticker_idx is None or ter_idx is None:
                    continue

                for _, row in data.iterrows():
                    t = str(row[ticker_idx]).strip() if pd.notna(row[ticker_idx]) else ""
                    ter = pd.to_numeric(row[ter_idx], errors="coerce")
                    if not t or pd.isna(ter) or t in exp_map:
                        continue
                    ter_pct = float(ter * 100.0) if float(ter) < 0.05 else float(ter)
                    exp_map[t] = ter_pct
            except Exception:
                continue

        fees_bps = []
        missing = 0
        for t in tickers:
            ter_pct = exp_map.get(t, np.nan)
            if pd.isna(ter_pct):
                missing += 1
                fees_bps.append(cfg.fees_core_fallback_bps_per_etf)
            else:
                fees_bps.append(float(ter_pct) * 100.0)

        est = float(np.mean(fees_bps))
        if missing:
            print(
                f"  ⚠  Frais Core: {missing}/{len(tickers)} ETF sans TER metadata, "
                f"fallback {cfg.fees_core_fallback_bps_per_etf:.0f} bps."
            )
        print(f"  Frais Core estimés (data-driven): {est:.1f} bps/an")
        return est
    except Exception as exc:
        print(f"  ⚠  Estimation frais Core impossible ({exc}); fallback {cfg.fees_core_bps:.1f} bps.")
        return cfg.fees_core_bps


def agreger_ret_satellite(sat_rets: pd.DataFrame, sat_weights: pd.Series) -> pd.Series:
    """Agrège les rendements satellite en renormalisant sur les fonds disponibles."""
    weights = sat_weights.reindex(sat_rets.columns).fillna(0.0).values
    out = np.full(len(sat_rets), np.nan, dtype=float)

    for i, row in enumerate(sat_rets.values):
        valid = np.isfinite(row)
        if not valid.any():
            continue
        wv = weights[valid]
        sw = wv.sum()
        if sw <= 1e-12:
            continue
        out[i] = float((wv / sw) @ row[valid])

    return pd.Series(out, index=sat_rets.index, name="sat_pocket")


def poids_satellite_equal_weight(
    sat_rets_calib: pd.DataFrame,
    core_rets_calib: pd.Series,
    cfg: FondConfig,
    blocs_dict: Dict[str, List[str]] | None = None,
) -> Tuple[pd.Series, str]:
    """Allocation equal-weight (1/n) de la poche satellite."""
    combined = pd.concat(
        [core_rets_calib.rename("core"), sat_rets_calib],
        axis=1,
        sort=False,
    ).sort_index()
    combined = combined.loc[cfg.calib_start:cfg.calib_end].dropna(subset=["core"])

    min_obs = max(60, int(0.30 * len(combined)))
    valid = [
        t for t in sat_rets_calib.columns
        if t in combined.columns and combined[t].dropna().shape[0] >= min_obs
    ]
    if not valid:
        raise ValueError("Aucun fonds satellite avec données suffisantes sur la calib window.")

    if blocs_dict:
        for bloc_name, bloc_tickers in blocs_dict.items():
            if not any(t in valid for t in bloc_tickers):
                print(f"  ⚠  Bloc '{bloc_name}' : aucun fonds retenu")

    n = len(valid)
    w = np.ones(n) / n

    print("\n  Mode d'allocation satellite : equal_weight")
    print(f"  Fonds retenus : {len(valid)} / {len(sat_rets_calib.columns)}")
    print(f"  Poids par fonds : {1/n:.1%}")
    print(f"  {'Ticker':<30s}  {'Poids':>6}")
    print("  " + "-" * 42)
    w_series = pd.Series(w, index=valid, name="weight")
    for t, wgt in w_series.sort_values(ascending=False).items():
        print(f"  {t:<30s}  {wgt:6.1%}")
    print(f"  {'TOTAL':<30s}  {w_series.sum():6.1%}")

    return w_series, "equal_weight"


def _ols(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """OLS y = α + β·x → (alpha_daily, beta)."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(b[0]), float(b[1])


def poids_satellite_decorr_first(
    sat_rets_calib: pd.DataFrame,
    core_rets_calib: pd.Series,
    cfg: FondConfig,
    blocs_dict: Dict[str, List[str]] | None = None,
    scores_is: pd.Series | None = None,
    mode: str = "beta_inverse",
) -> Tuple[pd.Series, str]:
    """Allocation décorrélation-first de la poche satellite."""
    combined = pd.concat(
        [core_rets_calib.rename("core"), sat_rets_calib],
        axis=1,
        sort=False,
    ).sort_index()
    combined = combined.loc[cfg.calib_start:cfg.calib_end].dropna(subset=["core"])

    min_obs = max(60, int(0.30 * len(combined)))
    valid = [
        t for t in sat_rets_calib.columns
        if t in combined.columns and combined[t].dropna().shape[0] >= min_obs
    ]
    if not valid:
        raise ValueError("Aucun fonds satellite avec données suffisantes sur la calib window.")

    if blocs_dict:
        for bloc_name, bloc_tickers in blocs_dict.items():
            if not any(t in valid for t in bloc_tickers):
                print(f"  ⚠  Bloc '{bloc_name}' : aucun fonds retenu")

    n = len(valid)

    betas: Dict[str, float] = {}
    for t in valid:
        ts = combined[t].dropna()
        if len(ts) >= 30:
            core_t = combined["core"].reindex(ts.index).dropna()
            ts_aligned = ts.reindex(core_t.index)
            if len(ts_aligned) >= 30:
                _, b = _ols(ts_aligned.values, core_t.values)
                betas[t] = b
            else:
                betas[t] = 1.0
        else:
            betas[t] = 1.0

    effective_mode = mode

    if mode == "score_prop":
        if scores_is is None or scores_is.reindex(valid).dropna().empty:
            print("  ⚠  scores_is absent ou tous NaN → fallback vers beta_inverse")
            effective_mode = "beta_inverse"

    if effective_mode == "beta_inverse":
        eps = 0.05
        inv_beta = np.array([1.0 / (abs(betas[t]) + eps) for t in valid])
        w_raw = inv_beta / inv_beta.sum()

    elif effective_mode == "score_prop":
        sc = scores_is.reindex(valid)
        sc = sc.fillna(float(sc.median()))
        sc_vals = sc.values.astype(float)
        sc_vals -= sc_vals.max()
        exp_sc = np.exp(sc_vals)
        w_raw = exp_sc / exp_sc.sum()

    elif effective_mode == "min_corr":
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError:
            print("  ⚠  scipy non disponible → fallback vers beta_inverse")
            effective_mode = "beta_inverse"
            eps = 0.05
            inv_beta = np.array([1.0 / (abs(betas[t]) + eps) for t in valid])
            w_raw = inv_beta / inv_beta.sum()
        else:
            sat_mat = combined[valid].dropna(how="all")
            cov_sat = sat_mat.cov().values.astype(float)
            if np.isnan(cov_sat).any():
                cov_sat = np.nan_to_num(cov_sat, nan=0.0)
                np.fill_diagonal(cov_sat, np.diag(cov_sat) + 1e-8)

            def _objective(w: np.ndarray) -> float:
                return float(w @ cov_sat @ w)

            def _jac(w: np.ndarray) -> np.ndarray:
                return 2.0 * cov_sat @ w

            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
            bounds = [(0.02, 0.60)] * n
            result = scipy_minimize(
                _objective,
                np.ones(n) / n,
                jac=_jac,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            if result.success:
                w_raw = np.maximum(result.x, 0.0)
                w_raw /= w_raw.sum()
            else:
                print(f"  ⚠  min_corr optimisation échouée ({result.message}) → fallback vers beta_inverse")
                effective_mode = "beta_inverse"
                eps = 0.05
                inv_beta = np.array([1.0 / (abs(betas[t]) + eps) for t in valid])
                w_raw = inv_beta / inv_beta.sum()

    else:
        raise ValueError(
            f"Mode d'allocation inconnu : '{mode}'. "
            "Valeurs valides : beta_inverse, score_prop, min_corr."
        )

    if blocs_dict and cfg.min_weight_per_bloc > 0.0:
        w_dict: Dict[str, float] = dict(zip(valid, w_raw.tolist()))
        for _ in range(10):
            changed = False
            for bloc_name, bloc_tickers in blocs_dict.items():
                bloc_in_valid = [t for t in bloc_tickers if t in w_dict]
                if not bloc_in_valid:
                    continue
                bloc_w = sum(w_dict[t] for t in bloc_in_valid)
                if bloc_w < cfg.min_weight_per_bloc - 1e-9:
                    deficit = cfg.min_weight_per_bloc - bloc_w
                    boost_per_fund = deficit / len(bloc_in_valid)
                    for t in bloc_in_valid:
                        w_dict[t] += boost_per_fund
                    others = [t for t in valid if t not in bloc_in_valid]
                    total_others = sum(w_dict[t] for t in others)
                    if total_others > deficit:
                        for t in others:
                            w_dict[t] -= deficit * (w_dict[t] / total_others)
                    changed = True
            total_w = sum(w_dict.values())
            if total_w > 1e-10:
                w_dict = {t: v / total_w for t, v in w_dict.items()}
            if not changed:
                break
        w_raw = np.array([w_dict[t] for t in valid])

    w_raw = np.maximum(w_raw, 0.0)
    w_raw /= w_raw.sum()

    print(f"\n  Mode d'allocation satellite : {effective_mode}")
    print(f"  {'Ticker':<30s}  {'Poids':>6}  {'Beta IS':>8}")
    print("  " + "-" * 52)
    w_series = pd.Series(w_raw, index=valid, name="weight")
    for t, w in w_series.sort_values(ascending=False).items():
        b = betas.get(t, float('nan'))
        print(f"  {t:<30s}  {w:6.1%}  {b:+8.3f}")
    print(f"  {'TOTAL':<30s}  {w_series.sum():6.1%}")

    return w_series, effective_mode


def calibrer_allocation(
    core_rets_calib: pd.Series,
    sat_pocket_calib: pd.Series,
    cfg: FondConfig,
) -> Tuple[float, float]:
    """Trouve w_core et w_sat qui approchent la cible de volatilité."""
    aligned = pd.concat(
        [core_rets_calib.rename("core"), sat_pocket_calib.rename("sat")],
        axis=1,
    ).dropna()

    vc = float(aligned["core"].var())
    vs = float(aligned["sat"].var())
    ccs = float(aligned.cov().at["core", "sat"])

    def portfolio_vol(w_c: float) -> float:
        w_s = 1.0 - w_c
        return np.sqrt(252.0 * (w_c**2 * vc + w_s**2 * vs + 2.0 * w_c * w_s * ccs))

    w_core_low = max(cfg.w_core_min, 1.0 - cfg.w_sat_abs_max)
    w_core_high = min(cfg.w_core_max, 1.0)
    if w_core_low > w_core_high:
        raise ValueError(
            "Contraintes incohérentes: bornes Core incompatibles avec le cap satellite. "
            f"w_core_low={w_core_low:.2%}, w_core_high={w_core_high:.2%}."
        )

    candidates = np.linspace(w_core_low, w_core_high, 500)
    vols = np.array([portfolio_vol(w) for w in candidates])

    feasible = (vols >= cfg.vol_target_min) & (vols <= cfg.vol_target_max)
    if feasible.any():
        # Choisir le poids qui vise au plus près la vol cible centrale,
        # plutôt qu'un simple milieu de borne de poids.
        cand_feas = candidates[feasible]
        vol_feas = vols[feasible]
        dists = np.abs(vol_feas - cfg.vol_target_mid)
        min_dist = float(np.min(dists))
        best = cand_feas[dists <= (min_dist + 1e-12)]
        # Tie-break conservateur: privilégier le poids Core le plus élevé.
        w_opt = float(np.max(best))
    else:
        if cfg.strict_vol_target:
            w_best = float(candidates[np.argmin(np.abs(vols - cfg.vol_target_mid))])
            v_best = portfolio_vol(w_best)
            raise ValueError(
                "Vol cible non atteignable dans la plage Core autorisée. "
                f"Cible [{cfg.vol_target_min:.0%}, {cfg.vol_target_max:.0%}], "
                f"meilleur point: w_core={w_best:.1%}, vol={v_best:.1%}."
            )
        w_opt = float(candidates[np.argmin(np.abs(vols - cfg.vol_target_mid))])
        v_at_opt = portfolio_vol(w_opt)
        print(
            f"  ⚠  Vol cible [{cfg.vol_target_min:.0%}, {cfg.vol_target_max:.0%}] non atteignable "
            f"dans w_core ∈ [{w_core_low:.0%}, {w_core_high:.0%}] → "
            f"w_core = {w_opt:.1%}, vol ≈ {v_at_opt:.1%}"
        )

    w_sat = 1.0 - w_opt
    if w_sat > cfg.w_sat_abs_max + 1e-12:
        raise ValueError(
            f"Contrainte satellite violée: w_sat={w_sat:.2%} > {cfg.w_sat_abs_max:.2%}."
        )

    return w_opt, w_sat


def exporter_tableau_contraintes(
    cfg: FondConfig,
    metrics: Dict,
    vol_calib: float,
    core_currency_ok: bool,
    sat_currency_ok: bool,
) -> None:
    """Exporte un tableau synthétique de conformité des contraintes mandat."""
    w_core = float(metrics["w_core"])
    w_sat = float(metrics["w_sat"])
    total_exp = w_core + w_sat
    vol_oos = float(metrics["vol_portfolio_ann"])
    fees_bps = float(metrics["fees_total_bps"])

    rows = [
        {
            "constraint": "Satellite <= 30%",
            "value": w_sat,
            "min": np.nan,
            "max": cfg.w_sat_abs_max,
            "ok": w_sat <= cfg.w_sat_abs_max + 1e-12,
        },
        {
            "constraint": "No leverage (gross <= 100%)",
            "value": total_exp,
            "min": np.nan,
            "max": 1.0,
            "ok": total_exp <= 1.0 + 1e-12,
        },
        {
            "constraint": "Vol cible calib [8%;12%]",
            "value": vol_calib,
            "min": cfg.vol_target_min,
            "max": cfg.vol_target_max,
            "ok": cfg.vol_target_min <= vol_calib <= cfg.vol_target_max,
        },
        {
            "constraint": "Vol cible OOS [8%;12%]",
            "value": vol_oos,
            "min": cfg.vol_target_min,
            "max": cfg.vol_target_max,
            "ok": cfg.vol_target_min <= vol_oos <= cfg.vol_target_max,
        },
        {
            "constraint": "Frais total <= 80 bps",
            "value": fees_bps,
            "min": np.nan,
            "max": cfg.fees_bps_max,
            "ok": fees_bps <= cfg.fees_bps_max + 1e-12,
        },
        {
            "constraint": "Core en EUR",
            "value": float(core_currency_ok),
            "min": np.nan,
            "max": np.nan,
            "ok": core_currency_ok,
        },
        {
            "constraint": "Satellite en EUR",
            "value": float(sat_currency_ok),
            "min": np.nan,
            "max": np.nan,
            "ok": sat_currency_ok,
        },
    ]

    out = pd.DataFrame(rows)
    out.to_csv(cfg.output_constraints_csv, index=False)
    print(f"  -> {cfg.output_constraints_csv}")


def backtest(
    core_rets: pd.Series,
    sat_prices: pd.DataFrame,
    sat_weights: pd.Series,
    w_core: float,
    w_sat: float,
    cfg: FondConfig,
) -> pd.DataFrame:
    """Backtest journalier buy-and-hold avec rebalancement trimestriel."""
    tickers = sat_weights.index.tolist()
    gross_target = w_core + w_sat

    sat_rets_full = sat_prices[tickers].ffill(limit=5).pct_change(fill_method=None)

    cr = core_rets.loc[cfg.backtest_start:cfg.backtest_end]
    sr = sat_rets_full.loc[cfg.backtest_start:cfg.backtest_end]

    dates = cr.index.intersection(sr.index)
    cr = cr.loc[dates]
    sr = sr.loc[dates]

    theta = sat_weights.reindex(tickers).fillna(0.0).values
    n_obs = len(dates)

    w_c = w_core
    w_s = theta * w_sat

    port_rets = np.empty(n_obs)
    sat_p_rets = np.empty(n_obs)
    last_rebal = 0

    for idx in range(n_obs):
        r_c = float(cr.iloc[idx])
        r_s = sr.iloc[idx].values
        valid = np.isfinite(r_s)
        r_s_eff = np.where(valid, r_s, 0.0)

        if valid.any():
            w_valid = w_s[valid]
            denom = w_valid.sum()
            sat_ret_t = float((w_valid / denom) @ r_s[valid]) if denom > 1e-10 else 0.0
            sat_p_rets[idx] = sat_ret_t
            port_rets[idx] = w_c * r_c + w_sat * sat_ret_t
        else:
            sat_p_rets[idx] = 0.0
            port_rets[idx] = w_c * r_c

        w_c_new = w_c * (1.0 + r_c)
        w_s_new = w_s * (1.0 + r_s_eff)
        tot = w_c_new + w_s_new.sum()
        if tot > 1e-10:
            w_c = gross_target * (w_c_new / tot)
            w_s = gross_target * (w_s_new / tot)

        if (idx - last_rebal) >= cfg.rebal_freq_days and idx < n_obs - 1:
            w_c = w_core
            w_s = theta * w_sat
            last_rebal = idx

    return pd.DataFrame(
        {
            "core_ret": cr.values,
            "sat_pocket_ret": sat_p_rets,
            "portfolio_ret": port_rets,
        },
        index=dates,
    )


def beta_rolling(y: pd.Series, x: pd.Series, window: int = 126) -> pd.Series:
    """Rolling OLS beta de y sur x."""
    cov_ = y.rolling(window).cov(x)
    var_ = x.rolling(window).var()
    return (cov_ / var_).rename("beta_rolling_126j")


def _yearly_perf(r: pd.Series) -> pd.Series:
    """Performance annuelle robuste selon version pandas."""
    try:
        return (1.0 + r).resample("YE").prod() - 1.0
    except Exception:
        return (1.0 + r).resample("Y").prod() - 1.0


def calculer_metriques(
    bt_df: pd.DataFrame,
    sat_weights: pd.Series,
    fees_bps: pd.Series,
    w_core: float,
    w_sat: float,
    core_fees_bps: float,
    cfg: FondConfig,
    rf_daily: pd.Series,
    rf_source: str,
) -> Dict:
    """Calcul complet des métriques sur la période de backtest."""
    r_p = bt_df["portfolio_ret"]
    r_c = bt_df["core_ret"]
    r_s = bt_df["sat_pocket_ret"]

    vol_p = float(r_p.std() * np.sqrt(252))
    vol_c = float(r_c.std() * np.sqrt(252))
    vol_s = float(r_s.std() * np.sqrt(252))

    ann_p = float((1.0 + r_p).prod() ** (252.0 / len(r_p)) - 1.0)
    ann_c = float((1.0 + r_c).prod() ** (252.0 / len(r_c)) - 1.0)
    ann_s = float((1.0 + r_s).prod() ** (252.0 / len(r_s)) - 1.0)

    rf_bt = rf_daily.reindex(bt_df.index).ffill().fillna(0.0)
    sharpe_p = sharpe_excess(r_p, rf_bt)
    sharpe_c = sharpe_excess(r_c, rf_bt)
    sharpe_s = sharpe_excess(r_s, rf_bt)
    excess_p_ann = float((r_p - rf_bt).mean() * 252.0)
    excess_c_ann = float((r_c - rf_bt).mean() * 252.0)
    excess_s_ann = float((r_s - rf_bt).mean() * 252.0)

    alpha_p_daily, beta_p = _ols(r_p.values, r_c.values)
    alpha_p_ann = float((1.0 + alpha_p_daily) ** 252 - 1.0)

    alpha_s_daily, beta_s_static = _ols(r_s.values, r_c.values)
    alpha_s_ann = float((1.0 + alpha_s_daily) ** 252 - 1.0)

    cum_wealth = (1.0 + r_p).cumprod()
    peak = cum_wealth.cummax()
    mdd = float(((cum_wealth / peak) - 1.0).min())

    rb = beta_rolling(r_s, r_c, window=126)
    beta_sat_roll_mean = float(rb.dropna().mean())
    beta_sat_roll_std = float(rb.dropna().std())

    fees_arr = fees_bps.reindex(sat_weights.index).fillna(cfg.fees_sat_default_bps).values
    fees_sat_wavg = float(sat_weights.values @ fees_arr)
    fees_total_bps = w_core * core_fees_bps + w_sat * fees_sat_wavg

    annual_df = pd.DataFrame({
        "portfolio": _yearly_perf(r_p),
        "core": _yearly_perf(r_c),
        "satellite": _yearly_perf(r_s),
    })
    annual_df.index = annual_df.index.year

    corr_core_sat = float(r_c.corr(r_s))

    return {
        "vol_portfolio_ann": vol_p,
        "ret_ann_portfolio": ann_p,
        "ret_ann_excess_portfolio": excess_p_ann,
        "sharpe_portfolio": sharpe_p,
        "max_drawdown": mdd,
        "alpha_portfolio_ann": alpha_p_ann,
        "beta_portfolio": beta_p,
        "vol_core_ann": vol_c,
        "ret_ann_core": ann_c,
        "ret_ann_excess_core": excess_c_ann,
        "sharpe_core": sharpe_c,
        "vol_satellite_ann": vol_s,
        "ret_ann_satellite": ann_s,
        "ret_ann_excess_satellite": excess_s_ann,
        "sharpe_satellite": sharpe_s,
        "alpha_satellite_ann": alpha_s_ann,
        "beta_satellite_static": beta_s_static,
        "beta_sat_rolling_mean": beta_sat_roll_mean,
        "beta_sat_rolling_std": beta_sat_roll_std,
        "corr_core_satellite": corr_core_sat,
        "w_core": w_core,
        "w_sat": w_sat,
        "fees_core_contrib_bps": w_core * core_fees_bps,
        "fees_sat_wavg_bps": fees_sat_wavg,
        "fees_sat_contrib_bps": w_sat * fees_sat_wavg,
        "fees_total_bps": fees_total_bps,
        "fees_ok": fees_total_bps <= cfg.fees_bps_max,
        "rf_source": rf_source,
        "_annual_df": annual_df,
        "_beta_rolling_series": rb,
    }


def main() -> None:
    cfg = FondConfig()

    print("=" * 65)
    print("  CONSTRUCTION DU FONDS CORE-SATELLITE")
    print(f"  Calibration : {cfg.calib_start} → {cfg.calib_end}")
    print(f"  Backtest    : {cfg.backtest_start} → {cfg.backtest_end}")
    print("=" * 65)

    print("\n[1] Chargement des données...")
    core_fees_bps = estimer_frais_core_bps(cfg)
    core_rets = charger_core_rets(cfg)
    print(f"    Core  : {len(core_rets)} obs. | {core_rets.index.min().date()} → {core_rets.index.max().date()}")

    sat_info = pd.read_csv(cfg.satellite_selected_csv)
    tickers_sat = sat_info["ticker"].astype(str).tolist()

    sat_currency_ok = True
    if "devise" in sat_info.columns:
        sat_currency_ok = sat_info["devise"].astype(str).str.lower().eq("euro").all()

    core_currency_ok = True
    core_finaux_path = project_root / "outputs" / "Core_finaux.csv"
    if core_finaux_path.exists():
        core_finaux = pd.read_csv(core_finaux_path)
        if "devise" in core_finaux.columns:
            core_currency_ok = core_finaux["devise"].astype(str).str.upper().eq("EUR").all()

    scores_is: pd.Series | None = None
    v3_path = Path(cfg.satellite_selected_v3_csv)
    if v3_path.exists():
        try:
            df_v3 = pd.read_csv(v3_path)
            if "ticker" in df_v3.columns and "score" in df_v3.columns:
                scores_is = df_v3.set_index("ticker")["score"].dropna()
                print(f"    Scores IS chargés depuis satellite_selected_v3.csv ({len(scores_is)} fonds)")
        except Exception as exc:
            print(f"  ⚠  Impossible de lire satellite_selected_v3.csv : {exc}")

    fees_bps = (sat_info.set_index("ticker")["expense_pct"] * 100.0).fillna(cfg.fees_sat_default_bps)

    sat_prices = charger_prix_satellite(tickers_sat, cfg).sort_index()
    print(f"    Satellite : {sat_prices.shape[1]} fonds | {sat_prices.index.min().date()} → {sat_prices.index.max().date()}")

    print("\n  Track record des fonds satellite sélectionnés :")
    print(f"  {'Ticker':<32s}  {'Premier prix':>12s}  {'Nb obs calib':>12s}  {'Nb obs OOS':>10s}")
    for t in tickers_sat:
        if t in sat_prices.columns:
            first_date = sat_prices[t].dropna().index.min()
            sub_calib = sat_prices[t].loc[cfg.calib_start:cfg.calib_end].dropna()
            sub_oos = sat_prices[t].loc[cfg.backtest_start:cfg.backtest_end].dropna()
            print(f"  {t:<32s}  {str(first_date.date()):>12s}  {len(sub_calib):>12d}  {len(sub_oos):>10d}")
        else:
            print(f"  {t:<32s}  {'ABSENT':>12s}")

    print("    Couverture par période :")
    tickers_with_backtest: List[str] = []
    for t in tickers_sat:
        if t in sat_prices.columns:
            sub_calib = sat_prices[t].loc[cfg.calib_start:cfg.calib_end].dropna()
            sub_oos = sat_prices[t].loc[cfg.backtest_start:cfg.backtest_end].dropna()
            ok_calib = len(sub_calib) >= cfg.min_calib_obs
            ok_oos = len(sub_oos) >= cfg.min_backtest_obs
            flag = "✓" if (ok_calib and ok_oos) else ("⚠ exclu (calib)" if not ok_calib else "⚠ exclu (backtest)")
            print(f"      {t:<30s}  calib={len(sub_calib):3d} obs  oos={len(sub_oos):4d} obs  {flag}")
            if ok_calib and ok_oos:
                tickers_with_backtest.append(t)

    excluded = [t for t in tickers_sat if t not in tickers_with_backtest]
    if excluded:
        print(f"\n  → Fonds exclus (données insuffisantes sur calib ou backtest) : {excluded}")

    sat_rets_full = sat_prices.ffill(limit=5).pct_change(fill_method=None).sort_index()
    sat_rets_calib = sat_rets_full[tickers_with_backtest].loc[cfg.calib_start:cfg.calib_end].sort_index()
    core_calib = core_rets.loc[cfg.calib_start:cfg.calib_end].sort_index()

    blocs_dict: Dict[str, List[str]] = {}
    for _, row in sat_info[sat_info["ticker"].isin(tickers_with_backtest)].iterrows():
        bloc = str(row.get("bloc", "Inconnu"))
        blocs_dict.setdefault(bloc, []).append(row["ticker"])
    print(f"\n  Blocs ({len(blocs_dict)}) :", {b: len(v) for b, v in blocs_dict.items()})

    print(f"\n[2] Allocation satellite (mode demandé : {cfg.satellite_alloc_mode})...")
    if cfg.satellite_alloc_mode == "equal_weight":
        sat_weights, alloc_mode_used = poids_satellite_equal_weight(
            sat_rets_calib,
            core_calib,
            cfg,
            blocs_dict,
        )
    else:
        sat_weights, alloc_mode_used = poids_satellite_decorr_first(
            sat_rets_calib,
            core_calib,
            cfg,
            blocs_dict,
            scores_is=scores_is,
            mode=cfg.satellite_alloc_mode,
        )

    sat_rets_calib_sel = sat_rets_calib[sat_weights.index]
    sat_pocket_calib = agreger_ret_satellite(sat_rets_calib_sel, sat_weights)
    aligned_calib = pd.concat(
        [core_calib.rename("core"), sat_pocket_calib.rename("sat_pocket")],
        axis=1,
        sort=False,
    ).sort_index().dropna()
    sat_pocket_calib = aligned_calib["sat_pocket"]
    core_calib_aligned = aligned_calib["core"]

    alpha_s_cal, beta_s_cal = _ols(sat_pocket_calib.values, core_calib_aligned.values)
    corr_calib = float(core_calib_aligned.corr(sat_pocket_calib))
    print("\n  Vérification calib 2019-2020 :")
    print(f"    β satellite vs Core = {beta_s_cal:+.3f}  (cible |β| ≤ {cfg.beta_sat_max:.2f})")
    print(f"    α satellite (ann.)  = {(1 + alpha_s_cal) ** 252 - 1:+.2%}")
    print(f"    ρ(Core, Satellite)  = {corr_calib:+.3f}  (décorrélation structurelle)")

    print("\n[3] Calibration de l'allocation Core / Satellite...")
    w_core, w_sat = calibrer_allocation(core_calib_aligned, sat_pocket_calib, cfg)

    vc = float(core_calib_aligned.var())
    vs = float(sat_pocket_calib.var())
    ccs = float(core_calib_aligned.cov(sat_pocket_calib))
    vol_total_calib = np.sqrt(252.0 * (w_core**2 * vc + w_sat**2 * vs + 2.0 * w_core * w_sat * ccs))

    print(f"  w_core = {w_core:.1%}  |  w_sat = {w_sat:.1%}")
    print(
        f"  Vol calib → Core: {np.sqrt(252 * vc):.1%}  | Satellite: {np.sqrt(252 * vs):.1%}  | "
        f"Total: {vol_total_calib:.1%}  (cible [{cfg.vol_target_min:.0%}, {cfg.vol_target_max:.0%}])"
    )

    scale_total = cfg.portfolio_scale
    w_core_eff = w_core
    w_sat_eff = w_sat

    if w_sat_eff > cfg.w_sat_abs_max + 1e-12:
        raise ValueError(
            f"Contrainte satellite violée après calibration: {w_sat_eff:.2%} > {cfg.w_sat_abs_max:.2%}"
        )
    if (w_core_eff + w_sat_eff) > 1.0 + 1e-12:
        raise ValueError(
            f"Contrainte sans levier violée: exposition totale={w_core_eff + w_sat_eff:.2%}"
        )

    print(f"  Exposition brute      Core {w_core_eff:.1%} | Satellite {w_sat_eff:.1%} | Total {w_core_eff + w_sat_eff:.1%}  (pas de levier)")
    print(f"  Vol calib : {vol_total_calib:.1%}")

    fees_arr_opt = fees_bps.reindex(sat_weights.index).fillna(cfg.fees_sat_default_bps).values
    fees_sat_wavg = float(sat_weights.values @ fees_arr_opt)
    fees_total_bps = w_core_eff * core_fees_bps + w_sat_eff * fees_sat_wavg
    fees_ok = fees_total_bps <= cfg.fees_bps_max

    print("\n[4] Frais estimés :")
    print(f"  Core      : {w_core_eff:.1%} × {core_fees_bps:.0f} bps = {w_core_eff * core_fees_bps:.1f} bps")
    print(f"  Satellite : {w_sat_eff:.1%} × {fees_sat_wavg:.0f} bps = {w_sat_eff * fees_sat_wavg:.1f} bps")
    print(f"  TOTAL     : {fees_total_bps:.1f} bps (budget {cfg.fees_bps_max:.0f} bps) {'✓' if fees_ok else '⚠  DÉPASSÉ'}")

    print("\n[5] Backtest OOS 2021-2025...")
    bt_df = backtest(core_rets, sat_prices, sat_weights, w_core_eff, w_sat_eff, cfg)
    print(f"  {len(bt_df)} observations | {bt_df.index.min().date()} → {bt_df.index.max().date()}")

    rf_daily, rf_source = get_bund_risk_free_daily(bt_df.index)
    print(f"  Taux sans risque (Bund proxy): {rf_source}")

    print("\n[6] Calcul des métriques de performance...")
    metrics = calculer_metriques(
        bt_df,
        sat_weights,
        fees_bps,
        w_core_eff,
        w_sat_eff,
        core_fees_bps,
        cfg,
        rf_daily,
        rf_source,
    )

    print("\n" + "=" * 65)
    print("  RÉSULTATS – BACKTEST 2021-2025")
    print("=" * 65)

    print("\n  PORTEFEUILLE TOTAL")
    print(f"    Allocation           Core {metrics['w_core']:.1%}  |  Satellite {metrics['w_sat']:.1%}")
    print(f"    Rendement annualisé  {metrics['ret_ann_portfolio']:+.2%}")
    print(
        f"    Volatilité           {metrics['vol_portfolio_ann']:.2%}  "
        f"(cible [{cfg.vol_target_min:.0%}, {cfg.vol_target_max:.0%}])  "
        f"{'✓' if cfg.vol_target_min <= metrics['vol_portfolio_ann'] <= cfg.vol_target_max else '⚠'}"
    )
    print(f"    Sharpe (exces rf)    {metrics['sharpe_portfolio']:.3f}")
    print(f"    Max Drawdown         {metrics['max_drawdown']:.2%}")

    print("\n  ALPHA / BETA vs CORE")
    print(f"    Alpha (portefeuille) {metrics['alpha_portfolio_ann']:+.2%}")
    print(f"    Beta  (portefeuille) {metrics['beta_portfolio']:+.3f}")

    print("\n  POCHE CORE")
    print(f"    Rendement annualisé  {metrics['ret_ann_core']:+.2%}")
    print(f"    Volatilité           {metrics['vol_core_ann']:.2%}")
    print(f"    Sharpe (exces rf)    {metrics['sharpe_core']:.3f}")

    print("\n  POCHE SATELLITE")
    print(f"    Rendement annualisé  {metrics['ret_ann_satellite']:+.2%}")
    print(f"    Volatilité           {metrics['vol_satellite_ann']:.2%}")
    print(f"    Sharpe (exces rf)    {metrics['sharpe_satellite']:.3f}")
    print(f"    Alpha vs Core (ann.) {metrics['alpha_satellite_ann']:+.2%}")
    print(f"    Beta statique        {metrics['beta_satellite_static']:+.3f}")
    print(f"    Beta rolling 6M – μ  {metrics['beta_sat_rolling_mean']:+.3f}  (cible ≈ 0)")
    print(f"    Beta rolling 6M – σ  {metrics['beta_sat_rolling_std']:.3f}  (stabilité : plus faible = mieux)")

    corr_oos = float(bt_df["core_ret"].corr(bt_df["sat_pocket_ret"]))
    print("\n  DÉCORRÉLATION STRUCTURELLE (OOS)")
    print(f"    ρ(Core, Satellite)   {corr_oos:+.3f}  ({'✓ décorrélé' if abs(corr_oos) < 0.30 else '⚠ corrélation élevée'})")

    print("\n  FRAIS")
    print(f"    Total estimé         {metrics['fees_total_bps']:.1f} bps (budget {cfg.fees_bps_max:.0f} bps) {'✓' if metrics['fees_ok'] else '⚠  DÉPASSÉ'}")

    print("\n  PERFORMANCE ANNUELLE :")
    annual_df = metrics["_annual_df"]
    header = f"    {'Année':>6}  {'Portefeuille':>13}  {'Core':>8}  {'Satellite':>10}  {'Excès':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for year, row in annual_df.iterrows():
        excess_y = row["portfolio"] - row["core"]
        print(
            f"    {year:>6}  {row['portfolio']:>+13.2%}  "
            f"{row['core']:>+8.2%}  {row['satellite']:>+10.2%}  {excess_y:>+8.2%}"
        )

    print("\n[7] Export des résultats...")
    bt_df.to_csv(cfg.output_returns_csv)
    print(f"  -> {cfg.output_returns_csv}")

    sat_rets_oos = sat_prices[sat_weights.index].ffill(limit=5).pct_change(fill_method=None)
    sat_rets_oos = sat_rets_oos.loc[cfg.backtest_start:cfg.backtest_end]
    sat_individual_path = project_root / "outputs" / "satellite_individual_returns.csv"
    sat_rets_oos.to_csv(sat_individual_path)
    print(f"  -> {sat_individual_path}")

    weights_out = sat_weights.to_frame("theta_satellite")
    weights_out["absolute_weight"] = sat_weights * w_sat
    weights_out["w_core"] = w_core_eff
    weights_out["w_sat"] = w_sat_eff
    weights_out["portfolio_scale"] = scale_total
    weights_out["allocation_mode"] = alloc_mode_used
    weights_out.to_csv(cfg.output_weights_csv)
    print(f"  -> {cfg.output_weights_csv}")

    scalar_metrics = {
        k: (float(v) if isinstance(v, bool) else v)
        for k, v in metrics.items()
        if not k.startswith("_")
    }
    pd.DataFrame([scalar_metrics]).T.rename(columns={0: "valeur"}).to_csv(cfg.output_metrics_csv)
    print(f"  -> {cfg.output_metrics_csv}")

    annual_df.to_csv(cfg.output_annual_csv)
    print(f"  -> {cfg.output_annual_csv}")

    metrics["_beta_rolling_series"].to_frame().to_csv(cfg.output_beta_rolling_csv)
    print(f"  -> {cfg.output_beta_rolling_csv}")

    print("\n[8] Export contrôle de conformité...")
    exporter_tableau_contraintes(
        cfg=cfg,
        metrics=metrics,
        vol_calib=float(vol_total_calib),
        core_currency_ok=bool(core_currency_ok),
        sat_currency_ok=bool(sat_currency_ok),
    )

    print("\n  ✓  Fonds Core-Satellite construit avec succès.")


if __name__ == "__main__":
    main()