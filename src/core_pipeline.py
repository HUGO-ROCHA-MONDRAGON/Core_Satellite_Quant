"""
Pipeline Core – v5 (Excel unifié wide format)

Sources de données :
  - 'univers_core_etf_eur_daily_wide_VF.xlsx'
      * Equity / Credit / Rates            : metadata (TER, devise, nom, exposition…)
      * Equity_Wide_Daily_Values            : prix daily wide (dates en lignes, tickers en colonnes)
      * Credit_Wide_Daily_Values            : idem
      * Rates_Wide_Daily_Values             : idem

Pipeline :
  1) Lecture prix Equity/Rates/Credit (format wide natif)
  2) Lecture metadata TER depuis les onglets descriptifs
  3) Filtrage structurel (date de démarrage, fréquence, expense ratio)
  4) Sélection 1 ETF « best » par thème (Equity, Rates, Credit)
  5) Backtest rolling OOS trimestriel (Max Sharpe, poids ∈ [5%, 50%])
  6) Export rendements journaliers du Core (3 ETFs sélectionnés)

Tous les ETF du fichier sont cotés en EUR (pas de hedged EUR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path


@dataclass(frozen=True)
class CoreConfig:
    """Paramètres du pipeline Core."""
    project_root: Path = Path(__file__).resolve().parent.parent

    # ── Fichier Excel unifié ───────────────────────
    core_excel: str = str(project_root / "univers_core_etf_eur_daily_wide_VF.xlsx")

    # Onglets prix (format wide : dates en lignes, tickers en colonnes)
    sheet_equity_prices: str = "Equity_Wide_Daily_Values"
    sheet_credit_prices: str = "Credit_Wide_Daily_Values"
    sheet_rates_prices: str = "Rates_Wide_Daily_Values"

    # Onglets metadata
    sheet_equity_meta: str = "Equity"
    sheet_credit_meta: str = "Credit"
    sheet_rates_meta: str = "Rates"

    # ── Filtres ─────────────────────────────────────
    max_start_date: str = "2019-01-01"
    max_avg_gap_days: float = 2.0
    total_fee_budget_bps: float = 80.0
    w_core_mid: float = 0.725
    satellite_expense_bps: float = 60.0
    require_expense_info: bool = False

    # ── Filtre exposition Equity ─────────────────────
    #    Ne retenir que les ETFs dont l'exposition contient un de ces mots-clés.
    #    Vide = pas de filtre (tout passe).
    equity_exposure_keywords: tuple = ("Europe",)

    # ── pick_best (IS = In-Sample) ──────────────────
    min_obs_pick_best: int = 250
    score_start: str = "2019-01-01"
    score_end: str = "2020-12-31"

    # ── Backtest ──────────────────────────────────
    lookback: int = 252
    rebal_freq: int = 63
    w_min: float = 0.05
    w_max: float = 0.50

    # ── Fenêtres IS / OOS ────────────────────────
    oos_start: str = "2021-01-01"
    oos_end: str = "2025-12-31"

    # ── Sorties ───────────────────────────────────
    output_core_daily_csv: str = str(project_root / "outputs" / "core_returns_daily_oos.csv")
    output_core_daily_is_csv: str = str(project_root / "outputs" / "core_returns_daily_is.csv")
    output_selected_core_csv: str = str(project_root / "outputs" / "core_selected_etfs.csv")
    output_core_finaux_csv: str = str(project_root / "outputs" / "Core_finaux.csv")

    @property
    def max_core_expense_pct(self) -> float:
        """Expense ratio max (%) pour un ETF core, dérivé du budget total."""
        w_sat = 1.0 - self.w_core_mid
        max_bps = (self.total_fee_budget_bps - w_sat * self.satellite_expense_bps) / self.w_core_mid
        return max_bps / 100.0


# ══════════════════════════════════════════════════
#  Lecture des données (nouveau format wide)
# ══════════════════════════════════════════════════

def _lire_wide_values(path: str, sheet: str) -> pd.DataFrame:
    """
    Lit un onglet *_Wide_Daily_Values :
      - Ligne 6 (0-indexed) = tickers Bloomberg (col 0 = "Bloomberg security")
      - Ligne 10+ = données (col 0 = dates, cols 1+ = prix)
    Retourne DataFrame wide : index=DatetimeIndex, columns=Bloomberg tickers.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    # Tickers depuis la ligne 6
    tickers_row = df.iloc[6].tolist()
    tickers = []
    for i, v in enumerate(tickers_row):
        if i == 0:
            continue  # col 0 = "Bloomberg security"
        if isinstance(v, str) and v.strip():
            tickers.append(v.strip())
        else:
            tickers.append(f"__col{i}__")

    # Données depuis la ligne 10
    data = df.iloc[10:].copy()
    data.columns = ["Date"] + tickers[:data.shape[1] - 1]

    # Parse dates
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    data = data.set_index("Date").sort_index()

    # Convertir en numérique
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Supprimer colonnes placeholder
    data = data[[c for c in data.columns if not c.startswith("__col")]]
    data.index.name = "Date"

    return data


def _lire_metadata(path: str, sheet: str) -> pd.DataFrame:
    """
    Lit un onglet metadata (Equity/Credit/Rates) :
      - Ligne 4 (0-indexed) = header
      - Ligne 5+ = données
    Retourne DataFrame avec Bloomberg ticker en index.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    # Header en ligne 4
    headers = df.iloc[4].tolist()
    # Normaliser
    col_map = {}
    for i, h in enumerate(headers):
        hs = str(h).strip().lower() if pd.notna(h) else f"col_{i}"
        if "bloomberg" in hs:
            col_map[i] = "ticker"
        elif "ter" in hs:
            col_map[i] = "ter_pct"
        elif "devise" in hs:
            col_map[i] = "devise"
        elif "nom" in hs:
            col_map[i] = "nom"
        elif "exposition" in hs or "indice" in hs:
            col_map[i] = "exposition"
        elif "provider" in hs:
            col_map[i] = "provider"
        elif "isin" in hs:
            col_map[i] = "isin"
        elif "encours" in hs:
            col_map[i] = "encours_eur_m"
        else:
            col_map[i] = hs

    data = df.iloc[5:].copy()
    data.columns = range(len(data.columns))
    data = data.rename(columns=col_map)

    if "ticker" in data.columns:
        data["ticker"] = data["ticker"].astype(str).str.strip()
        data = data.dropna(subset=["ticker"])
        data = data[data["ticker"] != "nan"]
        data = data.set_index("ticker")

    if "ter_pct" in data.columns:
        data["ter_pct"] = pd.to_numeric(data["ter_pct"], errors="coerce")
        # Si TER en décimal (0.002 = 0.2%), convertir en %
        if data["ter_pct"].dropna().max() < 0.1:
            data["ter_pct"] = data["ter_pct"] * 100.0

    if "encours_eur_m" in data.columns:
        data["encours_eur_m"] = pd.to_numeric(data["encours_eur_m"], errors="coerce")

    return data


def lire_theme(
    cfg: CoreConfig,
    theme_name: str,
    sheet_prices: str,
    sheet_meta: str,
    exposure_keywords: tuple = (),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lit prix + metadata d'un thème, applique filtres structurels.
    Si exposure_keywords est non-vide, ne retient que les ETFs dont le champ
    'exposition' dans la metadata contient au moins un des mots-clés.
    Retourne (wide_filtré, summary, metadata).
    """
    print(f"  Lecture {theme_name}...")
    wide = _lire_wide_values(cfg.core_excel, sheet_prices)
    meta = _lire_metadata(cfg.core_excel, sheet_meta)
    print(f"    -> {wide.shape[1]} ETFs | {wide.index.min().date()} à {wide.index.max().date()}")

    # ── Filtre exposition (si demandé) ────────────────────────────────────
    if exposure_keywords and "exposition" in meta.columns:
        allowed = []
        for t in wide.columns:
            if t in meta.index:
                expo = str(meta.at[t, "exposition"]).lower()
                if any(kw.lower() in expo for kw in exposure_keywords):
                    allowed.append(t)
            # Si pas de metadata et filtre actif → exclure (on ne peut pas vérifier)
        n_excl_expo = wide.shape[1] - len(allowed)
        if n_excl_expo > 0:
            print(f"    Filtre exposition ({', '.join(exposure_keywords)}) : "
                  f"{len(allowed)} retenus, {n_excl_expo} exclus")
        wide = wide[allowed]

    # Expense map depuis metadata
    expense_map: Dict[str, float] = {}
    if "ter_pct" in meta.columns:
        for t in meta.index:
            val = meta.at[t, "ter_pct"]
            if pd.notna(val):
                expense_map[t] = float(val)

    # Filtrage structurel
    max_date = pd.Timestamp(cfg.max_start_date)
    max_expense = cfg.max_core_expense_pct

    kept: List[str] = []
    rows: List[dict] = []

    for ticker in wide.columns:
        prices = wide[ticker].dropna()
        if len(prices) < 10:
            continue
        first_date = prices.index.min()
        n_obs = len(prices)
        avg_gap = prices.index.to_series().diff().dropna().dt.days.mean()
        expense = expense_map.get(ticker, np.nan)

        pass_date = first_date <= max_date
        pass_freq = avg_gap <= cfg.max_avg_gap_days
        if cfg.require_expense_info:
            pass_expense = (not np.isnan(expense)) and (expense <= max_expense)
        else:
            pass_expense = np.isnan(expense) or (expense <= max_expense)
        selected = pass_date and pass_freq and pass_expense

        rows.append(dict(
            ticker=ticker, first_date=first_date.strftime("%Y-%m-%d"),
            n_obs=n_obs, avg_gap=round(avg_gap, 2), expense_pct=expense,
            pass_date=pass_date, pass_freq=pass_freq, pass_expense=pass_expense,
            selected=selected,
        ))
        if selected:
            kept.append(ticker)

    summary = pd.DataFrame(rows)
    n_excl_date = (~summary["pass_date"]).sum() if len(summary) else 0
    n_excl_freq = (~summary["pass_freq"]).sum() if len(summary) else 0
    n_excl_exp = (~summary["pass_expense"]).sum() if len(summary) else 0
    print(f"    Filtrés : {len(kept)} retenus "
          f"(excl. date {n_excl_date}, freq {n_excl_freq}, frais {n_excl_exp})")

    wide_filtered = wide[kept].dropna(how="all").ffill()
    return wide_filtered, summary, meta


# ══════════════════════════════════════════════════
#  pick_best & optimisation
# ══════════════════════════════════════════════════

def pick_best_theme(
    theme_name: str,
    wide: pd.DataFrame,
    rets_log_calib: pd.DataFrame,
    min_obs: int,
) -> str:
    """
    Sélection du meilleur ETF pour un thème donné.
    Score = corrélation moyenne intra-thème + bonus obs − pénalité vol.
    """
    cands = [t for t in wide.columns
             if t in rets_log_calib.columns
             and rets_log_calib[t].dropna().shape[0] >= min(min_obs, len(rets_log_calib) // 2)]

    if len(cands) == 0:
        raise ValueError(f"Aucun candidat pour {theme_name} dans la fenêtre de calibration")

    if len(cands) == 1:
        return cands[0]

    R = rets_log_calib[cands].dropna(how="all")
    corr = R.corr()
    avg_corr = corr.mean(axis=1).values
    obs = np.array([rets_log_calib[t].dropna().shape[0] for t in cands], dtype=float)
    vol = R.std().values

    obs_z = (obs - obs.mean()) / (obs.std() + 1e-12)
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-12)

    score = avg_corr + 0.10 * obs_z - 0.10 * vol_z
    return cands[int(np.argmax(score))]


def optimiser_max_sharpe_contraint(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """Max Sharpe (rf=0) sous contraintes long-only et bornes [w_min, w_max]."""
    n = len(mu)
    x0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov @ w))
        if vol < 1e-12:
            return 1e10
        return -(float(w @ mu) / vol)

    bounds = [(w_min, w_max)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else x0


def backtest_rolling(
    prices: pd.DataFrame,
    lookback: int,
    rebal_freq: int,
    w_min: float,
    w_max: float,
    oos_start: str | None = None,
    oos_end: str | None = None,
    label: str = "OOS",
) -> pd.Series:
    """
    Backtest rolling trimestriel (Max Sharpe contraint).
    Si oos_start/oos_end sont fournis, ne retient que les rendements dans cette fenêtre.
    La calibration (lookback) utilise toujours les données précédant chaque rebalancement.
    """
    rets_log = np.log(prices).diff().dropna()
    dates = rets_log.index

    port_log_rets: List[float] = []
    port_dates: List[pd.Timestamp] = []

    for start in range(lookback, len(dates) - rebal_freq, rebal_freq):
        window = rets_log.iloc[start - lookback:start]
        oos = rets_log.iloc[start:start + rebal_freq]

        mu = window.mean().values * 252
        cov = window.cov().values * 252

        w = optimiser_max_sharpe_contraint(mu, cov, w_min, w_max)

        oos_port = oos.values @ w
        port_log_rets.extend(oos_port.tolist())
        port_dates.extend(oos.index.tolist())

    s = pd.Series(port_log_rets, index=pd.DatetimeIndex(port_dates), name=f"core_log_return_{label.lower()}")
    s = s.sort_index()

    # Filtrer sur la fenêtre demandée
    if oos_start:
        s = s.loc[oos_start:]
    if oos_end:
        s = s.loc[:oos_end]

    return s


def _print_perf_summary(log_rets: pd.Series, label: str) -> None:
    """Affiche un résumé de performance."""
    if len(log_rets) == 0:
        print(f"  {label}: aucune donnée")
        return
    cum = np.exp(log_rets.cumsum())
    perf = float(cum.iloc[-1] - 1)
    n_years = len(log_rets) / 252
    ann_ret = float((1 + perf) ** (1 / max(n_years, 0.01)) - 1)
    vol = float(log_rets.std() * np.sqrt(252))
    sharpe = ann_ret / vol if vol > 1e-10 else np.nan
    dd = float(((cum / cum.cummax()) - 1).min())
    print(f"  {label}: {log_rets.index.min().date()} → {log_rets.index.max().date()} | {len(log_rets)} jours")
    print(f"    Perf cumulée : {perf:+.2%}")
    print(f"    Ret annualisé: {ann_ret:+.2%}")
    print(f"    Vol annualisée: {vol:.2%}")
    print(f"    Sharpe        : {sharpe:.2f}")
    print(f"    Max Drawdown  : {dd:.2%}")


def main() -> None:
    cfg = CoreConfig()

    print("=" * 60)
    print("  PIPELINE CORE v5 – Format wide unifié")
    print(f"  Fichier : {Path(cfg.core_excel).name}")
    print(f"  IS (sélection + calib) : {cfg.score_start} → {cfg.score_end}")
    print(f"  OOS (validation)       : {cfg.oos_start} → {cfg.oos_end}")
    print("=" * 60)

    # ── 1) Lecture + filtrage des 3 thèmes ────────────────────────────────
    print("\n[1/6] Lecture et filtrage des données...")
    wide_eq, summary_eq, meta_eq = lire_theme(
        cfg, "Equity", cfg.sheet_equity_prices, cfg.sheet_equity_meta,
        exposure_keywords=cfg.equity_exposure_keywords)
    wide_rt, summary_rt, meta_rt = lire_theme(
        cfg, "Rates", cfg.sheet_rates_prices, cfg.sheet_rates_meta)
    wide_cr, summary_cr, meta_cr = lire_theme(
        cfg, "Credit", cfg.sheet_credit_prices, cfg.sheet_credit_meta)

    # ── 2) Sélection 1 ETF par thème ─────────────────────────────────────
    print(f"\n[2/6] Sélection pick_best (1 ETF / thème)...")
    print(f"  Fenêtre de scoring : {cfg.score_start} → {cfg.score_end}")

    rets_eq_calib = np.log(wide_eq).diff().loc[cfg.score_start:cfg.score_end]
    rets_rt_calib = np.log(wide_rt).diff().loc[cfg.score_start:cfg.score_end]
    rets_cr_calib = np.log(wide_cr).diff().loc[cfg.score_start:cfg.score_end]

    best_eq = pick_best_theme("Equity", wide_eq, rets_eq_calib, cfg.min_obs_pick_best)
    best_rt = pick_best_theme("Rates", wide_rt, rets_rt_calib, cfg.min_obs_pick_best)
    best_cr = pick_best_theme("Credit", wide_cr, rets_cr_calib, cfg.min_obs_pick_best)

    print(f"  -> Equity : {best_eq}")
    print(f"  -> Rates  : {best_rt}")
    print(f"  -> Credit : {best_cr}")

    core_etfs = [best_eq, best_rt, best_cr]
    core_themes = ["Equity", "Rates", "Credit"]

    # ── 3) Export sélection ───────────────────────────────────────────────
    print(f"\n[3/6] Export sélection Core...")
    Path(cfg.output_core_finaux_csv).parent.mkdir(parents=True, exist_ok=True)

    df_core_finaux = pd.DataFrame({"Ticker": core_etfs, "Theme": core_themes})
    df_core_finaux.to_csv(cfg.output_core_finaux_csv, index=False)
    print(f"  -> {cfg.output_core_finaux_csv}")

    pd.DataFrame({"core_etfs": core_etfs, "theme": core_themes})\
        .to_csv(cfg.output_selected_core_csv, index=False)
    print(f"  -> {cfg.output_selected_core_csv}")

    # ── 4) Wide combiné + log-rendements des 3 ETFs ──────────────────────
    print(f"\n[4/6] Construction du portefeuille Core...")
    wide_combined = pd.concat([
        wide_eq[[best_eq]],
        wide_rt[[best_rt]],
        wide_cr[[best_cr]],
    ], axis=1).sort_index().dropna(how="all").ffill()

    core_etf_log_daily = np.log(wide_combined).diff().dropna()
    etf_log_path = str(Path(cfg.output_core_daily_csv).parent / "core3_etf_daily_log_returns.csv")
    core_etf_log_daily.to_csv(etf_log_path)
    print(f"  -> {etf_log_path}")

    # ── 5) Backtest rolling : IS (2019-2020) + OOS (2021-2025) ─────────
    print(f"\n[5/7] Backtest rolling (Max Sharpe contraint)...")

    # IS : sélection et calibration sur 2019-2020
    print(f"\n  --- IS ({cfg.score_start} → {cfg.score_end}) ---")
    core_log_daily_is = backtest_rolling(
        wide_combined,
        lookback=cfg.lookback,
        rebal_freq=cfg.rebal_freq,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
        oos_start=cfg.score_start,
        oos_end=cfg.score_end,
        label="IS",
    )
    _print_perf_summary(core_log_daily_is, "IS")

    core_log_daily_is.to_frame().to_csv(cfg.output_core_daily_is_csv, index=True)
    print(f"    -> {cfg.output_core_daily_is_csv}")

    # OOS : validation sur 2021-2025
    print(f"\n  --- OOS ({cfg.oos_start} → {cfg.oos_end}) ---")
    core_log_daily_oos = backtest_rolling(
        wide_combined,
        lookback=cfg.lookback,
        rebal_freq=cfg.rebal_freq,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
        oos_start=cfg.oos_start,
        oos_end=cfg.oos_end,
        label="OOS",
    )
    _print_perf_summary(core_log_daily_oos, "OOS")

    core_log_daily_oos.to_frame().to_csv(cfg.output_core_daily_csv, index=True)
    print(f"    -> {cfg.output_core_daily_csv}")

    # ── 6) Corrélations inter-ETFs ───────────────────────────────────────
    print(f"\n[6/7] Corrélations inter-ETFs...")
    rets_3etf = core_etf_log_daily.dropna()
    for period, label in [(f"{cfg.score_start}/{cfg.score_end}", "IS"),
                          (f"{cfg.oos_start}/{cfg.oos_end}", "OOS")]:
        start, end = period.split("/")
        r = rets_3etf.loc[start:end]
        if len(r) > 30:
            corr = r.corr()
            print(f"\n  Corrélations {label} ({start} → {end}) :")
            for i, t1 in enumerate(corr.columns):
                for j, t2 in enumerate(corr.columns):
                    if j > i:
                        print(f"    {t1} vs {t2} : {corr.iloc[i, j]:.3f}")

    # ── 7) Résumé ────────────────────────────────────────────────────────
    print(f"\n[7/7] Résumé")
    print("=" * 60)
    for etf, theme in zip(core_etfs, core_themes):
        print(f"  {theme:10s} : {etf}")
    print(f"\n  IS  : {cfg.score_start} → {cfg.score_end}")
    print(f"  OOS : {cfg.oos_start} → {cfg.oos_end}")
    print("=" * 60)


if __name__ == "__main__":
    main()