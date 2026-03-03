"""
Pipeline Core (réutilisation du notebook du groupe, version "propre" pour PyCharm).

Ce script :
1) lit core_ETF.xlsx (avec 3 lignes de métadonnées en tête)
2) reconstruit une table de prix "wide" (Date x Tickers)
3) reconstruit le mapping ticker -> bucket
4) définit les thèmes Equity / Rates / Credit
5) sélectionne 1 ETF "best" par thème (pick_best)
6) construit un Core à 3 ETFs (Equity/Rates/Credit)
7) lance un backtest rolling trimestriel OOS (lookback 252, rebal 63)
   avec la méthode : Max Sharpe sous contraintes (poids ∈ [5%, 50%])
8) exporte les rendements mensuels du Core :
   - core_returns_monthly.csv (colonne core_portfolio_return)

Notes :
- On conserve la logique du notebook : estimation sur log-rendements.
- Les rendements mensuels sont calculés à partir des log-rendements OOS :
  r_m = exp(sum(log_r_dans_le_mois)) - 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from typing import Optional
from scipy.optimize import minimize


@dataclass(frozen=True)
class CoreConfig:
    """Paramètres du pipeline Core."""
    excel_path: str = "data/core_ETF.xlsx"
    sheet: int = 0

    # Sélection ETF "best" par thème
    min_obs_pick_best: int = 750

    # Backtest rolling
    lookback: int = 252
    rebal_freq: int = 63

    # Contraintes de poids (validées)
    w_min: float = 0.05
    w_max: float = 0.50

    # Sorties
    output_core_monthly_csv: str = "outputs/core_returns_monthly.csv"
    output_core_daily_csv: str = "outputs/core_returns_daily_oos.csv"
    output_selected_core_csv: str = "outputs/core_selected_etfs.csv"


def _clean(x) -> str:
    """Nettoyage simple des métadonnées."""
    if pd.isna(x):
        return ""
    return str(x).strip()



def _parse_dates_any(x: pd.Series) -> pd.DatetimeIndex:
    """
    Parse une colonne de dates robuste :
    - si déjà datetime -> ok
    - si string type '04/01/2010' -> dayfirst=True
    - si numérique -> interprété comme serial Excel (origin 1899-12-30)
    """
    # 1) tentative directe datetime
    dt = pd.to_datetime(x, errors="coerce", dayfirst=True)

    # 2) si beaucoup de NaN, tenter le mode serial Excel
    if dt.notna().sum() < max(5, int(0.20 * len(x))):
        num = pd.to_numeric(x, errors="coerce")
        dt2 = pd.to_datetime(num, errors="coerce", unit="D", origin="1899-12-30")
        # on combine : on garde dt si OK sinon dt2
        dt = dt.fillna(dt2)

    return pd.DatetimeIndex(dt)


def lire_excel_et_reconstruire_wide(cfg: CoreConfig) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Lecture Excel EXACTEMENT comme dans le notebook du collègue :
    - Calendrier de référence = 1ère colonne date
    - Pour chaque ETF : on prend uniquement la colonne Price (j+1)
    - Conversion dates : serial Excel [20000,60000] sinon parsing texte
    """
    print("[1/6] Lecture Excel + reconstruction wide / buckets (version notebook)...")

    path = cfg.excel_path
    sheet = cfg.sheet

    meta = pd.read_excel(path, sheet_name=sheet, header=None, nrows=3)

    tickers = meta.iloc[0].tolist()

    def clean(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    tickers = [clean(x) for x in tickers]

    df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)

    # 1) calendrier référence = 1ère colonne date
    ref_raw = df.iloc[:, 0]
    ref_num = pd.to_numeric(ref_raw, errors="coerce")
    mask_excel = ref_num.between(20000, 60000)

    ref_dates = ref_raw.copy()
    ref_dates.loc[mask_excel] = pd.to_datetime(ref_num[mask_excel], unit="D", origin="1899-12-30")
    ref_dates.loc[~mask_excel] = pd.to_datetime(ref_raw[~mask_excel], errors="coerce", dayfirst=True)

    # garder seulement les lignes où la date est valide
    valid = ref_dates.notna()
    df = df.loc[valid].reset_index(drop=True)
    ref_dates = pd.to_datetime(ref_dates.loc[valid]).reset_index(drop=True)

    # 2) wide = index = ref_dates, colonnes = tickers, valeurs = colonnes "Price" uniquement
    wide = pd.DataFrame(index=ref_dates)

    for j in range(0, df.shape[1], 2):
        price_col = j + 1
        if price_col >= df.shape[1]:
            break

        tkr = tickers[price_col] if price_col < len(tickers) and tickers[price_col] not in ("", "Date") else f"UNKNOWN_{j//2}"
        px = pd.to_numeric(df.iloc[:, price_col], errors="coerce").values
        wide[tkr] = px

    wide = wide.sort_index()
    wide = wide.ffill()

    # --- Buckets (ligne 1) + mapping ticker -> bucket (même logique notebook)
    buckets = meta.iloc[1].tolist()
    buckets = [clean(x) for x in buckets]

    ticker_to_bucket: Dict[str, str] = {}
    for j in range(0, df.shape[1], 2):
        price_col = j + 1
        if price_col >= df.shape[1]:
            break

        tkr = tickers[price_col] if price_col < len(tickers) and tickers[price_col] not in ("", "Date") else f"UNKNOWN_{j//2}"
        bkt = buckets[price_col] if price_col < len(buckets) else ""
        ticker_to_bucket[tkr] = bkt

    wide.index.name = "Date"

    print(f"  -> Période wide: {wide.index.min().date()} à {wide.index.max().date()}")
    print(f"  -> Nb tickers wide: {wide.shape[1]}")
    return wide, ticker_to_bucket


def definir_themes() -> Dict[str, Set[str]]:
    """Définit les thèmes comme dans le notebook."""
    return {
        "Equity": {"Equity DM (Core)", "Equity DM (Regional)"},
        "Rates": {"Rates EMU Govies (Core)", "Rates EMU Govies (Bucket)", "Rates EMU Govies (Linkers)"},
        "Credit": {
            "Credit EMU IG (Core)",
            "Credit EMU IG (Bucket)",
            "Credit EMU IG (Large Cap)",
            "Credit EMU IG (Covered)",
            "EMU Aggregate (IG)",
        },
    }


def pick_best(
    theme_name: str,
    wide: pd.DataFrame,
    rets_log: pd.DataFrame,
    ticker_to_bucket: Dict[str, str],
    themes: Dict[str, Set[str]],
    min_obs: int,
) -> str:
    """
    Réplique l'idée du notebook :
    - candidats = tickers dont le bucket appartient au thème
    - filtre historique min_obs
    - score = corrélation moyenne intra-thème + bonus nb obs - pénalité vol
    """
    cands = [t for t in wide.columns if ticker_to_bucket.get(t, "") in themes[theme_name]]
    cands = [t for t in cands if rets_log[t].dropna().shape[0] >= min_obs]

    if len(cands) == 0:
        raise ValueError(f"Aucun candidat pour {theme_name} avec min_obs={min_obs}")

    R = rets_log[cands].dropna(how="all")
    corr = R.corr()

    avg_corr = corr.mean(axis=1).values  # représentativité
    obs = np.array([rets_log[t].dropna().shape[0] for t in cands], dtype=float)
    vol = R.std().values  # vol mensuelle/journalière selon R (ici log daily)

    # Normalisations simples
    obs_z = (obs - obs.mean()) / (obs.std() + 1e-12)
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-12)

    score = avg_corr + 0.10 * obs_z - 0.10 * vol_z
    best_idx = int(np.argmax(score))
    return cands[best_idx]


def optimiser_max_sharpe_contraint(mu: np.ndarray, cov: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    """
    Optimisation Max Sharpe (rf=0) sous contraintes long-only et bornes [w_min, w_max].
    Fallback : équipondéré si échec.
    """
    n = len(mu)
    x0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov @ w))
        if vol < 1e-12:
            return 1e10
        ret = float(w @ mu)
        return -(ret / vol)

    bounds = [(w_min, w_max)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return x0
    return res.x


def backtest_rolling_oos(
    prices: pd.DataFrame,
    lookback: int,
    rebal_freq: int,
    w_min: float,
    w_max: float,
) -> pd.Series:
    """
    Backtest rolling trimestriel OOS :
    - On estime mu/cov sur une fenêtre lookback (log-rendements)
    - On optimise max sharpe sous contraintes
    - On applique les poids sur la période suivante (OOS) en log-rendements

    Sortie :
    - série de log-rendements OOS quotidiens du portefeuille Core
    """
    print("[4/6] Backtest rolling OOS (Max Sharpe contraint)...")

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

    s = pd.Series(port_log_rets, index=pd.DatetimeIndex(port_dates), name="core_log_return_oos")
    s = s.sort_index()
    print(f"  -> OOS: {s.index.min().date()} à {s.index.max().date()} | Nb jours: {len(s)}")
    return s


def log_daily_to_monthly_simple(log_daily: pd.Series) -> pd.Series:
    """
    Convertit des log-rendements journaliers en rendements simples mensuels :
    r_m = exp( somme(log_r_jour) ) - 1
    """
    print("[5/6] Conversion OOS journalier -> mensuel (r_m = exp(sum) - 1)...")
    monthly = np.exp(log_daily.resample("ME").sum()) - 1.0
    monthly = monthly.dropna()
    monthly.name = "core_portfolio_return"
    print(f"  -> Nb mois: {len(monthly)} | Début: {monthly.index.min().date()} | Fin: {monthly.index.max().date()}")
    return monthly


def main() -> None:
    cfg = CoreConfig()

    wide, ticker_to_bucket = lire_excel_et_reconstruire_wide(cfg)

    # Log-rendements pour sélection best
    rets_log = np.log(wide).diff()

    themes = definir_themes()

    print("[2/6] Sélection des meilleurs ETFs par thème...")
    best_eq = pick_best("Equity", wide, rets_log, ticker_to_bucket, themes, cfg.min_obs_pick_best)
    best_rt = pick_best("Rates", wide, rets_log, ticker_to_bucket, themes, cfg.min_obs_pick_best)
    best_cr = pick_best("Credit", wide, rets_log, ticker_to_bucket, themes, cfg.min_obs_pick_best)

    core_etfs = [best_eq, best_rt, best_cr]
    print(f"  -> Equity: {best_eq}")
    print(f"  -> Rates : {best_rt}")
    print(f"  -> Credit: {best_cr}")

    # Sauvegarde des ETF sélectionnés
    pd.DataFrame({"core_etfs": core_etfs}).to_csv(cfg.output_selected_core_csv, index=False)
    print(f"[3/6] ETFs Core sauvegardés dans: {cfg.output_selected_core_csv}")

    # Prix Core
    prices_core = wide[core_etfs].dropna(how="all").ffill()

    # Backtest rolling OOS
    core_log_daily_oos = backtest_rolling_oos(
        prices_core,
        lookback=cfg.lookback,
        rebal_freq=cfg.rebal_freq,
        w_min=cfg.w_min,
        w_max=cfg.w_max,
    )

    # Export daily OOS (utile debug)
    core_log_daily_oos.to_frame().to_csv(cfg.output_core_daily_csv, index=True)
    print(f"  -> Rendements log journaliers OOS exportés: {cfg.output_core_daily_csv}")

    # Conversion en mensuel (simple returns)
    core_monthly = log_daily_to_monthly_simple(core_log_daily_oos)

    # Export mensuel (format prêt à fusionner avec Satellite)
    core_monthly.to_frame().to_csv(cfg.output_core_monthly_csv, index=True)
    print(f"[6/6] Export OK: {cfg.output_core_monthly_csv}")


if __name__ == "__main__":
    main()