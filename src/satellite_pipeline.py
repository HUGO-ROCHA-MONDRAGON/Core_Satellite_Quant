"""
Pipeline Satellite v2 – Filtrage 4 niveaux + scoring + sélection.

Sources :
  - STRAT1/2/3_info.xlsx  : métadonnées Bloomberg (AUM, frais, stratégie, etc.)
  - STRAT1/2/3_price.xlsx : prix daily (paires date/prix par fonds)
  - outputs/core_returns_daily_oos.csv : rendements journaliers du Core (benchmark)
    - outputs/core_selected_etfs.csv + outputs/core3_etf_daily_log_returns.csv
        : benchmark équipondéré dynamique des 3 ETF Core sélectionnés

Structure des 3 blocs :
  Bloc 1 – Décorrélation / Convexité  : Dérivés, Marché baissier, Spécialisé
  Bloc 2 – Alpha avec bêta contrôlé   : CTA, Equity Hedge, L/S, Market Neutral…
  Bloc 3 – Diversification macro      : Commodités, ILS, ARP, Relative Value…

Pipeline de filtrage (toutes métriques calculées sur la fenêtre calib 2019-2020) :
    Niveau Beta – Filtre initial      : rolling beta 3 mois vs Core équipondéré,
                         |beta| <= 15% sur au moins 80% des jours d'analyse
  Niveau 0 – Structurel universel : AUM ≥ 100 M$, premier prix ≤ 01/01/2019,
             filtre devise (optionnel), exclusion de stratégies (optionnel)
  Niveau 1 – Frais & Volatilité    : plafonds par bloc/stratégie
  Niveau 2 – Qualité quantitative  : Sharpe, Alpha vs Core, Max Drawdown
  Niveau 3 – Comportemental        : Skewness, Kurtosis (sur prix calib),
                                      Concentration (info Bloomberg)

Score composite (z-scoré intra-bloc, calculé sur calib window 2019-2020) :
  Sharpe(30%) + (-|β_core|)(35%) + Alpha(20%) + (-Drawdown)(10%)
  + Skewness(3%) + (-Kurtosis)(2%)

Sélection finale : 3 fonds par bloc, max 2 par stratégie.
Anti look-ahead : tous les calculs quantitatifs sur la fenêtre 2019-2020 uniquement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
#  Poids du score composite (modifiables ici)
# ══════════════════════════════════════════════════════════════════════════════

SCORE_WEIGHTS: Dict[str, float] = {
    "sharpe":        0.30,
    "neg_abs_beta":  0.35,   # bas beta vs core = bonne décorrélation
    "alpha":         0.20,   # alpha vs core (annualisé)
    "neg_drawdown":  0.10,   # moins négatif = mieux
    "skew":          0.03,   # légère préférence positive
    "neg_kurtosis":  0.02,   # moins de fat tails = mieux
}


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlocConfig:
    """Paramètres de filtrage et de sélection pour un bloc satellite."""

    nom: str
    info_path: str
    price_path: str

    # ── Niveau 1 : Frais ──────────────────────────────────────────────────
    expense_max_default: float = 2.0          # % (frais max par défaut)
    expense_max_by_strategy: Dict[str, float] = field(default_factory=dict)

    # ── Niveau 1 : Volatilité (annualisée, calculée sur calib window) ────
    vol_max_default: float = 0.30
    vol_max_by_strategy: Dict[str, float] = field(default_factory=dict)

    # ── Niveau 2 : Qualité quantitative (sur calib window) ───────────────
    sharpe_min: float = -0.5
    alpha_min_annual: float = -0.10    # alpha annualisé vs core (plancher)
    drawdown_max: float = -0.80        # max drawdown (valeur négative)

    # ── Niveau 3 : Comportemental ─────────────────────────────────────────
    skew_min: float = -2.5             # skewness sur prix calib
    kurtosis_max: float = 10.0         # kurtosis sur prix calib
    concentration_max: float = 95.0    # % des 10 premières positions (info)

    # ── Sélection finale ──────────────────────────────────────────────────
    n_select: int = 3
    max_per_strategy: int = 2


@dataclass
class SatelliteConfig:
    """Configuration globale du pipeline satellite."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # ── Source rendements Core (benchmark alpha & beta) ──────────────────
    core_daily_csv: str = ""           # rempli dans __post_init__
    core_selected_csv: str = ""        # tickers Core retenus (CSV)
    core3_daily_log_returns_csv: str = ""  # rendements log des 3 ETF Core (CSV)

    # ── Fenêtre de calibration (anti look-ahead) ─────────────────────────
    calib_start: str = "2019-01-01"
    calib_end: str = "2020-12-31"

    # ── Niveau 0 – Filtres universels ────────────────────────────────────
    aum_min_m: float = 100.0
    max_start_date: str = "2019-01-01"
    allowed_currencies: List[str] = field(default_factory=list)
    excluded_strategies: List[str] = field(default_factory=list)

    # ── Rolling beta window ───────────────────────────────────────────────
    beta_rolling_days: int = 252

    # ── Filtre beta initial (vs core équipondéré des 3 ETF sélectionnés) ─
    beta_filter_window_days: int = 63
    beta_filter_max_abs: float = 0.25
    beta_filter_min_pass_ratio: float = 0.70

    # ── Sorties ───────────────────────────────────────────────────────────
    output_selected_csv: str = ""

    # ── Blocs ─────────────────────────────────────────────────────────────
    blocs: Dict[str, BlocConfig] = field(default_factory=dict)

    def __post_init__(self):
        r = self.project_root
        if not self.core_daily_csv:
            self.core_daily_csv = str(r / "outputs" / "core_returns_daily_oos.csv")
        if not self.core_selected_csv:
            self.core_selected_csv = str(r / "outputs" / "core_selected_etfs.csv")
        if not self.core3_daily_log_returns_csv:
            self.core3_daily_log_returns_csv = str(r / "outputs" / "core3_etf_daily_log_returns.csv")
        if not self.output_selected_csv:
            self.output_selected_csv = str(r / "outputs" / "satellite_selected.csv")
        if not self.blocs:
            self.blocs = _build_default_blocs(r)


def _build_default_blocs(project_root: Path) -> Dict[str, BlocConfig]:
    """Construit les configurations par défaut des 3 blocs."""
    return {
        "Bloc1": BlocConfig(
            nom="Bloc 1 – Décorrélation / Convexité",
            info_path=str(project_root / "data" / "STRAT1_info.xlsx"),
            price_path=str(project_root / "data" / "STRAT1_price.xlsx"),
            expense_max_default=2.0,
            vol_max_default=0.20,
            vol_max_by_strategy={},
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.80,
            skew_min=-2.0,
            kurtosis_max=10.0,
            concentration_max=95.0,
            n_select=3,
            max_per_strategy=2,
        ),
        "Bloc2": BlocConfig(
            nom="Bloc 2 – Alpha avec bêta contrôlé",
            info_path=str(project_root / "data" / "STRAT2_info.xlsx"),
            price_path=str(project_root / "data" / "STRAT2_price.xlsx"),
            expense_max_default=2.0,
            vol_max_default=0.10,
            vol_max_by_strategy={
                "Neutre au marché":            0.08,
                "CTA/futures gérés":           0.18,
                "Long Short":                  0.10,
                "Equity Hedge":                0.10,
                "Mené par les événements":     0.10,
                "Multi-stratégie":             0.10,
            },
            sharpe_min=-0.3,
            alpha_min_annual=-0.10,
            drawdown_max=-0.50,
            skew_min=-2.0,
            kurtosis_max=8.0,
            concentration_max=80.0,
            n_select=3,
            max_per_strategy=2,
        ),
        "Bloc3": BlocConfig(
            nom="Bloc 3 – Diversification macro & scénarios",
            info_path=str(project_root / "data" / "STRAT3_info.xlsx"),
            price_path=str(project_root / "data" / "STRAT3_price.xlsx"),
            expense_max_default=1.5,
            expense_max_by_strategy={
                "Titres adossés à des actifs": 1.8,
                "Prêts bancaires":             1.8,
            },
            vol_max_default=0.45,
            vol_max_by_strategy={
                "Energie":                         0.45,
                "Métaux industriels":              0.30,
                "Métaux précieux":                 0.30,
                "Protégé contre l'inflation":      0.20,
                "Titres adossés à des actifs":     0.08,
                "Prêts bancaires":                 0.08,
                "Obligataire Valeur relative":     0.10,
            },
            sharpe_min=-0.5,
            alpha_min_annual=-0.10,
            drawdown_max=-0.70,
            skew_min=-2.5,
            kurtosis_max=10.0,
            concentration_max=90.0,
            n_select=3,
            max_per_strategy=2,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Lecture des données
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dates(x: pd.Series) -> pd.Series:
    """Parse robuste de dates : datetime natif / string / serial Excel."""
    dt = pd.to_datetime(x, errors="coerce", dayfirst=False)
    if dt.notna().sum() < max(5, int(0.20 * len(x))):
        num = pd.to_numeric(x, errors="coerce")
        # Filtrer les valeurs hors plage raisonnable pour un serial Excel
        # (1 = 1900-01-01, ~80 000 ≈ 2119) avant la conversion pour éviter l'overflow
        mask_valid = num.notna() & (num >= 1) & (num <= 80_000)
        dt2 = pd.Series(pd.NaT, index=x.index)
        if mask_valid.any():
            dt2.loc[mask_valid] = pd.to_datetime(
                num.loc[mask_valid].astype(int),
                errors="coerce", unit="D", origin="1899-12-30"
            )
        dt = dt.fillna(dt2)
    return dt


def lire_prix_wide(path: str) -> pd.DataFrame:
    """
    Lit TOUTES les feuilles d'un fichier de prix satellite.
    Chaque feuille correspond à une stratégie.
    Structure par feuille : ligne 0 = tickers (paires date/prix), lignes 1+ = données.
    Retourne DataFrame wide : index=Date (DatetimeIndex), cols=tickers (toutes feuilles).
    """
    xls = pd.ExcelFile(path)
    seen: set = set()
    all_series: Dict[str, pd.Series] = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        tickers_row = df.iloc[0, :].tolist()
        data = df.iloc[1:]

        for j in range(0, df.shape[1], 2):
            ticker = tickers_row[j] if j < len(tickers_row) else None
            if not isinstance(ticker, str):
                continue
            ticker = ticker.strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)

            price_col = j + 1
            if price_col >= df.shape[1]:
                break

            dates = _parse_dates(data.iloc[:, j])
            prices = pd.to_numeric(data.iloc[:, price_col], errors="coerce")
            valid = dates.notna() & prices.notna()
            if valid.sum() < 5:
                continue

            s = pd.Series(
                prices[valid].values,
                index=pd.DatetimeIndex(dates[valid].values),
                name=ticker,
            ).sort_index()
            s = s[~s.index.duplicated(keep="first")]
            all_series[ticker] = s

    wide = pd.DataFrame(all_series).sort_index()
    wide.index.name = "Date"
    return wide


def lire_info(path: str) -> pd.DataFrame:
    """
    Lit un fichier info satellite. Normalise les colonnes clés.
    Retourne DataFrame indexé sur 'ticker'.
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        cl = col.lower()
        if "total actifs usd" in cl:
            rename_map[col] = "aum_usd_m"
        elif ("ratio des dépenses" in cl or "ratio des depenses" in cl
              or ("expense" in cl and col not in rename_map.values())):
            rename_map[col] = "expense_pct"
        elif "stratégie de fonds" in cl or "strategie de fonds" in cl:
            rename_map[col] = "strategie"
        elif "date de création" in cl or "date de creation" in cl:
            rename_map[col] = "date_creation"
        elif col.strip() == "Dev" or "devise" in cl:
            rename_map[col] = "devise"
        elif "% des 10" in cl or "premières positions" in cl:
            rename_map[col] = "concentration"

    df = df.rename(columns=rename_map)

    # Nettoyage numérique
    # Les expense ratios satellite sont déjà en pourcentage dans les Excel
    # (ex: 0.95 signifie 0.95%), donc aucune mise à l'échelle n'est appliquée.
    for col in ["aum_usd_m", "expense_pct", "concentration"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace("\xa0", "", regex=False)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )

    if "date_creation" in df.columns:
        df["date_creation"] = pd.to_datetime(df["date_creation"], errors="coerce", dayfirst=False)

    # Colonne ticker
    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    if "ticker" not in df.columns:
        df["ticker"] = df.iloc[:, 0]
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df.dropna(subset=["ticker"])

    return df.set_index("ticker")


def charger_core_returns(cfg: SatelliteConfig) -> pd.Series:
    """
    Charge les rendements journaliers du Core (log returns) et les convertit
    en rendements simples journaliers.
    """
    df = pd.read_csv(cfg.core_daily_csv, index_col=0, parse_dates=True)
    col = df.columns[0]
    simple = np.exp(df[col]) - 1.0
    simple.name = "core_return"
    return simple.sort_index().dropna()


def charger_core_eqw_returns_from_csv(cfg: SatelliteConfig) -> pd.Series:
    """
    Construit un benchmark Core équipondéré (3 ETF sélectionnés) depuis des CSV.
    Les rendements individuels sont lus dans core3_etf_daily_log_returns.csv,
    convertis en indices de prix, puis agrégés en portefeuille équipondéré.
    """
    selected = pd.read_csv(cfg.core_selected_csv)
    ticker_col = "core_etfs" if "core_etfs" in selected.columns else selected.columns[0]
    tickers = selected[ticker_col].astype(str).str.strip().dropna().tolist()
    if len(tickers) == 0:
        raise ValueError("core_selected_etfs.csv ne contient aucun ticker exploitable.")

    core3_log = pd.read_csv(cfg.core3_daily_log_returns_csv, index_col=0, parse_dates=True).sort_index()
    available = [t for t in tickers if t in core3_log.columns]
    if len(available) < 3:
        raise ValueError(
            "core3_etf_daily_log_returns.csv ne contient pas les 3 ETF sélectionnés. "
            f"Trouvés: {available}"
        )

    core3_log = core3_log[available[:3]].dropna(how="all")
    # Reconstruction d'indices de prix (base 100) à partir des log-rendements.
    core3_prices = (100.0 * np.exp(core3_log.cumsum())).ffill()
    core_eqw_price = core3_prices.mean(axis=1).rename("core_eqw_price")
    core_eqw_log = np.log(core_eqw_price).diff().dropna().rename("core_eqw_log_return")
    return core_eqw_log


# ══════════════════════════════════════════════════════════════════════════════
#  Calcul des métriques sur la fenêtre de calibration
# ══════════════════════════════════════════════════════════════════════════════

def _annualized_vol(daily_rets: pd.Series) -> float:
    return float(daily_rets.std() * np.sqrt(252))


def _annualized_sharpe(daily_rets: pd.Series) -> float:
    vol = _annualized_vol(daily_rets)
    if vol < 1e-10:
        return np.nan
    ann_ret = float((1 + daily_rets).prod() ** (252 / len(daily_rets)) - 1)
    return ann_ret / vol


def _max_drawdown(daily_rets: pd.Series) -> float:
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    return float(((cum / peak) - 1).min())


def _ols_alpha_beta(fund_rets: pd.Series, core_rets: pd.Series) -> Tuple[float, float]:
    """OLS : fund = alpha_daily + beta * core + eps → (alpha_annual, beta)."""
    aligned = pd.concat([fund_rets, core_rets], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan, np.nan
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    slope, intercept, *_ = stats.linregress(x, y)
    return float(intercept * 252), float(slope)


def calculer_metriques_calib(
    wide_prices: pd.DataFrame,
    core_rets: pd.Series,
    calib_start: str,
    calib_end: str,
) -> pd.DataFrame:
    """
    Calcule pour chaque fonds (sur calib window uniquement) :
    vol, sharpe, alpha_annual, beta_core, drawdown, skewness, kurtosis.
    """
    log_rets = np.log(wide_prices).diff().dropna(how="all")
    rets_calib = log_rets.loc[calib_start:calib_end]
    core_calib = core_rets.loc[calib_start:calib_end]

    rows = []
    for ticker in rets_calib.columns:
        s = rets_calib[ticker].dropna()
        if len(s) < 30:
            continue
        alpha, beta = _ols_alpha_beta(s, core_calib)
        rows.append(dict(
            ticker=ticker,
            vol_calib=_annualized_vol(s),
            sharpe_calib=_annualized_sharpe(s),
            alpha_annual=alpha,
            beta_core=beta,
            drawdown_calib=_max_drawdown(s),
            skew_calib=float(stats.skew(s)),
            kurtosis_calib=float(stats.kurtosis(s)),   # excess kurtosis
            n_obs_calib=len(s),
        ))

    return pd.DataFrame(rows).set_index("ticker")


def calculer_beta_rolling(
    wide_prices: pd.DataFrame,
    core_rets: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """
    Calcule le beta rolling (fenêtre = window jours) de chaque fonds vs Core.
    Retourne DataFrame : index=Date, cols=tickers.
    """
    fund_rets = np.log(wide_prices).diff().dropna(how="all")
    aligned = pd.concat([fund_rets, core_rets.rename("__core__")], axis=1).dropna(how="all")

    core_col = aligned["__core__"]
    fund_cols = aligned.drop(columns=["__core__"])

    betas: Dict[str, pd.Series] = {}
    for ticker in fund_cols.columns:
        s = aligned[[ticker, "__core__"]].dropna()
        if len(s) < window:
            continue
        cov = s[ticker].rolling(window).cov(s["__core__"])
        var = s["__core__"].rolling(window).var()
        betas[ticker] = (cov / var).where(var > 1e-12)

    return pd.DataFrame(betas)


# ══════════════════════════════════════════════════════════════════════════════
#  Niveaux de filtrage
# ══════════════════════════════════════════════════════════════════════════════

def filtrer_niveau0(
    info: pd.DataFrame,
    wide_prices: pd.DataFrame,
    cfg: SatelliteConfig,
) -> List[str]:
    """
    Niveau 0 – Structurel universel :
      - Ticker présent dans info ET dans prices
      - AUM ≥ aum_min_m (USD millions)
      - Premier prix ≤ max_start_date
      - Devise dans allowed_currencies (si liste non vide)
      - Stratégie hors excluded_strategies (si liste non vide)
    """
    valid = []
    max_date = pd.Timestamp(cfg.max_start_date)

    for ticker in wide_prices.columns:
        if ticker not in info.index:
            continue

        row = info.loc[ticker]

        aum = row.get("aum_usd_m", np.nan)
        if pd.isna(aum) or aum < cfg.aum_min_m:
            continue

        prices = wide_prices[ticker].dropna()
        if len(prices) == 0 or prices.index.min() > max_date:
            continue

        if cfg.allowed_currencies:
            devise = str(row.get("devise", "")).strip()
            if devise not in cfg.allowed_currencies:
                continue

        if cfg.excluded_strategies:
            strat = str(row.get("strategie", "")).strip()
            if strat in cfg.excluded_strategies:
                continue

        # Exclure les fonds sans expense ratio renseigné
        expense = row.get("expense_pct", np.nan)
        if pd.isna(expense):
            continue

        valid.append(ticker)
    return valid


def filtrer_niveau_beta_initial(
    tickers: List[str],
    beta_rolling: pd.DataFrame,
    cfg: SatelliteConfig,
) -> List[str]:
    """
    Filtre initial beta : conserve les fonds dont le beta rolling 3M vs core équipondéré
    respecte |beta| <= beta_filter_max_abs sur au moins beta_filter_min_pass_ratio
    des jours dans la fenêtre d'analyse.
    """
    valid: List[str] = []
    beta_window = beta_rolling.loc[cfg.calib_start:cfg.calib_end]

    for ticker in tickers:
        if ticker not in beta_window.columns:
            continue
        s = beta_window[ticker].dropna()
        if len(s) == 0:
            continue
        pass_ratio = float((s.abs() <= cfg.beta_filter_max_abs).mean())
        if pass_ratio >= cfg.beta_filter_min_pass_ratio:
            valid.append(ticker)
    return valid


def filtrer_niveau1(
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
) -> List[str]:
    """
    Niveau 1 – Frais & Volatilité (réalisée sur calib window) :
      - Expense ratio ≤ seuil (par stratégie ou défaut bloc)
      - Volatilité annualisée ≤ seuil (par stratégie ou défaut bloc)
    """
    valid = []
    for ticker in tickers:
        row_info = info.loc[ticker] if ticker in info.index else None
        strat = str(row_info.get("strategie", "")) if row_info is not None else ""

        expense_max = bloc_cfg.expense_max_by_strategy.get(strat, bloc_cfg.expense_max_default)
        expense = float(row_info.get("expense_pct", np.nan)) if row_info is not None else np.nan
        if not np.isnan(expense) and expense > expense_max:
            continue

        if ticker not in metrics.index:
            continue
        vol_max = bloc_cfg.vol_max_by_strategy.get(strat, bloc_cfg.vol_max_default)
        vol = metrics.at[ticker, "vol_calib"]
        if not np.isnan(vol) and vol > vol_max:
            continue

        valid.append(ticker)
    return valid


def filtrer_niveau2(
    tickers: List[str],
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
) -> List[str]:
    """
    Niveau 2 – Qualité quantitative (sur calib window) :
      - Sharpe ≥ sharpe_min
      - Alpha annualisé vs Core ≥ alpha_min_annual
      - Max drawdown ≥ drawdown_max  (ex: -0.80)
    """
    valid = []
    for ticker in tickers:
        if ticker not in metrics.index:
            continue
        row = metrics.loc[ticker]
        if not np.isnan(row["sharpe_calib"]) and row["sharpe_calib"] < bloc_cfg.sharpe_min:
            continue
        if not np.isnan(row["alpha_annual"]) and row["alpha_annual"] < bloc_cfg.alpha_min_annual:
            continue
        if not np.isnan(row["drawdown_calib"]) and row["drawdown_calib"] < bloc_cfg.drawdown_max:
            continue
        valid.append(ticker)
    return valid


def filtrer_niveau3(
    tickers: List[str],
    info: pd.DataFrame,
    metrics: pd.DataFrame,
    bloc_cfg: BlocConfig,
) -> List[str]:
    """
    Niveau 3 – Comportemental :
      - Skewness (calculée sur prix calib) ≥ skew_min
      - Kurtosis (calculée sur prix calib) ≤ kurtosis_max
      - Concentration (% des 10 premières positions, from info) ≤ concentration_max
    """
    valid = []
    for ticker in tickers:
        if ticker in metrics.index:
            skew = metrics.at[ticker, "skew_calib"]
            kurt = metrics.at[ticker, "kurtosis_calib"]
            if not np.isnan(skew) and skew < bloc_cfg.skew_min:
                continue
            if not np.isnan(kurt) and kurt > bloc_cfg.kurtosis_max:
                continue

        if ticker in info.index and "concentration" in info.columns:
            conc = info.at[ticker, "concentration"]
            if not np.isnan(conc) and conc > bloc_cfg.concentration_max:
                continue

        valid.append(ticker)
    return valid


# ══════════════════════════════════════════════════════════════════════════════
#  Score composite & sélection finale
# ══════════════════════════════════════════════════════════════════════════════

def _zscore_col(series: pd.Series) -> pd.Series:
    """Z-score (NaN remplacés par 0 après normalisation)."""
    mu, sigma = series.mean(), series.std()
    if sigma < 1e-10:
        return pd.Series(0.0, index=series.index)
    return ((series - mu) / sigma).fillna(0.0)


def scorer(
    tickers: List[str],
    metrics: pd.DataFrame,
    weights: Dict[str, float] = SCORE_WEIGHTS,
) -> pd.Series:
    """
    Score composite intra-bloc (z-scoré) basé sur les métriques calib window.
    """
    df = metrics.loc[[t for t in tickers if t in metrics.index]].copy()
    if df.empty:
        return pd.Series(dtype=float)

    score = (
        weights.get("sharpe", 0.30)        * _zscore_col(df["sharpe_calib"])
      + weights.get("neg_abs_beta", 0.35)  * _zscore_col(-df["beta_core"].abs())
      + weights.get("alpha", 0.20)         * _zscore_col(df["alpha_annual"])
      + weights.get("neg_drawdown", 0.10)  * _zscore_col(-df["drawdown_calib"])
      + weights.get("skew", 0.03)          * _zscore_col(df["skew_calib"])
      + weights.get("neg_kurtosis", 0.02)  * _zscore_col(-df["kurtosis_calib"])
    )
    return score.sort_values(ascending=False)


def selectionner(
    scores: pd.Series,
    info: pd.DataFrame,
    bloc_cfg: BlocConfig,
) -> List[str]:
    """
    Sélectionne n_select fonds (ordre décroissant des scores).
    Contrainte : max max_per_strategy fonds par stratégie.
    """
    selected: List[str] = []
    strategy_count: Dict[str, int] = {}

    for ticker in scores.index:
        if len(selected) >= bloc_cfg.n_select:
            break
        strat = (
            str(info.at[ticker, "strategie"])
            if (ticker in info.index and "strategie" in info.columns)
            else "Inconnu"
        )
        count = strategy_count.get(strat, 0)
        if count >= bloc_cfg.max_per_strategy:
            continue
        selected.append(ticker)
        strategy_count[strat] = count + 1

    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline par bloc + pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def traiter_bloc(
    bloc_name: str,
    bloc_cfg: BlocConfig,
    core_rets: pd.Series,
    core_eqw_rets: pd.Series,
    cfg: SatelliteConfig,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Traite un bloc complet : lecture → calibration → filtrage 4 niveaux → score → sélection.
    Retourne (selected_tickers, metrics_df, info_df, beta_rolling_df).
    """
    print(f"\n{'=' * 60}")
    print(f"  {bloc_cfg.nom}")
    print(f"{'=' * 60}")

    prices = lire_prix_wide(bloc_cfg.price_path)
    info = lire_info(bloc_cfg.info_path)

    common = [t for t in prices.columns if t in info.index]
    prices = prices[common]
    print(f"  [données] {len(common)} fonds avec prix ET info")

    print(f"  [métriques] Calcul sur calib {cfg.calib_start} → {cfg.calib_end}...")
    print(f"  [beta init] Rolling {cfg.beta_filter_window_days}j vs Core équipondéré...")
    beta_init = calculer_beta_rolling(prices, core_eqw_rets, cfg.beta_filter_window_days)
    t_beta = filtrer_niveau_beta_initial(common, beta_init, cfg)
    print(
        f"  [Niv.Beta] |beta| <= {cfg.beta_filter_max_abs:.0%} sur >= "
        f"{cfg.beta_filter_min_pass_ratio:.0%} jours   {len(t_beta):3d} / {len(common)}"
    )

    if not t_beta:
        print(f"  ⚠  Aucun fonds ne passe le filtre beta initial dans {bloc_name}.")
        return [], pd.DataFrame(), info, pd.DataFrame()

    prices = prices[t_beta]
    metrics = calculer_metriques_calib(prices, core_rets, cfg.calib_start, cfg.calib_end)
    print(f"             {len(metrics)} fonds avec données suffisantes")

    t0 = filtrer_niveau0(info, prices, cfg)
    print(f"  [Niv.0] AUM/Date/Devise/Excl.  {len(t0):3d} / {len(common)}")

    t1 = filtrer_niveau1(t0, info, metrics, bloc_cfg)
    print(f"  [Niv.1] Frais & Volatilité     {len(t1):3d} / {len(t0)}")

    t2 = filtrer_niveau2(t1, metrics, bloc_cfg)
    print(f"  [Niv.2] Sharpe/Alpha/Drawdown  {len(t2):3d} / {len(t1)}")

    t3 = filtrer_niveau3(t2, info, metrics, bloc_cfg)
    print(f"  [Niv.3] Skew/Kurt/Conc.        {len(t3):3d} / {len(t2)}")

    if not t3:
        print(f"  ⚠  Aucun fonds ne passe tous les filtres dans {bloc_name}.")
        return [], metrics, info, pd.DataFrame()

    scores = scorer(t3, metrics)
    selected = selectionner(scores, info, bloc_cfg)

    print(f"\n  [sélection] {bloc_name} – {len(selected)} fonds retenus :")
    for rank, ticker in enumerate(selected, 1):
        strat = info.at[ticker, "strategie"] if "strategie" in info.columns else "?"
        sc = scores.get(ticker, np.nan)
        beta = metrics.at[ticker, "beta_core"] if ticker in metrics.index else np.nan
        sh = metrics.at[ticker, "sharpe_calib"] if ticker in metrics.index else np.nan
        print(f"    #{rank}  {ticker:<28s}  strat={strat:<28s}  "
              f"score={sc:+.3f}  β={beta:+.3f}  Sharpe={sh:.3f}")

    print(f"\n  [beta rolling {cfg.beta_rolling_days}j]...")
    prices_sel = prices[[t for t in selected if t in prices.columns]]
    beta_roll = calculer_beta_rolling(prices_sel, core_rets, cfg.beta_rolling_days)

    return selected, metrics, info, beta_roll


def main() -> None:
    cfg = SatelliteConfig()

    print("=" * 60)
    print("  PIPELINE SATELLITE v2 – Filtrage & Sélection")
    print(f"  Fenêtre calib : {cfg.calib_start} → {cfg.calib_end}")
    print("=" * 60)

    print("\n[1] Chargement des rendements Core...")
    core_rets = charger_core_returns(cfg)
    print(f"    {len(core_rets)} obs. daily | "
          f"{core_rets.index.min().date()} → {core_rets.index.max().date()}")

    core_eqw_rets = charger_core_eqw_returns_from_csv(cfg)
    print("    Benchmark filtre beta : Core équipondéré (3 ETF sélectionnés via CSV)")
    print(f"    {len(core_eqw_rets)} obs. daily | "
          f"{core_eqw_rets.index.min().date()} → {core_eqw_rets.index.max().date()}")

    all_selected: Dict[str, List[str]] = {}
    all_info: Dict[str, pd.DataFrame] = {}
    all_metrics: Dict[str, pd.DataFrame] = {}
    all_betas: Dict[str, pd.DataFrame] = {}

    for bloc_name, bloc_cfg in cfg.blocs.items():
        sel, metrics, info, beta_roll = traiter_bloc(bloc_name, bloc_cfg, core_rets, core_eqw_rets, cfg)
        all_selected[bloc_name] = sel
        all_info[bloc_name] = info
        all_metrics[bloc_name] = metrics
        all_betas[bloc_name] = beta_roll

    # ── Export CSV sélection ─────────────────────────────────────────────
    print("\n\n[2] Export...")
    rows = []
    for bloc_name, tickers in all_selected.items():
        metrics = all_metrics[bloc_name]
        info = all_info[bloc_name]
        for ticker in tickers:
            row: Dict = {"bloc": bloc_name, "ticker": ticker}
            if ticker in info.index:
                for col in ["strategie", "devise", "expense_pct", "aum_usd_m"]:
                    row[col] = info.at[ticker, col] if col in info.columns else np.nan
            if ticker in metrics.index:
                for col in ["vol_calib", "sharpe_calib", "alpha_annual", "beta_core",
                            "drawdown_calib", "skew_calib", "kurtosis_calib", "n_obs_calib"]:
                    row[col] = metrics.at[ticker, col] if col in metrics.columns else np.nan
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(cfg.output_selected_csv, index=False)
    print(f"    -> {cfg.output_selected_csv}")

    # Export beta rolling par bloc
    out_dir = Path(cfg.output_selected_csv).parent
    for bloc_name, beta_roll in all_betas.items():
        if not beta_roll.empty:
            out = str(out_dir / f"satellite_beta_rolling_{bloc_name}.csv")
            beta_roll.to_csv(out)
            print(f"    -> {out}")

    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RÉSUMÉ – Fonds satellite sélectionnés")
    print("=" * 60)
    for bloc_name, tickers in all_selected.items():
        print(f"\n  {bloc_name} ({len(tickers)} fonds) :")
        for t in tickers:
            print(f"    • {t}")
    total = sum(len(v) for v in all_selected.values())
    print(f"\n  Total : {total} fonds satellite")


if __name__ == "__main__":
    main()