"""Dynamic Core/Satellite allocator with volatility targeting and deadband."""

import numpy as np
import pandas as pd


def rebalance_dates_from_index(index, rebalance_freq="ME"):
    """Return last available dates for each rebalancing period."""
    idx = pd.DatetimeIndex(index).sort_values()
    bins = pd.Series(1, index=idx).groupby(pd.Grouper(freq=rebalance_freq))
    dates = [grp.index[-1] for _, grp in bins if len(grp) > 0]
    return pd.DatetimeIndex(dates)


def portfolio_vol_ex_ante(w_sat, sigma_core, sigma_sat, rho_cs):
    """Annualized ex-ante portfolio volatility for one or many satellite weights."""
    w_core = 1.0 - w_sat
    var = (
        (w_core ** 2) * (sigma_core ** 2)
        + (w_sat ** 2) * (sigma_sat ** 2)
        + 2.0 * w_core * w_sat * rho_cs * sigma_core * sigma_sat
    )
    return np.sqrt(np.clip(var, 0.0, None))


def allocate_core_satellite_dynamic(
    returns,
    core_col="core",
    sat_col="satellite",
    lookback=63,
    rebalance_freq="ME",
    target_vol=0.10,
    deadband=0.005,
    w_sat_min=0.20,
    w_sat_max=0.30,
    w_sat_step=0.005,
    initial_w_sat=None,
    annualization=252,
    rf_annual=0.02,
    vol_floor=0.095,
    vol_cap=0.105,
    verbose=True,
):
    """
    Dynamic Core/Satellite allocation: maximize ex-ante Sharpe ratio
    subject to vol ∈ [vol_floor, vol_cap] and w_sat ∈ [w_sat_min, w_sat_max].

    Falls back to target_vol ± deadband if no weight satisfies vol constraints.

    Returns: (weights_daily, portfolio_returns, portfolio_vol_rolling, decisions_df)
    """
    if core_col not in returns.columns or sat_col not in returns.columns:
        raise ValueError(f"Colonnes requises absentes: {core_col}, {sat_col}")

    ret = returns[[core_col, sat_col]].copy().sort_index()
    ret.columns = ["core", "sat"]

    if initial_w_sat is None:
        initial_w_sat = 0.5 * (w_sat_min + w_sat_max)
    initial_w_sat = float(np.clip(initial_w_sat, w_sat_min, w_sat_max))

    w_grid = np.arange(w_sat_min, w_sat_max + 1e-12, w_sat_step)
    if len(w_grid) == 0:
        raise ValueError("La grille de poids est vide. Vérifie w_sat_min/w_sat_max/w_sat_step.")

    rf_daily = (1.0 + rf_annual) ** (1.0 / annualization) - 1.0

    vol_core = ret["core"].rolling(lookback, min_periods=lookback).std() * np.sqrt(annualization)
    vol_sat = ret["sat"].rolling(lookback, min_periods=lookback).std() * np.sqrt(annualization)
    corr_cs = ret["core"].rolling(lookback, min_periods=lookback).corr(ret["sat"])
    mu_core = ret["core"].rolling(lookback, min_periods=lookback).mean() * annualization
    mu_sat = ret["sat"].rolling(lookback, min_periods=lookback).mean() * annualization

    reb_dates = rebalance_dates_from_index(ret.index, rebalance_freq=rebalance_freq)

    current_w_sat = initial_w_sat
    decisions = []
    w_sat_rebal = pd.Series(index=reb_dates, dtype=float)

    for dt in reb_dates:
        sc = vol_core.loc[dt] if dt in vol_core.index else np.nan
        ss = vol_sat.loc[dt] if dt in vol_sat.index else np.nan
        rc = corr_cs.loc[dt] if dt in corr_cs.index else np.nan
        mc = mu_core.loc[dt] if dt in mu_core.index else np.nan
        ms = mu_sat.loc[dt] if dt in mu_sat.index else np.nan

        if pd.isna(sc) or pd.isna(ss) or pd.isna(rc) or pd.isna(mc) or pd.isna(ms):
            decision = "insufficient_data"
            new_w_sat = current_w_sat
            sigma_current = np.nan
            sigma_opt = np.nan
        else:
            sigma_current = float(portfolio_vol_ex_ante(np.array([current_w_sat]), sc, ss, rc)[0])

            # Compute vol and Sharpe for each grid point
            sigma_grid = portfolio_vol_ex_ante(w_grid, sc, ss, rc)
            mu_grid = (1.0 - w_grid) * mc + w_grid * ms
            sharpe_grid = (mu_grid - rf_annual) / np.where(sigma_grid > 0, sigma_grid, np.nan)

            # Filter by vol constraint [vol_floor, vol_cap]
            feasible = (sigma_grid >= vol_floor) & (sigma_grid <= vol_cap)

            if feasible.any():
                # Max Sharpe among feasible weights
                sharpe_feasible = np.where(feasible, sharpe_grid, -np.inf)
                idx_best = int(np.argmax(sharpe_feasible))
                new_w_sat = float(w_grid[idx_best])
                sigma_opt = float(sigma_grid[idx_best])

                if new_w_sat == current_w_sat:
                    decision = "keep_optimal"
                else:
                    decision = "rebalance"
            else:
                # Fallback: closest to target_vol (deadband logic)
                if (target_vol - deadband) <= sigma_current <= (target_vol + deadband):
                    decision = "keep_deadband"
                    new_w_sat = current_w_sat
                    sigma_opt = sigma_current
                else:
                    idx_best = int(np.argmin(np.abs(sigma_grid - target_vol)))
                    new_w_sat = float(w_grid[idx_best])
                    sigma_opt = float(sigma_grid[idx_best])
                    decision = "rebalance_fallback"

        if verbose:
            msg = f"[{dt.date()}] decision={decision:>18} | w_sat: {current_w_sat:.3f} -> {new_w_sat:.3f}"
            if not pd.isna(sigma_current):
                msg += f" | vol_cur={sigma_current:.2%} | vol_opt={sigma_opt:.2%}"
            print(msg)

        decisions.append({
            "date": dt, "decision": decision,
            "w_sat_prev": current_w_sat, "w_sat_new": new_w_sat,
            "sigma_current": sigma_current, "sigma_opt": sigma_opt,
        })

        current_w_sat = new_w_sat
        w_sat_rebal.loc[dt] = current_w_sat

    weights_daily = pd.DataFrame(index=ret.index)
    weights_daily["w_sat"] = np.nan
    weights_daily.iloc[0, 0] = initial_w_sat
    weights_daily.loc[w_sat_rebal.index, "w_sat"] = w_sat_rebal.values
    weights_daily["w_sat"] = weights_daily["w_sat"].ffill().clip(w_sat_min, w_sat_max)
    weights_daily["w_core"] = 1.0 - weights_daily["w_sat"]

    if not np.allclose((weights_daily["w_core"] + weights_daily["w_sat"]).values, 1.0, atol=1e-10):
        raise AssertionError("La contrainte w_core + w_sat = 1 n'est pas respectée.")

    w_lag = weights_daily.shift(1).fillna(weights_daily.iloc[0])
    portfolio_returns = w_lag["w_core"] * ret["core"] + w_lag["w_sat"] * ret["sat"]
    portfolio_returns.name = "core_sat_portfolio_return"

    portfolio_vol_rolling = (
        portfolio_returns.rolling(lookback, min_periods=lookback).std() * np.sqrt(annualization)
    )
    portfolio_vol_rolling.name = "core_sat_portfolio_vol_rolling"

    decisions_df = pd.DataFrame(decisions).set_index("date")

    return weights_daily[["w_core", "w_sat"]], portfolio_returns, portfolio_vol_rolling, decisions_df
