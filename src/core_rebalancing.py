"""
Rebalanceur Core : Gestion du rebalancement des poids cibles.

Concept : 
- Les poids cibles sont fixés au départ (ex: Efficient Vol 15%)
- À une fréquence définie (quotidienne, hebdomadaire, mensuelle, trimestrielle, annuelle),
  on recalcule la composition du portefeuille pour revenir aux poids cibles
- Cela implique de vendre les assets qui ont surperformé et d'acheter ceux qui ont sous-performé

Fréquences supportées :
- 'daily' : chaque jour de trading
- 'weekly' : chaque lundi
- 'monthly' : 1er de chaque mois
- 'quarterly' : 1er de chaque trimestre (jan, apr, jul, oct)
- 'annual' : 2 janvier de chaque année
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class CoreRebalancer:
    """
    Gère le rebalancement annuel d'une allocation Core fixe.
    
    Paramètres:
    -----------
    core_weights : dict
        {"XDWD GY Equity": 0.71, "EUNH GY Equity": 0.237, "XBLC GY Equity": 0.05}
    daily_returns : pd.DataFrame
        Index = dates, Columns = tickers, Values = rendements journaliers (simple %)
    rebalance_dates : list[str] or 'annual'
        Dates de rebalancement ou 'annual' pour janvier de chaque année
    """
    
    def __init__(
        self,
        core_weights: Dict[str, float],
        daily_returns: pd.DataFrame,
        rebalance_dates: str | list = 'annual',
        start_value: float = 100.0,
    ):
        self.core_weights = core_weights
        self.daily_returns = daily_returns.copy()
        self.start_value = start_value
        self.rebalance_dates = self._get_rebalance_dates(rebalance_dates)
        
    def _get_rebalance_dates(self, rebalance_dates):
        """
        Extrait les dates de rebalancement.
        
        Paramètres:
        -----------
        rebalance_dates : str or list
            'daily', 'weekly', 'monthly', 'quarterly', 'annual', ou liste de dates
        
        Retour:
        -------
        list of str (au format 'YYYY-MM-DD')
        """
        if isinstance(rebalance_dates, list):
            return rebalance_dates
        
        dates = pd.DatetimeIndex(self.daily_returns.index)
        
        if rebalance_dates == 'daily':
            # Tous les jours
            return [d.strftime('%Y-%m-%d') for d in dates]
        
        elif rebalance_dates == 'weekly':
            # Chaque lundi
            return [d.strftime('%Y-%m-%d') for d in dates if d.weekday() == 0]
        
        elif rebalance_dates == 'monthly':
            # 1er ou 2e jour de trading de chaque mois
            monthly_dates = []
            prev_month = None
            for d in dates:
                if d.month != prev_month:
                    monthly_dates.append(d.strftime('%Y-%m-%d'))
                    prev_month = d.month
            return monthly_dates
        
        elif rebalance_dates == 'quarterly':
            # 1er jour de trading de chaque trimestre
            quarterly_dates = []
            prev_quarter = None
            for d in dates:
                quarter = (d.month - 1) // 3
                year_quarter = (d.year, quarter)
                if year_quarter != prev_quarter:
                    quarterly_dates.append(d.strftime('%Y-%m-%d'))
                    prev_quarter = year_quarter
            return quarterly_dates
        
        elif rebalance_dates == 'annual':
            # 2 janvier de chaque année (ou 1er jour de trading de l'année)
            annual_dates = []
            prev_year = None
            for d in dates:
                if d.year != prev_year:
                    annual_dates.append(d.strftime('%Y-%m-%d'))
                    prev_year = d.year
            return annual_dates
        
        else:
            raise ValueError(f"Fréquence inconnue : {rebalance_dates}. "
                           f"Utilisez : 'daily', 'weekly', 'monthly', 'quarterly', 'annual' ou une liste de dates")

    
    def compute_portfolio_evolution(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calcule l'évolution du portefeuille avec rebalancement annuel.
        
        Retour:
        -------
        portfolio_values : pd.DataFrame
            Valeurs du portefeuille au fil du temps
        held_weights : pd.DataFrame  
            Poids réels du portefeuille (avant rebalancement)
        rebalance_transactions : pd.DataFrame
            Les ajustements de rebalancement à chaque date
        """
        # Initialisation
        tickers = list(self.core_weights.keys())
        dates = self.daily_returns.index
        
        # Prix cumulés (base 1.0)
        cum_returns = (1 + self.daily_returns).cumprod()
        
        # Initialiser les holdings en unités d'assets
        holdings = {}
        for ticker in tickers:
            holdings[ticker] = np.zeros(len(dates))
        
        portfolio_values = np.zeros(len(dates))
        held_weights_array = np.zeros((len(dates), len(tickers)))
        rebalance_log = []
        
        # Valeur initiale du portefeuille
        total_val = self.start_value
        
        # Initialiser les holdings au 1er jour avec poids cibles
        for i, ticker in enumerate(tickers):
            target_val = self.start_value * self.core_weights[ticker]
            price_0 = cum_returns[ticker].iloc[0]
            holdings[ticker][0] = target_val / price_0
            held_weights_array[0, i] = self.core_weights[ticker]
        
        portfolio_values[0] = self.start_value
        
        # Simulation jour par jour
        rebalance_dates_dt = pd.DatetimeIndex(self.rebalance_dates)
        rebalance_dates_set = set(rebalance_dates_dt.date)
        
        for t in range(1, len(dates)):
            current_date = dates[t]
            
            # Calculer la valeur du portefeuille (avant rebalancement)
            total_val = 0.0
            ticker_values = {}
            for i, ticker in enumerate(tickers):
                price_t = cum_returns[ticker].iloc[t]
                val_t = holdings[ticker][t-1] * price_t
                ticker_values[ticker] = val_t
                total_val += val_t
            
            # Conserver les holdings
            for ticker in tickers:
                holdings[ticker][t] = holdings[ticker][t-1]
            
            # Stocker les weights réels
            for i, ticker in enumerate(tickers):
                held_weights_array[t, i] = ticker_values[ticker] / total_val if total_val > 0 else 0
            
            portfolio_values[t] = total_val
            
            # Vérifier si c'est une date de rebalancement
            if current_date.date() in rebalance_dates_set:
                # Rebalancer pour revenir aux poids cibles
                rebal_dict = {
                    'date': current_date,
                    'portfolio_value_before': total_val,
                }
                
                for i, ticker in enumerate(tickers):
                    price_t = cum_returns[ticker].iloc[t]
                    target_val = total_val * self.core_weights[ticker]
                    current_val = ticker_values[ticker]
                    adjustment = (target_val - current_val) / price_t
                    
                    holdings[ticker][t] += adjustment
                    
                    rebal_dict[f'{ticker}_target_val'] = target_val
                    rebal_dict[f'{ticker}_current_val'] = current_val
                    rebal_dict[f'{ticker}_adjustment'] = adjustment
                
                rebalance_log.append(rebal_dict)
        
        # Convertir en DataFrames
        portfolio_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
        }).set_index('date')
        
        weights_df = pd.DataFrame(
            held_weights_array,
            index=dates,
            columns=tickers
        )
        
        rebalance_df = pd.DataFrame(rebalance_log) if rebalance_log else None
        
        return portfolio_df, weights_df, rebalance_df
    
    def summary_table(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée un tableau récapitulatif du rebalancement par année.
        
        Paramètres:
        -----------
        weights_df : pd.DataFrame
            DataFrame des poids réels (sortie de compute_portfolio_evolution)
        
        Retour:
        -------
        pd.DataFrame avec stats par année
        """
        weights_df['year'] = weights_df.index.year
        tickers = list(self.core_weights.keys())
        
        summary = []
        for year in weights_df['year'].unique():
            year_data = weights_df[weights_df['year'] == year]
            row = {'year': year}
            
            for ticker in tickers:
                target = self.core_weights[ticker]
                min_w = year_data[ticker].min()
                max_w = year_data[ticker].max()
                mean_w = year_data[ticker].mean()
                drift = max_w - min_w
                
                row[f'{ticker}_target'] = target
                row[f'{ticker}_drift'] = drift
                row[f'{ticker}_max_dev'] = max(abs(min_w - target), abs(max_w - target))
            
            summary.append(row)
        
        return pd.DataFrame(summary).set_index('year')


    
    def compute_tracking_error(
        self,
        portfolio_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calcule la tracking error et l'impact du rebalancing.
        
        Mesure la différence de performance entre :
        1. Portefeuille avec rebalancing (portfolio_df)
        2. Portefeuille sans rebalancing (poids fixes)
        
        Retour:
        -------
        Dict contenant:
            - 'tracking_error' : écart-type des rendements différentiels (annualisé)
            - 'cumulative_impact' : impact total cumulé (en %)
            - 'rebalancing_benefit' : bénéfice net du rebalancing
            - 'final_value_rebalanced' : valeur finale avec rebalancing
            - 'final_value_static' : valeur finale sans rebalancing
        """
        # Portefeuille avec rebalancing
        port_with_rebal = portfolio_df['portfolio_value'].values
        
        # Portefeuille statique (poids fixes, sans rebalancement)
        tickers = list(self.core_weights.keys())
        cum_returns = (1 + self.daily_returns).cumprod()
        
        port_static = np.zeros(len(self.daily_returns))
        port_static[0] = self.start_value
        
        for t in range(1, len(self.daily_returns)):
            val = 0.0
            for ticker in tickers:
                share_val = self.core_weights[ticker] * self.start_value
                price_ratio = cum_returns[ticker].iloc[t] / cum_returns[ticker].iloc[0]
                val += share_val * price_ratio
            port_static[t] = val
        
        # Rendements quotidiens
        ret_rebal = np.diff(port_with_rebal) / port_with_rebal[:-1]
        ret_static = np.diff(port_static) / port_static[:-1]
        
        # Rendements différentiels
        diff_ret = ret_rebal - ret_static
        
        # Tracking error annualisée
        tracking_error = np.std(diff_ret) * np.sqrt(252)
        
        # Impact cumulé
        cumulative_impact = (port_with_rebal[-1] / port_static[-1] - 1) * 100
        
        return {
            'tracking_error': float(tracking_error),
            'cumulative_impact': float(cumulative_impact),
            'rebalancing_benefit': float(cumulative_impact),  # positif = bénéfice
            'final_value_rebalanced': float(port_with_rebal[-1]),
            'final_value_static': float(port_static[-1]),
            'daily_differential_returns': diff_ret,
            'port_with_rebal': port_with_rebal,
            'port_static': port_static,
        }


def load_core_weights_from_strategies(
    strategies: Dict,
    strategy_name: str = "Efficient Vol 15%"
) -> Dict[str, float]:
    """
    Extrait les poids cibles d'une stratégie.
    
    Paramètres:
    -----------
    strategies : Dict
        Le dictionnaire retourné par efficient_frontier_core.main()
    strategy_name : str
        Nom de la stratégie (ex: "Efficient Vol 15%")
    
    Retour:
    -------
    Dict avec {ticker: weight} normalisé
    """
    if strategy_name not in strategies:
        raise ValueError(f"Stratégie '{strategy_name}' non trouvée. Disponibles: {list(strategies.keys())}")
    
    w = strategies[strategy_name]["w"]
    tickers = None
    
    # Récupérer les tickers depuis une autre stratégie si possible
    for strat_name, strat_data in strategies.items():
        if "w" in strat_data:
            # Récupérer depuis les données du backtest (on va devoir passer les tickers)
            break
    
    # On doit passer les tickers directement
    # Ce sera fait dans le wrapper
    return w


if __name__ == "__main__":
    print("Core Rebalancing Module")
