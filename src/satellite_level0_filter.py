"""
Satellite Level 0 Filter
Filtrage de base: devise et AUM minimum
"""

import pandas as pd
from pathlib import Path


def load_satellite_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Charge tous les fichiers STRAT_info.xlsx et les combine.
    
    Parameters
    ----------
    data_dir : str
        Répertoire contenant les fichiers Excel
        
    Returns
    -------
    pd.DataFrame
        DataFrame combiné avec colonne 'Strat' identifiant la source
    """
    all_data = []
    
    for strat_num in [1, 2, 3]:
        filepath = Path(data_dir) / f"STRAT{strat_num}_info.xlsx"
        df = pd.read_excel(filepath, sheet_name=0)
        df['Strat'] = f'STRAT{strat_num}'
        all_data.append(df)
    
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


def filter_level0(
    df: pd.DataFrame,
    devise: list,
    min_aum_usd: float,
    aum_col: str = 'Total actifs USD (M)',
    devise_col: str = 'Dev'
) -> pd.DataFrame:
    """
    Applique les filtres de niveau 0:
    - Devise (parmi les valeurs spécifiées)
    - AUM USD minimum
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des fonds
    devise : list
        Liste des devises acceptées (ex: ['Euro', 'Euro (BEF)'])
    min_aum_usd : float
        AUM minimum en millions USD
    aum_col : str
        Nom de la colonne AUM
    devise_col : str
        Nom de la colonne devise
        
    Returns
    -------
    pd.DataFrame
        DataFrame filtré
    """
    # Filtre devise
    mask_devise = df[devise_col].isin(devise)
    
    # Filtre AUM
    mask_aum = df[aum_col] > min_aum_usd
    
    # Combine les critères
    df_filtered = df[mask_devise & mask_aum].copy()
    
    return df_filtered


def get_summary_by_bloc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé du nombre de fonds par bloc (Strat).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des fonds
        
    Returns
    -------
    pd.DataFrame
        Résumé par Strat
    """
    summary = df.groupby('Strat').size().reset_index(name='Nombre de fonds')
    return summary


def filter_satellite_level0(
    data_dir: str = "data",
    devise: list = None,
    min_aum_usd: float = 50.0,
    verbose: bool = True
) -> tuple:
    """
    Pipeline complet de filtrage niveau 0.
    
    Parameters
    ----------
    data_dir : str
        Répertoire contenant les fichiers Excel
    devise : list
        Liste des devises acceptées (défaut: ['Euro', 'Euro (BEF)'])
    min_aum_usd : float
        AUM minimum en millions USD
    verbose : bool
        Afficher les résultats
        
    Returns
    -------
    tuple
        (df_filtered, summary)
    """
    if devise is None:
        devise = ['Euro', 'Euro (BEF)']
    
    # Charge les données
    df = load_satellite_data(data_dir)
    
    if verbose:
        print("="*80)
        print("SATELLITE LEVEL 0 FILTER - DEVISE ET AUM")
        print("="*80)
        print(f"\nUnivers initial: {len(df)} fonds")
    
    # Applique les filtres
    df_filtered = filter_level0(
        df,
        devise=devise,
        min_aum_usd=min_aum_usd
    )
    
    if verbose:
        print(f"\nCritères appliqués:")
        print(f"  - Devise: {devise}")
        print(f"  - AUM USD > {min_aum_usd}M")
        print(f"\nFonds restants: {len(df_filtered)}")
    
    # Résumé par bloc
    summary = get_summary_by_bloc(df_filtered)
    
    if verbose:
        print(f"\nComposition par bloc:")
        print(summary.to_string(index=False))
        print(f"\nTotal: {summary['Nombre de fonds'].sum()} fonds")
    
    return df_filtered, summary
