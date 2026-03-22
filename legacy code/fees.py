"""
Calcul de l'enveloppe de frais (TER) pour un portefeuille Core/Satellite.

Objectif du sujet :
- enveloppe totale <= 80 bps (0.80%)
- Core en ETF à faibles frais
- Satellite en fonds/gestion active, frais maîtrisés

Ce script calcule :
- TER Core (pondéré)
- TER Satellite (pondéré)
- TER total (pondéré)
et exporte un tableau pour le rapport.

Entrées :
- aucune (paramètres dans le code)
Sorties :
- outputs/fees_breakdown.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class FeesConfig:
    """Configuration des frais."""
    project_root: Path = Path(__file__).resolve().parent.parent
    output_dir: Path = project_root / "outputs"
    output_csv: Path = output_dir / "fees_breakdown.csv"

    fee_budget: float = 0.008  # 80 bps = 0.80%


def calcul_ter_pondere(weights: Dict[str, float], ters: Dict[str, float]) -> float:
    """
    Calcule le TER pondéré.
    - weights: poids par instrument (somme = 1 pour la poche considérée)
    - ters: TER par instrument (en décimal, ex 0.0020 pour 0.20%)
    """
    missing = [k for k in weights if k not in ters]
    if missing:
        raise ValueError(f"TER manquant pour: {missing}")

    total = 0.0
    for k, w in weights.items():
        total += float(w) * float(ters[k])
    return total


def main() -> None:
    cfg = FeesConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # 1) Paramètres à renseigner
    # --------------------------
    # Poids structurels (comme dans portfolio_engine.py)
    w_sat_total = 0.30
    w_core_total = 1.0 - w_sat_total

    # Poids internes (exemples : à adapter si besoin)
    # Core : 3 ETF (ici équipondéré ex-ante pour le calcul TER,
    #       sinon tu peux remplacer par une moyenne des poids effectifs si tu préfères)
    core_weights = {
        "SWDA": 1.0 / 3.0,
        "CBE3": 1.0 / 3.0,
        "ICOV": 1.0 / 3.0,
    }

    # Satellite : notre construction DBMF + GLD
    # (tu peux mettre 50/50 ou utiliser la moyenne des poids inv-vol)
    sat_weights = {
        "DBMF": 0.50,
        "GLD": 0.50,
    }

    # TER (en décimal)
    # IMPORTANT : ce sont des hypothèses à documenter dans le rapport.
    # Tu peux changer ces valeurs pour coller aux factsheets des supports choisis.
    ters = {
        # Core ETF (hypothèses typiques : 10-25 bps)
        "SWDA": 0.0020,  # 0.20%
        "CBE3": 0.0015,  # 0.15%
        "ICOV": 0.0020,  # 0.20%

        # Satellite proxies (DBMF/GLD sont des ETFs, mais DBMF représente une stratégie "active-like")
        "DBMF": 0.0085,  # 0.85% (ordre de grandeur managed futures ETF)
        "GLD": 0.0040,   # 0.40% (ETF/ETC or typique)
    }

    # --------------------------
    # 2) Calculs
    # --------------------------
    core_ter = calcul_ter_pondere(core_weights, ters)
    sat_ter = calcul_ter_pondere(sat_weights, ters)

    total_ter = w_core_total * core_ter + w_sat_total * sat_ter

    # --------------------------
    # 3) Table de décomposition
    # --------------------------
    rows = []

    # Core
    for k, w in core_weights.items():
        rows.append(
            {
                "Poche": "Core",
                "Instrument": k,
                "Poids_dans_poche": w,
                "Poids_dans_portefeuille": w * w_core_total,
                "TER": ters[k],
                "Contribution_TER_portefeuille": (w * w_core_total) * ters[k],
            }
        )

    # Satellite
    for k, w in sat_weights.items():
        rows.append(
            {
                "Poche": "Satellite",
                "Instrument": k,
                "Poids_dans_poche": w,
                "Poids_dans_portefeuille": w * w_sat_total,
                "TER": ters[k],
                "Contribution_TER_portefeuille": (w * w_sat_total) * ters[k],
            }
        )

    out = pd.DataFrame(rows)

    # Ajout lignes de synthèse
    summary = pd.DataFrame(
        [
            {
                "Poche": "SYNTHÈSE",
                "Instrument": "TER Core (pondéré)",
                "Poids_dans_poche": "",
                "Poids_dans_portefeuille": w_core_total,
                "TER": core_ter,
                "Contribution_TER_portefeuille": w_core_total * core_ter,
            },
            {
                "Poche": "SYNTHÈSE",
                "Instrument": "TER Satellite (pondéré)",
                "Poids_dans_poche": "",
                "Poids_dans_portefeuille": w_sat_total,
                "TER": sat_ter,
                "Contribution_TER_portefeuille": w_sat_total * sat_ter,
            },
            {
                "Poche": "SYNTHÈSE",
                "Instrument": "TER Total (pondéré)",
                "Poids_dans_poche": "",
                "Poids_dans_portefeuille": 1.0,
                "TER": total_ter,
                "Contribution_TER_portefeuille": total_ter,
            },
            {
                "Poche": "CONTRAINTE",
                "Instrument": "Budget frais (80 bps)",
                "Poids_dans_poche": "",
                "Poids_dans_portefeuille": "",
                "TER": cfg.fee_budget,
                "Contribution_TER_portefeuille": "",
            },
        ]
    )

    full = pd.concat([out, summary], ignore_index=True)

    full.to_csv(cfg.output_csv, index=False)

    # --------------------------
    # 4) Prints
    # --------------------------
    print("=== FRAIS (TER) ===")
    print(f"Poids Core total     : {w_core_total:.0%}")
    print(f"Poids Satellite total: {w_sat_total:.0%}\n")

    print(f"TER Core (pondéré)     : {core_ter:.4%}")
    print(f"TER Satellite (pondéré): {sat_ter:.4%}")
    print(f"TER Total (pondéré)    : {total_ter:.4%}")
    print(f"Budget frais           : {cfg.fee_budget:.4%}")

    if total_ter <= cfg.fee_budget:
        print("\n Contrainte respectée : TER total <= 80 bps")
    else:
        print("\n Contrainte non respectée : TER total > 80 bps")

    print(f"\n-> Export: {cfg.output_csv}")


if __name__ == "__main__":
    main()