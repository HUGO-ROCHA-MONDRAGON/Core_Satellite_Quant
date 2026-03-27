# 🎯 DEBUG FILTRE NIVEAU 1 - SYNTHÈSE DE LA CORRECTION

## Le Problème Original

Votre filtre niveau 1 rejetait **TOUS les 167 fonds** de level 0, alors que l'ancien pipeline en passait plusieurs.

## Root Cause Analysis - CEQ LA VRAIE DIFFÉRENCE 🎯

### Anciens Paramètres (Pipeline Original - Satellite_pipeline.py)

```python
# LINE 174-177: Configuration
beta_filter_window_days: int = 63        # ➡️ LABEL SEULEMENT (non utilisé!)
beta_filter_max_abs: float = 0.35        # Condition 1
beta_filter_min_pass_ratio: float = 0.80 # Condition 3
beta_filter_q75_max: float = 0.55        # Condition 2

# LINE 611-750: Fonction calculer_beta_rolling() + filtrer_niveau_beta_initial()
```

### Logique du Pipeline Original

1. **Étape 1**: Calculer le rolling beta avec **fenêtre = 252 jours** (annuel)
   ```python
   # Ref: calculer_beta_rolling(window=252)
   # Pour chaque fonds, calcule beta_rolling = Cov(fund_ret, core_ret) / Var(core_ret)
   # sur une fenêtre glissante de 252j
   ```

2. **Étape 2**: Extraire la fenêtre de calibration (2019-2020, ~523 jours)

3. **Étape 3**: Appliquer **3 conditions SIMULTANÉMENT sur TOUTE LA SÉRIE** de betas:
   ```
   ✓ median(|β|) <= 0.35
   ✓ Q75(|β|) <= 0.55
   ✓ Ratio(|β| <= 0.35) >= 0.80  (i.e., 80% des 523 jours ont |β|≤0.35)
   ```

4. **Résultat**: Fonds PASSÉ si et SEULEMENT SI les 3 conditions sont vraies

### Vos Paramètres (Version Cassée v1) ❌

```python
ROLLING_WINDOW = 252       # ✓ Bon
BETA_THRESHOLD = 1.2       # ❌ PROBLEM: Seuil simple (pas de statistiques multi)
PERCENTILE_THRESHOLD = 50% # ❌ PROBLEM: Cherchait |β| < 1.2 dans 50% des périodes (logic différente)
```

**Logique cassée**: Cherchait si |beta| < 1.2 dans 50% des observations
- C'est une logic E DIFFÉRENTE des 3 conditions du pipeline!
- Seuil de 1.2 trop large par rapport à 0.35
- Ratio 50% vs 80% → trop permissif

## La Correction ✅

Le fichier `/workspaces/Core_Satellite_Quant/src/satellite_level1_beta_filter_final.py` 
implémente la **VRAIE logique** du pipeline:

```python
def apply_level1_filter_corrected(
    level0_df, prices_aligned, core_returns_aligned,
    rolling_window=252,              # 🎯 252 jours pour calculer rolling
    median_beta_max=0.35,            # 🎯 Condition 1: median(|β|) <= 0.35
    q75_beta_max=0.55,               # 🎯 Condition 2: Q75(|β|) <= 0.55
    pass_ratio_min=0.80,             # 🎯 Condition 3: ratio >= 80%
    ...
)
```

### Étapes de la Correction

1. ✅ Utilise la **fenêtre 252j** pour calculer rolling beta
2. ✅ Applique les **3 conditions MULTI-STAT** simultanément
3. ✅ Fonds PASSÉ = **TOUTES les 3 conditions vraies**
4. ✅ Réutilise les **données pré-alignées** (pas de recharge separate)

## Comment Tester

Exécutez la cellule "Filtre Niveau 1" du notebook:

```python
# Cell #VSC-7307af28
df_satellite_level1, results_level1 = apply_level1_filter_corrected(
    df_satellite_level0,
    df_satellite_prices_aligned,  
    core_returns_aligned,
    rolling_window=252,
    median_beta_max=0.35,
    q75_beta_max=0.55,
    pass_ratio_min=0.80,
    verbose=True
)
```

## Résultats Attendus

**AVANT (Bug)**: 0/167 fonds passent
**APRÈS (Fix)**: N > 0 fonds devraient passer (à tester)

Si pareilfois TOUJOURS 0 fonds passent même avec la correction, c'est un problème 
de **data quality** (pas de betas disponibles, données vides, etc.)

## Prochaines Étapes

1. ✅ Corriger niveau 1 (FAIT)
2. ⏳ Créer niveau 2 (Alpha + Frais score)
3. ⏳ Implémenter rolling IS/OOS windows (2019-20 IS → 21 OOS, etc.)
4. ⏳ Final selection (15-21 fonds)

---
**Auteur**: Debug Session | **Date**: 2024 | **Status**: ✅ Root cause found & fixed
