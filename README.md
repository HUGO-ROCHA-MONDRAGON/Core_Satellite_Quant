# Core-Satellite Quant — Logique Financière de Construction du Portefeuille

> Document de référence décrivant **toute la logique financière** du fonds Core-Satellite EUR.
> Ce document ne contient aucune référence au code — il se concentre uniquement sur les choix
> d'investissement, les méthodologies quantitatives et les paramètres retenus.

---

## Table des matières

1. [Vue d'ensemble du fonds](#1-vue-densemble-du-fonds)
2. [Fenêtres temporelles IS / OOS](#2-fenêtres-temporelles-is--oos)
3. [Poche Core — Sélection des ETFs](#3-poche-core--sélection-des-etfs)
4. [Poche Core — Frontière efficiente et choix de la stratégie](#4-poche-core--frontière-efficiente-et-choix-de-la-stratégie)
5. [Poche Core — Simulation et rebalancement](#5-poche-core--simulation-et-rebalancement)
6. [Détection du régime de marché](#6-détection-du-régime-de-marché)
7. [Poche Satellite — Filtre structurel (Level 0)](#7-poche-satellite--filtre-structurel-level-0)
8. [Poche Satellite — Filtre beta rolling (Level 1)](#8-poche-satellite--filtre-beta-rolling-level-1)
9. [Poche Satellite — Scoring alpha + frais (Level 2)](#9-poche-satellite--scoring-alpha--frais-level-2)
10. [Poche Satellite — Scoring personnalisé par bloc](#10-poche-satellite--scoring-personnalisé-par-bloc)
11. [Poche Satellite — Sélection annuelle et trimestrielle glissante](#11-poche-satellite--sélection-annuelle-et-trimestrielle-glissante)
12. [Allocation dynamique intra-Satellite par régime](#12-allocation-dynamique-intra-satellite-par-régime)
13. [Performance Satellite et contributions par bloc](#13-performance-satellite-et-contributions-par-bloc)
14. [Comparaison Core vs Satellite](#14-comparaison-core-vs-satellite)
15. [Frais Satellite](#15-frais-satellite)
16. [Allocation dynamique Core / Satellite (vol-targeting)](#16-allocation-dynamique-core--satellite-vol-targeting)
17. [Portefeuille final — Dashboard et métriques](#17-portefeuille-final--dashboard-et-métriques)
18. [Garde-fous anti look-ahead bias](#18-garde-fous-anti-look-ahead-bias)

---

## 1. Vue d'ensemble du fonds

Ce fonds suit une **architecture Core-Satellite** :

- **Poche Core (70–75 %)** : exposition aux primes de risque de long terme via 3 ETFs cotés en EUR couvrant les grandes classes d'actifs (actions mondiales, obligations souveraines, crédit). Objectif : bêta de marché à faible coût.
- **Poche Satellite (25–30 %)** : génération d'alpha décorrélé du marché via des fonds alternatifs sélectionnés rigoureusement (CTA, market neutral, event driven, crédit structuré). Objectif : bêta ≈ 0 vis-à-vis du Core, alpha positif indépendant du cycle.

L'allocation entre Core et Satellite est **dynamique** : chaque mois, le poids Satellite est recalculé pour cibler une volatilité globale du portefeuille de **10 % annualisée**, avec une bande morte (deadband) de ±0,5 %.

| Paramètre | Valeur |
|-----------|--------|
| Volatilité cible totale | 10 % annualisé |
| Poids Satellite | 25 % – 30 % du portefeuille |
| Poids Core | 70 % – 75 % du portefeuille |
| Devise | EUR |
| Période de calibration (IS) | 2019-01-01 → 2020-12-31 |
| Période de validation (OOS) | 2021-01-01 → 2025-12-31 |

---

## 2. Fenêtres temporelles IS / OOS

| Fenêtre | Dates | Rôle |
|---------|-------|------|
| **In-Sample (IS)** | 01/01/2019 → 31/12/2020 | Calibration de tous les paramètres : poids Core via frontière efficiente, seuils de filtrage Satellite, seuils de régime |
| **Out-of-Sample (OOS)** | 01/01/2021 → 31/12/2025 | Application des règles calibrées, sans aucune recalibration. Évaluation de la performance réelle |

### Pourquoi 2019–2020 comme IS ?

Cette fenêtre est dite *"through-the-cycle"* car elle couvre :
- **2019** : marché haussier, faible volatilité
- **T1 2020** : crash COVID (−35 % en 5 semaines), contaminant toutes les classes d'actifs

Tout ETF ou fonds retenu sur cette fenêtre a donc été testé dans deux régimes opposés. Cela réduit considérablement le risque de sélection sur un unique régime favorable.

---

## 3. Poche Core — Sélection des ETFs

### Univers d'ETFs

L'univers de départ est constitué d'ETFs cotés en EUR, organisés en **3 thèmes** (blocs) :
- **Equity** : ETFs actions mondiales / européennes
- **Rates** : ETFs obligations souveraines
- **Credit** : ETFs obligations d'entreprises (investment grade)

### Filtres structurels pré-sélection

Avant d'entrer dans l'optimisation, chaque ETF doit satisfaire :

| Filtre | Critère | Justification |
|--------|---------|---------------|
| Date de lancement | ≤ 01/01/2019 | Au moins 2 ans de track record avant la période OOS |
| Cotation | Pas de gap majeur | Liquidité correcte |

### ETFs Core retenus

Pour chaque thème, **un seul ETF** est retenu — celui ayant le meilleur ratio rendement/risque sur l'IS. Les 3 ETFs retenus sont :

| Thème | Ticker Bloomberg | Nom |
|-------|-----------------|-----|
| Equity | XDWD GY Equity | Xtrackers MSCI World UCITS ETF |
| Rates | EUNH GY Equity | iShares Core EUR Govt Bond UCITS ETF |
| Credit | XBLC GY Equity | Xtrackers EUR Corporate Bond UCITS ETF |

---

## 4. Poche Core — Frontière efficiente et choix de la stratégie

### Méthodologie

Sur la **fenêtre IS (2019-2020)**, on estime les rendements et la matrice de covariance des 3 ETFs Core à partir des rendements log-journaliers, puis on trace la **frontière efficiente** sous contraintes de poids.

### Contraintes sur les poids

| Paramètre | Valeur |
|-----------|--------|
| Poids minimum par ETF | 5 % |
| Poids maximum par ETF | 90 % |

Chaque ETF doit avoir au minimum 5 % d'allocation et ne peut pas dépasser 90 %, ce qui empêche les solutions dégénérées (tout sur un seul actif).

### Stratégies comparées

Cinq types de stratégies sont évaluées sur la frontière IS, puis appliquées **fixement** sur la période OOS :

1. **Max Sharpe** : poids optimisés pour maximiser le ratio Sharpe IS
2. **Min Variance** : poids optimisés pour minimiser la variance IS
3. **Equal Weight** : 1/3 sur chaque ETF (benchmark naïf)
4. **Risk Parity** : poids proportionnels à $1/\sigma_i$ (inverse de la volatilité IS)
5. **Efficient Vol X %** : pour chaque cible de volatilité de 10 % à 20 % (par pas de 1 %), on trouve les poids qui maximisent le rendement sous la contrainte $\sigma_{portfolio} \leq X\%$

### Stratégie retenue : Efficient Vol 17 %

Après comparaison des performances OOS de toutes les stratégies, la stratégie **Efficient Vol 17 %** est sélectionnée. Elle offre le meilleur compromis rendement/risque OOS.

Les poids résultants (calibrés sur IS, appliqués fixement sur OOS) sont :

| ETF | Poids |
|-----|-------|
| **XDWD GY Equity** (actions) | **81,32 %** |
| **EUNH GY Equity** (taux) | **13,68 %** |
| **XBLC GY Equity** (crédit) | **5,00 %** |

L'allocation est fortement orientée actions (81 %), ce qui est cohérent avec une cible de volatilité élevée (17 %). Le crédit est au minimum (5 %) car son profil rendement/risque IS ne justifiait pas une allocation plus importante.

---

## 5. Poche Core — Simulation et rebalancement

### Principe

Une fois les poids cibles définis par la frontière efficiente, on simule le portefeuille Core sur toute la période OOS (2021→2025) en intégrant :
- Le **rebalancement** périodique vers les poids cibles
- Les **frais TER** propres à chaque ETF (déduits quotidiennement)
- Les **coûts de transaction** à chaque rebalancement

### Paramètres de simulation

| Paramètre | Valeur |
|-----------|--------|
| Fréquence de rebalancement | **Annuelle** (1er jour de trading de chaque année) |
| Coût de transaction | **10 bps par côté** (achat et vente) |
| Frais fixes additionnels | 0 bps |
| Valeur initiale du portefeuille | 100 (base) |
| Période simulée | 01/01/2021 → 31/12/2025 |

### Mécanique du rebalancement

À chaque date de rebalancement :
1. On observe la **valeur de marché** de chaque position (les poids ont dérivé depuis le dernier rebalancement du fait des performances relatives des ETFs)
2. On calcule l'**écart** entre poids effectifs et poids cibles
3. On achète/vend pour **revenir aux poids cibles**
4. Un **coût de transaction de 10 bps** est appliqué sur le montant en valeur absolue de chaque ajustement

Entre deux dates de rebalancement, les poids dérivent naturellement avec les marchés (pas de rebalancement intra-période).

### Application des TER

Les frais TER de chaque ETF sont déduits quotidiennement sous forme d'un *drag* :

$$r_{net,i}(t) = \frac{1 + r_{brut,i}(t)}{1 + TER_{daily,i}} - 1$$

où $TER_{daily,i} = (1 + TER_{annual,i})^{1/252} - 1$.

### Sortie

La simulation produit une **série de NAV (Net Asset Value)** quotidienne du portefeuille Core, intégrant rebalancement, frais et coûts de transaction. C'est cette série qui sert de benchmark pour toute la suite (sélection Satellite, comparaison Core vs Satellite, allocation finale).

---

## 6. Détection du régime de marché

Le portefeuille Satellite adapte son allocation interne (poids des 3 blocs) au **régime de marché** détecté sur les rendements du Core. Cette détection est **causale** : chaque décision ne repose que sur des données passées.

### Les 3 régimes

| Régime | Interprétation | Implication Satellite |
|--------|----------------|-----------------------|
| **Stress** | Vol élevée, corrélations accrues, momentum négatif | Surpondérer le bloc 1 (décorrélation / convexité) |
| **Neutre** | Conditions normales de marché | Équilibrer les 3 blocs |
| **Risk-on** | Vol basse, momentum positif | Surpondérer le bloc 3 (carry / crédit) |

### Construction de l'indicateur

L'indicateur de régime est composite, construit à partir de 3 métriques rolling sur les rendements du Core (pondérés par les poids de la stratégie sélectionnée, nets de TER) :

1. **Volatilité rolling 63 jours** ($\sigma_{63}$) : mesure de la turbulence récente
2. **Corrélation rolling equity-bonds 63 jours** ($\rho_{eq,bonds}$) : quand cette corrélation monte, les actifs sont entraînés ensemble — signe de stress
3. **Momentum rolling 63 jours** ($mom_{63}$) : rendement cumulé sur 63 jours — un momentum négatif signale une tendance baissière

### Score de régime

Chaque indicateur est converti en **z-score rolling** (fenêtre 315 jours, minimum 189 observations) pour le rendre comparable dans le temps. Le score de régime est ensuite :

$$RegimeScore(t) = z_{vol}(t-1) + z_{corr}(t-1) - z_{mom}(t-1)$$

**Interprétation** : un score élevé → vol élevée, corrélation élevée, momentum négatif → stress. Un score faible → le contraire → risk-on.

Le **décalage de 1 jour** sur toutes les composantes (shift) assure la causalité : le score à la date $t$ ne dépend que de données jusqu'à $t-1$.

### Seuils et classification

Les seuils de transition entre régimes sont **rolling et calibrés sur l'In-Sample** :

| Paramètre | Valeur |
|-----------|--------|
| Fenêtre rolling des seuils | 630 jours |
| Quantile bas (seuil Risk-on) | 30ᵉ percentile |
| Quantile haut (seuil Stress) | 70ᵉ percentile |

La classification est :
- $RegimeScore \leq q_{30\%}^{rolling}$ → **Risk-on**
- $RegimeScore \geq q_{70\%}^{rolling}$ → **Stress**
- entre les deux → **Neutre**

Les seuils rolling sont initialisés avec les quantiles IS (2019-2020), puis recalculés de façon glissante. Le shift(1) est appliqué aux seuils eux-mêmes pour garantir l'absence de look-ahead.

### Lissage causal

Pour éviter le *signal whipsaw* (oscillations rapides entre régimes), un **lissage de 7 jours minimum** est appliqué : un changement de régime ne prend effet que s'il persiste au moins 7 jours consécutifs. Ce lissage est causal (il ne regarde pas le futur).

### Optimisation sous grille

Les paramètres de régime (fenêtres d'indicateurs, de z-score, de seuils, quantiles) sont optimisés par **grid search** sur la période IS. L'objectif est de trouver la combinaison qui maximise une métrique de cohérence économique (ex. : le régime Stress doit effectivement correspondre à des périodes de vol élevée et drawdown important).

---

## 7. Poche Satellite — Filtre structurel (Level 0)

Les fonds candidats sont issus de 3 univers (STRAT1, STRAT2, STRAT3) correspondant aux 3 blocs du Satellite. Avant tout calcul quantitatif, un **filtre structurel** élimine les fonds inadaptés.

### Critères Level 0

| Critère | Seuil | Justification |
|---------|-------|---------------|
| **Devise** | Euro uniquement | Le fonds est domicilié EUR, on ne veut pas de risque de change non contrôlé |
| **AUM** (Actifs sous gestion) | > 50 M USD | Filtre de liquidité et de pérennité du gérant |

### Devise : précision importante

Seuls les fonds dont la devise est strictement **"Euro"** sont retenus. Les variantes historiques comme "Euro (BEF)" sont exclues pour éviter d'inclure des classes de parts d'un autre pays ou des fonds ayant changé de devise lors du passage à l'EUR.

### Résultat

Sur ~300+ fonds disponibles dans STRAT1+STRAT2+STRAT3, environ **166 fonds** passent le Level 0 après filtrage devise + AUM.

---

## 8. Poche Satellite — Filtre beta rolling (Level 1)

L'objectif central de la poche Satellite est la **décorrélation** vis-à-vis du Core. Le Level 1 applique un filtre de bêta rolling strict pour ne garder que les fonds structurellement à faible bêta.

### Le bêta rolling

Pour chaque fonds, on calcule un bêta rolling vis-à-vis du benchmark Core (la série pondérée des 3 ETFs selon la stratégie "Efficient Vol 17 %") sur une fenêtre glissante de **126 jours** (~6 mois).

$$\beta_{fund}(t) = \frac{Cov(r_{fund}, r_{core})}{Var(r_{core})} \quad \text{sur } [t-126, t]$$

### Conditions de passage

Un fonds passe le Level 1 si **les 3 conditions suivantes sont simultanément remplies** sur la fenêtre de calibration :

| Condition | Seuil | Signification |
|-----------|-------|---------------|
| Médiane du bêta absolu | ≤ **0,50** | Le bêta typique est contenu |
| 75ᵉ percentile du bêta absolu | ≤ **0,70** | Même dans les queues, le bêta reste raisonnable |
| Ratio de passage | ≥ **60 %** | Au moins 60 % du temps, le bêta absolu est ≤ 0,50 |

### Fenêtre de calibration

La fenêtre de calibration du Level 1 est **glissante** et dépend de la date de revue annuelle. Pour chaque revue annuelle (voir section 11), la fenêtre de calibration est :

$$[review\_date - 2\text{ ans}, \ review\_date - 1\text{ jour}]$$

Par exemple, pour la revue du 01/04/2021, la calibration utilise les données du 01/04/2019 au 31/03/2021. Le **–1 jour** est essentiel pour éviter tout look-ahead.

---

## 9. Poche Satellite — Scoring alpha + frais (Level 2)

Les fonds ayant passé le Level 1 sont classés par un **score composite alpha + frais** au sein de chaque STRAT.

### Calcul de l'alpha

Pour chaque fonds, sur la même fenêtre de calibration glissante :

$$\alpha_{annual} = r_{annual,fund} - \beta_{fund} \times r_{annual,core}$$

C'est l'alpha de Jensen : le rendement résiduel du fonds après neutralisation de sa composante de marché. Un fonds avec un alpha élevé génère du rendement indépendamment du Core.

### Calcul du score Level 2

$$score_{L2} = \alpha_{weight} \times z(\alpha_{annual}) + expense_{weight} \times z(-expense\_pct)$$

| Paramètre | Valeur |
|-----------|--------|
| $\alpha_{weight}$ | **0,6** (60 %) |
| $expense_{weight}$ | **0,4** (40 %) |
| Observations minimales pour le calcul d'alpha | 60 jours |
| Nombre de fonds retenus par STRAT | Top **7** |

Les composantes sont des **z-scores intra-STRAT** : on standardise les valeurs au sein de chaque bloc pour que les scores soient comparables entre fonds d'un même univers.

Le signe négatif sur le ratio de dépenses ($-expense\_pct$) signifie qu'un fonds moins cher a un meilleur score.

### Résultat

Après le Level 2, on conserve les **7 meilleurs fonds par STRAT** (soit 21 fonds au total dans le pool).

---

## 10. Poche Satellite — Scoring personnalisé par bloc

Après le Level 2, un **scoring personnalisé** est appliqué en fonction du bloc (STRAT) auquel appartient le fonds. Ce scoring reflète la philosophie d'investissement propre à chaque bloc.

### Métriques calculées pour le scoring

Pour chaque fonds, sur la fenêtre de calibration, on calcule en plus des métriques Level 2 :

| Métrique | Description |
|----------|-------------|
| $\beta_{stress}$ | Bêta du fonds calculé uniquement sur les jours de stress Core (bottom 20 % des rendements Core) |
| $\beta_{bloc1}$ | = $\beta_{stress}$ si disponible, sinon $\beta$ standard |
| $corr$ | Corrélation simple rendements fonds vs Core |
| $return_{stress}$ | Rendement annualisé du fonds sur les jours de stress uniquement |
| $sharpe$ | Ratio de Sharpe (rendement ann. / vol ann.) |
| $sortino$ | Ratio de Sortino (rendement ann. / vol downside) |
| $maxdd$ | Drawdown maximum |
| $skewness$ | Asymétrie de la distribution des rendements |
| $kurtosis$ | Aplatissement (queues lourdes) |
| $ret_{ann}$ | Rendement annualisé |
| $vol$ | Volatilité annualisée |

### Formules de scoring par bloc

Toutes les composantes sont des **z-scores intra-STRAT**.

#### STRAT1 — Décorrélation / Convexité

$$score_{STRAT1} = -0{,}40 \times z(|\beta_{bloc1}|) - 0{,}20 \times z(|corr|) + 0{,}20 \times z(return_{stress}) + 0{,}15 \times z(sharpe) - 0{,}15 \times z(maxdd)$$

**Philosophie** : ce bloc vise des fonds qui tiennent ou performent en période de stress. On pénalise fortement le bêta stress (40 %) et la corrélation (20 %), on récompense le rendement en stress (20 %) et le Sharpe (15 %), et on pénalise les drawdowns (15 %).

#### STRAT2 — Alpha Hunters

$$score_{STRAT2} = 0{,}60 \times z(\alpha_{annual}) + 0{,}30 \times z(-expense\_pct) + 0{,}10 \times z(sharpe) - 0{,}05 \times z(|corr|)$$

**Philosophie** : ce bloc recherche l'alpha pur. L'alpha domine le score (60 %), les frais sont un critère secondaire important (30 %), le Sharpe apporte un léger bonus (10 %), et une faible corrélation est marginalement recherchée (5 %).

#### STRAT3 — Growth + Quality (Carry / Crédit)

$$score_{STRAT3} = 0{,}55 \times z(\alpha_{annual}) + 0{,}35 \times z(-expense\_pct) + 0{,}10 \times z(ret_{ann}) - 0{,}05 \times z(kurtosis)$$

**Philosophie** : ce bloc vise le carry et le crédit structuré. L'alpha est central (55 %), les frais sont pénalisés (35 %), le rendement brut apporte un léger avantage (10 %), et les fonds à queues trop lourdes (kurtosis élevé) sont légèrement pénalisés (5 %).

---

## 11. Poche Satellite — Sélection annuelle et trimestrielle glissante

La sélection Satellite est un processus en **deux temps** : une revue annuelle du pool, puis des révisions trimestrielles avec logique de switching.

### Revue annuelle (constitution du pool)

| Paramètre | Valeur |
|-----------|--------|
| Date de la 1ère revue OOS | **01/04/2021** |
| Fréquence | Annuelle (chaque 1er avril) |
| Lookback pour les filtres/scores | **2 ans** glissants |

À chaque revue annuelle :
1. On applique le **Level 1** (bêta rolling) sur la fenêtre $[review - 2\text{ ans},\ review - 1\text{ jour}]$
2. On applique le **Level 2** (alpha + expense) sur la même fenêtre
3. On applique le **scoring personnalisé par bloc** sur la même fenêtre
4. On conserve les **top 7 par STRAT** → c'est le **pool actif** valide jusqu'à la prochaine revue annuelle

### Revue trimestrielle (sélection depuis le pool)

| Paramètre | Valeur |
|-----------|--------|
| Fréquence de la sélection | **Trimestrielle**, calée sur le calendrier QS-APR (avril, juillet, octobre, janvier) |
| Rescoring trimestriel | **Oui** — le pool est rescoré sur une fenêtre 2 ans glissante avant chaque sélection |
| Nombre max de fonds par STRAT | **2** |

À chaque date trimestrielle :
1. Le pool annuel est **rescoré** sur $[qdate - 2\text{ ans},\ qdate - 1\text{ jour}]$ avec les formules par bloc
2. Pour chaque STRAT, le **top 1** est sélectionné (meilleur score)
3. Un **2ᵉ fonds** est ajouté si les conditions de diversification sont remplies :

| Condition pour le 2ᵉ fonds | Seuil |
|-----------------------------|-------|
| Corrélation pairwise avec le top 1 | ≤ **0,70** |
| Écart de score vs top 1 | ≤ **1,5** |

### Logique de switching (buffer anti-turnover)

Pour limiter le turnover, un **buffer de score de 0,30** est appliqué aux transitions :

- Si le fonds détenu est le même que le candidat → **pas de changement**
- Si le candidat a un score supérieur au fonds détenu **+ 0,30** → **switch** vers le candidat
- Sinon → **conservation du fonds détenu** (hold)

Cela évite les changements de fonds pour de petites différences de score, réduisant les coûts de transaction effectifs.

### Dédupplication

Si le même ticker est sélectionné pour deux rangs différents dans un même STRAT (résultat d'un hold et d'un nouveau candidat identique), le système le détecte et sélectionne un fonds alternatif pour éviter de doubler la même exposition.

---

## 12. Allocation dynamique intra-Satellite par régime

Une fois les fonds Satellite sélectionnés (2 par STRAT maximum), les poids de chaque **bloc** varient dynamiquement en fonction du **régime de marché** détecté (section 6).

### Poids par régime

| Régime | Bloc 1 (STRAT1) — Décorrélation | Bloc 2 (STRAT2) — Alpha | Bloc 3 (STRAT3) — Carry |
|--------|----------------------------------|--------------------------|--------------------------|
| **Stress** | **50 %** | 30 % | 20 % |
| **Neutre** | 30 % | **40 %** | 30 % |
| **Risk-on** | 15 % | 35 % | **50 %** |

**Logique économique** :
- En **stress**, on surpondère massivement le bloc 1 (50 %) car les fonds de type CTA / convexité sont censés tenir ou profiter des tendances baissières
- En **neutre**, le bloc 2 (alpha décorrélé) est majoritaire (40 %), car c'est le contexte optimal pour les stratégies market neutral
- En **risk-on**, on surpondère le bloc 3 (50 %) car le carry / crédit structuré bénéficie d'un environnement de spreads comprimés et de faible volatilité

### Répartition intra-bloc

Au sein de chaque bloc, le poids est réparti **équitablement** entre les fonds actifs de ce trimestre. Si un bloc contient 2 fonds actifs, chacun reçoit la moitié du poids du bloc. Si un seul fonds est actif, il reçoit 100 % du poids du bloc.

### Gestion des blocs vides

Si un bloc n'a aucun fonds actif (aucun fonds n'a passé les filtres pour ce STRAT), son poids est **redistribué proportionnellement** aux blocs actifs. Les poids sont ensuite renormalisés pour sommer à 100 % de la poche Satellite.

---

## 13. Performance Satellite et contributions par bloc

### Calcul des rendements Satellite

Le rendement quotidien de la poche Satellite est la somme pondérée des rendements de chaque fonds actif :

$$r_{satellite}(t) = \sum_i w_i(t-1) \times r_i(t)$$

Les poids sont pris **au jour t−1** pour éviter le look-ahead : on utilise les poids de l'allocation connue la veille pour pondérer les rendements du jour.

### Contributions par bloc

Pour chaque bloc, la contribution quotidienne est :

$$contrib_{bloc}(t) = \sum_{i \in bloc} w_i(t) \times r_i(t)$$

Cela permet d'identifier quel bloc est le moteur de la performance Satellite — et si la logique d'allocation par régime ajoute effectivement de la valeur.

### Métriques sur le Satellite

On calcule :
- NAV cumulée du Satellite
- Drawdown glissant
- Volatilité rolling 63 jours
- Bêta rolling vs Core
- Rendement annualisé, Sharpe, Calmar

---

## 14. Comparaison Core vs Satellite

### Source des rendements Core

Le rendement Core utilisé pour la comparaison provient de la **NAV simulée du portefeuille Core** (section 5), pas de la moyenne simple des 3 ETFs. C'est donc le rendement réel du portefeuille Core après rebalancement et TER.

### Source des rendements Satellite

Le rendement Satellite est celui calculé à la section 13. Les poids Satellite t−1 sont appliqués aux rendements de prix t, avec un recouvrement sur les dates communes Core/Satellite.

### Métriques comparatives

| Métrique | Description |
|----------|-------------|
| Bêta statique (Satellite vs Core) | Sensibilité globale du Satellite au Core |
| Corrélation statique | Corrélation linéaire sur toute la période |
| Tracking error | Écart-type annualisé de $r_{satellite} - r_{core}$ |
| Alpha annualisé | Rendement excédentaire du Satellite après neutralisation du bêta |
| Bêta rolling 63 jours | Évolution du bêta dans le temps (vérifie que la décorrélation tient en OOS) |
| Corrélation rolling 63 et 126 jours | Évolution de la corrélation |
| Alpha rolling | Détection de périodes où le Satellite génère ou détruit de l'alpha |

### Dashboard 8 panneaux

L'analyse produit 8 graphiques :
1. NAV cumulée Core vs Satellite
2. Drawdown comparé
3. Bêta rolling 63j et 126j
4. Corrélation rolling 63j et 126j
5. Alpha rolling 63j et 126j
6. Rendements annuels comparés
7. Cross-métriques (Sharpe, Max DD, Calmar)
8. Scatter plot rendements quotidiens

---

## 15. Frais Satellite

### Données de frais

Les ratios de dépenses (TER) de chaque fonds sont extraits des fiches Bloomberg (champ `Ratio des dépenses`). Ils sont exprimés en pourcentage annuel.

### TER pondéré du Satellite

Chaque jour, le TER effectif du Satellite est la somme pondérée des TER individuels :

$$TER_{sat}(t) = \sum_i w_i(t) \times TER_i$$

Ce TER pondéré est exprimé en **bps/an** et suivi dans le temps. Il permet de vérifier que le fonds respecte un budget de frais raisonnable (objectif ≤ 80 bps).

### Imputation des TER manquants

Si le TER d'un fonds n'est pas disponible, il est imputé par la **médiane des TER** des autres fonds de la même STRAT.

---

## 16. Allocation dynamique Core / Satellite (vol-targeting)

### Principe

L'allocation entre Core et Satellite n'est pas fixe — elle est **dynamiquement ajustée chaque mois** pour cibler une **volatilité globale ex ante de 10 % annualisée**.

### Algorithme

Chaque fin de mois :

1. **Calcul des risques rolling** (fenêtre 63 jours) :
   - $\sigma_{core}$ : volatilité annualisée du Core
   - $\sigma_{sat}$ : volatilité annualisée du Satellite
   - $\rho_{core,sat}$ : corrélation Core/Satellite

2. **Volatilité ex-ante du portefeuille** pour un poids Satellite $w_{sat}$ :

$$\sigma_{pf} = \sqrt{(1-w_{sat})^2 \sigma_{core}^2 + w_{sat}^2 \sigma_{sat}^2 + 2(1-w_{sat})w_{sat}\rho\sigma_{core}\sigma_{sat}}$$

3. **Vérification de la bande morte (deadband)** :
   - Si $\sigma_{pf}$ avec les poids actuels est dans $[9{,}5\% \ ;\ 10{,}5\%]$ → **pas de rebalancement** (on conserve les poids)
   - Sinon → on cherche le meilleur $w_{sat}$ dans la grille

4. **Optimisation sur grille** :
   - On teste $w_{sat} \in \{25\%, 26\%, 27\%, 28\%, 29\%, 30\%\}$ (pas de 1 %)
   - On retient le $w_{sat}$ dont la vol ex-ante est **la plus proche** de la cible 10 %

5. **Application** : le nouveau $w_{sat}$ s'applique au jour suivant ($w_{core} = 1 - w_{sat}$)

### Paramètres

| Paramètre | Valeur |
|-----------|--------|
| Lookback vol/corr | **63 jours** (~3 mois) |
| Fréquence de rebalancement | **Mensuelle** (fin de mois) |
| Volatilité cible | **10 %** annualisé |
| Deadband | **± 0,5 %** (soit [9,5 % ; 10,5 %]) |
| $w_{sat}$ minimum | **25 %** |
| $w_{sat}$ maximum | **30 %** |
| Pas de la grille | **1 %** |
| $w_{sat}$ initial | **25 %** |

### Logique du deadband

Le deadband est un mécanisme anti-turnover : si la volatilité est déjà proche de la cible, on ne modifie pas les poids. Cela évite les micro-ajustements coûteux et réduit le bruit.

---

## 17. Portefeuille final — Dashboard et métriques

### Rendement du portefeuille global

$$r_{global}(t) = w_{core}(t-1) \times r_{core}(t) + w_{sat}(t-1) \times r_{sat}(t)$$

Les poids sont **laggés d'un jour** (t−1) — on investit avec les poids de la veille, on reçoit les rendements du jour.

### Métriques calculées

Pour chaque composante (portefeuille global, Core seul, Satellite seul) :

| Métrique | Formule |
|----------|---------|
| Rendement cumulé total | $\prod(1 + r_t) - 1$ |
| Rendement annualisé | $(1 + r_{cum})^{252/N} - 1$ |
| Volatilité annualisée | $\sigma_{daily} \times \sqrt{252}$ |
| Sharpe (rf = 0) | $\frac{r_{ann}}{\sigma_{ann}}$ |
| Max Drawdown | $\min\left(\frac{NAV_t}{NAV_{peak}} - 1\right)$ |
| Calmar | $\frac{r_{ann}}{|MaxDD|}$ |

### Métriques croisées sur le portefeuille global

| Métrique | Description |
|----------|-------------|
| Bêta global vs Core | Sensibilité du portefeuille au Core seul |
| Corrélation globale vs Core | Lien linéaire portefeuille / Core |
| Tracking error | Écart-type annualisé de $r_{global} - r_{core}$ |

### Estimation des frais totaux

Les frais estimés du portefeuille global intègrent :
- **Frais Core** : TER pondéré des 3 ETFs (en bps)
- **Frais Satellite** : TER pondéré quotidien des fonds actifs (en bps)
- **Frais globaux** : $w_{core} \times TER_{core} + w_{sat} \times TER_{sat}$

### Dashboard 6 panneaux

1. NAV cumulée : Core, Satellite, Portefeuille global
2. Drawdown du portefeuille global
3. Volatilité rolling 63j du portefeuille
4. Évolution des poids Core / Satellite dans le temps
5. Rendements annuels comparés
6. Frais estimés (Core, Satellite, Global)

---

## 18. Garde-fous anti look-ahead bias

La rigueur **anti look-ahead** est un pilier de ce projet. Voici les mécanismes en place :

### Poids exécutés t−1

Partout dans le pipeline, les **poids à la date t sont ceux déterminés avant t** :
- Rendement Satellite : $r(t) = \sum w_i(t-1) \times r_i(t)$
- Rendement global : $r(t) = w_{core}(t-1) \times r_{core}(t) + w_{sat}(t-1) \times r_{sat}(t)$
- Allocation vol-targeting : les poids décidés en fin de mois s'appliquent au mois suivant

### Fenêtres de calibration : calib_end = review_date − 1

À chaque revue annuelle ou trimestrielle Satellite, la fenêtre de calibration se termine **1 jour avant** la date de décision :

$$calib\_end = review\_date - 1\text{ jour}$$

Ainsi, le rendement du jour de la décision n'est jamais inclus dans l'estimation.

### Score de régime décalé

Le RegimeScore à la date t utilise :
- Des indicateurs calculés jusqu'à t (inclus)
- Mais le score lui-même est **shifté de 1 jour** : $RegimeScore(t) = RegimeScore_{raw}(t-1)$
- Les seuils rolling sont également shiftés

→ La classification de régime au jour t ne dépend que de données ≤ t−1.

### Poids Core calibrés sur l'IS uniquement

Les poids de la stratégie "Efficient Vol 17 %" sont déterminés **une seule fois** sur la fenêtre IS (2019-2020), puis appliqués **fixement** sur toute la période OOS. Aucune recalibration de ces poids n'a lieu en OOS.

### Limites connues

- **Survivorship bias** : le filtre Level 0 utilise des données instantanées (AUM, devise) qui reflètent l'état actuel des fonds, pas nécessairement l'état historique. Un fonds qui n'existait plus en 2021 pourrait être absent de la base sans qu'on le sache.
- **Snapshot AUM** : l'AUM utilisé est celui de la dernière extraction Bloomberg, pas l'AUM historique à chaque date de décision.

Ces limites sont inhérentes à l'utilisation de données Bloomberg statiques et ne constituent pas un look-ahead bias au sens strict (aucune donnée *future* n'est utilisée), mais plutôt un léger biais de survie.
