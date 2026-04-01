# Core-Satellite Quant — Fonds Quantitatif EUR

> Moteur de construction et de backtest d'un fonds Core-Satellite institutionnel,
> entièrement codé en Python, avec séparation stricte In-Sample / Out-of-Sample.

## Table des matières
1. [Vision du projet](#1-vision-du-projet)
2. [Philosophie d'investissement Core-Satellite](#2-philosophie-dinvestissement-core-satellite)
3. [Architecture de la poche Core](#3-architecture-de-la-poche-core)
4. [Sélection qualitative et quantitative des fonds Satellite](#4-sélection-qualitative-et-quantitative-des-fonds-satellite)
5. [Moteur de portefeuille et vol-targeting](#5-moteur-de-portefeuille-et-vol-targeting)
6. [Fenêtres temporelles — Choix IS/OOS et justification](#6-fenêtres-temporelles--choix-isoos-et-justification)
7. [Métriques et attribution de performance](#7-métriques-et-attribution-de-performance)
8. [Architecture technique du code](#8-architecture-technique-du-code)
9. [Installation et exécution](#9-installation-et-exécution)
10. [Structure des fichiers](#10-structure-des-fichiers)
11. [Notes méthodologiques importantes (anti-biais)](#11-notes-méthodologiques-importantes-anti-biais)
12. [Limites connues et pistes d'amélioration](#12-limites-connues-et-pistes-damélioration)

---

## 1. Vision du projet

Ce projet implémente un **fonds Core-Satellite institutionnel en EUR**, dont le but est de combiner deux sources de rendement complémentaires :

- **La poche Core (70–75 %)** : exposition bêta bon marché via des ETFs cotés en EUR couvrant les grandes classes d'actifs (actions européennes, taux, crédit). L'objectif est de capter les primes de risque de long terme à faible coût de gestion.
- **La poche Satellite (25–30 %)** : génération d'alpha décorrélé du marché via des fonds alternatifs sélectionnés rigoureusement (CTA, market neutral, event driven, crédit structuré). L'objectif est un bêta satellite ≈ 0 et un alpha positif indépendant du cycle.

**Cibles de gestion :**

| Paramètre | Cible |
|-----------|-------|
| Volatilité annualisée totale | 8–12 % |
| Frais totaux estimés | ≤ 80 bps/an |
| Bêta poche Satellite vs Core | ≈ 0 (objectif décorrélation-first) |
| Univers investissable Core | ETFs EUR cotés uniquement |
| Univers investissable Satellite | Fonds alternatifs domiciliés EUR |

Le code est **entièrement en Python**, structuré en pipelines modulaires avec une **séparation stricte In-Sample (IS) / Out-of-Sample (OOS)** : aucune décision de construction ou de calibration ne consulte de données postérieures à la date de décision (janvier 2021).

---

## 2. Philosophie d'investissement Core-Satellite

### La poche Core : stabilité et coût minimal

La poche Core représente **70–75 % du portefeuille**. Elle est construite autour de 3 ETFs EUR thématiques (Equity, Rates, Credit), sélectionnés parmi un univers filtré sur leur qualité structurelle (ancienneté, liquidité, TER). Les poids Core sont recalibrés chaque trimestre par Risk Parity avec tilt momentum Equity, sans jamais utiliser d'estimateur de rendement attendu (robustesse OOS).

### La poche Satellite : alpha décorrélé en 3 blocs

La poche Satellite représente **25–30 % du portefeuille** et est organisée en 3 blocs fonctionnels :

| Poche | % du portefeuille | Objectif | Instruments |
|-------|-------------------|----------|-------------|
| Core | 70–75 % | Bêta de marché peu coûteux | 3 ETFs EUR (Equity, Rates, Credit) |
| Satellite Bloc 1 | ~7–8 % | Décorrélation / Convexité | CTA Trend, Short Volatilité |
| Satellite Bloc 2 | ~10–12 % | Alpha Décorrélé | Event Driven, Merger Arb, L/S, Market Neutral |
| Satellite Bloc 3 | ~7–8 % | Carry / Crédit Structuré | ABS, CLO, FI Relative Value, Senior Loans |

**Le principe directeur est la décorrélation-first** : le critère n°1 pour chaque fonds satellite est un bêta rolling proche de zéro vis-à-vis du Core. Le rendement et le Sharpe ne sont des critères qu'en second lieu. Cette approche garantit que la poche Satellite ajoute réellement de la diversification au portefeuille, sans dupliquer le risque de la poche Core.

---

## 3. Architecture de la poche Core

### Univers des ETFs

Le fichier `univers_core_etf_eur_daily_wide_VF.xlsx` contient les prix quotidiens et les métadonnées de l'univers des ETFs Core, organisés en 3 onglets thématiques (Equity, Rates, Credit). Chaque onglet contient :
- Les prix en format *wide* (dates en lignes, tickers Bloomberg en colonnes)
- Les métadonnées (TER, date de lancement, exposition géographique)

### Filtres structurels

Avant la sélection, chaque ETF est soumis à des filtres structurels :

| Filtre | Valeur | Justification |
|--------|--------|---------------|
| Date de lancement | ≤ 2019-01-01 | Track record ≥ 2 ans avant OOS |
| Fréquence de cotation | ≤ 2 jours ouvrés de gap moyen | Liquidité acceptable |
| Budget TER | ≤ 80 bps/an (global) | Maîtrise des frais |
| Exposition Equity | Mots-clés "Europe" | Fonds EUR institutionnel |

### Sélection `pick_best_theme()`

Pour chaque thème (Equity, Rates, Credit), la fonction `pick_best_theme()` sélectionne le **meilleur ETF selon le Sharpe annualisé sur la fenêtre IS 2019-2020**. Cette fenêtre est dite *"through-the-cycle IS"* car elle inclut à la fois un régime haussier (2019) et le crash COVID (mars 2020).

### Optimisation rolling `risk_parity_tilt`

Les poids Core sont recalibrés chaque trimestre (63 jours ouvrés) par la méthode `risk_parity_tilt` :

1. **Risk Parity de base** : `w_i ∝ 1/σ_i` (inverse de la volatilité de chaque ETF sur le lookback de 252 jours)
2. **Tilt Equity dynamique** : si le momentum Equity sur 252 jours est positif, le poids Equity est porté au plafond (`equity_weight_ceiling = 60 %`). Sinon, il est limité au plancher (`equity_weight_floor = 30 %`).
3. **Covariance Ledoit-Wolf** : la matrice de covariance est estimée par shrinkage de Ledoit-Wolf (via scikit-learn) pour réduire l'erreur d'estimation.

**Paramètres clés :**

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `lookback` | 252 jours | Fenêtre d'estimation de la covariance |
| `rebal_freq` | 63 jours | Fréquence de rebalancement (trimestrielle) |
| `equity_weight_floor` | 30 % | Poids minimum Equity |
| `equity_weight_ceiling` | 60 % | Poids maximum Equity (régime haussier) |
| `momentum_window` | 252 jours | Fenêtre du momentum Equity |
| `use_ledoit_wolf` | True | Shrinkage Ledoit-Wolf sur la covariance |

**Pourquoi Risk Parity plutôt que Max Sharpe ?**

Le Max Sharpe dépend de l'estimateur de rendement attendu μ, qui est très bruyant sur des fenêtres courtes. En OOS, cela génère des poids instables et une performance dégradée. Le Risk Parity ne dépend que des volatilités et corrélations — des quantités plus stables — et offre une robustesse OOS supérieure documentée par la littérature académique.

---

## 4. Sélection qualitative et quantitative des fonds Satellite

### 4.1 Philosophie de sélection

La sélection des fonds Satellite repose sur une **approche en deux temps** :

1. **Pré-sélection qualitative** : une *shortlist* de 19 fonds est constituée manuellement, sur la base de critères qualitatifs (réputation du gérant, stratégie clairement définie, décorrélation prouvée en période de stress, domiciliation EUR). Cette shortlist réduit l'univers de recherche et évite les biais de data mining sur un univers trop large.

2. **Filtrage quantitatif** : les 19 fonds de la shortlist sont ensuite soumis à un pipeline de filtres quantitatifs calculés **exclusivement sur la fenêtre IS 2019-2020**.

**Critère de track record :** chaque fonds doit avoir un premier prix ≤ 01/01/2019. Cela garantit que tous les fonds de la shortlist ont traversé le crash COVID (mars 2020) dans leur track record IS, ce qui est un test de résistance implicite.

### 4.2 Les 3 blocs Satellite

#### Bloc 1 — Décorrélation / Convexité (~7–8 % du portefeuille)

Objectif : fonds qui performent ou tiennent en période de stress de marché. Le bêta est épisodique mais la médiane est proche de zéro. Ces fonds apportent de la **convexité** au portefeuille : ils tendent à bien performer précisément quand le Core souffre.

| Ticker Bloomberg | Nom / Gérant | Stratégie | Justification qualitative |
|---|---|---|---|
| DWMAEIA ID Equity | Dunn Capital — WMA EUR Acc | CTA Trend Following | Un des plus anciens CTA en activité (depuis 1984). Stratégie purement systématique sur futures diversifiés. Décorrélation prouvée en période de stress (2008, COVID). Convexité naturelle : le trend-following capte les grandes tendances de crise. |
| FINVPRI GR Equity | FinCam — Short Volatility EUR | Short Volatilité (prime de variance) | Capture la prime de variance (VIX vs vol réalisée) de façon systématique. Décorrélé du marché directionnel en régime normal. Risque de queue (short gamma) géré par un budget strict. Complément naturel du CTA en régime de faible vol. |

#### Bloc 2 — Alpha Décorrélé (~10–12 % du portefeuille)

Objectif : alpha pur, bêta market proche de zéro, Sharpe régulier indépendant du cycle.

| Ticker Bloomberg | Nom / Gérant | Stratégie | Justification qualitative |
|---|---|---|---|
| HFHPERC LX Equity | Syquant Capital — Helium Fund | Event Driven | Spécialiste de l'arbitrage d'événements corporate (M&A, spin-offs, restructurations) en Europe. Track record > 10 ans, Sharpe > 1 historiquement. Alpha purement idiosyncratique, non corrélé aux marchés directionnels. |
| EXCRISA LX Equity | Exane — Merger Arbitrage | Merger Arbitrage | Stratégie pure merger arb sur opérations annoncées. Spread capture avec risque de deal break géré. Corrélation quasi nulle avec les marchés equity et taux. Liquidité correcte (hebdomadaire). |
| PDAIEUR LX Equity | Pictet — Diversified Alpha | Multi-Stratégie Market Neutral | Multi-strat market neutral de Pictet AM. Diversification interne entre sub-strategies (stat arb, pairs trading, vol trading). Bêta structurellement proche de zéro par construction. |
| HSMSTIC LX Equity | HSBC — Multi-Strategy | Multi-Stratégie | Multi-strat diversifiée sur un univers large. Allocation dynamique entre stratégies selon le régime de marché. AUM important garantissant la stabilité opérationnelle. |
| REYLSEB LX Equity | RAM Active Investments — Lux Systematic Equity | Market Neutral | Market neutral systématique de RAM. Modèle quantitatif factoriel (value, momentum, quality) avec hedging bêta dynamique. Sharpe régulier, drawdowns contenus. |
| CARPPFE LX Equity | Carmignac Portfolio — Long-Short EU | Long/Short Europe | L/S actions européennes de Carmignac. Exposition nette faible (±20 %). Alpha stock-picking + hedging macro. Bien établi dans l'univers des fonds alternatifs EUR. |
| ELEARER LX Equity | Eleva Capital — Eleva Abs. Ret. Europe | Long/Short Europe | Boutique L/S EU de haute conviction. Équipe issue de Lehman/Rothschild. Processus fondamental rigoureux. Corrélation faible au marché grâce à la flexibilité de l'exposition nette. |
| LDAPB2E ID Equity | Dalton Investments — Pan Asia Long Short | Long/Short Asie | L/S Asie de Dalton, apportant une diversification géographique au portefeuille. Marché asiatique peu corrélé aux cycles européens. Expertise locale reconnue. |
| LUMNUDE LX Equity | Man GLG — Absolute Return | Market Neutral Man GLG | Market neutral systématique de Man GLG. Infrastructure quantitative de premier plan. Stratégie de statistique arb pur. |
| EXCERFD LX Equity | Exane — Ceres Fund | Long/Short Value Europe | L/S value Europe avec approche fondamentale. Complément à ELEARER (approche différente). Diversification du stock-picking intra-bloc 2. |

#### Bloc 3 — Carry / Crédit Structuré (~7–8 % du portefeuille)

Objectif : capture de primes de crédit et de liquidité, faible corrélation actions et taux directionnels.

| Ticker Bloomberg | Nom / Gérant | Stratégie | Justification qualitative |
|---|---|---|---|
| AEABSIA ID Equity | Aegon — ABS Senior | ABS Senior | Fonds ABS senior tranches investment grade. Carry régulier avec faible duration. Risque de crédit structuré diversifié (RMBS, auto, consumer). Décorrélé des équités. |
| BNPAOPP LX Equity | BNP Paribas — Asset-Backed Securities | ABS Diversifié | ABS diversifié sur plusieurs pays EUR. Complémentaire à Aegon (gérant et exposition géographique différents). Liquidité mensuelle, valorisation transparente. |
| CLOHQIA GR Equity | Lupus Alpha — CLO Senior | CLO Senior | Spécialiste CLO senior AAA-AA de Lupus Alpha. Carry CLO significatif vs taux sans risque. Risque de défaut corporate diversifié sur 200+ prêts. Cotation moins fréquente (tolérance stale ratio élargie documentée dans le code). |
| INICLBI GR Equity | Infinigon — CLO IG | CLO Investment Grade | CLO IG de Infinigon. Exposition complémentaire à Lupus Alpha. Rating IG moyen, carry supérieur aux corporate bonds classiques. |
| DISFCPE LX Equity | Danske Invest — FI Relative Value | Fixed Income Relative Value | FI Relative Value de Danske. Capture des spreads intra-marchés taux (swap spreads, basis trades). Décorrélé du marché directionnel actions. Sharpe régulier en toutes conditions. |
| LIOFVS1 ID Equity | Lion Capital — Credit RV | Crédit Relative Value | Credit RV sur obligations et CDS. Valorisation mensuelle (stale ratio très élevé → tolérance portée à 95 % dans le code, documenté). Alpha de structure de capital. |
| MUESEIH ID Equity | Muzinich — Senior Loans EUR | Senior Loans | Senior loans corporate EUR. Taux variable → protection naturelle contre la hausse des taux. Séniorité dans la structure de capital. Carry régulier avec faible vol. |

### 4.3 Raisons de la shortlist (19 fonds)

La pré-sélection qualitative d'une shortlist de 19 fonds avant tout filtre quantitatif est une décision méthodologique délibérée :

- **Éviter le data mining** : tester tous les fonds alternatifs EUR disponibles et retenir ceux qui passent les filtres quantitatifs IS revient à faire du data mining. La shortlist impose une sélection qualitative antérieure à tout calcul.
- **Critères qualitatifs appliqués** : track record ≥ 5 ans, gérant reconnu dans la stratégie, AUM ≥ 100 M USD, stratégie clairement définie et différenciée, décorrélation prouvée lors des crises passées, fonds domicilié EUR.
- **Robustesse** : la shortlist n'est pas sensible à de petites variations de paramètres quantitatifs. Seul un changement de vue qualitative (nouveau gérant, stratégie dérivée, problème opérationnel) justifie de la modifier.

### 4.4 Pipeline de filtrage quantitatif (fenêtre IS 2019-2020 uniquement)

Cinq niveaux de filtrage sont appliqués séquentiellement, dans cet ordre :

| Niveau | Critères | Description |
|--------|----------|-------------|
| **Niveau Bêta** | Filtre rolling bêta 3M vs Core équipondéré | `median(|β|) ≤ 35 %`, `q75(|β|) ≤ 55 %`, `|β| ≤ 35 %` sur ≥ 80 % des jours IS |
| **Niveau 0** | Structurel universel | AUM ≥ 100 M$, premier prix ≤ 01/01/2019, devise EUR |
| **Niveau 1** | Frais, Volatilité, Stale pricing | Frais ≤ seuil par stratégie, vol ∈ [vol_min, vol_max] par stratégie, stale ratio ≤ 10 % |
| **Niveau 2** | Qualité quantitative | Sharpe IS ≥ -0,5 ; alpha annualisé ≥ -10 % ; max drawdown IS ≥ -50 % à -80 % selon bloc ; corrélation IS ≤ 45 % |
| **Niveau 3** | Comportemental | Skewness IS ≥ -2,0 à -2,5 ; kurtosis IS ≤ 10 ; concentration ≤ 95 % |
| **Filtre pairwise** | Cohérence inter-fonds | Corrélation pairwise IS ≤ 70 % entre fonds du même bloc retenus |

### 4.5 Score composite décorrélation-first

Après filtrage, chaque fonds passant tous les niveaux reçoit un **score composite z-scoré intra-bloc** :

| Composante | Poids | Description |
|------------|-------|-------------|
| `-\|β_core\|` | 30 % | Priorité absolue à la décorrélation |
| `-corr_core` | 10 % | Corrélation linéaire IS vs Core |
| `Sortino` | 25 % | Rendement ajusté du risque de baisse |
| `ret_rel_covid` | 15 % | Rendement relatif mars–mai 2020 (résilience COVID) |
| `-dd_covid` | 10 % | Drawdown maximal pendant la crise COVID |
| `Skewness` | 5 % | Asymétrie des rendements |
| `-Kurtosis` | 5 % | Queues de distribution (risque de queues épaisses) |

Le signe négatif sur `-|β_core|` traduit le principe *décorrélation-first* : un bêta faible est récompensé. Le poids élevé de la résilience COVID (25 % combiné entre `ret_rel_covid` et `-dd_covid`) garantit que les fonds retenus ont effectivement tenu lors du seul crash majeur de la période IS.

### 4.6 Sélection finale : 2 + 3 + 2 fonds

La sélection finale retient **7 fonds satellite** : 2 du Bloc 1, 3 du Bloc 2 et 2 du Bloc 3. Pour chaque bloc, les 2 fonds suivants dans le classement sont exportés en tant que **réserves** dans `satellite_reserves.csv`. Ces réserves permettent de substituer un fonds si son track record est insuffisant en OOS ou si le gérant subit un événement opérationnel.

---

## 5. Moteur de portefeuille et vol-targeting

### Allocation Core / Satellite

Le module `portfolio_engine.py` implémente la fonction `calibrer_allocation()` qui :

1. Calcule les rendements du portefeuille brut Core + Satellite selon les poids cibles `(w_core, w_sat)`.
2. Applique un **vol-targeting** via `appliquer_vol_targeting()` pour ramener la volatilité annualisée dans la cible `[vol_target_min, vol_target_max] = [8 %, 12 %]`.
3. Respecte les contraintes : `w_core ∈ [70 %, 75 %]`, `w_sat ≤ 30 %`, somme = 100 %, pas de levier.

### Vol-targeting anti-look-ahead

Le signal de vol-targeting est calculé à la date *t* et appliqué aux rendements de la date *t+1* via `scale.shift(1)`. Ce décalage d'un jour est **essentiel** pour éviter un look-ahead bias : sans ce décalage, la volatilité du jour *t* serait utilisée pour pondérer le rendement du même jour *t*, ce qui n'est pas réalisable en production.

### Allocation satellite en mode `beta_inverse`

Par défaut, la poche Satellite est allouée en mode `beta_inverse` : le poids de chaque fonds satellite est **inversement proportionnel à son bêta absolu IS** :

```
w_i ∝ 1 / (|β_i| + ε)
```

Les fonds les plus décorrélés (bêta le plus proche de zéro) reçoivent donc les poids les plus élevés, renforçant le principe *décorrélation-first* dans l'allocation elle-même. L'epsilon `ε` évite les divisions par zéro pour les fonds strictement market-neutral.

**Autres modes disponibles :**
- `score_prop` : pondération proportionnelle au score composite IS (softmax des scores)
- `min_corr` : minimisation de la variance intra-satellite (optimisation SLSQP)

### Rebalancement

Le portefeuille est rebalancé **chaque trimestre** (63 jours ouvrés). Les poids Core sont recalibrés par Risk Parity rolling ; les poids Satellite restent fixes (poids IS appliqués tels quels en OOS).

---

## 6. Fenêtres temporelles — Choix IS/OOS et justification

| Fenêtre | Dates | Usage |
|---------|-------|-------|
| Warm-up Core | 2018-01-01 → 2018-12-31 | Initialisation du lookback 252j sans utiliser les données IS |
| In-Sample (IS) | 2019-01-01 → 2020-12-31 | Sélection ETFs Core, sélection fonds Satellite, calibration poids |
| Out-of-Sample (OOS) | 2021-01-01 → 2025-12-31 | Validation pure, aucun paramètre recalibré |

### Justification de la fenêtre IS 2019-2020

La fenêtre IS 2019-2020 a été choisie pour sa propriété *"through-the-cycle"* :

- **2019** : régime de bull market, hausse des actifs risqués, faible volatilité. Les fonds à fort bêta performent bien.
- **T1 2020** : crash COVID brutal (-35 % sur les indices en 5 semaines). Les fonds à bêta élevé sont sévèrement touchés. Les fonds sélectionnés dans ce cadre ont prouvé leur résistance.

Cette combinaison garantit que les fonds retenus ont été testés dans deux régimes opposés, réduisant le risque de sélection sur un unique régime favorable.

### Justification de `max_start_date = "2019-01-01"`

Chaque fonds satellite doit avoir son premier prix ≤ 01/01/2019. Cela garantit :
- Au moins 1 an de track record avant la crise COVID (mars 2020)
- Que les filtres quantitatifs IS sont calculés sur des données représentatives, incluant le crash COVID

### Note sur le chevauchement IS/OOS

La sélection des ETFs Core et la calibration rolling partagent les mêmes données source (depuis 2019). Ce n'est **pas un look-ahead bias** — aucune donnée future n'est utilisée dans les décisions de poids. Ce chevauchement reflète la pratique réelle d'un gérant calibrant son processus sur l'historique disponible au moment du lancement (janvier 2021).

Le `warm_up_start = "2018-01-01"` dans `core_pipeline.py` atténue ce chevauchement en permettant au lookback de s'initialiser avant la période IS : les 252 premières observations du lookback proviennent de 2018, et non de 2019.

---

## 7. Métriques et attribution de performance

### Métriques (`metrics.py`)

| Métrique | Définition | Note |
|----------|-----------|------|
| Volatilité annualisée | `σ_daily × √252` | Sur les rendements log-journaliers |
| Sharpe | `(1+r_moy)^252 - 1) / σ_ann` | `rf = 0` (voir note ci-dessous) |
| Sortino | `ret_ann / σ_downside_ann` | Seuls les rendements négatifs dans le dénominateur |
| Max Drawdown | `min((cum / peak) - 1)` | Drawdown maximum sur la période |
| Calmar | `ret_ann / |max_drawdown|` | Rendement sur risque de drawdown |

**Note sur le taux sans risque** : toutes les métriques Sharpe/alpha sont calculées en **excess-return** vs un proxy Bund 10 ans obtenu via `risk_free.py` (source FRED) avec fallback constant **2 % annualisé** si l'API est indisponible. Le proxy est aligné sur l'index des séries utilisées (IS et OOS) pour éviter toute incohérence de datation.

### Attribution (`attribution.py`)

Le module `attribution.py` implémente une **régression OLS rolling sur 36 mois** :

```
r_portfolio(t) = alpha_daily + beta × r_core(t) + eps(t)
```

- **Alpha annualisé** : `(1 + alpha_daily)^252 - 1` (composition géométrique, cohérente avec les rendements log)
- **Bêta rolling** : sensibilité du portefeuille au Core sur la fenêtre glissante de 36 mois
- **Attribution** : la contribution de la poche Satellite à l'alpha total est identifiable par la différence entre l'alpha total et l'alpha de la poche Core seule

---

## 8. Architecture technique du code

Les modules sont organisés en **pipeline séquentiel** : Core → Satellite → Construction → Rapports.

```
cross_asset/
├── data/
│   ├── univers_core_etf_eur_daily_wide_VF.xlsx  # Prix + metadata ETFs Core (3 thèmes)
│   ├── STRAT1_info.xlsx / STRAT1_price.xlsx     # Metadata + prix fonds Satellite Bloc 1
│   ├── STRAT2_info.xlsx / STRAT2_price.xlsx     # idem Bloc 2
│   └── STRAT3_info.xlsx / STRAT3_price.xlsx     # idem Bloc 3
├── src/
│   ├── risk_free.py                 # Proxy Bund 10Y + Sharpe excess
│   ├── core_pipeline_corrected.py   # Pipeline ETF Core : lecture, filtrage, sélection, backtest rolling
│   ├── satellite_pipeline_corrected.py # Pipeline Satellite : shortlist, filtres IS, scoring, sélection
│   ├── fond_construction_corrected.py # Construction fonds final : allocation Core+Sat, vol-targeting
│   ├── portfolio_engine.py          # Moteur d'allocation : Risk Parity, vol-targeting
│   ├── efficient_frontier_core.py   # Frontière efficiente (outil d'analyse/diagnostic)
│   ├── plots_report.py              # Rapport graphique complet (28 figures)
│   ├── fees.py                      # Estimation des frais totaux (bps/an)
│   └── attribution.py               # Attribution alpha/beta
├── outputs/                         # Fichiers générés automatiquement par les pipelines
│   ├── core_selected_etfs.csv
│   ├── core_returns_daily_is.csv / core_returns_daily_oos.csv
│   ├── core3_etf_daily_log_returns.csv / core3_etf_daily_simple_returns.csv
│   ├── satellite_selected.csv / satellite_selected_v3.csv / satellite_reserves.csv
│   ├── fond_returns_daily.csv / fond_weights.csv / fond_metrics.csv / fond_annual_perf.csv
│   └── figures/
├── tests/                           # Scripts d’analyses complémentaires (benchmarks/sensitivité)
├── main.ipynb                       # Notebook d'orchestration principal
├── requirements.txt
└── README.md
```

### Flux de données entre modules

```
core_pipeline.py
    → outputs/core_selected_etfs.csv
    → outputs/core_returns_daily_is.csv
    → outputs/core_returns_daily_oos.csv
    → outputs/core3_etf_daily_log_returns.csv

satellite_pipeline.py  (lit core3_etf_daily_log_returns.csv)
    → outputs/satellite_selected.csv
    → outputs/satellite_selected_v3.csv
    → outputs/satellite_reserves.csv

fond_construction.py  (lit core_returns_daily_oos.csv + satellite_selected_v3.csv + prix STRAT)
    → outputs/fond_returns_daily.csv
    → outputs/fond_weights.csv
    → outputs/fond_metrics.csv
    → outputs/fond_annual_perf.csv

plots_report.py  (lit tous les outputs)
    → rapports graphiques (24 figures)
```

---

## 9. Installation et exécution

```bash
# 1. Cloner le repo
git clone https://github.com/HUGO-ROCHA-MONDRAGON/Core_Satellite_Quant.git
cd Core_Satellite_Quant

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Exécuter le pipeline complet (ordre recommandé)
python -m src.core_pipeline_corrected        # Étape 1 : Sélection et backtest Core
python -m src.satellite_pipeline_corrected   # Étape 2 : Sélection fonds Satellite
python -m src.fond_construction_corrected    # Étape 3 : Construction fonds final
python -m src.plots_report                   # Étape 4 : Génération des graphiques

# Ou via le notebook
jupyter notebook main.ipynb
```

**Prérequis :** Python ≥ 3.10, les fichiers de données dans `data/` (non inclus dans le repo pour des raisons de confidentialité Bloomberg).

**Dépendances principales :**

| Librairie | Usage |
|-----------|-------|
| `pandas` | Manipulation des séries temporelles |
| `numpy` | Calcul matriciel et statistiques |
| `scipy` | Optimisation (SLSQP), régression OLS |
| `scikit-learn` | Ledoit-Wolf shrinkage (covariance) |
| `matplotlib` / `seaborn` | Visualisation |
| `openpyxl` | Lecture des fichiers Excel |

---

## 10. Structure des fichiers

```
Core_Satellite_Quant/
├── src/
│   ├── core_pipeline.py
│   ├── satellite_pipeline.py
│   ├── fond_construction.py
│   ├── portfolio_engine.py
│   ├── attribution.py
│   ├── efficient_frontier_core.py
│   ├── fees.py
│   └── plots_report.py
├── data/
│   ├── univers_core_etf_eur_daily_wide_VF.xlsx
│   ├── STRAT1_info.xlsx
│   ├── STRAT1_price.xlsx
│   ├── STRAT2_info.xlsx
│   ├── STRAT2_price.xlsx
│   ├── STRAT3_info.xlsx
│   └── STRAT3_price.xlsx
├── outputs/
│   ├── core_returns_daily_oos.csv
│   ├── core_returns_daily_is.csv
│   ├── core_selected_etfs.csv
│   ├── core3_etf_daily_log_returns.csv
│   ├── satellite_selected.csv
│   ├── satellite_selected_v3.csv
│   ├── satellite_reserves.csv
│   ├── fond_returns_daily.csv
│   ├── fond_weights.csv
│   ├── fond_metrics.csv
│   └── fond_annual_perf.csv
├── tests/
│   └── sensitivity_analysis.py
├── main.ipynb
├── requirements.txt
└── README.md
```

---

## 11. Notes méthodologiques importantes (anti-biais)

Le tableau suivant récapitule les points de contrôle anti-biais implémentés dans le code :

| Point de contrôle | Mécanisme | Statut |
|---|---|---|
| Séparation IS/OOS | `score_start/end` vs `oos_start/end` séparés dans `CoreConfig` et `FondConfig` | ✅ Propre |
| Vol-targeting sans look-ahead | `scale.shift(1)` — signal calculé à *t* appliqué à *t+1* | ✅ Propre |
| Métriques satellite uniquement IS | `calculer_metriques_calib()` restreinte à `[calib_start, calib_end]` | ✅ Propre |
| Backtest rolling Core | Poids calculés sur `window = rets.iloc[start-lookback:start]`, appliqués sur `oos = rets.iloc[start:start+rebal_freq]` | ✅ Propre |
| Bêta rolling satellite | Calculé sur toute l'histoire mais filtré sur IS uniquement pour la sélection | ✅ Propre |
| Warm-up Core | `warm_up_start = "2018-01-01"` permet l'initialisation du lookback avant IS | ✅ Documenté |
| Chevauchement sélection/calibration Core | Données IS partagées entre sélection ETF et calibration rolling — documenté, non look-ahead | ⚠️ Documenté |
| `ffill` limité | `ffill(limit=5)` sur prix satellites pour éviter masquage de trous prolongés | ✅ Propre |
| Détection TER décimal/% | Seuil de détection corrigé à 0,05 (robuste pour ETFs < 10 bps) | ✅ Corrigé |
| Audit track record | Affichage informatif du premier prix et nombre d'observations IS/OOS par fonds | ✅ Documenté |

---

## 12. Limites connues et pistes d'amélioration

1. **Shortlist codée en dur** : les 19 fonds sont sélectionnés qualitativement et codés dans `SATELLITE_SHORTLIST`. La robustesse à un changement de shortlist n'est pas testée automatiquement. *Piste :* ajouter un test de sensibilité sur la shortlist (ex. leave-one-out).

2. **`rf = 0` dans les Sharpe** : acceptable pour la construction du portefeuille, mais à paramétrer pour le reporting client institutionnel (€STR, OAT 10 ans ou autre taux de référence EUR).

3. **Dates COVID hardcodées** : `covid_start = "2020-02-01"`, `covid_end = "2020-05-31"` sont correctes mais hardcodées dans `calculer_metriques_calib()`. *Piste :* les ajouter en paramètres de `SatelliteConfig` pour faciliter la ré-utilisation sur d'autres épisodes de stress.

4. **Univers Core limité à l'Europe** : `equity_exposure_keywords = ("Europe",)` est une contrainte de conception du fonds EUR institutionnel. Un fonds global nécessiterait une gestion du risque de change et un ajustement de cette contrainte.

5. **Absence de coûts de transaction dans le backtest** : les coûts de rebalancement (bid-ask spread ETF, frais d'entrée fonds alternatifs, impact de marché) ne sont pas modélisés en dehors du budget TER. *Piste :* ajouter un modèle de coûts proportionnels au turnover trimestriel.

6. **Fonds à valorisation mensuelle** : certains fonds Satellite (Bloc 3) ont une valorisation mensuelle. Le `ffill(limit=5)` actuel peut sous-estimer leur stale ratio réel. Ces fonds ont des tolérances de stale ratio élargies documentées dans le code (`STALE_OVERRIDE` dans `satellite_pipeline.py`).

