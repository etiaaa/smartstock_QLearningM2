# WeExplainYou — Le projet expliqué de A à Z

Ce document raconte ce qu'on a construit, pourquoi, comment, et avec quels résultats. Il complète le `README.md` (qui sert de cadrage) en détaillant la démarche technique et les chiffres obtenus.

---

## 1. La structure du projet

```
Hackathon Deep Q0Learning/
├── README.md                  cadrage et lien avec la consigne
├── WeExplainYou.md            ce document
├── requirements.txt           dépendances Python
├── app_web.py                 application web de démonstration (Flask)
├── templates/
│   └── index.html             interface du dashboard
├── env/
│   ├── restaurant_env.py      environnement mono-produit
│   ├── restaurant_env_multi.py environnement multi-produits (3 produits, analyse)
│   ├── restaurant_env_demo.py  environnement démo (4 produits, app web)
│   ├── test_env.py            test manuel mono-produit
│   └── test_env_multi.py      test + démo de l'explosion combinatoire
├── agents/
│   ├── q_learning.py          agent Q-learning tabulaire
│   ├── dqn.py                 agent Deep Q-Network
│   ├── train_q_learning.py    entraînement Q-learning + courbes
│   ├── train_dqn.py           entraînement DQN + courbes
│   ├── tune_q_learning.py     grid search Q-learning (9 configs)
│   ├── tune_dqn.py            grid search DQN (5 configs)
│   ├── evaluate.py            évaluation comparative Q-learning vs baselines
│   ├── compare_qlearning_dqn.py comparaison finale Q-learning vs DQN
│   ├── analyze_policies.py    extraction et visualisation des politiques apprises
│   └── robustness_tests.py    tests de robustesse (5 scénarios de perturbation)
├── baselines/
│   └── naive_policies.py      politiques de référence (random, fixed, moving avg)
└── results/
    ├── q_learning_curve.png   courbe Q-learning
    ├── dqn_curve.png          courbe DQN + loss
    ├── comparison.png         comparaison finale
    ├── q_table.npy            Q-table sauvegardée
    └── dqn_weights.pt         poids du réseau DQN
```

---

## 2. L'environnement de simulation

### Mono-produit (`RestaurantEnvSingle`)

Un seul ingrédient frais (par exemple le saumon). Chaque jour, l'agent choisit une quantité à commander, l'environnement génère une demande, sert les clients, fait vieillir le stock et jette ce qui est périmé.

**Paramètres retenus après itérations :**

| Paramètre | Valeur | Pourquoi |
|---|---|---|
| `max_stock` | 20 | capacité raisonnable d'un petit frigo pro |
| `shelf_life` | 2 jours | force le vieillissement rapide → vrai arbitrage |
| `max_order` | 12 | suffisant pour couvrir un rush |
| `margin` | 5.0 | marge de référence par unité vendue |
| `waste_penalty` | 6.0 | **plus cher que la marge** : jeter doit faire mal |
| `stockout_penalty` | 4.0 | un peu moins que le gaspillage |
| `episode_length` | 30 jours | un mois de service |

**Demande générée** : base 4 + effet jour de la semaine + bruit ±2 + 5% chance de rush (+5) + 5% chance de jour creux (-4). Cette volatilité empêche toute politique fixe de fonctionner parfaitement.

**État observable** : `(stock_total, jours_avant_péremption_du_plus_vieux_lot, jour_de_la_semaine)`. État volontairement compact pour que la Q-table reste petite.

### Multi-produits (`RestaurantEnvMulti`)

Trois produits aux profils contrastés :

| Produit | Marge | DLC | Coût gaspi | Coût rupture | Demande base |
|---|---|---|---|---|---|
| Saumon | 12.0 | 2 jours | 9.0 | 6.0 | 3 |
| Légumes | 4.0 | 4 jours | 1.5 | 2.0 | 5 |
| Pain | 2.0 | 1 jour | 0.8 | 3.0 | 6 |

Cette asymétrie est ce qui rend le problème intéressant : pour le saumon il faut être très conservateur (jeter coûte cher), pour le pain il faut éviter les ruptures (DLC ultra-courte, marge faible mais clients sensibles).

**État observable** : 22 dimensions (stock + âge moyen + 3 dernières demandes par produit + jour de la semaine en one-hot).

**Espace d'actions** : 9³ = 729 actions combinatoires (0 à 8 unités à commander pour chacun des 3 produits).

---

## 3. Pourquoi le DQN devient mécaniquement nécessaire

C'est le cœur de la justification demandée par le sujet. Avec une discrétisation à 5 niveaux par dimension d'état :

- États par produit : 5⁵ = 3 125
- États au total (3 produits) : ≈ 3 × 10¹¹
- Cases dans la Q-table : 3 × 10¹¹ × 729 ≈ **2 × 10¹⁴**
- Mémoire requise : **plusieurs centaines de To**

Une Q-table sur ce problème est physiquement impossible à stocker, et encore moins à apprendre. Le DQN n'est pas un effet de mode : il est mécaniquement nécessaire.

---

## 4. Q-learning tabulaire

### L'agent (`q_learning.py`)

- Q-table stockée dans un `defaultdict(lambda: np.zeros(n_actions))` → ne crée une entrée que pour les états visités
- Politique ε-greedy
- Mise à jour de Bellman classique : `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]`
- Décroissance exponentielle d'ε en fin d'épisode

### Tuning des hyperparamètres

On a testé 9 configurations sur les MÊMES 200 épisodes d'évaluation (seeds fixes). Voici le résultat brut :

| Configuration | α | γ | ε_decay | Épisodes | Reward |
|---|---|---|---|---|---|
| Baseline | 0.10 | 0.95 | 0.995 | 5000 | 844.7 |
| α bas (0.05) | 0.05 | 0.95 | 0.995 | 5000 | 822.0 |
| α haut (0.20) | 0.20 | 0.95 | 0.995 | 5000 | 843.0 |
| **γ bas (0.80)** | **0.10** | **0.80** | **0.995** | **5000** | **854.2** ⭐ |
| γ haut (0.99) | 0.10 | 0.99 | 0.995 | 5000 | 836.2 |
| ε decay lent (0.999) | 0.10 | 0.95 | 0.999 | 5000 | 835.6 |
| ε decay rapide (0.99) | 0.10 | 0.95 | 0.99 | 5000 | 820.1 |
| Long (10 000 ép) | 0.10 | 0.95 | 0.997 | 10000 | 827.9 |
| « Tout maximisé » | 0.10 | 0.99 | 0.999 | 10000 | 814.7 |

### Configuration finale retenue

```
α = 0.10
γ = 0.80     ← découverte clé
ε : 1.0 → 0.01 avec decay 0.995
5000 épisodes
```

### Pourquoi γ = 0.80 et pas 0.95 ?

C'est contre-intuitif et c'est l'observation la plus intéressante du tuning. Notre épisode dure 30 jours mais la péremption est de 2 jours seulement : ce qui se passera dans 5 jours n'a quasiment aucune influence sur la décision d'aujourd'hui, parce que le stock présent sera de toute façon vendu ou jeté avant. Un γ trop élevé pousse l'agent à planifier loin alors que ce n'est pas pertinent, ce qui ajoute du bruit dans l'apprentissage. Avec γ = 0.80, l'agent se concentre sur le très court terme, ce qui correspond à la réalité métier du problème.

---

## 5. Deep Q-Network

### L'agent (`dqn.py`)

Trois composants exigés par le sujet, tous présents :

1. **Réseau Q** : MLP simple en PyTorch
   - 22 entrées (dim de l'état)
   - 2 couches cachées de 128 neurones avec ReLU
   - 729 sorties (une valeur Q par action)

2. **Replay Buffer** : `deque` de capacité 30 000 transitions. À chaque pas, on stocke `(s, a, r, s', done)` puis on échantillonne un mini-batch aléatoire pour l'entraînement. Casse la corrélation temporelle entre transitions consécutives.

3. **Target Network** : copie périodique du réseau Q. Sert à calculer la cible Bellman pendant les mises à jour, avec des poids figés. Évite le « moving target problem » et stabilise l'apprentissage.

Détails techniques additionnels :
- Optimiseur Adam (lr = 5e-4)
- Loss Huber (`SmoothL1Loss`) plus robuste aux outliers que la MSE
- Gradient clipping à norme 10.0 pour éviter les explosions

### Tuning systématique (5 configurations)

Comme un entraînement DQN est plus long qu'un Q-learning, on a testé un ensemble plus restreint de 5 configurations sur 800 épisodes chacune (script `tune_dqn.py`). Les seeds d'évaluation sont fixes pour rendre la comparaison juste.

| Configuration | γ | lr | batch | target_update | Reward |
|---|---|---|---|---|---|
| **lr haut (1e-3)** | 0.90 | 1e-3 | 128 | 500 | **2573.9** ⭐ |
| Baseline | 0.90 | 5e-4 | 128 | 500 | 2532.8 |
| target rapide (200) | 0.90 | 5e-4 | 128 | 200 | 2530.5 |
| batch petit (64) | 0.90 | 5e-4 | 64 | 500 | 2455.2 |
| γ haut (0.95) | 0.95 | 5e-4 | 128 | 500 | 2363.4 |

Trois enseignements de ce tuning :

- **γ haut dégrade** (2363 vs 2532). C'est cohérent avec ce qu'on a observé en Q-learning : ce problème est intrinsèquement court terme à cause de la péremption rapide. Un facteur d'actualisation élevé ajoute du bruit dans l'apprentissage.
- **Le batch size compte** : passer de 128 à 64 fait perdre 80 points. Avec 729 actions, des gradients moins bruités sont essentiels.
- **lr=1e-3 est légèrement meilleur sur 800 épisodes**, mais on garde lr=5e-4 pour le run final à 1500 épisodes : un lr plus prudent est plus stable sur des entraînements plus longs, et l'écart de 40 points est dans la marge de variance.

### Instabilité initiale et corrections

Premier entraînement (4 produits, 800 épisodes, lr=1e-3, batch=64) → résultats très instables, oscillations entre 969 et 1785. Diagnostic :
- 6561 actions (4 produits) trop grand pour un seul argmax fiable
- Trop peu d'épisodes
- Batch trop petit, lr trop élevé

### Configuration finale retenue

| Paramètre | Valeur |
|---|---|
| Architecture | MLP 22 → 128 → 128 → 729 |
| Optimiseur | Adam |
| Learning rate | 5e-4 |
| γ | 0.90 |
| Buffer size | 30 000 |
| Batch size | 128 |
| Target update | tous les 500 pas |
| ε | 1.0 → 0.05, decay 0.995 |
| Épisodes | 1500 |

Avec cette configuration, la courbe d'apprentissage est nette et monotone :

| Phase | Épisode | Reward (moy 30) |
|---|---|---|
| Démarrage (ε≈0.8) | 50 | 1287 |
| Apprentissage rapide | 200 | 1970 |
| Plateau | 400-500 | 2040 |
| Affinage | 700 | 2262 |
| **Convergence stable** | 1000-1500 | **2200-2350** |

Progression totale : **+72 % entre le démarrage et la convergence**.

---

## 6. Résultats finaux et comparaison

### Sur l'environnement mono-produit

| Politique | Reward | % gaspi | % rupture |
|---|---|---|---|
| Random | 695.6 | 8.1 % | 12.0 % |
| Fixed q=7 | 872.4 | 5.3 % | 4.2 % |
| MovingAvg(7j) | 783.5 | 4.2 % | 9.9 % |
| **Q-learning** | **862.0** | 7.6 % | **3.3 %** |

Sur ce problème simple, le Q-learning frôle la baseline « commander 7 chaque jour » et obtient le meilleur taux de rupture de toutes les politiques. Le tabulaire fait correctement son travail.

### Sur l'environnement multi-produits

| Politique | Reward | % gaspi | % rupture |
|---|---|---|---|
| Random | 1199.4 | 0.7 % | 38.0 % |
| Q-learning (discrétisé) | **−450.8** | 0.5 % | **74.5 %** |
| **DQN** | **2527.1** | 4.5 % | **7.8 %** |

**Le Q-learning s'effondre.** Récompense négative, 74,5 % de la demande en rupture, performance pire que le hasard. La discrétisation grossière de l'état continu ne capture plus la richesse du problème, et l'agent apprend des décisions incohérentes.

**Le DQN explose tout.** 2527 de récompense, soit +110 % vs random, +660 % vs Q-learning sur le même problème. C'est la démonstration mécanique qu'on cherchait.

### La conclusion qu'on défend

Sur un problème simple à un seul produit, Q-learning et DQN sont équivalents et le tabulaire est même préférable pour son interprétabilité. Mais dès qu'on étend à 3 produits aux profils contrastés, le Q-learning passe de 862 à −450 (effondrement total) tandis que le DQN atteint 2527. Le passage au DQN n'est donc pas un effet de mode : c'est une nécessité mécanique liée à l'explosion combinatoire de l'espace d'états.

---

### Grille comparative Q-learning vs DQN

Synthèse structurée selon les critères du hackathon, remplie avec nos résultats réels.

| Critère | Q-learning tabulaire | DQN |
|---|---|---|
| **Espace d'états** | Discret, 90 états visités sur 441 possibles (21 niveaux de stock × 3 niveaux de péremption × 7 jours). État compact à 3 dimensions. | Continu, 22 dimensions (5 features par produit × 3 produits + 7 one-hot jour). |
| **Convergence** | Rapide : converge en ~500 épisodes sur le mono-produit. 5000 épisodes suffisent largement. | Plus lente : progression nette jusqu'à ~800 épisodes, stabilisation vers 1000-1500. |
| **Stabilité** | Stable par nature : pas de gradient, pas de réseau. La Q-table ne diverge pas. Variance ±84 sur l'évaluation. | Sensible aux hyperparamètres. Premier entraînement instable (oscillations 969-1785) avant correction. Variance ±129 après tuning. |
| **Généralisation** | Aucune. Mémorisation pure : face à un état non visité, l'agent retourne une valeur par défaut. S'effondre sur le multi-produits (-450 de reward). | Bonne. Le réseau interpole entre états proches. Gère le multi-produits (2527 de reward) et les changements structurels (péremption -1 jour : -21% seulement). |
| **Complexité** | Faible. Un dictionnaire, une règle de mise à jour, aucun framework. Entraînement en ~5 secondes. | Élevée. Réseau PyTorch, replay buffer, target network, gradient clipping. Entraînement en ~3 minutes. |
| **Applicabilité métier** | Adapté au cas simple mono-produit. Interprétable : on peut lire la politique état par état. Utile pour comprendre le problème, insuffisant pour un déploiement multi-produits. | Adapté au cas réaliste multi-produits. Apprend des politiques différenciées par produit (saumon prudent, pain agressif). Moins interprétable mais plus performant et plus robuste. |

---

## 7. Tests de robustesse

Pour évaluer la fiabilité des deux agents en conditions réalistes, on les entraîne une seule fois sur l'environnement de référence, puis on les évalue sur 5 variantes perturbées SANS les réentraîner. C'est le vrai test : un agent déployé dans un restaurant ne sera pas réentraîné chaque fois que la demande change.

### Résultats Q-learning (mono-produit, 4 scénarios)

Le test « péremption -1 jour » a été retiré pour le Q-learning : le changement de shelf_life crée des états absents de la Q-table, ce qui relève de la généralisation hors-distribution et non de la robustesse. On garde ce test uniquement pour le DQN, qui opère sur un état continu.

| Scénario | Reward | % gaspi | % rupture | Delta vs ref |
|---|---|---|---|---|
| Référence | 857.6 | 7.5% | 3.4% | — |
| Bruit x2 | 785.1 | 9.7% | 5.5% | -8.5% |
| Demande +30% | 1032.7 | 1.0% | 10.6% | +20.4% |
| Demande -30% | 509.1 | 25.4% | 0.5% | -40.6% |
| Events extrêmes x2 | 825.2 | 8.1% | 4.7% | -3.8% |

### Résultats DQN (multi-produits, 5 scénarios)

| Scénario | Reward | % gaspi | % rupture | Delta vs ref |
|---|---|---|---|---|
| Référence | 2551.1 | 4.5% | 7.5% | — |
| Bruit x2 | 2357.5 | 7.0% | 10.5% | -7.6% |
| Demande +30% | 2498.5 | 3.1% | 15.2% | -2.1% |
| Demande -30% | 1303.0 | 32.8% | 1.0% | -48.9% |
| Events extrêmes x2 | 2403.0 | 6.2% | 10.3% | -5.8% |
| Péremption -1 jour | 2001.5 | 16.2% | 5.7% | -21.5% |

### Analyse

**Attention méthodologique** : les deux agents opèrent sur des environnements différents (mono-produit vs multi-produits), avec des perturbations implémentées différemment. On ne peut donc pas comparer directement les pourcentages de dégradation entre les deux. Chaque agent est évalué par rapport à sa propre référence.

**Q-learning (mono-produit)** : l'agent résiste bien au bruit accru (-8.5%) et aux événements extrêmes (-3.8%). La hausse de demande est un cas favorable (+20.4% car plus de ventes). Le point faible est la baisse de demande (-40.6%) où l'agent surstocke par habitude et gaspille 25% des produits.

**DQN (multi-produits)** : l'agent résiste bien au bruit (-7.6%), aux événements extrêmes (-5.8%) et même à la hausse de demande (-2.1% seulement). Le point faible est aussi la baisse de demande (-48.9%) avec 33% de gaspillage. Le DQN gère en plus le scénario péremption -1 jour (-21.5%), non testable sur le Q-learning car le changement de shelf_life crée des états absents de la Q-table.

**Limite commune** : la baisse de demande est le pire scénario pour les deux agents. C'est attendu : un agent entraîné sur une distribution fixe ne peut pas anticiper un changement de tendance. En conditions réelles, un réentraînement périodique serait nécessaire.

Le graphique est dans `results/robustness.png`.

Le point faible commun aux deux agents est la baisse de demande (-30%) : les deux surstockent par habitude et le gaspillage monte. C'est une limite attendue de tout agent entraîné sur une distribution fixe, et c'est un argument pour du réentraînement périodique en conditions réelles.

Le graphique comparatif est dans `results/robustness.png`.

---

## 8. Politiques apprises — ce que chaque modèle a vraiment appris

Les chiffres ne suffisent pas : il faut montrer la politique apprise par chaque modèle pour pouvoir la défendre. C'est l'objet du script `analyze_policies.py`, qui produit cinq fichiers dans `results/`.

### 7.1 Politique du Q-learning (mono-produit)

Le Q-learning a exploré **90 états** sur les ~600 théoriquement possibles. Les autres correspondent à des situations physiquement inatteignables avec une politique raisonnable. Pour chaque état visité, on extrait l'action préférée et sa Q-value. Le résultat complet est dans `results/policy_qlearning.txt`.

Trois comportements observés sur la politique extraite :

- **Stock haut → commande basse.** Exemple lundi : stock=12 → commander 2 ; samedi stock=12 → commander 0. L'agent a compris qu'inutile d'accumuler.
- **Effet jour de la semaine net.** Les Q-values sont plus élevées en milieu/fin de semaine (mercredi-samedi : Q ≈ 140-150) qu'en début (lundi : Q ≈ 90-115). L'agent valorise plus les états en période de forte demande.
- **Politique différenciée selon le jour.** Pour le même état stock=0, l'agent commande 12 unités le samedi mais 9 le lundi. Il anticipe le rush du week-end.

### 7.2 Politique du DQN (multi-produits)

L'état du DQN étant continu en 22 dimensions, on ne peut pas lister exhaustivement la politique. On observe à la place le comportement moyen sur 50 épisodes de test. Résultats complets dans `results/policy_dqn.txt`.

**Quantités moyennes commandées par jour et par produit :**

| Jour | Saumon | Légumes | Pain |
|---|---|---|---|
| Lundi | 3.23 | 7.79 | 8.00 |
| Mardi | 1.33 | 7.94 | 6.10 |
| Mercredi | 6.89 | 7.33 | 7.93 |
| Jeudi | 6.32 | 7.12 | 7.71 |
| Vendredi | 6.56 | 7.37 | 7.89 |
| Samedi | 7.28 | 5.74 | 6.50 |
| Dimanche | 5.28 | 7.83 | 7.91 |

Le DQN a appris **trois politiques différentes selon le profil du produit** :

- **Saumon** (marge 12, DLC 2 jours, gaspillage cher) : commande basse en début de semaine (1.33-3.23), puis montée progressive vers le week-end (7.28 le samedi). L'agent est prudent et anticipe le rush sans accumuler.
- **Légumes** (marge 4, DLC 4 jours, demande régulière) : commande stable autour de 7-8 toute la semaine. La DLC longue permet de maintenir un stock constant sans risque.
- **Pain** (marge 2, DLC 1 jour, rupture cher) : commande quasi-constante à 7-8. L'agent ne peut pas stocker (DLC=1) donc il commande chaque jour ce qu'il pense vendre.

C'est exactement la finesse qu'on espérait : **le DQN n'a pas appris une politique uniforme, il a compris la spécificité métier de chaque produit**. C'est la vraie valeur ajoutée par rapport au Q-learning, qui ne pourrait jamais capturer ces dynamiques différenciées.

### 7.3 Trajectoires d'épisodes

Pour visualiser concrètement la politique en action, on joue un épisode complet de 30 jours avec chaque agent (même seed pour rester comparable).

- `results/episode_qlearning.png` : stock, commandes, demande, ventes, gaspillage, ruptures jour par jour pour le Q-learning sur le mono-produit.
- `results/episode_dqn.png` : pour chacun des 3 produits, évolution simultanée du stock, des commandes et de la demande sur l'épisode.

L'analyse complète de ces trajectoires est sauvegardée dans `results/policy_analysis.md`.

---

## 9. Impact métier chiffré

Pour rendre la valeur business concrète, on compare le DQN à trois politiques de commande fixes qu'un restaurateur pourrait appliquer intuitivement (commander 3, 5 ou 7 unités par produit par jour).

| Politique | Reward/mois | Servi | Ruptures | Gaspillé | % rupture |
|---|---|---|---|---|---|
| Prudente (3/prod/j) | 605 | 269 | 327 | 0 | 54.8% |
| Modérée (5/prod/j) | 1969 | 434 | 162 | 12 | 27.2% |
| Agressive (7/prod/j) | 2086 | 543 | 53 | 72 | 8.8% |
| **DQN** | **2532** | **550** | **46** | **26** | **7.7%** |

Le DQN trouve un équilibre que les politiques fixes ne peuvent pas atteindre : il sert autant que la politique agressive (550 vs 543) mais gaspille presque 3 fois moins (26 vs 72 unités). Il combine le meilleur des deux mondes : peu de ruptures ET peu de gaspillage.

**Gain par rapport à une gestion intuitive (politique modérée)** : +563 euros/mois, soit environ 6 750 euros/an, avec 116 ruptures en moins par mois. Le surcoût en gaspillage est de 15 unités, largement compensé par les ventes supplémentaires.

**Gain par rapport à la politique agressive** : même niveau de service mais -46 unités gaspillées par mois (économie directe sur le coût d'achat des ingrédients).

---

## 10. Choix du modèle final

**Pour un déploiement réel sur un restaurant gérant plusieurs produits : DQN.**

Justifications :
- **Performance** : seule approche qui dépasse significativement les politiques naïves sur le problème réaliste.
- **Scalabilité** : ajouter un produit ne fait que rajouter quelques dimensions à l'état, là où le Q-learning verrait sa table multiplier sa taille.
- **Adaptation** : le réseau généralise à des configurations de stock jamais vues exactement à l'entraînement, contrairement au tabulaire.

**Limites honnêtes à mentionner devant le jury :**
- Convergence plus longue et plus instable que le tabulaire (besoin de tuning).
- Boîte noire : on perd l'interprétabilité de la Q-table.
- Sensibilité aux hyperparamètres : un mauvais lr ou batch size peut tout casser.
- Pas encore testé sur données réelles ; la simulation reste une approximation de la demande d'un vrai restaurant.

---

## 11. Application web de démonstration

Pour concrétiser le projet en produit démontrable, une application web a été développée en Flask. Elle permet de simuler la gestion d'un restaurant en temps réel avec un agent DQN.

### Environnement démo (4 produits)

L'app utilise un environnement dédié (`restaurant_env_demo.py`) à 4 produits (Saumon, Légumes, Pain, Fromage) avec `max_order=5` par produit, soit 6^4 = 1296 actions. L'agent démo a été entraîné avec un réseau plus large (hidden=256) sur 3000 épisodes, avec un epsilon minimum de 0.10 pour favoriser l'exploration. Résultat : l'agent utilise **14 actions différentes** sur un épisode de 30 jours et adapte ses commandes à l'état du stock et au jour de la semaine.

### Fonctionnalités de l'app

- Dashboard avec 5 métriques en temps réel : ventes réalisées, taux de service, gaspillage, ruptures, score RL
- 4 cartes produit affichant le stock actuel, la dernière demande, la DLC, la marge et la recommandation de l'agent pour le lendemain
- Simulation jour par jour, par semaine ou mois entier
- Détail quotidien par produit (commande, demande, servi, gaspillé, stock restant)
- Graphique de performance jour par jour
- Comparaison finale agent DQN vs politique fixe (sans IA)

### Lancement

```bash
python app_web.py
# Puis ouvrir http://localhost:5000
```

---

## 12. Comment relancer le projet

```bash
# Installation
pip install -r requirements.txt

# Test des environnements
python env/test_env.py
python env/test_env_multi.py

# Entraînement Q-learning et courbe
python agents/train_q_learning.py

# Tuning des hyperparamètres Q-learning
python agents/tune_q_learning.py

# Entraînement DQN et courbe
python agents/train_dqn.py

# Tuning des hyperparamètres DQN (plus long, ~5 entraînements)
python agents/tune_dqn.py

# Comparaison finale Q-learning vs DQN
python agents/compare_qlearning_dqn.py

# Analyse des politiques apprises
python agents/analyze_policies.py

# Tests de robustesse (5 scenarios de perturbation)
python agents/robustness_tests.py

# Application web de démonstration
python app_web.py
# Ouvrir http://localhost:5000
```

Tous les résultats (courbes, modèles sauvegardés) sont écrits dans `results/`.
