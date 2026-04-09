"""
Environnement de simulation : gestion du stock d'un produit frais dans un restaurant.

Version mono-produit, utilisée pour entraîner le Q-learning tabulaire.
La version multi-produits (pour le DQN) sera dans un fichier séparé.

API inspirée de Gymnasium :
    - reset() : remet l'environnement à zéro et retourne l'état initial
    - step(action) : applique une action, retourne (next_state, reward, done, info)
"""

import numpy as np


class RestaurantEnvSingle:
    """
    Simulation simplifiée d'un restaurant qui gère le stock d'un seul produit frais.

    Chaque jour, l'agent décide combien d'unités commander pour le service du lendemain.
    Le stock vieillit, les unités périmées sont jetées, et la demande est servie tant
    qu'il reste du stock. La récompense reflète la marge des ventes, pénalisée par le
    gaspillage et les ruptures.
    """

    def __init__(
        self,
        max_stock=20,           # capacité maximale du stock (en unités)
        shelf_life=2,           # durée de vie d'une unité après livraison (en jours)
        max_order=12,           # quantité max commandable en une fois
        margin=5.0,             # marge gagnée pour chaque unité vendue
        waste_penalty=6.0,      # pénalité forte : jeter coute plus cher que la marge
        stockout_penalty=4.0,   # pénalité pour chaque unité demandée mais non servie
        episode_length=30,      # nombre de jours par épisode (≈ un mois)
        seed=None,
    ):
        self.max_stock = max_stock
        self.shelf_life = shelf_life
        self.max_order = max_order
        self.margin = margin
        self.waste_penalty = waste_penalty
        self.stockout_penalty = stockout_penalty
        self.episode_length = episode_length

        self.rng = np.random.default_rng(seed)

        # L'espace d'actions est discret : l'agent peut commander entre 0 et max_order unités
        self.n_actions = max_order + 1

        self.reset()

    # ------------------------------------------------------------------
    # Génération de la demande
    # ------------------------------------------------------------------
    def _sample_demand(self, day_of_week):
        """
        Demande quotidienne volatile : base + effet jour + gros bruit + événements rares.

        - Lundi/mardi : demande faible
        - Mercredi/jeudi : demande moyenne
        - Vendredi/samedi : demande forte
        - Dimanche : moyenne
        - 10% de chance d'un evenement (rush ou jour creux imprevisible)
        """
        base = 4
        weekly_effect = [0, 1, 2, 3, 5, 6, 3]  # lundi → dimanche
        noise = self.rng.integers(-2, 3)       # bruit dans {-2,-1,0,1,2}
        demand = base + weekly_effect[day_of_week] + noise

        # Evenement rare : 5% rush (+5), 5% jour creux (-4)
        roll = self.rng.random()
        if roll < 0.05:
            demand += 5
        elif roll < 0.10:
            demand -= 4

        return max(0, demand)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        """Remet l'environnement à l'état initial et retourne l'observation."""
        # Le stock est une liste de lots, chacun étant un dict {qty, age}.
        # qty = nombre d'unités dans le lot, age = nombre de jours depuis la livraison.
        # Quand age >= shelf_life, le lot est jeté.
        self.stock_lots = []  # liste de [qty, age]

        self.day = 0
        self.day_of_week = 0  # on commence un lundi
        self.last_demand = 0
        self.done = False

        return self._get_state()

    # ------------------------------------------------------------------
    # État observable par l'agent
    # ------------------------------------------------------------------
    def _get_state(self):
        """
        Retourne l'observation que voit l'agent.

        Pour le Q-learning tabulaire, on garde un état compact et discret :
            (stock_total, jours_avant_péremption_du_plus_vieux_lot, jour_de_la_semaine)

        C'est suffisant pour apprendre une bonne politique sans faire exploser la Q-table.
        """
        stock_total = sum(qty for qty, _ in self.stock_lots)

        if len(self.stock_lots) > 0:
            oldest_age = max(age for _, age in self.stock_lots)
            days_left = max(0, self.shelf_life - oldest_age)
        else:
            days_left = self.shelf_life

        return (stock_total, days_left, self.day_of_week)

    # ------------------------------------------------------------------
    # Step : un jour de simulation
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Applique une action (= quantité à commander) et avance d'un jour.

        Déroulement d'une journée :
            1. La commande passée hier est livrée le matin → ajoutée au stock
            2. La demande du jour est tirée
            3. On sert les clients (FIFO sur les lots)
            4. Les lots vieillissent d'un jour, ceux qui dépassent shelf_life sont jetés
            5. On calcule la récompense
            6. On passe au jour suivant
        """
        if self.done:
            raise RuntimeError("Episode terminé, appeler reset() avant de continuer.")

        # 1. Livraison de la commande (en respectant la capacité max)
        order = min(action, self.max_order)
        current_total = sum(qty for qty, _ in self.stock_lots)
        order = min(order, self.max_stock - current_total)  # on ne dépasse jamais la capa
        if order > 0:
            # Nouveau lot d'âge 0 (livré aujourd'hui)
            self.stock_lots.append([order, 0])

        # 2. Demande du jour
        demand = self._sample_demand(self.day_of_week)
        self.last_demand = demand

        # 3. Servir les clients en FIFO (les lots les plus vieux partent en premier)
        # On trie par âge décroissant pour servir les plus vieux d'abord
        self.stock_lots.sort(key=lambda lot: -lot[1])
        served = 0
        remaining_demand = demand
        for lot in self.stock_lots:
            if remaining_demand <= 0:
                break
            take = min(lot[0], remaining_demand)
            lot[0] -= take
            served += take
            remaining_demand -= take
        # On nettoie les lots vides
        self.stock_lots = [lot for lot in self.stock_lots if lot[0] > 0]

        stockout = demand - served  # demande non servie

        # 4. Vieillissement : tous les lots prennent un jour de plus
        for lot in self.stock_lots:
            lot[1] += 1

        # 5. Péremption : on jette tous les lots dont l'âge >= shelf_life
        wasted = sum(qty for qty, age in self.stock_lots if age >= self.shelf_life)
        self.stock_lots = [lot for lot in self.stock_lots if lot[1] < self.shelf_life]

        # 6. Récompense
        reward = (
            served * self.margin
            - wasted * self.waste_penalty
            - stockout * self.stockout_penalty
        )

        # 7. Avancement du temps
        self.day += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        if self.day >= self.episode_length:
            self.done = True

        info = {
            "demand": demand,
            "served": served,
            "stockout": stockout,
            "wasted": wasted,
            "stock_after": sum(qty for qty, _ in self.stock_lots),
        }

        return self._get_state(), reward, self.done, info
