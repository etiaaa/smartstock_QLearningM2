"""
Environnement multi-produits : version étendue de RestaurantEnvSingle.

Le restaurant gère plusieurs ingrédients en parallèle, chacun avec ses propres
paramètres métier : marge, durée de péremption, pénalités, demande de base.

C'est cette version qui justifie le passage au DQN :
    - état continu de dimension élevée
    - espace d'actions combinatoire (n_actions ^ n_produits)
    - une Q-table devient impraticable
"""

import numpy as np


# Définition des produits avec des paramètres métier réalistes
DEFAULT_PRODUCTS = [
    {
        "name": "Saumon",
        "shelf_life": 2,
        "margin": 12.0,
        "waste_penalty": 9.0,
        "stockout_penalty": 6.0,
        "demand_base": 3,
        "weekly_effect": [0, 0, 1, 1, 3, 4, 2],
    },
    {
        "name": "Legumes",
        "shelf_life": 4,
        "margin": 4.0,
        "waste_penalty": 1.5,
        "stockout_penalty": 2.0,
        "demand_base": 5,
        "weekly_effect": [0, 1, 1, 2, 3, 4, 2],
    },
    {
        "name": "Pain",
        "shelf_life": 1,
        "margin": 2.0,
        "waste_penalty": 0.8,
        "stockout_penalty": 3.0,
        "demand_base": 6,
        "weekly_effect": [1, 1, 2, 2, 4, 5, 3],
    },
]
# Note : on garde 3 produits (Saumon, Légumes, Pain) pour limiter
# l'espace d'actions à 9^3 = 729 (au lieu de 9^4 = 6561 avec 4 produits).
# Les 3 produits choisis sont les plus contrastés : haute marge / courte DLC,
# moyenne marge / longue DLC, faible marge / DLC ultra-courte.


class RestaurantEnvMulti:
    """
    Restaurant simulé avec plusieurs produits frais en parallèle.

    Observation : vecteur continu (NumPy float32) qui concatène pour chaque produit
        - le stock total
        - l'âge moyen du stock (proxy de l'urgence de péremption)
        - la demande des 3 derniers jours
    Plus le jour de la semaine encodé en one-hot (7 dims).

    Action : entier qui code une combinaison de quantités à commander pour chaque
    produit. Pour simplifier, on discrétise chaque produit en (max_order+1) niveaux
    et on encode l'action multi-dim en un seul entier via base (max_order+1).
    """

    def __init__(
        self,
        products=None,
        max_stock_per_product=15,
        max_order_per_product=8,
        episode_length=30,
        seed=None,
    ):
        self.products = products if products is not None else DEFAULT_PRODUCTS
        self.n_products = len(self.products)
        self.max_stock = max_stock_per_product
        self.max_order = max_order_per_product
        self.episode_length = episode_length

        self.rng = np.random.default_rng(seed)

        # Espace d'actions discret combinatoire
        self.n_actions = (self.max_order + 1) ** self.n_products

        # Dimension de l'observation
        # Pour chaque produit : stock (1) + age moyen (1) + 3 derniers jours (3) = 5
        # + jour de la semaine one-hot (7)
        self.obs_dim = 5 * self.n_products + 7

        self.reset()

    # ------------------------------------------------------------------
    # Encodage / décodage d'action
    # ------------------------------------------------------------------
    def decode_action(self, action_int):
        """Convertit un entier d'action en un vecteur de quantités par produit."""
        quantities = []
        a = action_int
        base = self.max_order + 1
        for _ in range(self.n_products):
            quantities.append(a % base)
            a //= base
        return quantities

    # ------------------------------------------------------------------
    # Demande
    # ------------------------------------------------------------------
    def _sample_demand(self, product, day_of_week):
        base = product["demand_base"]
        weekly = product["weekly_effect"][day_of_week]
        noise = self.rng.integers(-1, 2)
        # Petits événements aléatoires (rush ou jour creux)
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly + noise + bonus)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        # Pour chaque produit : liste de lots [qty, age]
        self.stocks = [[] for _ in range(self.n_products)]
        # Historique des demandes (3 derniers jours par produit)
        self.demand_history = [[0, 0, 0] for _ in range(self.n_products)]

        self.day = 0
        self.day_of_week = 0
        self.done = False

        return self._get_state()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_state(self):
        features = []
        for i, product in enumerate(self.products):
            lots = self.stocks[i]
            stock_total = sum(qty for qty, _ in lots)
            if lots:
                avg_age = np.mean([age for _, age in lots])
            else:
                avg_age = 0.0
            features.append(stock_total / self.max_stock)        # normalisé [0,1]
            features.append(avg_age / max(1, product["shelf_life"]))
            features.extend([d / 10.0 for d in self.demand_history[i]])

        # Jour de la semaine en one-hot
        dow = [0.0] * 7
        dow[self.day_of_week] = 1.0
        features.extend(dow)

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action_int):
        if self.done:
            raise RuntimeError("Episode terminé, appeler reset().")

        quantities = self.decode_action(action_int)

        total_reward = 0.0
        info = {"per_product": []}

        for i, product in enumerate(self.products):
            lots = self.stocks[i]

            # 1. Livraison
            order = quantities[i]
            current = sum(qty for qty, _ in lots)
            order = min(order, self.max_stock - current)
            if order > 0:
                lots.append([order, 0])

            # 2. Demande
            demand = self._sample_demand(product, self.day_of_week)

            # 3. Service FIFO (les plus vieux d'abord)
            lots.sort(key=lambda lot: -lot[1])
            served = 0
            remaining = demand
            for lot in lots:
                if remaining <= 0:
                    break
                take = min(lot[0], remaining)
                lot[0] -= take
                served += take
                remaining -= take
            lots = [lot for lot in lots if lot[0] > 0]

            stockout = demand - served

            # 4. Vieillissement
            for lot in lots:
                lot[1] += 1

            # 5. Péremption
            wasted = sum(qty for qty, age in lots if age >= product["shelf_life"])
            lots = [lot for lot in lots if lot[1] < product["shelf_life"]]

            self.stocks[i] = lots

            # 6. Récompense pour ce produit (asymétrique par produit)
            r = (
                served * product["margin"]
                - wasted * product["waste_penalty"]
                - stockout * product["stockout_penalty"]
            )
            total_reward += r

            # 7. Mise à jour de l'historique (on garde 3 derniers jours)
            self.demand_history[i] = self.demand_history[i][1:] + [demand]

            info["per_product"].append({
                "name": product["name"],
                "ordered": order,
                "demand": demand,
                "served": served,
                "stockout": stockout,
                "wasted": wasted,
            })

        # 8. Avancement du temps
        self.day += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        if self.day >= self.episode_length:
            self.done = True

        return self._get_state(), total_reward, self.done, info
