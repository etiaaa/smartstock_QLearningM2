"""
Environnement 5 produits pour la démo web.

max_order=3 par produit → 4^5 = 1024 actions (gérable pour le DQN).
5 produits aux profils contrastés pour rendre la démo visuelle et parlante.
"""

import numpy as np

DEMO_PRODUCTS = [
    {
        "name": "Saumon",
        "emoji": "🐟",
        "shelf_life": 2,
        "margin": 12.0,
        "waste_penalty": 9.0,
        "stockout_penalty": 6.0,
        "demand_base": 2,
        "weekly_effect": [0, 0, 1, 1, 2, 3, 1],
    },
    {
        "name": "Legumes",
        "emoji": "🥬",
        "shelf_life": 4,
        "margin": 4.0,
        "waste_penalty": 1.5,
        "stockout_penalty": 2.0,
        "demand_base": 3,
        "weekly_effect": [0, 0, 1, 1, 2, 3, 1],
    },
    {
        "name": "Pain",
        "emoji": "🥖",
        "shelf_life": 1,
        "margin": 2.0,
        "waste_penalty": 0.8,
        "stockout_penalty": 3.0,
        "demand_base": 3,
        "weekly_effect": [0, 0, 1, 1, 2, 3, 1],
    },
    {
        "name": "Fromage",
        "emoji": "🧀",
        "shelf_life": 7,
        "margin": 5.0,
        "waste_penalty": 4.0,
        "stockout_penalty": 2.0,
        "demand_base": 1,
        "weekly_effect": [0, 0, 0, 1, 1, 2, 1],
    },
]


class RestaurantEnvDemo:
    def __init__(self, products=None, max_stock_per_product=12,
                 max_order_per_product=5, episode_length=30, seed=None):
        self.products = products if products is not None else DEMO_PRODUCTS
        self.n_products = len(self.products)
        self.max_stock = max_stock_per_product
        self.max_order = max_order_per_product
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.n_actions = (self.max_order + 1) ** self.n_products
        self.obs_dim = 5 * self.n_products + 7
        self.reset()

    def decode_action(self, action_int):
        quantities = []
        a = action_int
        base = self.max_order + 1
        for _ in range(self.n_products):
            quantities.append(a % base)
            a //= base
        return quantities

    def _sample_demand(self, product, day_of_week):
        base = product["demand_base"]
        weekly = product["weekly_effect"][day_of_week]
        noise = self.rng.integers(-1, 2)
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly + noise + bonus)

    def reset(self):
        self.stocks = [[] for _ in range(self.n_products)]
        self.demand_history = [[0, 0, 0] for _ in range(self.n_products)]
        self.day = 0
        self.day_of_week = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        features = []
        for i, product in enumerate(self.products):
            lots = self.stocks[i]
            stock_total = sum(qty for qty, _ in lots)
            avg_age = np.mean([age for _, age in lots]) if lots else 0.0
            features.append(stock_total / self.max_stock)
            features.append(avg_age / max(1, product["shelf_life"]))
            features.extend([d / 10.0 for d in self.demand_history[i]])
        dow = [0.0] * 7
        dow[self.day_of_week] = 1.0
        features.extend(dow)
        return np.array(features, dtype=np.float32)

    def step(self, action_int):
        if self.done:
            raise RuntimeError("Episode termine.")
        quantities = self.decode_action(action_int)
        total_reward = 0.0
        info = {"per_product": []}
        for i, product in enumerate(self.products):
            lots = self.stocks[i]
            order = quantities[i]
            current = sum(qty for qty, _ in lots)
            order = min(order, self.max_stock - current)
            if order > 0:
                lots.append([order, 0])
            demand = self._sample_demand(product, self.day_of_week)
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
            for lot in lots:
                lot[1] += 1
            wasted = sum(qty for qty, age in lots if age >= product["shelf_life"])
            lots = [lot for lot in lots if lot[1] < product["shelf_life"]]
            self.stocks[i] = lots
            r = served * product["margin"] - wasted * product["waste_penalty"] - stockout * product["stockout_penalty"]
            total_reward += r
            self.demand_history[i] = self.demand_history[i][1:] + [demand]
            info["per_product"].append({
                "name": product["name"],
                "ordered": order, "demand": demand,
                "served": served, "stockout": stockout, "wasted": wasted,
                "stock_after": sum(qty for qty, _ in self.stocks[i]),
            })
        self.day += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        if self.day >= self.episode_length:
            self.done = True
        return self._get_state(), total_reward, self.done, info
