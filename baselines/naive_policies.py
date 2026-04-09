"""
Politiques naïves de référence pour comparer notre agent Q-learning.

Ces politiques ne font AUCUN apprentissage. Elles servent de point de comparaison
pour démontrer la valeur ajoutée du RL.

Trois politiques :
    - RandomPolicy   : action aléatoire (borne basse de la performance)
    - FixedPolicy    : commande la même quantité chaque jour
    - MovingAvgPolicy: commande la moyenne des ventes des N derniers jours
"""

import numpy as np
from collections import deque


class RandomPolicy:
    def __init__(self, n_actions, seed=None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def choose_action(self, state, info=None):
        return int(self.rng.integers(0, self.n_actions))


class FixedPolicy:
    """Commande toujours la même quantité, peu importe l'état."""
    def __init__(self, fixed_quantity):
        self.q = fixed_quantity

    def reset(self):
        pass

    def choose_action(self, state, info=None):
        return self.q


class MovingAvgPolicy:
    """
    Commande la moyenne arrondie des ventes des N derniers jours.

    C'est la baseline "intelligente sans RL" : si la demande moyenne est de 7,
    on commande 7. Simple, naturel, et difficile à battre quand la demande
    est régulière.
    """
    def __init__(self, n_actions, window=7, default=5):
        self.n_actions = n_actions
        self.window = window
        self.default = default
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def choose_action(self, state, info=None):
        if len(self.history) == 0:
            return self.default
        avg = int(round(np.mean(self.history)))
        return max(0, min(self.n_actions - 1, avg))

    def observe(self, demand):
        """À appeler après chaque step pour mémoriser la demande observée."""
        self.history.append(demand)
