"""
Agent Q-learning tabulaire pour l'environnement RestaurantEnvSingle.

Le Q-learning apprend une fonction Q(s, a) qui estime la récompense cumulée
attendue si on prend l'action a dans l'état s, puis qu'on suit la politique
optimale ensuite.

La Q-table est ici un dictionnaire {état -> vecteur de valeurs Q (une par action)}.
On utilise un dict plutôt qu'un array NumPy parce que l'espace d'états est
potentiellement creux : on ne crée une entrée que pour les états réellement visités.
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        n_actions,
        alpha=0.1,           # taux d'apprentissage
        gamma=0.95,          # facteur d'actualisation
        epsilon_start=1.0,   # exploration initiale (100 % aléatoire)
        epsilon_end=0.01,    # exploration finale (1 %)
        epsilon_decay=0.995, # facteur de décroissance par épisode
        seed=None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table : à chaque état non visité, on initialise un vecteur de zéros
        self.Q = defaultdict(lambda: np.zeros(n_actions))

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Choix d'action : politique ε-greedy
    # ------------------------------------------------------------------
    def choose_action(self, state):
        """
        Avec probabilité ε : action aléatoire (exploration).
        Sinon : action qui maximise Q(s, a) (exploitation).
        """
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Mise à jour de Bellman
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """
        Règle de mise à jour du Q-learning :
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Si l'épisode est terminé, il n'y a pas de récompense future :
            Q(s, a) ← Q(s, a) + α * [r - Q(s, a)]
        """
        current_q = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] = current_q + self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # Décroissance de l'exploration en fin d'épisode
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
