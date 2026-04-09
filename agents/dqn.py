"""
Agent Deep Q-Network (DQN) pour l'environnement RestaurantEnvMulti.

Le DQN remplace la Q-table par un réseau de neurones qui prend l'état en entrée
et produit une valeur Q estimée pour chaque action possible.

Trois composants exigés par le sujet :
    1. Réseau Q (MLP simple)
    2. Replay Buffer (mémoire de transitions passées)
    3. Target Network (réseau Q cible mis à jour à fréquence fixe)

Pourquoi ces deux mécanismes ?
    - Replay Buffer : casse la corrélation temporelle entre transitions consécutives.
      Sans lui, le réseau apprend sur des données très corrélées et devient instable.
    - Target Network : fournit une cible Q stable pendant la mise à jour. Sans lui,
      la cible bouge en même temps que le réseau qu'on entraîne, ce qui cree des
      boucles d'instabilité ("moving target problem").
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------------------------------------------------
# Réseau Q : MLP simple
# ----------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# Replay Buffer
# ----------------------------------------------------------------------
class ReplayBuffer:
    """
    Mémoire FIFO de transitions (s, a, r, s', done) jusqu'à `capacity`.
    On échantillonne ensuite des mini-batches aléatoires pour l'entraînement.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# Agent DQN
# ----------------------------------------------------------------------
class DQNAgent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden=128,
        gamma=0.95,
        lr=1e-3,
        buffer_size=20000,
        batch_size=64,
        target_update_freq=200,    # mise à jour du target net (en pas, pas en épisodes)
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss : plus robuste que MSE

        self.buffer = ReplayBuffer(buffer_size)
        self.step_count = 0

    # ------------------------------------------------------------------
    # Choix d'action
    # ------------------------------------------------------------------
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_net(s)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Stockage et entraînement
    # ------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Une mise à jour du réseau si on a assez de transitions en mémoire."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Q(s, a) actuel
        q_pred = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Cible : r + gamma * max_a' Q_target(s', a') (sauf si épisode terminé)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clipping du gradient pour stabiliser
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Mise à jour périodique du target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
