"""
Tuning des hyperparamètres du DQN.

Comme un entraînement DQN est plus coûteux qu'un Q-learning, on teste un nombre
limité de configurations (4) plutôt qu'une grille exhaustive. Chaque config est
entraînée puis évaluée sur 50 épisodes de test (mêmes seeds pour toutes).
"""

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from restaurant_env_multi import RestaurantEnvMulti
from dqn import DQNAgent


N_EVAL = 50
EVAL_SEED_BASE = 7000


def train_dqn(gamma, lr, batch_size, target_update, n_episodes):
    env = RestaurantEnvMulti(seed=0)
    agent = DQNAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        hidden=128, gamma=gamma, lr=lr,
        buffer_size=30000, batch_size=batch_size,
        target_update_freq=target_update,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
        seed=0,
    )
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
        agent.decay_epsilon()
    agent.epsilon = 0.0
    return agent


def evaluate(agent):
    rewards = []
    for i in range(N_EVAL):
        env = RestaurantEnvMulti(seed=EVAL_SEED_BASE + i)
        state = env.reset()
        total = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


def main():
    # On reduit a 800 episodes pour ne pas exploser le temps de calcul
    # (l'idee est de comparer les configs entre elles, pas d'atteindre la conv finale)
    N_EP = 800

    configs = [
        # (nom,                gamma, lr,    batch, target_update)
        ("Baseline",            0.90, 5e-4,  128,  500),
        ("gamma haut (0.95)",   0.95, 5e-4,  128,  500),
        ("lr haut (1e-3)",      0.90, 1e-3,  128,  500),
        ("batch petit (64)",    0.90, 5e-4,   64,  500),
        ("target rapide (200)", 0.90, 5e-4,  128,  200),
    ]

    print("=" * 80)
    print(f"TUNING DQN - {len(configs)} configurations - {N_EP} episodes / config")
    print(f"Evaluation : {N_EVAL} episodes (seeds {EVAL_SEED_BASE}-{EVAL_SEED_BASE+N_EVAL-1})")
    print("=" * 80)
    print(f"{'Config':<24}{'gamma':>8}{'lr':>10}{'batch':>8}{'target':>10}{'Reward':>12}{'± std':>10}")
    print("-" * 80)

    results = []
    for name, gamma, lr, batch, target in configs:
        print(f"  Entrainement : {name}...")
        agent = train_dqn(gamma, lr, batch, target, N_EP)
        r_mean, r_std = evaluate(agent)
        results.append((name, gamma, lr, batch, target, r_mean, r_std))
        print(f"{name:<24}{gamma:>8.2f}{lr:>10.0e}{batch:>8}{target:>10}"
              f"{r_mean:>12.1f}{r_std:>10.1f}")

    print("-" * 80)
    results_sorted = sorted(results, key=lambda x: -x[5])
    print("\nCLASSEMENT (par recompense moyenne) :")
    for rank, r in enumerate(results_sorted, 1):
        marker = " <-- MEILLEUR" if rank == 1 else ""
        print(f"  {rank}. {r[0]:<24} reward={r[5]:.1f}{marker}")
    print("=" * 80)


if __name__ == "__main__":
    main()
