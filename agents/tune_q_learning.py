"""
Tuning des hyperparamètres du Q-learning.

On teste plusieurs configurations en faisant varier alpha, gamma, epsilon_decay
et le nombre d'épisodes d'entraînement. Pour chaque config :
    - on entraîne un agent avec la config
    - on l'évalue sur 200 épisodes de test (mêmes seeds pour toutes les configs)
    - on enregistre la récompense moyenne, le % gaspi et le % rupture

À la fin, on affiche le classement et on identifie la meilleure configuration.
"""

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from restaurant_env import RestaurantEnvSingle
from q_learning import QLearningAgent


N_EVAL_EPISODES = 200
EVAL_SEED_BASE = 1000


def train_agent(alpha, gamma, eps_decay, n_episodes):
    env = RestaurantEnvSingle(seed=0)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=alpha, gamma=gamma,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=eps_decay,
        seed=0,
    )
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
    agent.epsilon = 0.0  # exploitation pure pour l'évaluation
    return agent


def evaluate_agent(agent):
    rewards, wasted_pct, stockout_pct = [], [], []
    for i in range(N_EVAL_EPISODES):
        env = RestaurantEnvSingle(seed=EVAL_SEED_BASE + i)
        state = env.reset()
        total_r, total_d, total_w, total_s, total_o = 0, 0, 0, 0, 0
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            total_r += reward
            total_d += info["demand"]
            total_w += info["wasted"]
            total_s += info["stockout"]
            total_o += action
        rewards.append(total_r)
        wasted_pct.append(100 * total_w / max(1, total_o))
        stockout_pct.append(100 * total_s / max(1, total_d))
    return np.mean(rewards), np.std(rewards), np.mean(wasted_pct), np.mean(stockout_pct)


def main():
    # Configurations à tester. On part de la baseline puis on fait varier
    # un paramètre à la fois pour voir son effet.
    configs = [
        # (nom,                   alpha, gamma, eps_decay, n_episodes)
        ("Baseline",                0.10, 0.95, 0.995,  5000),
        ("alpha bas (0.05)",        0.05, 0.95, 0.995,  5000),
        ("alpha haut (0.20)",       0.20, 0.95, 0.995,  5000),
        ("gamma bas (0.80)",        0.10, 0.80, 0.995,  5000),
        ("gamma haut (0.99)",       0.10, 0.99, 0.995,  5000),
        ("eps decay lent (0.999)",  0.10, 0.95, 0.999,  5000),
        ("eps decay rapide (0.99)", 0.10, 0.95, 0.99,   5000),
        ("Long (10000 ep)",         0.10, 0.95, 0.997, 10000),
        ("Optimise propose",        0.10, 0.99, 0.999, 10000),
    ]

    print("=" * 90)
    print(f"TUNING Q-LEARNING - {len(configs)} configurations testees")
    print(f"Evaluation : {N_EVAL_EPISODES} episodes (seeds {EVAL_SEED_BASE}-{EVAL_SEED_BASE+N_EVAL_EPISODES-1})")
    print("=" * 90)
    print(f"{'Config':<28}{'alpha':>7}{'gamma':>7}{'decay':>8}{'eps':>7}"
          f"{'Reward':>10}{'gaspi':>9}{'rupture':>10}")
    print("-" * 90)

    results = []
    for name, alpha, gamma, eps_decay, n_ep in configs:
        agent = train_agent(alpha, gamma, eps_decay, n_ep)
        r_mean, r_std, w_pct, s_pct = evaluate_agent(agent)
        results.append((name, alpha, gamma, eps_decay, n_ep, r_mean, w_pct, s_pct))
        print(f"{name:<28}{alpha:>7.2f}{gamma:>7.2f}{eps_decay:>8.3f}{n_ep:>7}"
              f"{r_mean:>10.1f}{w_pct:>8.1f}%{s_pct:>9.1f}%")

    print("-" * 90)

    # Classement par récompense moyenne
    results_sorted = sorted(results, key=lambda x: -x[5])
    print("\nCLASSEMENT (par recompense moyenne) :")
    for rank, r in enumerate(results_sorted, 1):
        marker = " <-- MEILLEUR" if rank == 1 else ""
        print(f"  {rank}. {r[0]:<28} reward={r[5]:.1f}{marker}")
    print("=" * 90)


if __name__ == "__main__":
    main()
