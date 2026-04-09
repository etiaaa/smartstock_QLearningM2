"""
Évaluation comparative : Q-learning vs baselines naïves.

Toutes les politiques sont évaluées sur EXACTEMENT les mêmes épisodes
(même seed → même séquence de demande). C'est la seule façon de comparer
honnêtement leurs performances.

Métriques mesurées :
    - récompense cumulée moyenne
    - taux de gaspillage (unités jetées / total commandé)
    - taux de rupture (unités non servies / demande totale)
"""

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

from restaurant_env import RestaurantEnvSingle
from q_learning import QLearningAgent
from naive_policies import RandomPolicy, FixedPolicy, MovingAvgPolicy


N_EVAL_EPISODES = 200
EVAL_SEED_BASE = 1000  # seeds différentes de l'entraînement


def run_episode(env, policy, is_qlearning=False, is_moving_avg=False):
    """Joue un épisode complet et retourne les métriques agrégées."""
    state = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    total_reward = 0
    total_demand = 0
    total_served = 0
    total_wasted = 0
    total_stockout = 0
    total_ordered = 0

    done = False
    while not done:
        action = policy.choose_action(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_demand += info["demand"]
        total_served += info["served"]
        total_wasted += info["wasted"]
        total_stockout += info["stockout"]
        total_ordered += action

        if is_moving_avg:
            policy.observe(info["demand"])

        state = next_state

    return {
        "reward": total_reward,
        "demand": total_demand,
        "served": total_served,
        "wasted": total_wasted,
        "stockout": total_stockout,
        "ordered": total_ordered,
    }


def evaluate_policy(name, policy_factory, is_moving_avg=False):
    """Évalue une politique sur N_EVAL_EPISODES épisodes avec des seeds fixes."""
    rewards, wasted_pct, stockout_pct = [], [], []

    for i in range(N_EVAL_EPISODES):
        env = RestaurantEnvSingle(seed=EVAL_SEED_BASE + i)
        policy = policy_factory()
        m = run_episode(env, policy, is_moving_avg=is_moving_avg)

        rewards.append(m["reward"])
        ordered = max(1, m["ordered"])
        demand = max(1, m["demand"])
        wasted_pct.append(100 * m["wasted"] / ordered)
        stockout_pct.append(100 * m["stockout"] / demand)

    return {
        "name": name,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "wasted_pct": np.mean(wasted_pct),
        "stockout_pct": np.mean(stockout_pct),
    }


def train_qlearning_agent():
    """Réentraîne un agent Q-learning depuis zéro pour évaluation."""
    env = RestaurantEnvSingle(seed=0)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.1, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        seed=0,
    )
    for _ in range(5000):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
    # On force epsilon=0 pour l'évaluation (exploitation pure)
    agent.epsilon = 0.0
    return agent


def main():
    print("Reentrainement de l'agent Q-learning pour evaluation...")
    qagent = train_qlearning_agent()
    print("OK\n")

    n_actions = RestaurantEnvSingle().n_actions

    results = []
    results.append(evaluate_policy(
        "Random", lambda: RandomPolicy(n_actions, seed=42)
    ))
    results.append(evaluate_policy(
        "Fixed (q=5)", lambda: FixedPolicy(5)
    ))
    results.append(evaluate_policy(
        "Fixed (q=7)", lambda: FixedPolicy(7)
    ))
    results.append(evaluate_policy(
        "MovingAvg(7j)", lambda: MovingAvgPolicy(n_actions, window=7),
        is_moving_avg=True
    ))
    results.append(evaluate_policy(
        "Q-learning", lambda: qagent
    ))

    print("=" * 75)
    print(f"EVALUATION sur {N_EVAL_EPISODES} episodes (memes seeds pour toutes)")
    print("=" * 75)
    print(f"{'Politique':<18}{'Reward moy':>14}{'± std':>10}{'% gaspi':>12}{'% rupture':>12}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<18}"
              f"{r['reward_mean']:>14.1f}"
              f"{r['reward_std']:>10.1f}"
              f"{r['wasted_pct']:>11.1f}%"
              f"{r['stockout_pct']:>11.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
