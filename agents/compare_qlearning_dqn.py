"""
Comparaison finale Q-learning vs DQN.

Objectif : répondre à la question centrale du hackathon — "pourquoi le DQN
devient-il nécessaire ?"

On compare les deux approches sur les DEUX environnements :
    1. Mono-produit (RestaurantEnvSingle)  -> Q-learning a son terrain de jeu
    2. Multi-produits (RestaurantEnvMulti) -> Q-learning explose, DQN gère

Pour chaque env, on évalue :
    - politiques baselines (random, fixed, moving_avg)
    - Q-learning (entraîné from scratch ici, paramètres tunés)
    - DQN (entraîné from scratch ici, paramètres tunés)

Sortie : tableau comparatif + graphique de comparaison.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

from restaurant_env import RestaurantEnvSingle
from restaurant_env_multi import RestaurantEnvMulti
from q_learning import QLearningAgent
from dqn import DQNAgent
from naive_policies import RandomPolicy, FixedPolicy, MovingAvgPolicy


N_EVAL = 100
EVAL_SEED_BASE = 5000


# ----------------------------------------------------------------------
# Évaluation générique
# ----------------------------------------------------------------------
def eval_episode(env, choose_action_fn, observe_demand_fn=None):
    state = env.reset()
    total_reward = 0
    total_demand = 0
    total_wasted = 0
    total_stockout = 0
    done = False
    while not done:
        action = choose_action_fn(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if "demand" in info:  # mono-produit
            total_demand += info["demand"]
            total_wasted += info["wasted"]
            total_stockout += info["stockout"]
            if observe_demand_fn is not None:
                observe_demand_fn(info["demand"])
        else:  # multi-produits
            for p in info["per_product"]:
                total_demand += p["demand"]
                total_wasted += p["wasted"]
                total_stockout += p["stockout"]
    return total_reward, total_wasted, total_stockout, total_demand


def evaluate(name, env_factory, policy_factory):
    rewards, wpcts, spcts = [], [], []
    for i in range(N_EVAL):
        env = env_factory(EVAL_SEED_BASE + i)
        policy = policy_factory()
        observe_fn = getattr(policy, "observe", None)
        r, w, s, d = eval_episode(env, policy.choose_action, observe_fn)
        rewards.append(r)
        wpcts.append(100 * w / max(1, d))
        spcts.append(100 * s / max(1, d))
    return {
        "name": name,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "wasted": np.mean(wpcts),
        "stockout": np.mean(spcts),
        "rewards": rewards,
    }


# ----------------------------------------------------------------------
# Wrappers pour avoir une interface uniforme
# ----------------------------------------------------------------------
class QLearningWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.agent.epsilon = 0.0
    def choose_action(self, state):
        return self.agent.choose_action(state)


class DQNWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.agent.epsilon = 0.0
    def choose_action(self, state):
        return self.agent.choose_action(state)


# ----------------------------------------------------------------------
# Entraînements
# ----------------------------------------------------------------------
def train_qlearning_single():
    env = RestaurantEnvSingle(seed=0)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.1, gamma=0.80,  # config optimale trouvée au tuning
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        seed=0,
    )
    for _ in range(5000):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
    return agent


def train_qlearning_multi():
    """
    Tentative de Q-learning sur le multi-produits.

    On force la discrétisation de l'état continu pour que la Q-table puisse
    fonctionner. Mais avec 729 actions et un état discrétisé en 5 niveaux par
    dimension, ça va galérer — c'est exactement ce qu'on veut démontrer.
    """
    env = RestaurantEnvMulti(seed=0)

    def discretize(state):
        # Discrétise chaque dimension en 5 niveaux puis renvoie un tuple
        return tuple((state * 5).astype(int).tolist())

    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.1, gamma=0.90,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997,
        seed=0,
    )
    rewards = []
    for ep in range(1000):
        state = env.reset()
        s_disc = discretize(state)
        done = False
        ep_reward = 0
        while not done:
            action = agent.choose_action(s_disc)
            next_state, reward, done, _ = env.step(action)
            ns_disc = discretize(next_state)
            agent.update(s_disc, action, reward, ns_disc, done)
            s_disc = ns_disc
            ep_reward += reward
        agent.decay_epsilon()
        rewards.append(ep_reward)

    return agent, rewards, discretize


def train_dqn_multi():
    env = RestaurantEnvMulti(seed=0)
    agent = DQNAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        hidden=128, gamma=0.90, lr=5e-4,
        buffer_size=30000, batch_size=128, target_update_freq=500,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
        seed=0,
    )
    rewards = []
    for ep in range(1500):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            ep_reward += reward
        agent.decay_epsilon()
        rewards.append(ep_reward)
    return agent, rewards


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("COMPARAISON Q-LEARNING vs DQN")
    print("=" * 80)

    # ---------- Mono-produit ----------
    print("\n[1/3] Entrainement Q-learning sur env mono-produit...")
    qagent_single = train_qlearning_single()

    print("[2/3] Entrainement Q-learning sur env multi-produits...")
    qagent_multi, q_rewards_multi, discretize_fn = train_qlearning_multi()

    print("[3/3] Entrainement DQN sur env multi-produits...")
    dqn_agent, dqn_rewards = train_dqn_multi()

    # ---------- Évaluations ----------
    print("\nEvaluation sur mono-produit (100 episodes)...")
    n_actions_single = RestaurantEnvSingle().n_actions
    results_single = []
    results_single.append(evaluate(
        "Random", lambda s: RestaurantEnvSingle(seed=s),
        lambda: RandomPolicy(n_actions_single, seed=42)))
    results_single.append(evaluate(
        "Fixed q=7", lambda s: RestaurantEnvSingle(seed=s),
        lambda: FixedPolicy(7)))
    results_single.append(evaluate(
        "MovingAvg(7j)", lambda s: RestaurantEnvSingle(seed=s),
        lambda: MovingAvgPolicy(n_actions_single, window=7)))
    results_single.append(evaluate(
        "Q-learning", lambda s: RestaurantEnvSingle(seed=s),
        lambda: QLearningWrapper(qagent_single)))

    print("Evaluation sur multi-produits (100 episodes)...")
    n_actions_multi = RestaurantEnvMulti().n_actions
    results_multi = []
    results_multi.append(evaluate(
        "Random", lambda s: RestaurantEnvMulti(seed=s),
        lambda: RandomPolicy(n_actions_multi, seed=42)))

    # Q-learning sur multi : utilise discretisation
    class QLearningMultiWrapper:
        def __init__(self, agent, disc_fn):
            self.agent = agent
            self.agent.epsilon = 0.0
            self.disc_fn = disc_fn
        def choose_action(self, state):
            return self.agent.choose_action(self.disc_fn(state))

    results_multi.append(evaluate(
        "Q-learning", lambda s: RestaurantEnvMulti(seed=s),
        lambda: QLearningMultiWrapper(qagent_multi, discretize_fn)))
    results_multi.append(evaluate(
        "DQN", lambda s: RestaurantEnvMulti(seed=s),
        lambda: DQNWrapper(dqn_agent)))

    # ---------- Affichage ----------
    def print_table(title, results):
        print(f"\n{title}")
        print("-" * 80)
        print(f"{'Politique':<18}{'Reward':>14}{'± std':>10}{'% gaspi':>12}{'% rupture':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['name']:<18}{r['reward_mean']:>14.1f}{r['reward_std']:>10.1f}"
                  f"{r['wasted']:>11.1f}%{r['stockout']:>11.1f}%")

    print("\n" + "=" * 80)
    print("RESULTATS FINAUX")
    print("=" * 80)
    print_table("MONO-PRODUIT (Q-learning a l'avantage)", results_single)
    print_table("MULTI-PRODUITS (DQN devient necessaire)", results_multi)

    # ---------- Graphiques ----------
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Courbes d'apprentissage
    def smooth(v, w=30):
        if len(v) < w: return v
        return np.convolve(v, np.ones(w)/w, mode="valid")

    axes[0].plot(q_rewards_multi, alpha=0.3, color="C0")
    axes[0].plot(smooth(q_rewards_multi), label="Q-learning", color="C0", linewidth=2)
    axes[0].plot(dqn_rewards, alpha=0.3, color="C1")
    axes[0].plot(smooth(dqn_rewards), label="DQN", color="C1", linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Recompense cumulee")
    axes[0].set_title("Apprentissage sur env multi-produits")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bar chart final
    names_m = [r["name"] for r in results_multi]
    rewards_m = [r["reward_mean"] for r in results_multi]
    colors = ["gray", "C0", "C1"]
    axes[1].bar(names_m, rewards_m, color=colors)
    axes[1].set_ylabel("Recompense moyenne (eval 100 ep)")
    axes[1].set_title("Performance finale (multi-produits)")
    axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(rewards_m):
        axes[1].text(i, v + 30, f"{v:.0f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison.png"), dpi=120)
    print(f"\nGraphique sauvegarde : {os.path.join(results_dir, 'comparison.png')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
