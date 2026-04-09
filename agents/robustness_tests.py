"""
Tests de robustesse des agents Q-learning et DQN.

On entraîne chaque agent UNE SEULE FOIS sur l'environnement de référence, puis on
l'évalue sur des variantes perturbées SANS le réentraîner. C'est le vrai test :
un agent robuste doit maintenir ses performances même quand les conditions changent.

5 scénarios de perturbation testés :
    1. Demande plus volatile (bruit x2)
    2. Demande en hausse (+30%)
    3. Demande en baisse (-30%)
    4. Événements extrêmes fréquents (20% au lieu de 10%)
    5. Péremption plus courte (-1 jour)

Pour chaque scénario, on mesure : reward, % gaspillage, % rupture.
On compare à la performance de référence (env normal).
"""

import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from restaurant_env import RestaurantEnvSingle
from restaurant_env_multi import RestaurantEnvMulti, DEFAULT_PRODUCTS
from q_learning import QLearningAgent
from dqn import DQNAgent


N_EVAL = 100
EVAL_SEED_BASE = 8000


# ======================================================================
# Environnements perturbés — mono-produit
# ======================================================================
class NoisyEnvSingle(RestaurantEnvSingle):
    """Demande avec bruit x2 (±4 au lieu de ±2)."""
    def _sample_demand(self, day_of_week):
        base = 4
        weekly_effect = [0, 1, 2, 3, 5, 6, 3]
        noise = self.rng.integers(-4, 5)  # bruit x2
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly_effect[day_of_week] + noise + bonus)


class HighDemandEnvSingle(RestaurantEnvSingle):
    """Demande en hausse de 30% (base 4 → 5, effets +30%)."""
    def _sample_demand(self, day_of_week):
        base = 5
        weekly_effect = [0, 1, 3, 4, 7, 8, 4]  # +30% arrondi
        noise = self.rng.integers(-2, 3)
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly_effect[day_of_week] + noise + bonus)


class LowDemandEnvSingle(RestaurantEnvSingle):
    """Demande en baisse de 30% (base 3, effets -30%)."""
    def _sample_demand(self, day_of_week):
        base = 3
        weekly_effect = [0, 1, 1, 2, 4, 4, 2]  # -30% arrondi
        noise = self.rng.integers(-2, 3)
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly_effect[day_of_week] + noise + bonus)


class FrequentEventsEnvSingle(RestaurantEnvSingle):
    """Événements extrêmes 2x plus fréquents (10% rush, 10% creux)."""
    def _sample_demand(self, day_of_week):
        base = 4
        weekly_effect = [0, 1, 2, 3, 5, 6, 3]
        noise = self.rng.integers(-2, 3)
        roll = self.rng.random()
        bonus = 5 if roll < 0.10 else (-4 if roll < 0.20 else 0)
        return max(0, base + weekly_effect[day_of_week] + noise + bonus)


class ShortShelfEnvSingle(RestaurantEnvSingle):
    """Péremption réduite à 1 jour (au lieu de 2)."""
    def __init__(self, **kwargs):
        kwargs["shelf_life"] = 1
        super().__init__(**kwargs)


# ======================================================================
# Environnements perturbés — multi-produits
# ======================================================================
class NoisyEnvMulti(RestaurantEnvMulti):
    """Demande avec bruit x2."""
    def _sample_demand(self, product, day_of_week):
        base = product["demand_base"]
        weekly = product["weekly_effect"][day_of_week]
        noise = self.rng.integers(-3, 4)  # bruit x2
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly + noise + bonus)


class HighDemandEnvMulti(RestaurantEnvMulti):
    """Demande en hausse de 30%."""
    def _sample_demand(self, product, day_of_week):
        base = int(product["demand_base"] * 1.3)
        weekly = int(product["weekly_effect"][day_of_week] * 1.3)
        noise = self.rng.integers(-1, 2)
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly + noise + bonus)


class LowDemandEnvMulti(RestaurantEnvMulti):
    """Demande en baisse de 30%."""
    def _sample_demand(self, product, day_of_week):
        base = int(product["demand_base"] * 0.7)
        weekly = int(product["weekly_effect"][day_of_week] * 0.7)
        noise = self.rng.integers(-1, 2)
        roll = self.rng.random()
        bonus = 3 if roll < 0.05 else (-2 if roll < 0.10 else 0)
        return max(0, base + weekly + noise + bonus)


class FrequentEventsEnvMulti(RestaurantEnvMulti):
    """Événements extrêmes 2x plus fréquents."""
    def _sample_demand(self, product, day_of_week):
        base = product["demand_base"]
        weekly = product["weekly_effect"][day_of_week]
        noise = self.rng.integers(-1, 2)
        roll = self.rng.random()
        bonus = 5 if roll < 0.10 else (-4 if roll < 0.20 else 0)
        return max(0, base + weekly + noise + bonus)


class ShortShelfEnvMulti(RestaurantEnvMulti):
    """Péremption réduite de 1 jour pour chaque produit."""
    def __init__(self, **kwargs):
        products = copy.deepcopy(DEFAULT_PRODUCTS)
        for p in products:
            p["shelf_life"] = max(1, p["shelf_life"] - 1)
        kwargs["products"] = products
        super().__init__(**kwargs)


# ======================================================================
# Entraînement
# ======================================================================
def train_qlearning():
    env = RestaurantEnvSingle(seed=0)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.1, gamma=0.80,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        seed=0,
    )
    for _ in range(5000):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            ns, r, done, _ = env.step(action)
            agent.update(state, action, r, ns, done)
            state = ns
        agent.decay_epsilon()
    agent.epsilon = 0.0
    return agent


def train_dqn():
    env = RestaurantEnvMulti(seed=0)
    agent = DQNAgent(
        obs_dim=env.obs_dim, n_actions=env.n_actions,
        hidden=128, gamma=0.90, lr=5e-4,
        buffer_size=30000, batch_size=128, target_update_freq=500,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
        seed=0,
    )
    for _ in range(1500):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            ns, r, done, _ = env.step(action)
            agent.remember(state, action, r, ns, done)
            agent.train_step()
            state = ns
        agent.decay_epsilon()
    agent.epsilon = 0.0
    return agent


# ======================================================================
# Évaluation générique
# ======================================================================
def evaluate_on_env(agent, env_class, is_multi=False, **env_kwargs):
    rewards, wpcts, spcts = [], [], []
    for i in range(N_EVAL):
        env = env_class(seed=EVAL_SEED_BASE + i, **env_kwargs)
        state = env.reset()
        total_r, total_d, total_w, total_s = 0, 0, 0, 0
        done = False
        while not done:
            action = agent.choose_action(state)
            state, r, done, info = env.step(action)
            total_r += r
            if is_multi:
                for p in info["per_product"]:
                    total_d += p["demand"]
                    total_w += p["wasted"]
                    total_s += p["stockout"]
            else:
                total_d += info["demand"]
                total_w += info["wasted"]
                total_s += info["stockout"]
        rewards.append(total_r)
        wpcts.append(100 * total_w / max(1, total_d))
        spcts.append(100 * total_s / max(1, total_d))

    return np.mean(rewards), np.std(rewards), np.mean(wpcts), np.mean(spcts)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 90)
    print("TESTS DE ROBUSTESSE")
    print("=" * 90)

    print("\n[1/2] Entrainement Q-learning (mono-produit, env de reference)...")
    qagent = train_qlearning()
    print("      OK")

    print("[2/2] Entrainement DQN (multi-produits, env de reference)...")
    dqn_agent = train_dqn()
    print("      OK")

    # ------------------------------------------------------------------
    # Scénarios mono-produit (Q-learning)
    # ------------------------------------------------------------------
    scenarios_single = [
        ("Reference (normal)",    RestaurantEnvSingle),
        ("Bruit x2",             NoisyEnvSingle),
        ("Demande +30%",         HighDemandEnvSingle),
        ("Demande -30%",         LowDemandEnvSingle),
        ("Events extremes x2",   FrequentEventsEnvSingle),
        # Peremption -1 jour retiré : le changement de shelf_life crée des états
        # absents de la Q-table, ce qui n'est pas un test de robustesse mais un
        # problème de généralisation hors-distribution.
    ]

    print("\n" + "=" * 90)
    print("Q-LEARNING (mono-produit) — Agent entraine une fois, evalue sur 6 scenarios")
    print("=" * 90)
    print(f"{'Scenario':<24}{'Reward':>10}{'± std':>10}{'% gaspi':>10}{'% rupture':>10}{'Delta':>10}")
    print("-" * 90)

    ref_reward_q = None
    results_q = []
    for name, env_cls in scenarios_single:
        r_mean, r_std, w, s = evaluate_on_env(qagent, env_cls, is_multi=False)
        if ref_reward_q is None:
            ref_reward_q = r_mean
        delta = ((r_mean - ref_reward_q) / abs(ref_reward_q)) * 100
        results_q.append((name, r_mean, r_std, w, s, delta))
        print(f"{name:<24}{r_mean:>10.1f}{r_std:>10.1f}{w:>9.1f}%{s:>9.1f}%{delta:>+9.1f}%")

    # ------------------------------------------------------------------
    # Scénarios multi-produits (DQN)
    # ------------------------------------------------------------------
    scenarios_multi = [
        ("Reference (normal)",    RestaurantEnvMulti),
        ("Bruit x2",             NoisyEnvMulti),
        ("Demande +30%",         HighDemandEnvMulti),
        ("Demande -30%",         LowDemandEnvMulti),
        ("Events extremes x2",   FrequentEventsEnvMulti),
        ("Peremption -1 jour",   ShortShelfEnvMulti),
    ]

    print("\n" + "=" * 90)
    print("DQN (multi-produits) — Agent entraine une fois, evalue sur 6 scenarios")
    print("=" * 90)
    print(f"{'Scenario':<24}{'Reward':>10}{'± std':>10}{'% gaspi':>10}{'% rupture':>10}{'Delta':>10}")
    print("-" * 90)

    ref_reward_dqn = None
    results_dqn = []
    for name, env_cls in scenarios_multi:
        r_mean, r_std, w, s = evaluate_on_env(dqn_agent, env_cls, is_multi=True)
        if ref_reward_dqn is None:
            ref_reward_dqn = r_mean
        delta = ((r_mean - ref_reward_dqn) / abs(ref_reward_dqn)) * 100
        results_dqn.append((name, r_mean, r_std, w, s, delta))
        print(f"{name:<24}{r_mean:>10.1f}{r_std:>10.1f}{w:>9.1f}%{s:>9.1f}%{delta:>+9.1f}%")

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Q-learning
    names_q = [r[0] for r in results_q]
    rewards_q = [r[1] for r in results_q]
    stds_q = [r[2] for r in results_q]
    colors_q = ["#2ecc71" if i == 0 else ("#e74c3c" if r < ref_reward_q * 0.8 else "#3498db")
                for i, r in enumerate(rewards_q)]
    bars_q = axes[0].barh(names_q, rewards_q, xerr=stds_q, color=colors_q, alpha=0.8)
    axes[0].axvline(x=ref_reward_q, color="green", linestyle="--", alpha=0.5, label="Reference")
    axes[0].set_xlabel("Recompense moyenne")
    axes[0].set_title("Robustesse Q-learning (mono-produit)")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="x")
    for bar, r in zip(bars_q, results_q):
        axes[0].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                     f"{r[5]:+.1f}%", va="center", fontsize=9, fontweight="bold")

    # DQN
    names_d = [r[0] for r in results_dqn]
    rewards_d = [r[1] for r in results_dqn]
    stds_d = [r[2] for r in results_dqn]
    colors_d = ["#2ecc71" if i == 0 else ("#e74c3c" if r < ref_reward_dqn * 0.8 else "#3498db")
                for i, r in enumerate(rewards_d)]
    bars_d = axes[1].barh(names_d, rewards_d, xerr=stds_d, color=colors_d, alpha=0.8)
    axes[1].axvline(x=ref_reward_dqn, color="green", linestyle="--", alpha=0.5, label="Reference")
    axes[1].set_xlabel("Recompense moyenne")
    axes[1].set_title("Robustesse DQN (multi-produits)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="x")
    for bar, r in zip(bars_d, results_dqn):
        axes[1].text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                     f"{r[5]:+.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(results_dir, "robustness.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nGraphique sauvegarde : {out_path}")

    # ------------------------------------------------------------------
    # Synthèse
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SYNTHESE")
    print("=" * 90)

    # On ne compare que les scénarios communs aux deux agents (4 scénarios de demande)
    # et on distingue gains et dégradations
    common_names = ["Bruit x2", "Demande +30%", "Demande -30%", "Events extremes x2"]

    common_q = [r for r in results_q[1:] if r[0] in common_names]
    common_dqn = [r for r in results_dqn[1:] if r[0] in common_names]

    print(f"\nComparaison sur les 4 scenarios communs :")
    print(f"{'Scenario':<24}{'Q-learning':>14}{'DQN':>14}")
    print("-" * 52)
    for rq, rd in zip(common_q, common_dqn):
        print(f"{rq[0]:<24}{rq[5]:>+13.1f}%{rd[5]:>+13.1f}%")

    degradations_q = [r[5] for r in common_q if r[5] < 0]
    degradations_dqn = [r[5] for r in common_dqn if r[5] < 0]
    avg_deg_q = np.mean(degradations_q) if degradations_q else 0
    avg_deg_dqn = np.mean(degradations_dqn) if degradations_dqn else 0

    print(f"\nDegradation moyenne (scenarios negatifs uniquement) :")
    print(f"  Q-learning : {avg_deg_q:.1f}%")
    print(f"  DQN        : {avg_deg_dqn:.1f}%")

    # Scénario péremption uniquement testable sur DQN
    shelf_dqn = [r for r in results_dqn[1:] if r[0] == "Peremption -1 jour"]
    if shelf_dqn:
        print(f"\nScenario peremption -1 jour (DQN uniquement) : {shelf_dqn[0][5]:+.1f}%")
        print(f"  Non testable sur Q-learning (etats hors-distribution)")

    print(f"\nNote : les deux agents operent sur des environnements differents")
    print(f"(mono vs multi-produits). Les pourcentages ne sont pas directement")
    print(f"comparables. Chaque agent est evalue par rapport a sa propre reference.")
    print(f"\nLimite commune : la baisse de demande est le pire scenario pour les deux.")
    print(f"Avantage DQN : gere les changements structurels (peremption), impossible")
    print(f"pour le Q-learning a cause de sa representation discrete des etats.")
    print("=" * 90)


if __name__ == "__main__":
    main()
