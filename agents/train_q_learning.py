"""
Boucle d'entraînement du Q-learning tabulaire sur RestaurantEnvSingle.

Produit :
    - une courbe de récompense par épisode (results/q_learning_curve.png)
    - une Q-table sauvegardée (results/q_table.npy)
    - quelques statistiques résumées dans la console
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ajout des dossiers parents au path pour pouvoir importer env/ et agents/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from restaurant_env import RestaurantEnvSingle
from q_learning import QLearningAgent


# ----------------------------------------------------------------------
# Hyperparamètres
# ----------------------------------------------------------------------
N_EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995


def smooth(values, window=100):
    """Moyenne glissante pour lisser la courbe de récompense."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    env = RestaurantEnvSingle(seed=0)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPS_START,
        epsilon_end=EPS_END,
        epsilon_decay=EPS_DECAY,
        seed=0,
    )

    rewards_per_episode = []

    print(f"Entrainement Q-learning : {N_EPISODES} episodes")
    print(f"alpha={ALPHA}, gamma={GAMMA}, eps {EPS_START}->{EPS_END} (decay={EPS_DECAY})")
    print("-" * 60)

    for episode in range(N_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(episode_reward)

        # Log toutes les 500 épisodes
        if (episode + 1) % 500 == 0:
            recent_mean = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1:>5} | "
                  f"reward (moy 100 derniers) = {recent_mean:>7.1f} | "
                  f"eps = {agent.epsilon:.3f} | "
                  f"taille Q-table = {len(agent.Q)}")

    print("-" * 60)
    print(f"Recompense finale (moy 100 derniers) : {np.mean(rewards_per_episode[-100:]):.1f}")
    print(f"Etats explores : {len(agent.Q)}")

    # ------------------------------------------------------------------
    # Sauvegarde de la courbe
    # ------------------------------------------------------------------
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    smoothed = smooth(rewards_per_episode, window=100)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, alpha=0.3, label="Recompense brute")
    plt.plot(range(len(smoothed)), smoothed, label="Moyenne glissante (100)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Recompense cumulee")
    plt.title("Q-learning tabulaire - Courbe d'apprentissage")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "q_learning_curve.png")
    plt.savefig(out_path, dpi=120)
    print(f"Courbe sauvegardee : {out_path}")

    # Sauvegarde de la Q-table sous forme de dict standard
    q_dict = {str(k): v for k, v in agent.Q.items()}
    np.save(os.path.join(results_dir, "q_table.npy"), q_dict, allow_pickle=True)
    print(f"Q-table sauvegardee : {os.path.join(results_dir, 'q_table.npy')}")


if __name__ == "__main__":
    main()
