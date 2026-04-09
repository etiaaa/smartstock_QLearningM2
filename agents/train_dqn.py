"""
Boucle d'entraînement du DQN sur RestaurantEnvMulti.

Produit :
    - une courbe de récompense par épisode (results/dqn_curve.png)
    - les poids du réseau Q (results/dqn_weights.pt)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from restaurant_env_multi import RestaurantEnvMulti
from dqn import DQNAgent


# ----------------------------------------------------------------------
# Hyperparamètres
# ----------------------------------------------------------------------
N_EPISODES = 1500
GAMMA = 0.90        # cohérent avec ce qu'on a appris en Q-learning : court terme
LR = 5e-4           # plus bas pour stabiliser
BUFFER_SIZE = 30000
BATCH_SIZE = 128    # batch plus gros = gradients plus stables
TARGET_UPDATE = 500 # mise à jour moins frequente du target net
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995   # decay un peu plus lent
HIDDEN = 128


def smooth(values, window=30):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    env = RestaurantEnvMulti(seed=0)
    agent = DQNAgent(
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        hidden=HIDDEN,
        gamma=GAMMA,
        lr=LR,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        epsilon_start=EPS_START,
        epsilon_end=EPS_END,
        epsilon_decay=EPS_DECAY,
        seed=0,
    )

    print(f"Entrainement DQN : {N_EPISODES} episodes")
    print(f"obs_dim={env.obs_dim} | n_actions={env.n_actions} | device={agent.device}")
    print(f"gamma={GAMMA} lr={LR} batch={BATCH_SIZE} target_update={TARGET_UPDATE}")
    print("-" * 70)

    rewards_per_episode = []
    losses = []

    for episode in range(N_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(episode_reward)
        losses.append(np.mean(episode_losses) if episode_losses else 0.0)

        if (episode + 1) % 50 == 0:
            recent = np.mean(rewards_per_episode[-30:])
            print(f"Episode {episode + 1:>4} | "
                  f"reward (moy 30) = {recent:>8.1f} | "
                  f"eps = {agent.epsilon:.3f} | "
                  f"buffer = {len(agent.buffer)}")

    print("-" * 70)
    print(f"Recompense finale (moy 30 derniers) : {np.mean(rewards_per_episode[-30:]):.1f}")

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    smoothed = smooth(rewards_per_episode, window=30)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(rewards_per_episode, alpha=0.3, label="Brut")
    axes[0].plot(range(len(smoothed)), smoothed, label="Moyenne 30", linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Recompense cumulee")
    axes[0].set_title("DQN - Courbe d'apprentissage")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(losses, alpha=0.6, color="orange")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss moyenne")
    axes[1].set_title("DQN - Loss par episode")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "dqn_curve.png")
    plt.savefig(out_path, dpi=120)
    print(f"Courbe sauvegardee : {out_path}")

    torch.save(agent.q_net.state_dict(), os.path.join(results_dir, "dqn_weights.pt"))
    print(f"Poids sauvegardes : {os.path.join(results_dir, 'dqn_weights.pt')}")


if __name__ == "__main__":
    main()
