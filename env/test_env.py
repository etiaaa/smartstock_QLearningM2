"""
Test manuel de l'environnement RestaurantEnvSingle.

On fait tourner un épisode complet avec une politique aléatoire et on affiche
chaque journée pour vérifier que les chiffres ont du sens.
"""

import numpy as np
from restaurant_env import RestaurantEnvSingle


def main():
    env = RestaurantEnvSingle(seed=42)
    state = env.reset()

    print("=" * 70)
    print("TEST ENVIRONNEMENT - Politique aleatoire sur 30 jours")
    print("=" * 70)
    print(f"{'Jour':<5}{'JdS':<5}{'Action':<8}{'Demande':<9}{'Servi':<7}"
          f"{'Rupture':<9}{'Gaspille':<10}{'Stock':<7}{'Reward':<10}")
    print("-" * 70)

    rng = np.random.default_rng(0)
    force_max = False  # passer à True pour tester la politique "commander max"
    total_reward = 0
    total_wasted = 0
    total_stockout = 0
    total_served = 0

    done = False
    day = 0
    while not done:
        action = env.n_actions - 1 if force_max else rng.integers(0, env.n_actions)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_wasted += info["wasted"]
        total_stockout += info["stockout"]
        total_served += info["served"]

        print(f"{day:<5}{state[2]:<5}{action:<8}{info['demand']:<9}"
              f"{info['served']:<7}{info['stockout']:<9}{info['wasted']:<10}"
              f"{info['stock_after']:<7}{reward:<10.1f}")

        state = next_state
        day += 1

    print("-" * 70)
    print(f"Recompense totale : {total_reward:.1f}")
    print(f"Total servi       : {total_served}")
    print(f"Total ruptures    : {total_stockout}")
    print(f"Total gaspille    : {total_wasted}")
    print("=" * 70)


if __name__ == "__main__":
    main()
