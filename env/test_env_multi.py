"""
Test de l'environnement multi-produits + démonstration de l'explosion combinatoire.

Ce script existe pour deux raisons :
    1. Vérifier que l'environnement multi-produits tourne correctement
    2. Calculer la taille théorique d'une Q-table sur ce problème, et montrer
       qu'elle est impraticable -> ce qui justifie le passage au DQN.
"""

import numpy as np
from restaurant_env_multi import RestaurantEnvMulti


def main():
    env = RestaurantEnvMulti(seed=42)

    print("=" * 70)
    print("ENVIRONNEMENT MULTI-PRODUITS - Caracteristiques")
    print("=" * 70)
    print(f"Nombre de produits      : {env.n_products}")
    for p in env.products:
        print(f"  - {p['name']:<10} marge={p['margin']:<5} "
              f"shelf={p['shelf_life']} demande_base={p['demand_base']}")
    print(f"Dimension observation   : {env.obs_dim}")
    print(f"Nombre d'actions        : {env.n_actions}")

    # Calcul de la taille theorique d'une Q-table
    # Si on discrétise chaque dimension de l'état en 5 niveaux :
    #   stock (5) x age (5) x demande_j-1 (5) x demande_j-2 (5) x demande_j-3 (5)
    #   = 5^5 par produit = 3125
    # Pour 4 produits : 3125^4 ≈ 9.5e13
    # Et il faut multiplier par n_actions et par 7 (jour de la semaine)
    n_levels = 5
    states_per_product = n_levels ** 5
    n_states = (states_per_product ** env.n_products) * 7
    qtable_size = n_states * env.n_actions

    print()
    print("=" * 70)
    print("EXPLOSION COMBINATOIRE - Pourquoi une Q-table devient impraticable")
    print("=" * 70)
    print(f"Discretisation a {n_levels} niveaux par dimension :")
    print(f"  - Etats par produit       : {states_per_product:,}")
    print(f"  - Etats au total (4 prod) : {n_states:,}")
    print(f"  - Actions                 : {env.n_actions:,}")
    print(f"  - Cases dans la Q-table   : {qtable_size:,}")
    print()
    print(f"En memoire (float32)        : {qtable_size * 4 / 1e9:.2f} GB")
    print()
    print("Conclusion : impossible de stocker, encore moins d'apprendre.")
    print("On passe au DQN qui apprend une approximation Q(s,a) via un reseau.")
    print("=" * 70)

    # Test rapide : 30 jours avec actions aleatoires
    print("\nTest rapide : 1 episode avec actions aleatoires")
    print("-" * 70)
    state = env.reset()
    total_reward = 0
    rng = np.random.default_rng(0)
    while not env.done:
        action = rng.integers(0, env.n_actions)
        state, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Recompense totale : {total_reward:.1f}")
    print(f"Dimension etat    : {state.shape}")


if __name__ == "__main__":
    main()
