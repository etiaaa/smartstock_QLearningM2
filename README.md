# SmartStock — Optimiser les stocks d'un restaurant par RL

> Pour le détail technique, les hyperparamètres, les résultats chiffrés et l'analyse des politiques apprises, voir [`WeExplainYou.md`](WeExplainYou.md).

## 1. Ce qu'on attend du projet

Le hackathon demande de construire un agent d'apprentissage par renforcement sur un environnement simulé, puis de comparer deux approches sur ce même environnement : un Q-learning tabulaire et un Deep Q-Network. Le DQN doit intégrer un replay buffer et un target network, et son utilisation doit être justifiée par un vrai besoin de généralisation.

L'évaluation porte sur cinq points : la maîtrise du problème RL, la qualité de la modélisation MDP, la propreté du code, l'analyse comparative des deux algorithmes, et la valeur métier dégagée.

## 2. Le sujet qu'on a choisi

Un agent qui apprend à gérer les commandes d'ingrédients frais d'un restaurant. Chaque jour, il décide combien commander de chaque produit pour le service du lendemain, en tenant compte des stocks restants et des dates de péremption. Son objectif : minimiser à la fois les ruptures et le gaspillage.

## 3. Pourquoi ce sujet

**C'est faisable.** L'environnement tient en quelques dizaines de lignes de NumPy. Pas de dataset externe, pas de moteur graphique, pas de physique.

**C'est riche en RL.** Commander trop, c'est jeter. Commander trop peu, c'est perdre des clients. La demande varie selon le jour de la semaine et la saison, donc aucune règle fixe ne fonctionne. C'est exactement le type de problème où un agent qui apprend prend tout son sens.

**C'est parfait pour comparer Q-learning et DQN.** En version mono-produit, l'espace d'états est petit : la Q-table suffit. En version multi-produits, l'espace explose et la Q-table devient impraticable. Le passage au DQN n'est donc pas un effet de mode, il est mécaniquement nécessaire — et c'est précisément ce que demande le sujet.

**Ça parle à tout le monde.** Le gaspillage alimentaire en restauration, c'est 1,5 million de tonnes par an en France selon l'ADEME. Les grandes chaînes ont des outils, les 200 000 restaurateurs indépendants n'en ont pas. Le besoin est réel et chiffrable.

## 4. Modélisation MDP

**État.** Stocks par produit, jours restants avant péremption de chaque lot, demande des derniers jours, jour de la semaine.

**Action.** Quantité à commander pour chaque produit (espace discret).

**Transition.** Le stock évolue avec la livraison, les ventes effectives (limitées par le stock disponible), et les pertes par péremption en FIFO.

**Récompense.** Marge des ventes, moins pénalité de rupture, moins pénalité de gaspillage. Les deux pénalités sont asymétriques et **différenciées par produit** : jeter du saumon coûte plus cher que jeter du pain, mais une rupture de pain frustre plus de clients qu'une rupture de saumon. Cette asymétrie est ce qui rend le problème intéressant à apprendre.

## 5. Stack et démarche

Python, NumPy, PyTorch, Matplotlib. Le DQN utilise replay buffer et target network, comme exigé.

On procède en deux étapes :
1. **Mono-produit + Q-learning tabulaire.** Baseline simple, interprétable, qui donne un premier point de référence.
2. **Multi-produits + DQN.** L'extension fait exploser l'espace d'états et justifie naturellement le passage au réseau de neurones.

Les deux agents seront comparés à deux baselines naïves (commande fixe, moyenne mobile) sur trois métriques : récompense cumulée, taux de gaspillage, taux de rupture.

## 6. Cible

Restaurateurs indépendants — bistrots, brasseries, restaurants de quartier. Pas les chaînes, qui sont déjà équipées.

## 7. Livrables Jour 1

- Environnement de simulation (mono- et multi-produits)
- Q-learning tabulaire entraîné et évalué
- DQN entraîné et évalué (replay buffer + target network)
- Courbes d'apprentissage et comparaison contre baselines
- Analyse Q-learning vs DQN et choix du modèle final
- Document de cadrage

## 8. Travail réalisé Jour 2

- Tests de robustesse : 5 scénarios de perturbation (bruit, hausse/baisse de demande, événements extrêmes, péremption réduite) pour évaluer la fiabilité des agents en conditions changeantes
- Grille comparative formelle Q-learning vs DQN sur 6 critères (espace d'états, convergence, stabilité, généralisation, complexité, applicabilité métier)
- Impact métier chiffré : comparaison DQN vs politique fixe en termes de ventes, gaspillage et ruptures
- Application web de démonstration (`python app_web.py` puis ouvrir `http://localhost:5000`) : dashboard interactif permettant de simuler la gestion d'un restaurant jour par jour avec 4 produits (Saumon, Légumes, Pain, Fromage), visualisation des stocks, recommandations de l'agent, et comparaison avec une gestion sans IA
- Préparation de la soutenance
