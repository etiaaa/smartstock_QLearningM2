"""
SmartStock — Application de démonstration

Interface desktop qui simule un restaurant géré par notre agent DQN.
Le restaurateur voit jour par jour les recommandations de l'agent,
l'état du stock, le gaspillage évité et la marge générée.

Lance avec : python app_smartstock.py
"""

import sys
import os
import tkinter as tk
from tkinter import ttk
import numpy as np
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from restaurant_env_multi import RestaurantEnvMulti
from dqn import DQNAgent
import torch


DAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
PRODUCTS = ["Saumon", "Legumes", "Pain"]
PRODUCT_COLORS = ["#e74c3c", "#2ecc71", "#f39c12"]
BG = "#1a1a2e"
BG_CARD = "#16213e"
FG = "#eaeaea"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
SUCCESS = "#2ecc71"
WARNING = "#f39c12"
DANGER = "#e74c3c"


class SmartStockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartStock — Gestion intelligente des stocks")
        self.root.geometry("1100x720")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.agent = None
        self.env = None
        self.state = None
        self.day = 0
        self.done = False
        self.history = []
        self.baseline_history = []

        # Totaux
        self.total_reward = 0
        self.total_waste = 0
        self.total_stockout = 0
        self.total_served = 0
        self.total_demand = 0
        self.baseline_reward = 0
        self.baseline_waste = 0
        self.baseline_stockout = 0

        self._build_ui()
        self._load_model()

    # ------------------------------------------------------------------
    # Chargement du modèle
    # ------------------------------------------------------------------
    def _load_model(self):
        self.status_var.set("Chargement du modele DQN...")
        self.root.update()

        self.env = RestaurantEnvMulti(seed=42)
        self.agent = DQNAgent(
            obs_dim=self.env.obs_dim,
            n_actions=self.env.n_actions,
            hidden=128, gamma=0.90, lr=5e-4,
            buffer_size=1, batch_size=1,
            target_update_freq=1,
            seed=0,
        )

        weights_path = os.path.join(ROOT, "results", "dqn_weights.pt")
        if os.path.exists(weights_path):
            self.agent.q_net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.agent.q_net.eval()
            self.agent.epsilon = 0.0
            self.status_var.set("Modele charge — Pret")
        else:
            self.status_var.set("Poids non trouves — Entrainement en cours...")
            self.root.update()
            self._train_agent()
            self.status_var.set("Entrainement termine — Pret")

        self._reset_simulation()

    def _train_agent(self):
        env = RestaurantEnvMulti(seed=0)
        self.agent = DQNAgent(
            obs_dim=env.obs_dim, n_actions=env.n_actions,
            hidden=128, gamma=0.90, lr=5e-4,
            buffer_size=30000, batch_size=128, target_update_freq=500,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, seed=0,
        )
        for ep in range(1500):
            s = env.reset()
            d = False
            while not d:
                a = self.agent.choose_action(s)
                ns, r, d, _ = env.step(a)
                self.agent.remember(s, a, r, ns, d)
                self.agent.train_step()
                s = ns
            self.agent.decay_epsilon()
            if (ep + 1) % 100 == 0:
                self.status_var.set(f"Entrainement... {ep+1}/1500")
                self.root.update()
        self.agent.epsilon = 0.0

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Titre
        title_frame = tk.Frame(self.root, bg=BG)
        title_frame.pack(fill="x", padx=20, pady=(15, 5))
        tk.Label(title_frame, text="SmartStock", font=("Helvetica", 24, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(title_frame, text="  Gestion intelligente des stocks par RL",
                 font=("Helvetica", 12), fg="#888", bg=BG).pack(side="left", pady=(8, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Initialisation...")
        tk.Label(title_frame, textvariable=self.status_var, font=("Helvetica", 10),
                 fg=WARNING, bg=BG).pack(side="right")

        # Corps principal
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # Colonne gauche : état du jour
        left = tk.Frame(main, bg=BG, width=360)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        self._build_day_card(left)
        self._build_stock_card(left)
        self._build_recommendation_card(left)

        # Colonne droite : résultats
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_scores_card(right)
        self._build_log_card(right)
        self._build_comparison_card(right)

        # Boutons
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.btn_next = tk.Button(btn_frame, text="Jour suivant", font=("Helvetica", 12, "bold"),
                                   bg=ACCENT, fg="white", relief="flat", padx=20, pady=8,
                                   command=self._next_day)
        self.btn_next.pack(side="left", padx=(0, 10))

        self.btn_run = tk.Button(btn_frame, text="Simuler le mois entier", font=("Helvetica", 11),
                                  bg=ACCENT2, fg="white", relief="flat", padx=20, pady=8,
                                  command=self._run_month)
        self.btn_run.pack(side="left", padx=(0, 10))

        self.btn_reset = tk.Button(btn_frame, text="Nouveau mois", font=("Helvetica", 11),
                                    bg="#333", fg="white", relief="flat", padx=20, pady=8,
                                    command=self._reset_simulation)
        self.btn_reset.pack(side="left")

    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=BG_CARD, relief="flat", bd=0)
        frame.pack(fill="x", pady=(0, 8))
        tk.Label(frame, text=title, font=("Helvetica", 11, "bold"),
                 fg=ACCENT, bg=BG_CARD, anchor="w").pack(fill="x", padx=12, pady=(8, 4))
        inner = tk.Frame(frame, bg=BG_CARD)
        inner.pack(fill="x", padx=12, pady=(0, 10))
        return inner

    def _build_day_card(self, parent):
        inner = self._card(parent, "Aujourd'hui")
        self.day_label = tk.Label(inner, text="Jour 1 — Lundi", font=("Helvetica", 16, "bold"),
                                   fg=FG, bg=BG_CARD)
        self.day_label.pack(anchor="w")
        self.episode_label = tk.Label(inner, text="Episode en cours", font=("Helvetica", 10),
                                       fg="#888", bg=BG_CARD)
        self.episode_label.pack(anchor="w")

    def _build_stock_card(self, parent):
        inner = self._card(parent, "Stock actuel")
        self.stock_labels = []
        for i, prod in enumerate(PRODUCTS):
            row = tk.Frame(inner, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=prod, font=("Helvetica", 11), fg=PRODUCT_COLORS[i],
                     bg=BG_CARD, width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="0 unites", font=("Helvetica", 11, "bold"),
                           fg=FG, bg=BG_CARD)
            lbl.pack(side="left", padx=(10, 0))
            bar = tk.Canvas(row, width=120, height=14, bg="#333", highlightthickness=0)
            bar.pack(side="right")
            self.stock_labels.append((lbl, bar, PRODUCT_COLORS[i]))

    def _build_recommendation_card(self, parent):
        inner = self._card(parent, "Recommandation de l'agent")
        self.reco_labels = []
        for i, prod in enumerate(PRODUCTS):
            row = tk.Frame(inner, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=prod, font=("Helvetica", 11), fg=PRODUCT_COLORS[i],
                     bg=BG_CARD, width=10, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="Commander 0", font=("Helvetica", 11, "bold"),
                           fg=SUCCESS, bg=BG_CARD)
            lbl.pack(side="left", padx=(10, 0))
            self.reco_labels.append(lbl)

    def _build_scores_card(self, parent):
        inner = self._card(parent, "Bilan du mois (agent DQN)")
        grid = tk.Frame(inner, bg=BG_CARD)
        grid.pack(fill="x")

        self.score_vars = {}
        metrics = [
            ("Marge", SUCCESS), ("Gaspille", DANGER),
            ("Ruptures", WARNING), ("Clients servis", FG),
        ]
        for i, (name, color) in enumerate(metrics):
            col = tk.Frame(grid, bg=BG_CARD)
            col.pack(side="left", expand=True, fill="x")
            var = tk.StringVar(value="0")
            tk.Label(col, textvariable=var, font=("Helvetica", 18, "bold"),
                     fg=color, bg=BG_CARD).pack()
            tk.Label(col, text=name, font=("Helvetica", 9), fg="#888", bg=BG_CARD).pack()
            self.score_vars[name] = var

    def _build_log_card(self, parent):
        inner = self._card(parent, "Journal des operations")
        self.log_text = tk.Text(inner, font=("Consolas", 9), bg="#0d1117", fg=FG,
                                 height=12, relief="flat", state="disabled")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("header", foreground=ACCENT)
        self.log_text.tag_config("waste", foreground=DANGER)
        self.log_text.tag_config("stockout", foreground=WARNING)
        self.log_text.tag_config("good", foreground=SUCCESS)

    def _build_comparison_card(self, parent):
        inner = self._card(parent, "DQN vs Politique fixe (5/produit/jour)")
        self.comparison_var = tk.StringVar(value="Lancez la simulation pour voir la comparaison")
        tk.Label(inner, textvariable=self.comparison_var, font=("Helvetica", 10),
                 fg=FG, bg=BG_CARD, wraplength=650, justify="left").pack(anchor="w")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _reset_simulation(self):
        seed = np.random.randint(0, 10000)
        self.env = RestaurantEnvMulti(seed=seed)
        self.state = self.env.reset()
        self.day = 0
        self.done = False
        self.history = []
        self.total_reward = 0
        self.total_waste = 0
        self.total_stockout = 0
        self.total_served = 0
        self.total_demand = 0

        # Baseline : politique fixe 5/produit/jour sur le même seed
        env_b = RestaurantEnvMulti(seed=seed)
        s = env_b.reset()
        fixed_action = 5 + 5 * 9 + 5 * 81
        self.baseline_reward = 0
        self.baseline_waste = 0
        self.baseline_stockout = 0
        d = False
        while not d:
            s, r, d, info = env_b.step(fixed_action)
            self.baseline_reward += r
            for p in info["per_product"]:
                self.baseline_waste += p["wasted"]
                self.baseline_stockout += p["stockout"]

        self._update_display()
        self._clear_log()
        self._log("Nouveau mois demarre. L'agent est pret.\n", "header")
        self.btn_next.config(state="normal")
        self.btn_run.config(state="normal")
        self.comparison_var.set("Lancez la simulation pour voir la comparaison")
        self.score_vars["Marge"].set("0")
        self.score_vars["Gaspille"].set("0")
        self.score_vars["Ruptures"].set("0")
        self.score_vars["Clients servis"].set("0")

    def _next_day(self):
        if self.done or self.agent is None:
            return

        action = self.agent.choose_action(self.state)
        quantities = self.env.decode_action(action)

        next_state, reward, done, info = self.env.step(action)

        # Log
        day_name = DAYS_FR[self.env.day_of_week - 1 if self.env.day_of_week > 0 else 6]
        self._log(f"--- Jour {self.day + 1} ({day_name}) ---\n", "header")
        for i, prod in enumerate(PRODUCTS):
            p = info["per_product"][i]
            self._log(f"  {prod}: commande {quantities[i]}, demande {p['demand']}, servi {p['served']}", "good")
            if p["wasted"] > 0:
                self._log(f", jete {p['wasted']}", "waste")
            if p["stockout"] > 0:
                self._log(f", rupture {p['stockout']}", "stockout")
            self._log("\n")

        self._log(f"  Reward du jour : {reward:.1f}\n\n")

        self.total_reward += reward
        for p in info["per_product"]:
            self.total_waste += p["wasted"]
            self.total_stockout += p["stockout"]
            self.total_served += p["served"]
            self.total_demand += p["demand"]

        self.state = next_state
        self.day += 1
        self.done = done

        self._update_display()
        self._update_recommendation()
        self._update_scores()

        if self.done:
            self._end_simulation()

    def _run_month(self):
        while not self.done:
            self._next_day()
            self.root.update()

    def _end_simulation(self):
        self.btn_next.config(state="disabled")
        self.btn_run.config(state="disabled")
        self._log("=== FIN DU MOIS ===\n", "header")
        self._log(f"Marge totale : {self.total_reward:.0f}\n", "good")
        self._log(f"Gaspillage : {self.total_waste} unites\n", "waste")
        self._log(f"Ruptures : {self.total_stockout} unites\n", "stockout")
        self.status_var.set("Mois termine — Cliquez 'Nouveau mois' pour relancer")

        # Comparaison
        diff_reward = self.total_reward - self.baseline_reward
        diff_waste = self.baseline_waste - self.total_waste
        diff_stockout = self.baseline_stockout - self.total_stockout
        sign_r = "+" if diff_reward >= 0 else ""
        sign_w = "+" if diff_waste >= 0 else ""
        sign_s = "+" if diff_stockout >= 0 else ""
        self.comparison_var.set(
            f"DQN : {self.total_reward:.0f} de marge | "
            f"{self.total_waste} gaspilles | {self.total_stockout} ruptures\n"
            f"Fixe(5) : {self.baseline_reward:.0f} de marge | "
            f"{self.baseline_waste} gaspilles | {self.baseline_stockout} ruptures\n"
            f"Difference : {sign_r}{diff_reward:.0f} marge | "
            f"{sign_w}{diff_waste} gaspillage | {sign_s}{diff_stockout} ruptures"
        )

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------
    def _update_display(self):
        day_name = DAYS_FR[self.env.day_of_week]
        self.day_label.config(text=f"Jour {self.day + 1} — {day_name}")
        self.episode_label.config(text=f"Jour {self.day + 1}/30")

        for i in range(len(PRODUCTS)):
            stock = sum(qty for qty, _ in self.env.stocks[i])
            max_s = self.env.max_stock
            self.stock_labels[i][0].config(text=f"{stock} unites")
            bar = self.stock_labels[i][1]
            color = self.stock_labels[i][2]
            bar.delete("all")
            w = int(120 * stock / max(1, max_s))
            bar.create_rectangle(0, 0, w, 14, fill=color, outline="")

    def _update_recommendation(self):
        if self.done or self.agent is None:
            return
        action = self.agent.choose_action(self.state)
        quantities = self.env.decode_action(action)
        for i, lbl in enumerate(self.reco_labels):
            lbl.config(text=f"Commander {quantities[i]}")

    def _update_scores(self):
        self.score_vars["Marge"].set(f"{self.total_reward:.0f}")
        self.score_vars["Gaspille"].set(f"{self.total_waste}")
        self.score_vars["Ruptures"].set(f"{self.total_stockout}")
        self.score_vars["Clients servis"].set(f"{self.total_served}")

    def _log(self, text, tag=None):
        self.log_text.config(state="normal")
        if tag:
            self.log_text.insert("end", text, tag)
        else:
            self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")


def main():
    root = tk.Tk()
    app = SmartStockApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
