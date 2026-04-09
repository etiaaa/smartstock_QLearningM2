"""
SmartStock — Application web de démonstration.

Lance avec : python app_web.py
Ouvre ensuite : http://localhost:5000
"""

import sys
import os
import json
import numpy as np
from flask import Flask, render_template, jsonify

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from restaurant_env_demo import RestaurantEnvDemo
DEMO_PRODUCTS = RestaurantEnvDemo().products
from dqn import DQNAgent
import torch

app = Flask(__name__)


def to_native(obj):
    """Convertit les types NumPy en types Python natifs pour la sérialisation JSON."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

DAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

# État global
agent = None
env = None
state = None
sim_data = {}


def train_demo_agent():
    print("Entrainement de l'agent DQN (5 produits, ~2 min)...")
    e = RestaurantEnvDemo(seed=0)
    ag = DQNAgent(
        obs_dim=e.obs_dim, n_actions=e.n_actions,
        hidden=256, gamma=0.90, lr=3e-4,
        buffer_size=30000, batch_size=128, target_update_freq=300,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, seed=0,
    )
    for ep in range(1200):
        s = e.reset(); d = False
        while not d:
            a = ag.choose_action(s)
            ns, r, d, _ = e.step(a)
            ag.remember(s, a, r, ns, d)
            ag.train_step()
            s = ns
        ag.decay_epsilon()
        if (ep + 1) % 200 == 0:
            print(f"  Episode {ep+1}/1200")
    ag.epsilon = 0.0
    weights_path = os.path.join(ROOT, "results", "dqn_demo_weights.pt")
    torch.save(ag.q_net.state_dict(), weights_path)
    print(f"  Poids sauvegardes : {weights_path}")
    return ag


def load_agent():
    global agent
    e = RestaurantEnvDemo(seed=0)
    agent = DQNAgent(
        obs_dim=e.obs_dim, n_actions=e.n_actions,
        hidden=256, gamma=0.90, lr=3e-4,
        buffer_size=1, batch_size=1, target_update_freq=1, seed=0,
    )
    weights_path = os.path.join(ROOT, "results", "dqn_demo_weights.pt")
    if os.path.exists(weights_path):
        agent.q_net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        agent.q_net.eval()
        agent.epsilon = 0.0
        print("Agent charge depuis les poids sauvegardes.")
    else:
        agent = train_demo_agent()


def reset_sim():
    global env, state, sim_data
    seed = int(np.random.randint(0, 10000))
    env = RestaurantEnvDemo(seed=seed)
    state = env.reset()
    sim_data = {
        "day": 0, "done": False, "seed": seed,
        "total_reward": 0, "total_waste": 0, "total_stockout": 0,
        "total_served": 0, "total_demand": 0,
        "history": [],
        "rewards_per_day": [],
        "waste_per_day": [],
        "stockout_per_day": [],
    }
    # Baseline fixe sur le même seed
    env_b = RestaurantEnvDemo(seed=seed)
    s = env_b.reset()
    base_act = sum((2 * ((env_b.max_order + 1) ** i)) for i in range(env_b.n_products))
    br, bw, bs = 0, 0, 0
    d = False
    while not d:
        s, r, d, info = env_b.step(base_act)
        br += r
        for p in info["per_product"]:
            bw += p["wasted"]
            bs += p["stockout"]
    sim_data["baseline"] = {"reward": br, "waste": bw, "stockout": bs}


@app.route("/")
def index():
    products_info = []
    for p in DEMO_PRODUCTS:
        products_info.append({
            "name": p["name"], "emoji": p["emoji"],
            "shelf_life": p["shelf_life"],
            "margin": p["margin"],
        })
    return render_template("index.html", products=products_info)


@app.route("/api/reset")
def api_reset():
    reset_sim()
    return jsonify(to_native(get_state_data()))


@app.route("/api/step")
def api_step():
    global state, sim_data
    if sim_data["done"]:
        return jsonify({"error": "Episode termine"})

    action = agent.choose_action(state)
    quantities = env.decode_action(action)
    next_state, reward, done, info = env.step(action)

    day_data = {
        "day": sim_data["day"] + 1,
        "day_name": DAYS_FR[env.day_of_week - 1 if env.day_of_week > 0 else 6],
        "reward": round(reward, 1),
        "products": [],
    }
    day_waste = 0
    day_stockout = 0
    for i, p in enumerate(info["per_product"]):
        day_data["products"].append({
            "name": p["name"],
            "ordered": quantities[i],
            "demand": p["demand"],
            "served": p["served"],
            "stockout": p["stockout"],
            "wasted": p["wasted"],
            "stock_after": p["stock_after"],
        })
        sim_data["total_waste"] += p["wasted"]
        sim_data["total_stockout"] += p["stockout"]
        sim_data["total_served"] += p["served"]
        sim_data["total_demand"] += p["demand"]
        day_waste += p["wasted"]
        day_stockout += p["stockout"]

    sim_data["total_reward"] += reward
    sim_data["day"] += 1
    sim_data["done"] = done
    sim_data["history"].append(day_data)
    sim_data["rewards_per_day"].append(round(reward, 1))
    sim_data["waste_per_day"].append(day_waste)
    sim_data["stockout_per_day"].append(day_stockout)

    state = next_state
    return jsonify(to_native(get_state_data()))


@app.route("/api/run_all")
def api_run_all():
    global state, sim_data
    while not sim_data["done"]:
        action = agent.choose_action(state)
        quantities = env.decode_action(action)
        next_state, reward, done, info = env.step(action)
        day_data = {
            "day": sim_data["day"] + 1,
            "day_name": DAYS_FR[env.day_of_week - 1 if env.day_of_week > 0 else 6],
            "reward": round(reward, 1),
            "products": [],
        }
        day_waste = 0
        day_stockout = 0
        for i, p in enumerate(info["per_product"]):
            day_data["products"].append({
                "name": p["name"], "ordered": quantities[i],
                "demand": p["demand"], "served": p["served"],
                "stockout": p["stockout"], "wasted": p["wasted"],
                "stock_after": p["stock_after"],
            })
            sim_data["total_waste"] += p["wasted"]
            sim_data["total_stockout"] += p["stockout"]
            sim_data["total_served"] += p["served"]
            sim_data["total_demand"] += p["demand"]
            day_waste += p["wasted"]
            day_stockout += p["stockout"]
        sim_data["total_reward"] += reward
        sim_data["day"] += 1
        sim_data["done"] = done
        sim_data["history"].append(day_data)
        sim_data["rewards_per_day"].append(round(reward, 1))
        sim_data["waste_per_day"].append(day_waste)
        sim_data["stockout_per_day"].append(day_stockout)
        state = next_state
    return jsonify(to_native(get_state_data()))


def get_state_data():
    stocks = []
    for i, p in enumerate(DEMO_PRODUCTS):
        stock = sum(qty for qty, _ in env.stocks[i])
        stocks.append({"name": p["name"], "stock": stock, "max": env.max_stock})

    reco = []
    if not sim_data["done"] and agent is not None:
        action = agent.choose_action(state)
        quantities = env.decode_action(action)
        for i, p in enumerate(DEMO_PRODUCTS):
            reco.append({"name": p["name"], "qty": quantities[i]})

    pct_served = round(100 * sim_data["total_served"] / max(1, sim_data["total_demand"]), 1)

    return {
        "day": sim_data["day"],
        "day_name": DAYS_FR[env.day_of_week],
        "done": sim_data["done"],
        "stocks": stocks,
        "recommendation": reco,
        "total_reward": round(sim_data["total_reward"], 0),
        "total_waste": sim_data["total_waste"],
        "total_stockout": sim_data["total_stockout"],
        "total_served": sim_data["total_served"],
        "pct_served": pct_served,
        "rewards_per_day": sim_data["rewards_per_day"],
        "waste_per_day": sim_data["waste_per_day"],
        "stockout_per_day": sim_data["stockout_per_day"],
        "baseline": sim_data.get("baseline", {}),
        "last_day": sim_data["history"][-1] if sim_data["history"] else None,
    }


if __name__ == "__main__":
    load_agent()
    reset_sim()
    print("\n  SmartStock demarre sur http://localhost:5000\n")
    app.run(debug=False, port=5000, use_reloader=False)
