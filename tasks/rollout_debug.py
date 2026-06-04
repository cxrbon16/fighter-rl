"""
Rollout debugger: runs one episode, saves per-step JSON telemetry + PNG plots.

Usage (from repo root):
    python -m tasks.rollout_debug --model PATH [--task dogfight|curriculum] [--episodes N] [--out DIR] [--random]

    --model   Path to .zip model file (omit or use --random for random policy)
    --vecnorm Path to VecNormalize .pkl (curriculum models only)
    --task    dogfight (default) | curriculum
    --episodes Number of episodes to record (default: 1)
    --out     Output directory (default: tasks/debug_runs/<timestamp>)
    --random  Force random policy even if model is provided

Output per episode:
    <out>/ep<N>_telemetry.json   — per-step telemetry for both agents
    <out>/ep<N>_overview.png     — altitude, distance, speed, tracking, energy, top-down path
    <out>/ep<N>_rewards.png      — all reward components for agent_1
    <out>/ep<N>_summary.txt      — episode outcome and key stats
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(task: str, reward_weights=None):
    if task == "curriculum":
        from tasks.curriculum_dogfight.dogfight import SelfPlayDogfightEnv
        base = SelfPlayDogfightEnv(render_mode="debug", reward_weights=reward_weights)
        stack = 8
    else:
        from tasks.dogfight.dogfight import SelfPlayDogfightEnv
        base = SelfPlayDogfightEnv(render_mode="debug")
        stack = 4

    env = ss.black_death_v3(base)
    env = ss.frame_stack_v1(env, stack_size=stack)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    return env


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_episode(model, env, max_steps: int = 10800):
    obs = env.reset()
    log = {"agent_1": [], "agent_2": []}
    outcome = "truncation"

    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([env.action_space.sample() for _ in range(env.num_envs)])

        obs, rewards, dones, infos = env.step(action)

        for idx, agent in enumerate(["agent_1", "agent_2"]):
            info = infos[idx]
            rc = info.get("reward_components", {})
            log[agent].append({
                "step": step,
                "lat":          info.get("lat", 0.0),
                "lon":          info.get("lon", 0.0),
                "alt_m":        info.get("alt_m", 0.0),
                "alt_ft":       info.get("alt_ft", 0.0),
                "roll_deg":     info.get("roll_deg", 0.0),
                "pitch_deg":    info.get("pitch_deg", 0.0),
                "yaw_deg":      info.get("yaw_deg", 0.0),
                "airspeed_kts": info.get("airspeed_kts", 0.0),
                "airspeed_ms":  info.get("airspeed_ms", 0.0),
                "dist_ft":      info.get("dist_ft", 0.0),
                "energy":       info.get("energy", 0.0),
                "tracking_time": info.get("tracking_time", 0),
                "reward":       float(rewards[idx]),
                "reward_components": {k: float(v) for k, v in rc.items()},
            })

        if any(dones):
            # Determine outcome from last reward components
            rc1 = log["agent_1"][-1]["reward_components"]
            rc2 = log["agent_2"][-1]["reward_components"]
            if rc1.get("crash_penalty", 0) < 0 or rc2.get("crash_penalty", 0) < 0:
                outcome = "crash"
            elif rc1.get("out_of_bounds_penalty", 0) < 0 or rc2.get("out_of_bounds_penalty", 0) < 0:
                outcome = "out_of_bounds"
            elif rc1.get("victory_reward", 0) > 0:
                outcome = "agent_1_wins"
            elif rc2.get("victory_reward", 0) > 0:
                outcome = "agent_2_wins"
            break

    return log, outcome


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(log, out_path: str, outcome: str):
    a1, a2 = log["agent_1"], log["agent_2"]
    s1 = [f["step"] for f in a1]
    s2 = [f["step"] for f in a2]

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(f"Dogfight Rollout  |  outcome: {outcome}  |  {len(s1)} steps", fontsize=13, fontweight="bold")

    # --- Altitude ---
    ax = axes[0, 0]
    ax.plot(s1, [f["alt_m"] for f in a1], label="Agent 1", color="royalblue")
    ax.plot(s2, [f["alt_m"] for f in a2], label="Agent 2", color="tomato")
    ax.axhline(304.8, color="red", linestyle="--", lw=0.8, label="Crash floor (1000 ft)")
    ax.set_title("Altitude (m)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Inter-agent distance ---
    ax = axes[0, 1]
    dist_m = [f["dist_ft"] * 0.3048 for f in a1]
    ax.plot(s1, dist_m, color="purple")
    ax.axhline(3000 * 0.3048, color="orange", linestyle="--", lw=0.8, label="WEZ (3000 ft)")
    ax.axhline(150000 * 0.3048, color="red", linestyle="--", lw=0.8, label="OOB (150k ft)")
    ax.set_title("Inter-agent Distance (m)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Airspeed ---
    ax = axes[1, 0]
    ax.plot(s1, [f["airspeed_kts"] for f in a1], label="Agent 1", color="royalblue")
    ax.plot(s2, [f["airspeed_kts"] for f in a2], label="Agent 2", color="tomato")
    ax.set_title("Airspeed (kts)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- WEZ tracking time ---
    ax = axes[1, 1]
    ax.plot(s1, [f["tracking_time"] for f in a1], label="Agent 1", color="royalblue")
    ax.plot(s2, [f["tracking_time"] for f in a2], label="Agent 2", color="tomato")
    ax.axhline(20, color="green", linestyle="--", lw=0.8, label="Victory threshold")
    ax.set_title("WEZ Tracking Time")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Specific energy ---
    ax = axes[2, 0]
    ax.plot(s1, [f["energy"] for f in a1], label="Agent 1", color="royalblue")
    ax.plot(s2, [f["energy"] for f in a2], label="Agent 2", color="tomato")
    ax.set_title("Specific Energy")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Top-down flight path ---
    ax = axes[2, 1]
    lat0 = a1[0]["lat"]
    lon0 = a1[0]["lon"]
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * np.cos(np.radians(lat0))

    def to_xy(frames):
        xs = [(f["lon"] - lon0) * lon_to_m for f in frames]
        ys = [(f["lat"] - lat0) * lat_to_m for f in frames]
        return xs, ys

    x1, y1 = to_xy(a1)
    x2, y2 = to_xy(a2)
    ax.plot(x1, y1, color="royalblue", alpha=0.7, lw=0.8, label="Agent 1")
    ax.plot(x2, y2, color="tomato",    alpha=0.7, lw=0.8, label="Agent 2")
    ax.plot(x1[0], y1[0], "o", color="royalblue", ms=7)
    ax.plot(x2[0], y2[0], "o", color="tomato",    ms=7)
    ax.plot(x1[-1], y1[-1], "x", color="royalblue", ms=9, mew=2)
    ax.plot(x2[-1], y2[-1], "x", color="tomato",    ms=9, mew=2)
    ax.set_title("Top-down Path  (o=start, x=end)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rewards(log, out_path: str):
    a1 = log["agent_1"]
    rc_keys = list(a1[0].get("reward_components", {}).keys())
    if not rc_keys:
        return

    steps = [f["step"] for f in a1]
    n = len(rc_keys)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows))
    axes = axes.flatten()
    fig.suptitle("Reward Components — Agent 1", fontsize=12, fontweight="bold")

    for i, key in enumerate(rc_keys):
        vals = [f["reward_components"].get(key, 0.0) for f in a1]
        axes[i].plot(steps, vals, lw=0.8)
        total = sum(vals)
        axes[i].set_title(f"{key}  (total: {total:.1f})", fontsize=9)
        axes[i].set_xlabel("Step", fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def write_summary(log, outcome, out_path: str, model_path: str):
    a1, a2 = log["agent_1"], log["agent_2"]
    n_steps = len(a1)

    def stats(vals):
        arr = np.array(vals)
        return f"min={arr.min():.1f}  max={arr.max():.1f}  mean={arr.mean():.1f}"

    lines = [
        f"=== Episode Summary ===",
        f"Model:    {model_path}",
        f"Outcome:  {outcome}",
        f"Steps:    {n_steps}",
        "",
        "--- Agent 1 ---",
        f"  Altitude (m):     {stats([f['alt_m'] for f in a1])}",
        f"  Airspeed (kts):   {stats([f['airspeed_kts'] for f in a1])}",
        f"  Distance (m):     {stats([f['dist_ft']*0.3048 for f in a1])}",
        f"  Tracking time:    {stats([f['tracking_time'] for f in a1])}",
        f"  Specific energy:  {stats([f['energy'] for f in a1])}",
        f"  Total reward:     {sum(f['reward'] for f in a1):.2f}",
        "",
        "--- Agent 2 ---",
        f"  Altitude (m):     {stats([f['alt_m'] for f in a2])}",
        f"  Airspeed (kts):   {stats([f['airspeed_kts'] for f in a2])}",
        f"  Total reward:     {sum(f['reward'] for f in a2):.2f}",
        "",
        "--- Reward component totals (Agent 1) ---",
    ]
    if a1 and a1[0].get("reward_components"):
        rc_keys = list(a1[0]["reward_components"].keys())
        for key in rc_keys:
            total = sum(f["reward_components"].get(key, 0) for f in a1)
            lines.append(f"  {key:<30} {total:+.2f}")

    text = "\n".join(lines)
    Path(out_path).write_text(text)
    print(f"  Saved: {out_path}")
    print()
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dogfight rollout debugger")
    parser.add_argument("--model",    type=str, default=None, help="Path to .zip model")
    parser.add_argument("--vecnorm",  type=str, default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--task",     type=str, default="dogfight", choices=["dogfight", "curriculum"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out",      type=str, default=None)
    parser.add_argument("--random",   action="store_true", help="Force random policy")
    args = parser.parse_args()

    out_dir = args.out or f"tasks/debug_runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    env = make_env(args.task)

    model = None
    if args.model and not args.random:
        model_path = args.model
        if args.vecnorm and os.path.exists(args.vecnorm):
            env = VecNormalize.load(args.vecnorm, env)
            env.training = False
            env.norm_reward = False
            print(f"Loaded VecNormalize: {args.vecnorm}")
        model = PPO.load(model_path, env=env)
        print(f"Loaded model: {model_path}")
    else:
        model_path = "random_policy"
        print("Using random policy.")

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1} / {args.episodes} ---")
        log, outcome = run_episode(model, env)

        prefix = os.path.join(out_dir, f"ep{ep+1:02d}")
        json_path = f"{prefix}_telemetry.json"
        with open(json_path, "w") as f:
            json.dump(log, f)
        print(f"  Saved: {json_path}")

        plot_overview(log, f"{prefix}_overview.png", outcome)
        plot_rewards(log, f"{prefix}_rewards.png")
        write_summary(log, outcome, f"{prefix}_summary.txt", model_path)

    env.close()
    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
