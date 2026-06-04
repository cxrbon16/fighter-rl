# Dogfight RL — Experiment Log

Working memory for the self-play dogfight training experiments. One section per run,
newest at the bottom.

The metric that matters most after Exp 1: not "does it ever get a kill" (it does, by
~7M steps) but **does the win rate HOLD and CLIMB past 8M** instead of decaying.
Watch `metrics/avg_tracking_time`, `rewards/wez_reward`, `rewards/victory_reward`.

Shared baseline unless noted: standard self-play (`tasks/dogfight/ppo_training.py`),
F-16, 2-agent shared-policy self-play, PPO MLP pi=[256,128,64] / vf=[256,256,128] Tanh,
gamma 0.99, frame-stack 4, 64 parallel envs, VecNormalize(norm_obs=False,
norm_reward=True, clip 10), all-1.0 `DEFAULT_REWARD_WEIGHTS` except where listed.
TensorBoard at `tasks/dogfight/tensorboard/<run_id>/`, W&B project `f16-dogfight-selfplay`.

---

## Exp 1 — reward + optimizer fix, cold start

- Run: `20260604_080614`
- Status: killed at **13.6M / 30M** budget
- Date: 2026-06-04

**Goal:** validate that fixing the reward gradient + optimizer makes a from-scratch
agent actually dogfight (the old 3-phase curriculum never reached the WEZ once in 245M steps).

**Config (vs the old broken baseline):**

| Knob | Value |
|---|---|
| learning_rate | 3e-4 (was 1e-5) |
| ent_coef | 0.01 (was 0.1) |
| n_epochs | 5 |
| offensive reward | range-gated, `closeness = clip((30000-dist)/24000, 0, 1)` (saturates at 6000 ft) |
| closing reward | `clip(dist_delta/80, +-0.5)` |
| WEZ gate | dist < 3000 ft AND ATA < 15 deg, hold 15 steps |
| action penalty | magnitude `-0.03*(ail^2 + elev^2)` |
| delta_energy weight | 1.0 |
| sim | 120 Hz / 8 substeps |
| num_cpus | 8 |

**Result — four phases:**

| Phase | Steps | Summary | ep_rew | crash | wez | victory |
|---|---|---|---|---|---|---|
| A | <5M | cold start, crashes every episode, learning survival | -117 | -100 | 0 | 0 |
| B | 5-7M | survival solved (crash->0 @5.77M), closing hard | +262 | -25 | 0 | 0 |
| C | 7-8M | FIRST KILLS ever (first @7.34M), peak return | **+350** | 0 | 32 | 100 |
| D | 8-13.6M | regression: kills decayed, energy-hoard/park, crashes returned | 296 | -36 | 3 | 9 |

Milestones: crash -> 0 at 5.77M; first WEZ + victory at 7.34M.

**Verdict:**
- PROVEN: the reward + optimizer fix works. From scratch: survival -> closing -> kills by 7.3M. The old curriculum never reached the WEZ in 245M steps.
- PROBLEM 1 (parking / energy-hoarding): `delta_energy` weight 1.0 (+135/ep) rewarded hoarding; offensive reward saturated at 6000 ft so there was no gradient into the WEZ. Agent parked nose-on at ~6000 ft. -> fixed in Exp 2 (A/B).
- PROBLEM 2 (no consolidation): kills peaked ~7-8M then DESTABILIZED (victory fired once at 10.5M then ~0; crashes returned, -100 in 4 of last 6 points; ep_rew drifted 350 -> 276). Likely self-play non-stationarity (shared policy, no opponent pool -> moving target). -> deferred to a later experiment (frozen-opponent league).

---

## Exp 2 — A/B reward rebalance (latest fixes), 50M

- Run: (fill in run_id at launch)
- Status: PLANNED
- Date: 2026-06-04

**Goal:** test whether removing the parking / energy-hoard attractor (A/B) lets the
kills **consolidate past 8M** instead of regressing as in Exp 1. Isolates the A/B
effect; the opponent league is intentionally deferred to keep one variable at a time.

**Hypothesis:** with `delta_energy` cut and the offensive reward ramping all the way
into the WEZ, parking at 6000 ft is no longer reward-maximal, so the agent commits to
the kill and the win rate holds / climbs past 8M.

**Config = Exp 1 + these deltas (all committed):**

| Knob | Exp 1 | Exp 2 |
|---|---|---|
| delta_energy weight (A) | 1.0 | **0.1** |
| offensive closeness (B) | `/24000` (flat to WEZ) | `/27000` (ramps to full at ~3000 ft) |
| action penalty | magnitude | **rate-based** (taxes jitter, not sustained turns) |
| sim | 120 Hz / 8 sub | **60 Hz / 4 sub** (~1.5x faster, same decision cadence) |
| num_cpus | 8 | 10 |
| budget | 30M (killed @13.6M) | **50M** |

Unchanged from Exp 1: lr 3e-4, ent_coef 0.01, n_epochs 5, WEZ 15 deg / 15-step hold,
g_limit weight 1.0, all other weights 1.0.

**Launch:** `python -m tasks.dogfight.ppo_training`

**Watch:**
- PRIMARY: does win rate (`wez_reward` / `victory_reward`) HOLD and CLIMB past 8M? (Exp 1 decayed here)
- `avg_tracking_time` trending up (Exp 1 peaked 0.18)
- `crash_penalty` staying ~0 past 8M (Exp 1 regressed to -36)
- `delta_energy_reward` now small (cap ~+13.5/ep) and not dominating
- `avg_distance_ft` settling below 3000 ft when in the WEZ

**Decision gate:**
- Kills consolidate -> A/B was the main cause; next push harder (tighten WEZ gate 15->10 deg, or scale up).
- Still destabilizes late -> it is self-play non-stationarity -> Exp 3 = frozen-opponent pool (league).

**Result:** (fill after run)

**Verdict:** (fill after run)
