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

- Run: `20260604_092001`
- Status: COMPLETE @ 50.3M
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

**Result (interim @ 10.5M):**
- GOOD: survival solved faster + stable (crash -> 0 at **3.1M** vs Exp 1's 5.77M, stays 0, no crash regression). Energy hoarding eliminated (A worked: `delta_energy_reward` ~0-11 vs Exp 1's +135). `ep_rew_mean` climbing to ~200, full-length episodes.
- FLAG: **no kills yet** (wez/victory/tracking all 0) - Exp 1 had its first kill by 7.34M. Not pressing to the merge: `avg_distance_ft` oscillating 16k-48k, never near 3000 ft. Possible over-correction by B (cut medium-range pull without the close-in kill taking over).
- Watch: if still no kills by ~15-20M, B over-corrected -> restore some medium-range pull or strengthen close-in wez/offensive.

**Result (final @ 50.3M):**

| Phase | Steps | ep_rew | crash | wez | victory | avg_dist |
|---|---|---|---|---|---|---|
| Cold start | 0-2.6M | -464 | -100 | 0 | 0 | 52k-101k ft |
| Survival solved | ~5.8M | -79 | 0 | 0 | 0 | 36k ft |
| Closing learning | 8-31M | 170-280 | 0 | 0 | 0 | 33k->9k ft |
| Plateau | 31-50M | 260-370 | 0 | 0 | 0 | 9k-35k ft |
| Only WEZ touch | 46.66M | — | 0 | 30 | 0 | — |

- A fix confirmed: `delta_energy_reward` capped at 13.5/ep (vs 135 in Exp 1), energy hoarding eliminated.
- B fix over-corrected: ONE WEZ touch in 50M steps (vs first kill at 7.34M in Exp 1). The `/27000` ramp removed the medium-range gradient entirely; agent settled at 9-30k ft with no incentive to close further.
- avg_tracking_time peaked at 0.083 (8 steps) — well below the 15-step kill threshold, and only once.
- Optimizer healthy: approx_kl 0.003-0.005, explained_variance 0.99. std crept 1.01→1.26 (policy getting noisier without converging on a kill strategy).

**Hypothesis verdict: WRONG.** The offensive reward was not strong or well-shaped enough to pull the agent into kill range. Removing the medium-range gradient (B) left no incentive between 6k-30k ft. The offensive reward design needs a rethink for Exp 3.

**Verdict:** B was the bug. A (delta_energy 0.1) is a keeper. Next experiment must restore or replace the medium-range offensive pull while keeping pressure all the way into the WEZ — the B ramp needs to saturate at a closer range (e.g. restore `/24000` from Exp 1, or add a separate WEZ-proximity bonus on top).

---

## Exp 3 — restore offensive closeness to `/24000`, 50M

- Run: `20260605_053643`
- Status: RUNNING (~7.3M / 50M)
- Date: 2026-06-05

**Goal:** verify that the parking in Exp 1 was caused by `delta_energy=1.0` (now fixed at 0.1), not by the `/24000` closeness shape. Restore Exp 1's offensive gradient to get kills back, while keeping Exp 2's energy fix to prevent parking.

**Hypothesis:** with `delta_energy=0.1` removing the parking attractor, the `/24000` shape (which produced kills at 7.34M in Exp 1) will again produce kills — and this time they will be stable past 8M because the agent has no incentive to park at 6k ft.

**Config = Exp 2 + one delta:**

| Knob | Exp 2 | Exp 3 |
|---|---|---|
| offensive closeness | `/27000` (ramps to WEZ, over-corrected) | **`/24000`** (saturates at ~6k ft, Exp 1 value) |

Unchanged from Exp 2: lr 3e-4, ent_coef 0.01, n_epochs 5, delta_energy weight 0.1, rate-based action penalty, WEZ 15 deg / 15-step hold, g_limit weight 1.0, 60 Hz / 4 substeps, 10 CPUs, budget 50M.

**Watch:**
- PRIMARY: do kills appear by ~8M? (Exp 1 got first kill at 7.34M)
- Does win rate HOLD and CLIMB past 8M? (Exp 1 decayed here; Exp 2 never got kills)
- `delta_energy_reward` stays small (≤13.5/ep) — confirm parking not returning
- `avg_tracking_time` trending up and not decaying
- `crash_penalty` stays 0 past 5M

**Decision gate:**
- Kills appear and consolidate past 8M → reward shaping solved; next = scale up or tighten WEZ gate.
- Kills appear but decay again past 8M → non-stationarity is the remaining issue → Exp 4 = frozen-opponent pool (league).
- No kills by 15M → `/24000` alone is not enough; need additional WEZ-proximity signal.

**Result:** (fill after run)
