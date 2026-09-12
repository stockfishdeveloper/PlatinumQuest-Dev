# HANDOFF — Edge-Avoidance Reward Shaping

**Date**: 2026-09-11
**Status**: Design complete + edge detector implemented + user has verified 3D visualization. **Reward-shaping code NOT yet wired into `train_ppo.py`.** Next agent picks up here.

---

## TL;DR — What to do next

1. Load pre-computed edge points from `ml_agent/edge_maps/edges_<mapname>.npy` at PPO server startup
2. Build a KDTree from those points for fast nearest-edge queries
3. Add potential-based edge-avoidance shaping to `PPOServer.compute_reward()` in [train_ppo.py](train_ppo.py)
4. Test on KingOfTheMarble (KOTM). Watch that gems don't crash and OOB rate drops
5. If it works on KOTM, run edge detector on Prophetic (already done, files in `edge_maps/`) and confirm transfer

---

## The problem (short version)

Model on KingOfTheMarble plateaued at ~30 gems/game. Main failure mode: **the marble walks straight into interior holes** — not overshooting past gems and falling, but literally rolling into voids like it doesn't perceive them.

**Diagnostic** ([generate_policy_diagnostic.py](generate_policy_diagnostic.py)) confirmed:
- Actor direction arrows point at gems THROUGH voids (no curving around holes)
- Critic V(s) shows no dark zones at void edges
- Model is genuinely oblivious to hole geometry

**Root cause** (per external AI consultation, confirmed our analysis): **credit-assignment gap**. Current freefall penalty fires DURING the fall (30-100 frames after the bad decision). GAE discount at γ=0.999, λ=0.95 → `0.949^50 = 7.3%` credit reaches approach decisions. The gradient signal never gets to the "steer away from hole" decision points before the fall.

---

## The solution (potential-based edge shaping)

**Precompute edge points**: For any map, parse `.dif` geometry → find floor surfaces → rasterize per Z-level → detect cells with non-walkable neighbors → those are edge points. **Already implemented in [generate_edge_map.py](generate_edge_map.py)**.

**At runtime**, load edge points into KDTree. Per step:

```python
# In PPOServer.__init__:
from scipy.spatial import cKDTree
edge_pts = np.load('edge_maps/edges_KingOfTheMarble_Hunt.npy')  # shape (N, 3)
self.edge_kdtree = cKDTree(edge_pts)
self.EDGE_SHAPING_COEF = 50.0      # tune
self.EDGE_SHAPING_RANGE = 3.0      # decay distance in world units
self.last_edge_phi = 0.0
self.last_edge_dist = 999.0         # for respawn detection

# In compute_reward(), add BEFORE the return:
# 6. Edge-proximity shaping (potential-based, Bellman-consistent)
# Gives dense per-step gradient during approach to edges, refunds on retreat.
if len(raw_obs) >= 3:
    # raw_obs[0:3] is camera-relative marble position (yaw locked at 0 -> world pos)
    marble_xyz = raw_obs[0:3]
    d, _ = self.edge_kdtree.query(marble_xyz)
    phi_new = -self.EDGE_SHAPING_COEF * math.exp(-d / self.EDGE_SHAPING_RANGE)

    # Detect OOB/respawn jumps to avoid rewarding teleports
    if oob or abs(d - self.last_edge_dist) > 10.0:
        # Reset — don't credit the position jump
        shaping = 0.0
    else:
        shaping = phi_new - self.last_edge_phi

    reward += shaping
    self.game_edge_shaping_sum += shaping   # diagnostic accumulator
    self.rollout_edge_shaping += shaping    # for Upd log line
    self.last_edge_phi = phi_new
    self.last_edge_dist = d
```

**Also add** to `__init__`:
- `self.game_edge_shaping_sum = 0.0` (and reset on game-end log line, both spots)
- `self.rollout_edge_shaping = 0.0` (reset in rollout counter reset section)
- `self.EDGE_SHAPING_COEF`, `self.EDGE_SHAPING_RANGE` config constants

**Also add** to `[GAME END]` log line and `Upd N` log line:
- `edge_shape: {game_edge_shaping_sum:.0f}` in GAME END
- `ES=(-{-rollout_edge_shaping:.0f})` in Upd line

---

## Why potential-based (not raw distance)

Raw penalty per step (naive `reward -= exp(-d/r)`) would train the marble to stay far from ALL edges forever — but gems are often NEAR edges (KOTM has yellow gems at corner tab tips). The marble would refuse to approach edge gems.

Potential-based form `Φ(new) - Φ(old)`:
- Moving toward edge = negative shaping
- Moving away from edge = positive shaping
- Cycle back to same state = net zero
- Trajectory ending in OOB = accumulated negative NEVER refunded → attributable to approach

Bellman-consistent → doesn't distort optimal policy. Only costs the marble net reward when it ACTUALLY falls off.

---

## Numbers we expect (KOTM, with default coef=50, range=3)

Per-step shaping during approach at various distances:
- d=10 → -0.05 raw
- d=5 → -0.32
- d=3 → -0.62
- d=1 → -1.21

Full approach (d=10 → 0) accumulates ~-50 raw before OOB. Falls of length ~100 frames add capped freefall (100 × 0.75 = -75) plus OOB (-25). Grand total per hole-fall ~-150 raw. Gem reward is +200. Grab-and-dive still nets +50 (was +125 before shaping), meaningfully worse.

**Watch for**: brake_rate collapse to 60%+, pickup speed dropping below 6.0, or gems crashing below 20. Any of those = coef too high, drop to 20-30.

---

## The critical rule (must follow)

**Solutions must be map-agnostic.** The user has rejected raycasts and edge-shaping proposals that were flat-map specific. The edge-detection approach IS map-agnostic because:
- `generate_edge_map.py` parses ANY `.dif` (uses vendored `hxDif.py`)
- Handles multi-level maps (verified on Prophetic: 17 Z-levels, 17,501 edge points)
- No per-map hand tuning
- Deployment on a new map = one command: `python generate_edge_map.py <mapname>`

**Do not** propose obs changes or oracle behaviors that require per-map calibration. See [feedback_must_transfer_to_other_maps.md](../.claude-memory/feedback_must_transfer_to_other_maps.md) in the user's persistent memory.

---

## Files already created

| File | Purpose |
|---|---|
| [hxDif.py](hxDif.py) | Vendored MBP `.dif` parser (RandomityGuy/io_dif). Handles v44 |
| [generate_map_topdown.py](generate_map_topdown.py) | Parse `.dif` + `.mcs` → floors → top-down matplotlib PNG |
| [generate_edge_map.py](generate_edge_map.py) | Extract edge points from any `.dif`. Outputs `.npy`, `.html` (Plotly 3D), `.png` |
| [generate_policy_diagnostic.py](generate_policy_diagnostic.py) | Probe actor direction + critic V(s) at every walkable cell |
| [generate_brake_heatmap_map.py](generate_brake_heatmap_map.py) | Brake probability heatmap for any map |
| `edge_maps/edges_KingOfTheMarble_Hunt.npy` | 1,576 edge points ready to load |
| `edge_maps/edges_KingOfTheMarble_Hunt.html` | Interactive 3D viz (verified by user) |
| `edge_maps/edges_prophetic.npy` | 17,501 edge points for Prophetic (transfer test) |
| `edge_maps/edges_prophetic.html` | Interactive 3D viz for Prophetic |

The user VERIFIED the KOTM edge visualization looks correct.

---

## Not yet done (do these)

1. **Add scipy to imports** in `train_ppo.py` if not already there:
   ```python
   from scipy.spatial import cKDTree
   ```
2. **Wire the edge shaping into `PPOServer.__init__` and `compute_reward()`** per the code above.
3. **Auto-detect the current map's edge file** — probably needs a way to know which map is being trained on. Simplest: hardcode `edges_KingOfTheMarble_Hunt.npy` for now, or add a CLI flag. The mission name comes from the game side — we don't currently pass it to Python. Could add a startup message from CS that names the map.
4. **Update `analyze_log.py`** to parse the new `edge_shape:` field from GAME END lines (per user's rule about keeping analyze_log.py in sync).
5. **Test on KOTM**: start training from latest checkpoint (`update_80530.pth` or wherever), watch first ~500 updates for:
   - `ES=` values in `Upd N` lines: should be moderately negative (-500 to -2000 raw per rollout)
   - Critic VL shouldn't spike above 15 sustained
   - Gems/game shouldn't crash
   - After 1000+ updates, `generate_policy_diagnostic.py` should start showing dark zones in critic V(s) near voids

---

## What the user cares about most

1. **Not falling into holes.** The whole point of this exercise. Success = OOB rate drops from current 4-5/rollout to 1-2/rollout AND gems hold or improve from current ~30 avg.
2. **Map-agnostic.** Anything you build must work on complex maps (Prophetic-style labyrinths) without hand tuning. The edge detector already handles this.
3. **Not collapsing the policy.** The user has watched panic-brake collapse multiple times. If gems crash below 20 in the first 500 updates, back off `EDGE_SHAPING_COEF`.
4. **Verifiable diagnostics.** The user wants to see what's happening, not trust a black box. Keep the FF/ES/etc. metrics in log lines and expose new signals as we go.

---

## Model checkpoint state (as of this handoff)

- Latest checkpoint: `models/checkpoints/update_80530.pth` (roughly — check `ls -t models/checkpoints/ | head`)
- Prior peak: `update_72820.pth` was rolled back to at one point
- Discarded folders in `models/checkpoints/` contain quarantined bad-trajectory checkpoints — don't touch
- Current performance: ~30 gems/game plateau, best single 53, best 50-window ~34
- Camera yaw: LOCKED at 0 radians — the model sees consistent world coordinates every frame
- Obs dim: 59 (35 base + 24 frame history)

---

## Current reward config (in `train_ppo.py`)

```python
GEM_REWARD           = 200
OOB_PENALTY          = 25       # keep here; edge shaping is the main OOB-related pressure
GEM_GAP_PENALTY_K    = 0.0      # disabled
SLOW_PICKUP_BONUS    = 0        # disabled
OVERSHOOT_LAMBDA     = 0.0      # disabled
JUMP_COST            = 0.3

FREE_FALL_VZ_THRESHOLD  = 8.0
FREE_FALL_COEF          = 2.0
FREE_FALL_MAX_PER_STEP  = 0.75  # keep — this is per-fall dense penalty
```

New to add:
```python
EDGE_SHAPING_COEF    = 50.0     # start here, may need to tune down
EDGE_SHAPING_RANGE   = 3.0      # world units
```

---

## Read these too

- [CONVERSATION_HISTORY.md](CONVERSATION_HISTORY.md) — condensed record of what we tried in the last 2 weeks and why each thing failed
- [feedback_must_transfer_to_other_maps.md](../.claude-memory) — the user's persistent rule about map-agnostic solutions
- [BRAKE_LEARNING_PROBLEM.md](BRAKE_LEARNING_PROBLEM.md) — earlier consultation doc, still useful context
