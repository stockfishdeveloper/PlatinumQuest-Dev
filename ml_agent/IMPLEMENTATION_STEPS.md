# Implementation Steps — Edge-Avoidance Reward Shaping

Concrete step-by-step guide for the next agent to wire the edge-avoidance shaping into training. Read [HANDOFF_EDGE_SHAPING.md](HANDOFF_EDGE_SHAPING.md) first for context.

## Step 0: Confirm environment

```powershell
cd C:\Users\doug\OneDrive\Documents\GitHub\PlatinumQuest-Dev\ml_agent
python -c "from scipy.spatial import cKDTree; print('scipy OK')"
```

If scipy is missing: `pip install scipy`.

Also verify the edge files exist:
```powershell
ls edge_maps\*.npy
```
Should see `edges_KingOfTheMarble_Hunt.npy` (1,576 points) and `edges_prophetic.npy` (17,501 points).

## Step 1: Add config constants to `PPOServer.__init__`

Find the reward-config block in [train_ppo.py](train_ppo.py) around the `FREE_FALL_*` constants (currently near line ~730). Add:

```python
# Edge-proximity shaping (potential-based).
# Precomputed edge points loaded from edge_maps/edges_<mapname>.npy.
# Per-step reward = Phi(d_new) - Phi(d_old) where Phi(d) = -coef * exp(-d/range).
# Bellman-consistent (approach cost refunded on retreat), gives dense
# gradient at approach frames (NOT deferred to the fall event itself).
# The credit-assignment fix for the "marble walks straight into holes" problem.
self.EDGE_SHAPING_COEF = 50.0    # depth of the potential well at the brink
self.EDGE_SHAPING_RANGE = 3.0    # world units; exp decay scale
```

## Step 2: Load edge points at server startup

At the end of `PPOServer.__init__` (after all other config), add:

```python
# Load precomputed edge points and build KDTree for fast nearest-edge queries.
# Auto-detect current map from the CS side is future work; hardcoded for now.
import numpy as np
from scipy.spatial import cKDTree
edge_path = 'edge_maps/edges_KingOfTheMarble_Hunt.npy'
if os.path.exists(edge_path):
    edge_pts = np.load(edge_path)
    self.edge_kdtree = cKDTree(edge_pts)
    self.log(f'Edge shaping: loaded {len(edge_pts):,} edge points from {edge_path}')
else:
    self.edge_kdtree = None
    self.log(f'Edge shaping: DISABLED (no {edge_path})')

# State for per-step potential difference
self.last_edge_phi = 0.0
self.last_edge_dist = 999.0
```

## Step 3: Add diagnostic accumulators

Near the other `self.game_*` and `self.rollout_*` initializations, add:

```python
# Edge shaping diagnostics
self.game_edge_shaping_sum = 0.0
self.rollout_edge_shaping = 0.0
```

You'll need to reset these in TWO places (search for existing accumulator resets like `self.game_freefall_penalty_sum = 0.0`):
- End of `compute_reward` game-end branch (around line ~1244)
- Rollout counter reset section in `run_ppo_update()` (around line ~1602)

## Step 4: Wire the shaping into `compute_reward()`

Find the end of `compute_reward()` in [train_ppo.py](train_ppo.py) — just before `return reward`. Add BEFORE the return:

```python
# 6. Edge-proximity shaping (potential-based, Bellman-consistent).
# Fires per-step during approach to any edge/void boundary; refunds on retreat.
# Credit-assignment fix — gives gradient AT approach decision frames rather
# than only during/after the fall (which was too discounted to reach approach).
if self.edge_kdtree is not None and len(raw_obs) >= 3:
    marble_xyz = np.array([raw_obs[0], raw_obs[1], raw_obs[2]])
    d, _ = self.edge_kdtree.query(marble_xyz)
    phi_new = -self.EDGE_SHAPING_COEF * math.exp(-d / self.EDGE_SHAPING_RANGE)

    # Detect respawn / OOB position jumps — never credit them as "moved away"
    if oob or abs(d - self.last_edge_dist) > 10.0:
        shaping = 0.0
    else:
        shaping = phi_new - self.last_edge_phi

    reward += shaping
    self.game_edge_shaping_sum += shaping
    self.rollout_edge_shaping += shaping
    self.last_edge_phi = phi_new
    self.last_edge_dist = d
```

Also add `import math` at top of file if not already imported.

## Step 5: Update logs

**GAME END log line** (around line ~1225 in `train_ppo.py`):

Find:
```python
self.log(f"[GAME END] total gems this game: {total_game_gems}pts ... freefall: {self.game_freefall_penalty_sum:.0f} ({freefall_pct:.1f}% of steps)")
```

Add `edge_shape:` at the end of the format string:
```python
... freefall: {self.game_freefall_penalty_sum:.0f} ({freefall_pct:.1f}% of steps) edge_shape: {self.game_edge_shaping_sum:.0f}")
```

**Upd log line** (around line ~1575): find the `ff_str` line, add an equivalent:
```python
es_str = f" ES={self.rollout_edge_shaping:.0f}"
```
and include `{es_str}` in the formatted log line.

## Step 6: Update `analyze_log.py`

Per user's rule about keeping analyze_log.py in sync — add regex for `edge_shape: ([-0-9]+)` and include in the summary output.

## Step 7: Test on KOTM

Start training from latest checkpoint. In the first 10 `Upd` lines you should see:
- `ES=` values between -500 and -3000 raw per rollout (roughly proportional to time spent near edges)
- Critic `VL` staying under 15 (if it spikes higher, drop `EDGE_SHAPING_COEF` to 20-30)
- `gems=` per rollout should stay in normal range (2-10)

In the first `[GAME END]` line:
- `edge_shape: <negative value>` should be around -500 to -3000
- Compare to `freefall: <value>` — edge_shape should be similar magnitude or larger

Watch for collapse warning signs (would tell you to lower coef):
- `brake_rate` jumping above 55%
- `pickup_speed` below 5
- `avg_steps/gem` above 700
- `gems/game` below 20 for multiple consecutive games

If those fire in the first 500 updates, halve `EDGE_SHAPING_COEF` to 25 and try again.

## Step 8: Verify learning

After ~1000-2000 updates, run:
```powershell
python generate_policy_diagnostic.py
```

**Success signal**: critic V(s) map (right panel) should now show dark zones near voids (cells where the model has learned "state is bad"). Compare to previous diagnostic outputs (there are old ones in the repo). If dark rings appear around voids, the model has learned void geometry — which is exactly what all previous approaches failed to achieve.

The actor direction arrows may also start curving around voids toward gems, though this could take more updates.

## Step 9: Test transfer

If KOTM works, run:
```powershell
python generate_edge_map.py <newmap>
```

The edge file will be auto-generated. Change the hardcoded edge_path in Step 2 to load the new map's edge file, and start training on the new map from the KOTM-trained checkpoint. Model should adapt.

## Rollback plan

If it goes badly and you need to disable edge shaping without touching much code:
1. Set `self.EDGE_SHAPING_COEF = 0.0` — shaping becomes zero everywhere
2. Or delete the `edges_KingOfTheMarble_Hunt.npy` — kdtree stays None, code no-ops

Roll back checkpoints via the standard pattern (move damaged ones to a discarded_ folder in `models/checkpoints/`).
