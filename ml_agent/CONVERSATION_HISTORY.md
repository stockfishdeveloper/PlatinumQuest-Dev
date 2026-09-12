# Conversation History — What We Tried (May 2026)

Condensed record of the previous ~2 weeks of iterations on training a PPO agent for KingOfTheMarble Hunt map. Useful context for next agent.

## Setting

- Marble Blast Platinum ML agent
- Was working on FlatWithJump (~65 gems/game) and moved to KingOfTheMarble to force map-transferable learning
- User wants a policy that generalizes to complex maps (labyrinths, half-pipes, multi-level)
- **Hard rule**: solutions must transfer to any map — no per-map oracles or hand-tuned obs

## The core problem on KOTM

Model plateaued at ~30 gems/game. Fundamental failure mode: **marble walks straight into interior voids**, not just overshooting past gems. Multiple diagnostics confirmed the model is oblivious to hole geometry:
- Actor direction arrows point AT gems THROUGH voids (no path curving)
- Critic V(s) shows no dark zones at void edges
- Behavior confirmed by user watching the marble in-game

## What we tried (in order, all essentially failed)

### 1. Higher OOB penalty
Tried OOB = 25 → 50 → 100. **Result**: model paid the bigger cost but didn't change behavior. At OOB=100, brake_rate spiked (panic-brake) but OOBs barely dropped. Rolled back to 25.

### 2. Freefall penalty (dense per-step during fall)
Added `-max(0, -vz - 8) * coef` while marble is in free fall.
- **Cap=5.0**: catastrophic — vz reaches -68 on KOTM (I predicted -25, was wrong), penalty hit -58k/rollout, critic VL spiked to 3337, policy collapsed to panic-brake / freeze in place, gems 30 → 3
- **Cap=1.5**: slow collapse over ~200 games. OOBs dropped 5→1 but gems crashed 30→3. Marble learned "barely move to avoid falls"
- **Cap=0.75**: sweet spot at first (35.9 peak gems) but slow-collapse pattern still visible (brake climbing 28%→41%)
- **Cap=0.5**: too weak, essentially ignored — OOB rate flat

### 3. Trust region tightening
Set `target_kl=1.0`, `clip_epsilon=0.1`, `max_grad_norm=0.3`. Combined with `critic_lr=1e-5`. Prevented critic explosion during interventions but didn't fix the core problem.

### 4. Log_std architectural clamps
Tightened `LOG_STD_MAX=-0.5` (35° direction cap) and `THROTTLE_LOG_STD_MAX=-0.9` (0.41 throttle cap). Prevented entropy runaway.

### 5. Negative entropy_coef
Set `entropy_coef = -0.001` to actively suppress entropy (was +0.001 → +0.005 originally, which allowed entropy runaway). Helped stability but didn't fix void-avoidance.

### 6. Cancel gem reward on grab-and-dive (proposed but not implemented)
Idea: if OOB within N steps of gem pickup, cancel the gem reward. Never actually wired up.

### 7. Fourier features on position (proposed)
My theory: MLP has spectral bias, can't learn high-frequency spatial functions (like "hole at these XY coordinates") from raw continuous position. Fourier features would fix representation. **User pushed back**: better to fix credit assignment than representation.

### 8. Committed stable training (~50k updates target)
Ran ~8k updates with frozen config. Model didn't collapse but also didn't improve. Plateau at 30 gems/game with slow drift toward more conservative play (brake creeping up, pickup speed dropping). **Concluded**: stable training alone won't fix void-blindness because gradient signal never reaches approach decisions.

### 9. External AI consultation (2026-09-11)
Consulted another AI. **Their key insight**: with γ=0.999, λ=0.95, GAE credit at 50 steps back is `0.949^50 ≈ 7%`. **The freefall penalty fires 50-100 frames AFTER the actual bad decision.** Credit assignment gap explains everything.

Their proposed fix: **potential-based edge-avoidance shaping** using geometry extracted from `.dif`. Fires during APPROACH to edges (dense gradient at the right frames). Bellman-consistent. Map-agnostic.

We agreed this is the right path. **Edge detection implemented and verified. Reward shaping code not yet written.** That's the handoff task.

## Things that ARE working (don't touch)

- Camera yaw locked at 0 radians (verified)
- Obs contains fixed-yaw world position — sufficient in principle for spatial learning
- Frame history obs (24 dims)
- Critic separated from actor
- γ=0.999 for long horizon
- The vendored `.dif` parser (`hxDif.py`) handles MBP v44
- Multi-level Z-clustering in `generate_edge_map.py`
- 3D Plotly visualization for verification

## Things to avoid re-proposing

- **Raycasts as new obs**: user rejected multiple times. Wants representational/reward fixes not new sensors.
- **BC from hand-coded oracle**: won't transfer to other maps.
- **`target_speed = sqrt(2*A_MAX*d)` obs**: flat-map only, won't work on slopes.
- **More reward shaping magnitude tweaks alone**: we've explored the whole knob, always same tradeoff (too weak = ignored, too strong = collapse).
- **Bigger OOB penalty alone**: caused panic-brake, doesn't fix credit assignment.

## The user's frustration triggers

- Claiming the model has "hit its ceiling" or that we should "accept current performance" → user WILL push back. They believe the model can be better; don't give up.
- Suggesting raycasts or fancy obs additions → they've been rejected repeatedly.
- Making a claim without checking the actual code (I did this twice — once wrong about camera rotation, once wrong about vz magnitude in falls). VERIFY before asserting.

## Verified facts (don't re-relitigate)

- Camera yaw is LOCKED at 0 in `mlAgent.cs` — every action msg ends with `,0` and `setMarbleCamYaw(0)` is called
- Position obs is camera-relative, but with yaw=0 it equals world position
- vz during KOTM falls reaches -65 to -68 (measured from actual training)
- KOTM has ~28 OOBs per full game (3-min round at 3x speed = ~11375 agent steps)
- OOB triggers when marble drops below the bounds z threshold; there's a fall of ~100 frames before OOB fires
- Fall duration averages ~70-150 frames per fall from measured data
