# Lip skill planner integration, 2026-11-03 (milestone 5)

Code: nav/marble_dynamics/lip_skill.py (opportunity cells from mesh lips, private-simulator solve, production re-forecast as a
LandingContinuation), search.py (LIP_SKILL, BLAST_CONSERVE_U, lip plan lock, ranking audit), reliability_eval.py (LIP_SWITCH_U:
a kept non-pickup plan gives way to a clearly cheaper lip plan; --no-lip-skill control), tools/edge_skill/execution.py
(exit_window_ticks / require_backup / horizon_replies for live solves), physics/lip_skill_proposer_v6a.pt (frozen checkpoint).

Offline full rounds (tools/sim_bench/full_round.py, pool qr3_explore_baseline, seed 482017593):
- pair1/: ten maps x spawn seeds 1-2, skill off vs on, before the lock and switch rules: 20/20 pairs identical (admitted lip plans
  never replaced the running plan: the v47 kept-plan rule).
- pair2/: Fractured x6, Cragmire x2, Duplex x2 with the lock and switch rules: 2 executions in 10 pairs, both +5 (Fractured s1
  103 -> 108, s5 111 -> 116), all other pairs identical; mean +1.0 over these three maps, 0 elsewhere.
- live_teacher_v1.jsonl: 49 live states captured from the lip arm, teacher budget 768: 11 solvable.

Diagnosis: opportunities inside the trained envelope (level lip 0.35-1.5 u above a level run-up, level platform >= 3.5 u deep)
are rare in play (Fractured ~30 offers a round, Cragmire ~30 all declined, others 0-3); the rise histogram shows the climbs that
cost Blast on Sprawl and Cragmire are 1.5-6.5 u lips (outside the envelope) and the 0.5-1 u lips there sit on sloped ground.
When admitted, a lip climb reaches the platform about 1.7 s later than a Blast-and-jump plan, so it wins only without Blast or
when Blast is priced >= 20 u (BLAST_CONSERVE_U, default 0: no measured benefit).
