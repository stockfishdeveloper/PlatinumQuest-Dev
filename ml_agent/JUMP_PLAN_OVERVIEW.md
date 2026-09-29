# Teaching the marble to jump: the plan in plain words

September 2026. This is the happy path: what we will do, in what order, and what each step should deliver if
everything works as intended. The full technical specification is `KOTMJUMP_NAVIGATION_DESIGN.md`, and the
first hands-on step is described in `KOTMJUMP_START_HERE.md`.

## Where we are

Our AI navigator plays King of the Marble at about **154 points** a round. A strong human reaches **172-177**
on a good round. The human isn't faster on the straights; the speeds are about the same. The difference is
the route. The human cuts across holes with 12 to 15 jumps a round, while our navigator jumps about twice and
rolls the long way around. About two thirds of the gap is those missing shortcuts.

Two days of tuning rewards and training settings have not moved the score: every attempt ended at or below
the best model we already had. The current navigator was taught jumping with hand-made rules built for this
one map, and it never really learned when a jump pays off. We need a different approach, and one that works
on every map, not just this one.

## The idea

Teach the computer the physics first, then let it plan with that knowledge, the way a good player does.

1. **Learn the physics from the game itself.** We put the marble into thousands of situations (rolling toward
   a hole at different speeds and angles, jumping at different moments, steering in the air) and record
   exactly what the game does. From those recordings, a model learns to answer questions like "if I jump
   right now and steer left in the air, where do I land, how fast am I going, and do I pass through that
   gem on the way?" It answers in a fraction of a millisecond, so it can be asked hundreds of questions every
   time the marble has to decide what to do.
2. **Plan routes with it.** When a gem needs collecting, a planner uses the physics model to try out many
   possible approaches from where the marble actually is: roll around, or speed up toward that lip and jump?
   It starts from the marble and works toward the gem, and it also works backward from the gem ("what
   launch would fly through it, or land close enough to make the trip shorter?") so it finds the jump
   quickly. It keeps the fastest route that the model is confident will work, whether or not that route
   includes a jump. Most of the gain on King of the Marble comes from jumping a hole to reach a gem on the
   floor beyond it, not from floating gems; the floating gems are just the test that forces the skill.
3. **Drive with feedback.** The marble follows the first step of the plan, sees what really happened, and
   re-plans. That happens 15 times a second, so small errors get corrected before they matter. If the marble
   picks up a floating gem in mid-air, it keeps its landing plan and doesn't panic when a new group of gems
   appears.
4. **Keep what already works.** We will test two ways of driving. In one, the new planner drives everything.
   In the other, today's navigator keeps driving normally, and the planner takes over only for a planned
   jump manoeuvre, then hands control back once the marble has landed somewhere safe. Today's navigator is
   good at ordinary rolling, so this second version is likely to win first.

Nothing is written for a specific map. The physics model only ever looks at the marble and the ground around
it, so what it learns about a ramp or a gap on one map applies to the same kind of ramp or gap anywhere.

## What it looks like in practice

The marble is rolling at a normal pace, 15 units from a hole with a gem floating over its middle.

* The planner asks the physics model about a couple of hundred possible approaches. One stands out: "turn
  slightly right, speed up for a second and a half, jump at the lip, hold forward in the air, then brake just
  before landing." The model says it passes through the gem and lands safely on the far side, faster than any
  way around.
* The marble starts that approach. Every fifteenth of a second it checks where it really is and adjusts:
  a little faster here, a slightly different angle there.
* Just before the lip, it checks one last time that the jump still works from its actual speed and direction,
  then jumps.
* In the air it collects the gem, lands, and carries on toward the next one.

## What stays the same

* Today's navigator, its best saved model and its training setup are not touched. It stays the baseline that
  every new result is compared against.
* The rules we play by: nothing written for a particular map, rewards come only from the game's own scoring,
  and every result is judged by real rounds of the game, not by training graphs.
* **Horizon** and **Archipelago** are set aside as test maps and are never used for learning, so we can
  honestly check whether the physics carries over to maps the model has never seen.
* Training of today's navigator stays stopped while the first experiment runs.

## The roadmap

The time estimates are rough and assume things go well. Each stage has to pass its check before the next one
starts.

| Stage | What we do | How we know it worked | Expected outcome | Rough time |
|---|---|---|---|---|
| 0. Setup | Measure today's navigator on the jump test map. Build a practice copy of that map with a single floating gem that is always there. | The practice map works and the baseline is recorded. | We know how today's navigator does on floating gems (probably poorly: it has never tried). | 1 session |
| 1. Record jumps | Put the marble near one hole in thousands of different ways, jump, and record exactly what happens. | Repeating the same jump gives the same result, and the records match the game. | A trustworthy set of recorded jumps. | 1 session |
| 2. First learned jumps | Train a small model on those jumps and let it pick how to jump from starting positions it has never seen. Compare it with simple rules and random choices. | On at least 100 new starting positions, it collects the gem and lands safely clearly more often than every simple alternative. | Proof that a learned physics model can choose good jumps. | 1 session |
| 3. The full physics model | Build the proper version: exact measurement of edges and surfaces, a model for rolling as well as flying, and data from more of the game's maps. | Its predictions match the game closely, including right at the edges of holes, and we know where it is still unsure. | A physics model that understands rolling, ramps, jumps and landings on several maps. | 1-2 weeks |
| 4. Jumps from anywhere | Add the planner and the feedback driving, so the marble sets up its own approach from wherever it happens to be. | Collects the floating gem and lands safely at least 80 % of the time, from 200 new starting positions, at all four floating gems. | The marble plans and makes the jump by itself. | 1-2 weeks |
| 5. Reliable and smart | Push reliability up, compare different routes to the same gem, handle whole groups of gems, and test handing control back and forth with today's navigator. | At least 95 % success, faster completion times counting every failure, and handovers that don't cause falls. | The marble finds shortcuts and uses them reliably. | ~2 weeks |
| 6. Real rounds | Play full rounds on the jump test map and on King of the Marble, 8 rounds each at 3x speed, for each way of driving. | King of the Marble is at least as good as today (no more than 2 points lower, no extra falls), and the floating gems are collected. | Higher King of the Marble scores as the marble cuts across holes like a human player, moving toward the 170 mark. | ~1 week |
| 7. New maps | Freeze the physics, then test on maps it has never seen, including Horizon and Archipelago. | It plays those maps well without learning anything from them. | The physics carries over to the rest of the game's maps. | 1-2 weeks |

On the happy path, stages 0 to 2 take about a week, and the first real rounds with the new system (stage 6)
arrive about two months in.

**Progress, 2026-09-27:** stages 0 to 2 are done and stage 2 passed. From 251 new starting positions where some
jump works, the learned model collected the gem and landed safely 68 % of the time; the best simple rule managed
19 %. Today's navigator, on the jump test map, scores 4.6 points a round and gets stuck after its first three
gems. Stage 3 starts when the operator says so.

**Progress, 2026-09-28 morning:** stage 3 ran overnight. The physics model was trained on 2.4 million short
recordings from 11 maps. It predicts the marble's next 64 ms to about 0.003 units, and just as well on two maps it
never saw. Choosing the jumps of stage 2's test without ever seeing that data, it succeeds 52 % of the time (the
model trained only there: 68 %; the best simple rule: 19 %). Stage 3's checks are met; stage 4 (planning the run-up
from anywhere) starts when the operator says so.

**Progress, 2026-09-28 afternoon:** stage 4 is done and its check passed. The marble now plans its own run-up and
jump: every 64 ms it tries a few hundred possible input sequences in the physics model, keeps the ones that take the
gem and land with room to spare, and sends the first input of the best. From 233 starting spots it had never been
tuned on (at rest, rolling toward the gem, across it or away from it, 4 to 18 units out), it took the floating gem
and landed safely 93 % of the time (the bar was 80 %), and at least 90 % for each of the four gems. When it decided
to jump for the gem, it got it 187 times out of 188; the misses come from circling without committing, or rolling
too close to a hole's edge.

**Progress, 2026-09-29 morning:** stage 5 ran overnight, with one clear success and one miss. The success: in full
3-minute rounds of the jump map, today's navigator drives and hands over to the planner only for the floating gems.
The score went from about 5 points a round (the navigator alone, stuck at the first floating gem and falling about
33 times a round) to 73 on average, and up to 87; the planner alone managed 65 in its one round without a fall. The miss: on 228 new starting spots for a single
floating gem, the planner succeeded 90.8 % of the time, short of the 95 % bar for each gem. Tuning had reached 95 % on
the practice starts but did not carry over. The main remaining failure: after taking the gem the marble lands at speed
on the narrow strip between two holes and rolls into the next one.

## Moments to watch

The best discoveries on this project have come from watching the game at normal speed. Natural points to
watch:

* After stage 2: the marble jumping at the practice hole, with the model's prediction drawn beside what
  really happened.
* After stage 4: the marble setting up its own run-up and taking the floating gem from different starting
  spots.
* After stage 6: full rounds, side by side with today's navigator.

## After this plan

Once the physics model and planner work across maps, the same machinery extends naturally:

* **More maps:** add the game's other Hunt maps a few at a time, each checked in the game before it is used.
* **Powerups:** a Super Speed or Super Jump is just another move the planner can consider. We have already
  shown a Super Speed can be fired in any direction.
* **Other players:** multiplayer and competition come after single-player navigation works everywhere.

## If a step doesn't work

This is the happy path, but each stage has a clear stopping rule. If a check fails, we stop and find out why
(bad measurements, missing situations in the data, or the model itself) before going further. We don't paper
over it and move on.
