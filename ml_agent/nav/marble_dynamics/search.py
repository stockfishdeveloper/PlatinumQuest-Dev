"""General phased control search and persistent discrete-action commitment."""
import copy
import math
import time
from collections import OrderedDict

import numpy as np

from nav.marble_dynamics.bridge import predict_replies,reply
from nav.marble_dynamics.gem_demo import Proposal, candidate_cost, support
from nav.joystick import action_to_joystick
from nav.marble_dynamics.navigator import NavigatorProposal
from nav.gems import visible_gems
from nav.protocol import RAW_TIME_LEFT
from nav.routing import snap_to_walkable,RoutePlanner,goal_key,path_waypoint
import nav.marble_dynamics.gem_demo as _gem_demo
from nav.marble_dynamics.blast import available as blast_available
from nav.marble_dynamics.planning_budget import PlanningBudgetExceeded


ROUTE_SPEED_LAW=True     # v16 (2026-10-15): stopping law on the remaining route length, not the next waypoint
PICKUP_MIN_SPEED=6.      # u/s floor on the target speed toward a gem (0 = brake to a stop at the gem, pre-v16)
THROUGH_GEM=True         # v22 (2026-10-15): score past a pickup toward the tour's next gem; 'through' proposals
THROUGH_SWITCH_U=.6      # u: a through proposal turns to the next gem once its path passes this close to the gem
POWERUP_THROUGH=False    # 2026-10-26 (under test): a powerup fetched on the way to a gem gets the through proposals too (keep the speed
                         # through the pickup toward the gem; reliability_eval.pick names the gem as its next target). Sprawl human
                         # demo: into the Super Speed at 10.9 u/s with a jump, kick in the air -> 37 u/s; the planner braked to 4.9
THROUGH_REACH_U=18.      # u: through proposals are added only when the gem is this close along the route
OPEN_TOP=0.              # --open-top (v36, under test): top target speed far from drops (0 = off; 13.5 with the flag) ...
OPEN_NEAR_U=4.           # ... 11 u/s within this walk-grid distance of a drop ...
OPEN_FAR_U=8.            # ... rising linearly to OPEN_TOP at this distance
REFINE_CANDIDATES=0      # opt-in bounded legal-control perturbations, scored by the existing physics gates
REFINE_FAILURE_ONLY=True # preserve useful ordinary plans; explore when their forecast fails or makes no progress
THROUGH_BOX=False        # 2026-10-19 (operator: keep momentum through gems): through plans switch to the next gem when the
                         # marble enters the gem's pickup box (the game's test), not within THROUGH_SWITCH_U of its centre
WALL_BOUNCE=False        # 2026-10-19 (operator: use wall hits): bank-shot plans off a wall toward the next gem (wall_bounce.py);
                         # on 21:25 with reliability_eval.WALL_TOUR (sim 120 starts x 10 maps -0.07 s a group, 6 better /
                         # 2 worse); ROLLED BACK 21:47 (operator) after weak live rounds; on the todo list (ml_agent/TODO.md)
CORNER_PROPOSALS=False   # v38 (--corner-proposals, under test): through proposals that reach the gem at the turn's speed
USE_AWAY_U=0.            # 2026-10-16: a use that collects but ends this much farther from the next gem is rejected (0 = off; v40: 8)
SS_RESERVE_LAST_U=0.     # 2026-10-16: no Super Speed plans for the last gem of a group closer than this (0 = off; v40: 20) ...
SS_RESERVE_TIME_MS=6000. # ... unless the round has less than this left (then it is now or never)
SS_EARLY_U=math.inf      # 2026-10-16: Super Speed 'use now' plans judged early on legs at least this long (inf = off; v40: 20)
POW_LAUNCH=True          # 2026-10-25 night (operator approved): take-off launches. The (use, jump) families above press the jump first, so
                         # the kick (2 decisions after the use) fires ~0.2 s into the air: flat and short. Human demo TreasureBox: jump +
                         # Super Speed on the SAME tick on a ramp: 31.5 u/s with vz 14, a 47 u flight onto the gem (planner probe from
                         # the human's states: these timings reach it, peak z 9.8 vs 6.5) ...
POW_LAUNCH_AT=(0,2,4,6,9,12,16,20)   # ... at these replies ahead (the forecast picks the spot: a ramp's top, a lip) ...
POW_LAUNCH_LEADS=(0,1)   # ... with the use pressed this many replies before the jump ...
POW_LAUNCH_FLIGHT=40     # ... each forecast running this many replies past the jump (flight and landing) ...
POW_LAUNCH_TYPES=(1,2)   # ... for a held Super Jump or Super Speed ...
POW_LAUNCH_MIN_LEG_U=20. # ... on a leg at least this long along the route (sim_launch1, launches on every leg: trips -2.5 % but group clears
                         # +1.3 %, points -14: inside a cluster a kick overshoots and the extra plans take search time)
POW_LAUNCH_LOCK_LEAD=3   # a launch plan is held (PlanMemory.locked) only from this many replies before its first timed input (the
                         # receding horizon picks the spot meanwhile; locked from acceptance, use_launch_12_13 held a northward flight
                         # for a second while an aimed launch reached the gem: human-start sim, 8.4 s)
POW_LAUNCH_LEAD_K=25.    # 2026-10-26 night: an aimed Super Speed launch leads its kick for the speed already carried (the kick ADDS 25 u/s
                         # along the aim): aimed at the gem, 9 u/s north on the TreasureBox ramp bent the flight 10 deg left into a box
                         # 10 u short; the human aimed right (kick 16.3, 18.2) and landed on it (lead_aim; 0 = aim at the gem)
LEAD_AIM_MAX=.9          # lead_aim: largest sideways share of the kick (the rest goes toward the goal)
DIRECT_LEAD=False        # 2026-10-26 night (under test): the direct_use_* Super Speed kicks lead for the carried speed too (lead_aim), so a
                         # running marble can kick into a sharp turn (human technique: 90-150 deg turns after a pickup)
POW_LAUNCH_LAND=True     # 2026-10-26 night: an aimed launch's flight steers its landing point onto the gem (land_action): the forecasts of
                         # the aimed TreasureBox launches passed 0.9-3.8 u from the gem (range errors; the air control braked toward the
                         # 11 u/s target speed); the human curved the flight onto it with the air control
LAND_G=20.               # land_action: gravity (u/s^2, straight down) and air acceleration per unit of stick (the engine's defaults) ...
LAND_AIR=5.
LAND_DZ=.3               # ... the landing height: the gem's z plus this (the marble centre inside the pickup box over a floor gem) ...
LAND_MIN_VZ=.5           # ... airborne only: |vz| above this (on the ground the ordinary controller drives) ...
LAND_MIN_T=.1            # ... and at least this long (s) before the landing ...
LAND_AFTER_USE=3         # ... from this many replies after the use (the kick fires 2 decisions after it)
POW_LAUNCH_ROLL=False    # 2026-10-26 (under test): Super Speed launches without a jump (aimed, lead, landing control) at POW_LAUNCH_AT
LAND_SHORT=False         # 2026-10-26 (under test, Python forecasts only): aimed launches land short of the gem by the braking distance
LAND_SHORT_ARRIVE=11.    # from their ground speed down to this (u/s) at LAND_SHORT_DECEL u/s^2 and roll in (Sprawl human demo: landed
LAND_SHORT_DECEL=7.      # ~22 u short at ~29 u/s, first gem at 8 u/s; the aimed launches landing ON the gem overshot and rolled off)
SS_RIDE=False            # OFF 22:58 (sim_ssA, 108 Super Speed starts: 7.3 percent slower, smacks 75 -> 118; opt-in --ss-ride). 2026-10-24 (operator: "every second of wasted time costs us $1000"; live Bowl: a held Super Speed fired only after
                         # the marble slowed from 29 to 4.8 u/s): plans that RIDE the kick (no braking toward the arrival speed after it,
                         # the fast family's carried-speed floor up to SS_RIDE_TOP) and head for the tour's next gem after the pickup;
                         # while a Super Speed is held no use margin, and a use plan may end above 18 u/s with its tail on the floor (the
                         # braking backup still guards edges); Super Speed plans always enter the cluster re-ranking (control --no-ss-ride)
SS_RIDE_TOP=40.          # u/s: speed kept toward the aim point by a riding plan
SS_RIDE_USES=(0,3)       # replies of the kick for the riding plans
NAV_SPAWN_OFF=False      # 2026-10-24 night (operator: clusters as fast as possible): no frozen-navigator proposal when the target
                         # is within NAV_SPAWN_U along the route and NAV_SPAWN_GEMS+ gems lie within 12 u of it (8 TreasureBox pool
                         # rounds: its in-spawn jumps collected 44 percent against 73 for the jump families; 217 of 592 presses)
NAV_SPAWN_U=8.
NAV_SPAWN_GEMS=3
BLAST_GATE_ALWAYS=False  # the pad_hops Blast gate (no Blast for a low pad gem within 8 u) without the spawn-chain plans (in-spawn
                         # Blast presses collected 20 percent: 46 presses)
EXTRA_JUMP_TIMES=(1,3,4,6,7,8)   # 2026-10-18 v79: jump times tried only when no candidate passed the gates (see search)
EXTRA_JUMP_S=.5          # s those may run past the search deadline
BRAKING_BACKUP=True      # v22 (2026-10-15): the chosen plan must leave a braking backup that stays on the floor
SAFE_HOLD=True           # v22: with no useful plan, execute the best gate-checked plan instead of the unchecked actor
BACKUP_REPLIES=20        # replies of full braking after the plan's end (1.28 s)
BACKUP_CHECKS=4          # at most this many of the best useful candidates are checked per search
PICKUP_STOP_SHORT_U=3.   # last-resort 'stop_short' goal this far before the gem
PICKUP_STOP_STREAK=24    # consecutive stop decisions AT REST (1.5 s parked) before the ordinary actor fallback resumes
PICKUP_STOP_EDGE_S=1.5   # stop only when the free roll leaves the floor within this many seconds (edge_ahead)
PICKUP_STOP_MIN_SPEED=0. # v6 tried 6 u/s: Cragmire parking gone but the Cube/Duplex falls came back (falls 7 -> 9, 17:31); 0 = v5
                         # (3 stuck rescues, -37). Recorded fall starts under 6 u/s: two Cragmire ledge cases and Duplex 3043,
                         # none of which v3-v5 prevented.
PICKUP_STOP_EDGE_DEPTH_U=30.  # v5: 'leaves the floor' = no floor within 30 u below (a drop to a lower floor is not a fall;
                              # all 23 recorded fall starts still test true, Cragmire's v4 stops drop from 163 to 108 of 295)
TRAVEL_MODE_FROM_U=25.   # 2026-10-24 (operator: "in absolutely no hurry to get to the next spawn"): a gem this far along the route = a trip
TRAVEL_SKIP_SLOW=False   # ... on such trips no throttle_* / accelerate_then_brake candidates (cruise: 50-72 deg/s of turning, losing speed)
TRAVEL_TURN_U_PER_RAD=0. # ... and a plan without a pickup pays this many u per radian its heading turns (steering plans snake at 56 deg/s
                         # against 12-31 for the fast families; the human cruises at 29 deg/s and 16 u/s against the planner's 50 and 11)
CLUSTER_PENALTIES=True   # 2026-10-23 (operator: "eradicate" pad-side smacks, ledge drops after a pickup, Super Speed for one gem);
                         # engine.cluster_penalties overrides (control --no-cluster-penalties). Measured on every candidate's 16 ms ticks:
BUMP_COST_U=50.          # (sim 10-23 round 5: 50 with the re-search, AWAY 2, edge hits penalised, pad arrival = impacts -53 percent, 1 percent faster) u per side impact: one tick (16 ms) losing more than BUMP_TICK_DV of horizontal speed, mostly sideways (braking
BUMP_TICK_DV=1.2         # removes at most ~0.5 u/s a tick, a landing or a ramp turns speed upward); plans built to use walls are exempt
BUMP_EXEMPT_WALL=True    # deliberate wall rebounds (labels containing 'wall') are exempt ...
BUMP_EXEMPT_EDGE=False   # ... and the edge-hit (ledge lip) family (labels starting 'edge_hit'): NOT exempt since 10-23 (20 percent of the smacks)
LEDGE_DROP_U=30.         # u when the plan collects its gem and then drops LEDGE_DROP_DZ below it while the tour's next gem is on that level
LEDGE_DROP_DZ=1.5        # (pools: 2.4 such drops a round, the next pickup 3.1 s later on average against 1.1 s without a drop)
LEDGE_TARGET_LEVEL=True  # also a plan without a pickup that leaves its target gem's level downward (the marble on that level now)
AWAY_U_PER_U_S=2.        # u per u/s of horizontal speed AWAY from the tour's next gem AWAY_AFTER_REPLIES after the pickup (momentum
AWAY_AFTER_REPLIES=8     # carrying the marble off past the gem: a Sprawl trace rolled on downhill and climbed back for 3 s)
SS_ONE_GEM_U=8.          # u when a plan spends a Super Speed, collects at most one gem, and SS_CLUSTER_GEMS more gems are within
SS_CLUSTER_GEMS=2        # SS_CLUSTER_U of its target (a cluster: the kick is better kept for the trip to the next group)
SS_CLUSTER_U=15.
BACKUP_PENALTY_U=150.    # cost added to a useful plan without a braking backup (above the -100 pickup credit)
FALL_FLOOR_U=15.         # u: airborne after the braking backup with no floor this far below = a fall
STOP_SPEED=2.            # u/s: when no useful plan exists, 'stop_N' (full braking from reply N) is tried above this speed


_FIELDS=OrderedDict()


def cached_route(terrain,goal):
    """A RoutePlanner on `goal` whose walk-distance field comes from a small cache (one Dijkstra per new goal)."""
    key=(id(terrain),goal_key(goal))
    field=_FIELDS.get(key)
    if field is None:
        field=terrain.goal_field(goal[0],goal[1],z=goal[2]);_FIELDS[key]=field
        while len(_FIELDS)>48:_FIELDS.popitem(last=False)
    else:_FIELDS.move_to_end(key)
    route=RoutePlanner(terrain);route.field=field;route.key=goal_key(goal);route.waypoint=None
    return route


def through_context(engine,route,gem,item):
    """(route to the next gem, its position, walk distance gem -> next) when the tour names a next gem for `gem`.

    Only for a real gem target outside a crossing phase; None keeps the pre-v22 scoring.
    """
    if not THROUGH_GEM or getattr(engine,'crossing_phase',False):return None
    if item is None or not (item.get('gem') or (POWERUP_THROUGH and item.get('powerup'))):return None
    nxt=(getattr(engine,'next_for',None) or {}).get(goal_key(gem))
    if nxt is None:return None
    ngem=np.asarray(nxt[:3],dtype=float);here=np.asarray(gem[:3],dtype=float)
    if np.linalg.norm(ngem-here)<.1:return None
    key=(goal_key(here),goal_key(ngem),id(route.terrain))
    cache=getattr(engine,'_through_cache',None)
    if cache is None or cache[0]!=key:
        nroute=cached_route(route.terrain,ngem);leg=float(nroute.distance(here))
        engine._through_cache=cache=(key,(nroute,ngem,leg) if math.isfinite(leg) else None)
    return cache[1]


CORNER_SLOW=True          # ON by default since 2026-10-25 05:25 (pair pf4 +1.0 a round, TreasureBox +5 / +31, Fractured +16 / +9; sim clear -3.1 percent, smacks -28 percent); 2026-10-25 (cluster speed): a second corner pair arriving at CORNER_SLOW_SHARE of the corner speed (floor
CORNER_SLOW_SHARE=.6      # 1.5 u/s): TreasureBox drops at 60-90 deg took 0.83 s against the human's 0.43 (arrival 5.5 vs 3.9 u/s)


def corner_speed(route,position,gem,through):
    """v38: speed at the gem for the turn toward the next gem, cap * cos^2(turn / 2) (full speed straight on, the
    1.5 u/s floor for a reversal); None for turns under 30 degrees (the plain through proposals cover them).
    Approach = marble -> gem, exit = gem -> the next gem's route waypoint."""
    nroute,ngem,leg=through;g=np.asarray(gem[:3],dtype=float);p=np.asarray(position,dtype=float)
    a=g[:2]-p[:2]
    try:wp,_=path_waypoint(route.terrain,float(g[0]),float(g[1]),nroute.field,ngem,mz=float(g[2]))
    except Exception:wp=ngem
    b=np.asarray(wp[:2],dtype=float)-g[:2]
    na,nb=float(np.linalg.norm(a)),float(np.linalg.norm(b))
    if na<.5 or nb<.5:return None
    c=float(a@b)/(na*nb)
    if c>math.cos(math.radians(30)):return None
    return max(1.5,_gem_demo.TARGET_SPEED_CAP*(1+c)/2)


TRAVEL_TOP=16.            # (--travel-top sets it at launch) 2026-10-24: 16 with the cluster cap left at 11 (sim: trips -3 percent, episodes -2.3 percent, falls 5 -> 2); 13.5 before; 10-22 step C (16 with cap 13.5) tied in game. 2026-10-17 v56 (operator: "traversing between spawns faster"): top target speed on long legs (0 = off).
TRAVEL_NEAR_U=12.         # The top ramps from TARGET_SPEED_CAP at this distance to the target gem (route length, or straight for
TRAVEL_FAR_U=20.          # direct proposals) up to TRAVEL_TOP at this one; closer, the cluster cap and the stopping law hold.


ARRIVAL_MIN_SPEED=0.      # 2026-10-17 v62 (118.5 vs v58 121.6, not adopted): pickup minimum speed at the gem when its leg started >= ARRIVAL_FAR_U away (0 = off;
ARRIVAL_FAR_U=20.         # the end of a long trip; clusters keep PICKUP_MIN_SPEED). v60 tried 8 everywhere: travel -7 %, clusters slower.


def leg_top(terrain,state,distance):
    """Proposal top for a leg 'distance' from its gem: the travel ramp, else search.open_top (None = TARGET_SPEED_CAP)."""
    if TRAVEL_TOP and distance>TRAVEL_NEAR_U:
        cap=_gem_demo.TARGET_SPEED_CAP
        return cap+(TRAVEL_TOP-cap)*min(1.,(distance-TRAVEL_NEAR_U)/(TRAVEL_FAR_U-TRAVEL_NEAR_U))
    return open_top(terrain,state)


def open_top(terrain,state):
    """--open-top (v36): top target speed from the walk grid's distance to the nearest drop at the marble's cell."""
    if not OPEN_TOP:return None
    edge=getattr(terrain,'edge_dist',None)
    if edge is None:return None
    p=state['position'];j,i=terrain.cell_of(p[0],p[1]);k=terrain.level_at(j,i,float(p[2])-float(state.get('radius',.19)))
    if k<0:return None
    d=float(edge[k,j,i]);base=11.
    if d<=OPEN_NEAR_U:return None
    return base+(OPEN_TOP-base)*min(1.,(d-OPEN_NEAR_U)/(OPEN_FAR_U-OPEN_NEAR_U))


FAST_TOPS=(16.,20.)       # 2026-10-18 (operator: "can we get it to use more speed?"): top target speeds of the fast families
                          # (route and direct aims each; () = off)
FAST_MIN_SPEED=9.         # ... their floor on the target speed at the gem (the other families: PICKUP_MIN_SPEED)
FAST_PROPOSALS=True       # False reproduces the old compiled normal-speed duplicates for matched diagnostic controls


class SequenceProposal:
    def __init__(self,route,gem,steps,offset=0.,scale=1.,jump=-1,use=-1,turn_after=6,brake_after=None,direct=False,blast=-1,through=None,arrive=None,fast=None):
        self.route=copy.copy(route);self.gem=np.asarray(gem);self.steps=steps
        self.arrive=arrive   # v38: speed at the gem for a corner (through proposals only)
        self.offset=offset;self.scale=scale;self.jump=jump;self.use=use
        self.turn_after=turn_after;self.brake_after=brake_after;self.direct=direct
        self.blast=blast
        # through = through_context(...): drive through the gem without braking
        # for it, then follow the route to the next gem (v22).
        self.through=through;self.next_route=copy.copy(through[0]) if through is not None else None
        self.passed=False;self.previous=None
        self.min_speed=None   # v62: pickup minimum speed for this plan (None = PICKUP_MIN_SPEED)
        self.box=None         # THROUGH_BOX: the gem's pickup box [lo xyz, hi xyz] (None = the THROUGH_SWITCH_U centre test)
        self.fast=fast        # fast family: top target speed; never brakes below the speed carried toward the aim point
        self.ride=False       # 2026-10-24 RideProposal only (the compiled controller runs plain SequenceProposals, where it is False)
        self.direct_from=None # 2026-10-25 night: from this reply on the plan aims straight at the gem (a launch: the route to the take-off
                              # spot, then the jump and the kick toward the gem); None = never (compiled: ABI 12, Q_DIRECT_FROM)
        self.lead_k=0.        # 2026-10-26 night: aiming straight at the gem, the use reply's aim leads for the carried speed (lead_aim with
                              # this kick speed; 0 = off; compiled: ABI 13, Q_LEAD_K)
        self.land=False       # 2026-10-26 night: aiming straight at the gem, LAND_AFTER_USE replies after the use on, the flight steers its
                              # landing point onto the gem (land_action; compiled: ABI 13, Q_LAND..)

    def observe_ticks(self, states):
        if self.box is not None and not self.passed:
            self.passed = _gem_demo.tick_pickup(states, {'bounds': self.box}) is not None

    def __call__(self,state,index):
        if self.through is not None:
            p=np.asarray(state['position'],dtype=float)
            if index==0:self.passed=False;self.previous=None
            if not self.passed and self.box is not None:
                # Only completed native tick sweeps may mark this collected.
                # The enclosing forecast calls observe_ticks after physics.
                pass
            elif not self.passed and self.previous is not None:
                # closest approach of the last step's path to the gem
                a=self.previous;ab=p-a;t=float(np.clip(((self.gem[:3]-a)@ab)/max(float(ab@ab),1e-12),0.,1.))
                self.passed=float(np.linalg.norm(a+t*ab-self.gem[:3]))<THROUGH_SWITCH_U
            elif not self.passed:self.passed=float(np.linalg.norm(p-self.gem[:3]))<THROUGH_SWITCH_U
            self.previous=p
            if self.passed:
                ngem=self.through[1];goal=self.next_route.navigate(p,ngem)[0]
                remaining=float(self.next_route.distance(p))
                distance=remaining if math.isfinite(remaining) else float(np.linalg.norm(ngem[:2]-p[:2]))
                floor,top=PICKUP_MIN_SPEED,leg_top(self.route.terrain,state,distance)
                if self.ride and self.fast:
                    # 2026-10-24 riding Super Speed plans (RideProposal): the speed carried toward the next gem is kept
                    d=np.asarray(goal[:2],dtype=float)-p[:2];n=float(np.linalg.norm(d))
                    along=float(np.asarray(state['velocity'],dtype=float)[:2]@d)/n if n>1e-6 else 0.
                    floor=max(floor,min(along,float(self.fast)));top=max(_gem_demo.TARGET_SPEED_CAP if top is None else top,floor)
                action=Proposal(goal,0.,speed_distance=distance,min_speed=floor,top=top)(state,0)
                action[:4]=[round(v*self.scale,6) for v in action[:4]]
                action[8]=0
                return action
        direct=self.direct or (self.direct_from is not None and index>=self.direct_from)
        if self.land and direct and self.use>=0 and index>=self.use+LAND_AFTER_USE:
            action=land_action(self.gem,state,getattr(self,'land_short',False))
            if action is not None:
                action[:4]=[round(v*self.scale,6) for v in action[:4]]
                action[8]=int(index==self.blast)
                return action
        goal=self.gem if direct else self.route.navigate(state['position'],self.gem)[0]
        # Phases vary steering, throttle, airborne controls and braking. They
        # describe legal controls, never a named trick or a map-specific launch.
        speed_distance=None
        if ROUTE_SPEED_LAW and not direct:
            # The aim point is the next waypoint, a few units ahead: braking
            # for it capped every route-following candidate at 5-6 u/s (v12b
            # journals: mean 4-7 u/s, pickups at 4-6 u/s vs the human's 11.4).
            remaining=float(self.route.distance(state['position']))
            speed_distance=remaining if math.isfinite(remaining) else float(np.linalg.norm(self.gem[:2]-np.asarray(state['position'])[:2]))
        min_speed=(PICKUP_MIN_SPEED if self.min_speed is None else self.min_speed) if not (self.brake_after is not None and index>=self.brake_after) else 0.
        target_distance=speed_distance if speed_distance is not None else float(np.linalg.norm(self.gem[:2]-np.asarray(state['position'])[:2]))
        if self.through is not None:
            # the gem needs no stop: brake only for what follows it
            base=speed_distance if speed_distance is not None else float(np.linalg.norm(self.gem[:2]-np.asarray(state['position'])[:2]))
            if self.arrive is not None:
                # v38 corner: the stopping law brought down to self.arrive at the gem (not to 0)
                speed_distance=base+self.arrive*self.arrive/(2*_gem_demo.BRAKE_DECEL);min_speed=0.
            else:speed_distance=base+self.through[2]
        if self.fast:
            # fast family: the speed already carried toward the aim point (a slope's gift) is kept up to the top
            p=np.asarray(state['position'],dtype=float);v=np.asarray(state['velocity'],dtype=float)
            d=np.asarray(goal[:2],dtype=float)-p[:2];n=float(np.linalg.norm(d))
            along=float(v[:2]@d)/n if n>1e-6 else 0.
            min_speed=max(FAST_MIN_SPEED,min(along,float(self.fast)))
        proposal=Proposal(goal,self.offset if index<self.turn_after else 0.,
                          jump_at=self.jump-index,use_at=self.use-index,
                          brake=self.brake_after is not None and index>=self.brake_after,
                          speed_distance=speed_distance,min_speed=min_speed,
                          top=float(self.fast) if self.fast else leg_top(self.route.terrain,state,target_distance))
        action=proposal(state,0)
        action[:4]=[round(v*self.scale,6) for v in action[:4]]
        action[8]=int(index==self.blast)
        if self.lead_k and direct and index==self.use:action[7]=lead_aim(goal,state,self.lead_k)
        return action


def lead_aim(goal,state,k):
    """The aim (yaw) of a kick adding k u/s along it whose sum with the carried ground velocity heads at `goal` (its sideways
    share capped at LEAD_AIM_MAX). Plain float arithmetic: marble_sim.cpp's lead_aim matches it bit for bit."""
    p=state['position'];v=state['velocity']
    dx=float(goal[0])-float(p[0]);dy=float(goal[1])-float(p[1]);n=math.sqrt(dx*dx+dy*dy)
    if not n>1e-6:return math.atan2(dx,dy)
    ux=dx/n;uy=dy/n
    s=min(LEAD_AIM_MAX,max(-LEAD_AIM_MAX,(float(v[0])*uy-float(v[1])*ux)/k))
    c=math.sqrt(1.-s*s)
    return math.atan2(c*ux-s*uy,c*uy+s*ux)


def land_action(gem,state,short=False):
    """In the air: the reply whose air control moves the free-flight landing point (where the marble comes down to the gem's z +
    LAND_DZ) onto the gem: the acceleration 2 * miss / t^2, at most the full stick (LAND_AIR on the camera diagonal, sqrt 2).
    None on the ground or close to the landing. Plain float arithmetic: marble_sim.cpp's land_action matches it bit for bit.
    `short` (LAND_SHORT, Python only): the landing point is set back from the gem along the line to it by the braking distance
    from the current ground speed down to LAND_SHORT_ARRIVE (the human's long Super Speed flights land short and roll in)."""
    p=state['position'];v=state['velocity']
    px,py,pz=float(p[0]),float(p[1]),float(p[2]);vx,vy,vz=float(v[0]),float(v[1]),float(v[2])
    if not abs(vz)>LAND_MIN_VZ:return None
    disc=vz*vz+2.*LAND_G*(pz-(float(gem[2])+LAND_DZ))
    if not disc>0.:return None
    t=(vz+math.sqrt(disc))/LAND_G
    if not t>LAND_MIN_T:return None
    tx,ty=float(gem[0]),float(gem[1])
    if short:
        dx,dy=tx-px,ty-py;dist=math.sqrt(dx*dx+dy*dy);sp=math.sqrt(vx*vx+vy*vy)
        back=min(max(0.,(sp*sp-LAND_SHORT_ARRIVE*LAND_SHORT_ARRIVE)/(2.*LAND_SHORT_DECEL)),max(0.,dist-2.))
        if dist>1e-6:tx-=dx/dist*back;ty-=dy/dist*back
    ax=2.*(tx-(px+vx*t))/(t*t);ay=2.*(ty-(py+vy*t))/(t*t)
    m=math.sqrt(ax*ax+ay*ay)
    throttle=min(1.,m/(LAND_AIR*math.sqrt(2.))) if m>.05 else 0.
    return reply(action_to_joystick(ax,ay,throttle,0,0,vx,vy),0,math.atan2(float(gem[0])-px,float(gem[1])-py))


class RideProposal(SequenceProposal):
    """SS_RIDE (2026-10-24): a Super Speed plan that rides the kick: the fast family's carried-speed floor (top SS_RIDE_TOP) before
    the pickup and, with `through`, toward the tour's next gem after it. A subclass, so the compiled controller declines it and
    the per-reply forecast runs it (it does not know the post-pickup floor)."""
    def __init__(self,route,gem,steps,use,direct=False,through=None):
        super().__init__(route,gem,steps,use=use,direct=direct,through=through,fast=SS_RIDE_TOP)
        self.ride=True


class BrakeContinuation:
    """Keep a predicted control prefix, then predict ordinary feedback braking."""
    def __init__(self,prefix,gem,extra_replies):
        self.prefix=prefix;self.gem=gem;self.steps=len(prefix)+extra_replies

    def __call__(self,state,index):
        if index<len(self.prefix):return self.prefix[index]
        return Proposal(self.gem,brake=True)(state,0)


class LandingContinuation:
    """Keep a predicted control prefix, then follow the ordinary route until landing.

    An airborne candidate (a jump or blast over an obstacle) used to be rejected
    at the horizon because it had not landed yet. Continue it with the same
    general route-following controls and let the usual checks score the landing.
    """
    def __init__(self,prefix,route,gem,extra_replies):
        self.prefix=prefix;self.follow=SequenceProposal(route,gem,len(prefix)+extra_replies)
        self.steps=len(prefix)+extra_replies

    def __call__(self,state,index):
        if index<len(self.prefix):return self.prefix[index]
        return self.follow(state,index)


GAP_RUN=True              # 2026-10-25 gap crossings (crossings.GAP_ROUTES): the cross phase of a gap route forecasts only
                          # GapRunProposal plans and keeps those that land on the far side (sim Fractured, the generic jump
                          # families: re-planned 'jump in 8 replies' never jumped, the stopping law toward the landing goal
                          # braked the run-up, 12 falls in 44 trips)
GAP_RUN_TOP=15.           # the fastest run (full throttle; rolling tops out near 15 u/s)
GAP_RUN_SPEED_ADD=(1.,2.5)   # the other runs hold the speed the gap needs (crossing v_need) plus these (sim: full-speed runs over a
                             # 3.5 u gap flew past the small island beyond)
GAP_RUN_LOOK_U=4.         # the run aims this far ahead on the crossing line (an offset marble converges onto the line)
GAP_RUN_LEAD_S=.08        # the jump is pressed this much travel time before the take-off point (a reply acts ~1.25 replies later)
GAP_RUN_LIP_U=(-.6,0.)    # take-off points relative to the lip (launch cell centre + GAP_LIP_AHEAD_U)
GAP_RUN_BLAST_AFTER=2     # a gap that needs the Blast fires it this many replies after the jump press (still rising)
GAP_RUN_AIR_REPLIES=40    # forecast replies after the take-off: the flight, the landing and the first metres beyond
GAP_RUN_MAX_REPLIES=96


class GapRunProposal:
    """A gap crossing as one plan: run along the crossing line holding `speed`, press jump when the marble reaches the lip (a
    place, not a reply count, so every re-plan keeps the same take-off point), Blast `blast_after` replies later when the gap
    needs it, keep pushing toward the landing goal while over the gap, then (over the far side) follow `after` = (route, gem)
    with the stopping law, or stop at the landing goal. Not a SequenceProposal subclass: the per-reply forecast runs it."""
    def __init__(self,crossing,route,gem,steps,lip_u=0.,blast_after=-1,after=None,speed=GAP_RUN_TOP):
        self.launch=np.asarray(crossing['launch'][:3],dtype=float);self.u=np.asarray(crossing['direction'][:2],dtype=float)
        self.land=np.asarray(crossing.get('land_goal',crossing['land'])[:3],dtype=float)
        self.land_along=float((self.land[:2]-self.launch[:2])@self.u)
        self.edge_along=float((np.asarray(crossing['land'][:2],dtype=float)-self.launch[:2])@self.u)   # the far side's edge
        lip=crossing.get('lip_along')   # where the floor ends on the crossing line (gap_run_specs), else the cell estimate
        self.trigger=(GAP_LIP_AHEAD_U if lip is None else float(lip))+float(lip_u);self.blast_after=int(blast_after);self.speed=float(speed)
        self.route=copy.copy(route);self.gem=np.asarray(gem[:3],dtype=float);self.steps=int(steps)
        self.after=after;self.jumped=None

    def __call__(self,state,index):
        if index==0:self.jumped=None
        p=np.asarray(state['position'],dtype=float);v=np.asarray(state['velocity'],dtype=float)
        along=float((p[:2]-self.launch[:2])@self.u)
        if self.jumped is None:
            press=along+max(0.,float(v[:2]@self.u))*GAP_RUN_LEAD_S>=self.trigger
            ahead=max(along+1.,min(along+GAP_RUN_LOOK_U,self.land_along))
            aim=np.r_[self.launch[:2]+self.u*ahead,self.launch[2]]
            action=Proposal(aim,0.,jump_at=0 if press else -1,speed_distance=1e3,min_speed=self.speed,top=self.speed)(state,0)
            if press:self.jumped=index
            action[8]=0
            return action
        k=index-self.jumped
        if along<self.edge_along+.5:
            action=Proposal(self.land,0.,speed_distance=1e3,min_speed=self.speed,top=self.speed)(state,0)   # over the gap: full reach
        else:
            # over the far side: brake (the forecast then proves the landing can stop there; the searches after the take-off
            # choose the way on. Following the route on at 9 u/s ran off the small island beyond an 8 u gap, sim 854)
            action=Proposal(self.land,0.,brake=True)(state,0)
        action[8]=int(k==self.blast_after)
        return action


def lip_along(mesh,crossing,radius=.2):
    """Where the floor ends on the crossing line, as a distance along the crossing from the launch cell's centre (the walking
    cells are 1 u apart, so the real edge lies anywhere within about half a cell of it): the first point, marching from 1.5 u
    behind the centre in 0.05 u steps, with no floor within radius + 0.6 u under the launch height. None if no edge is found."""
    L=np.asarray(crossing['launch'][:3],dtype=float);u=np.asarray(crossing['direction'][:2],dtype=float)
    for s in np.arange(-1.5,GAP_LIP_SCAN_U,.05):
        q=L[:2]+u*float(s)
        hit=mesh.raycast(np.array([q[0],q[1],L[2]+.5]),np.array([q[0],q[1],L[2]-radius-.6]),cull_backfaces=True)
        if hit is None or hit['normal'][2]<=.4:return float(s)
    return None


GAP_LIP_SCAN_U=2.5        # lip_along searches this far past the launch cell's centre
GAP_RUN_LIP_EDGE_U=(-.5,-.2)  # with the measured edge: jump presses this far before it (after GAP_RUN_LEAD_S of travel)


def gap_run_specs(engine,route,dynamics,gem,crossing):
    """The cross phase's plans for a gap route: GapRunProposal at the GAP_RUN_LIP_U take-off points, with the Blast when the gap
    needs it. Each forecast runs the whole crossing (the run-up time to the lip + GAP_RUN_AIR_REPLIES)."""
    from nav.marble_dynamics.gap_links import ROLL_ACC
    if 'lip_along' not in crossing:
        try:crossing['lip_along']=lip_along(engine.mesh,crossing,float(dynamics['native']['state'].get('radius',.2) or .2))
        except Exception:crossing['lip_along']=None
    st=dynamics['native']['state'];p=np.asarray(st['position'],dtype=float);v=np.asarray(st['velocity'],dtype=float)
    u=np.asarray(crossing['direction'][:2],dtype=float);rel=p[:2]-np.asarray(crossing['launch'][:2],dtype=float)
    d=max(0.,-float(rel@u))+GAP_LIP_AHEAD_U+abs(float(rel[0]*u[1]-rel[1]*u[0]))
    v0=max(0.,float(v[:2]@u))
    def replies(speed):
        # replies to the lip holding `speed` (from v0, at the roll acceleration; a faster start slows down to it), + the flight
        top=max(speed,v0,.5);d_top=(top*top-v0*v0)/(2.*ROLL_ACC)
        t=(top-v0)/ROLL_ACC+(d-d_top)/top if d>d_top else (-v0+math.sqrt(v0*v0+2.*ROLL_ACC*d))/ROLL_ACC
        return int(min(GAP_RUN_MAX_REPLIES,math.ceil(1.3*t/.064)+GAP_RUN_AIR_REPLIES))
    real=crossing.get('goal')
    after=(cached_route(route.terrain,np.asarray(real[:3],dtype=float)),np.asarray(real[:3],dtype=float)) if real is not None else None
    need=float(crossing.get('needed_meter',0.) or 0.)>0.
    if need and not blast_available(dynamics.get('blast')):return []
    v_need=float(crossing.get('v_need',0.) or 0.)
    speeds=sorted({round(min(GAP_RUN_TOP,v_need+a),2) for a in GAP_RUN_SPEED_ADD}|{GAP_RUN_TOP})
    specs=[]
    for speed in speeds:
        for lip in (GAP_RUN_LIP_EDGE_U if crossing.get('lip_along') is not None else GAP_RUN_LIP_U):
            label=('gap_run_blast_' if need else 'gap_run_jump_')+'%g_%g'%(speed,lip)
            specs.append((label,GapRunProposal(crossing,route,gem,replies(speed),lip_u=lip,blast_after=GAP_RUN_BLAST_AFTER if need else -1,after=after,speed=speed)))
    return specs


def interleave_specs(specs):
    """Round-robin the candidate families so a deadline never starves one of them.

    Families are drive, jump, blast, powerup use, steering and the learned
    proposal; within a family the original order is kept. No gate changes.
    """
    def family(label):
        base=label[7:] if label.startswith('direct_') else label
        if base.startswith('blast'):return 'blast'
        if base.startswith('use'):return 'use'
        if base.startswith('jump'):return 'jump'
        if base.startswith('steering'):return 'steer'
        if base.startswith('frozen_navigator'):return 'nav'
        return 'drive'
    groups={}
    for item in specs:groups.setdefault(family(item[0]),[]).append(item)
    order=['drive','jump','blast','use','steer','nav'];result=[]
    while any(groups.get(k) for k in order):
        for k in order:
            if groups.get(k):result.append(groups[k].pop(0))
    return result


def terminal_support(mesh,tail):
    """Measure settled motion relative to the contacted surface, including slopes.

    World vertical velocity is nonzero during ordinary downhill rolling. Keep
    four supported states, the same separation/speed limits and upward normals;
    test bounce/separation along each actual surface normal instead.
    """
    speeds=[]
    for state in tail:
        p=np.asarray(state['position']);r=state['radius']
        hit=mesh.raycast(p+[0,0,.05],p-[0,0,r+.10],cull_backfaces=True)
        if hit is None or hit['normal'][2]<=.4:return False,False,None
        n=np.asarray(hit['normal']);n=n/np.linalg.norm(n)
        speeds.append(abs(float(np.asarray(state['velocity'])@n)))
    supported=len(tail)>=4
    stable=supported and all(s<2. for s in speeds) and all(np.linalg.norm(s['velocity'])<18. for s in tail)
    return supported,stable,max(speeds,default=None)


def _tick_arrays(native):
    """(positions, velocities) of a forecast's 16 ms tick states: compiled rows (nav/native NativeRows) or state dicts."""
    rows=getattr(native,'rows',None)
    if rows is not None:return np.asarray(rows[:,0:3],dtype=float),np.asarray(rows[:,3:6],dtype=float)
    if not native:return None,None
    return (np.asarray([s['position'] for s in native],dtype=float),np.asarray([s['velocity'] for s in native],dtype=float))


def side_impacts(native):
    """2026-10-23: side impacts in a forecast: ticks losing more than BUMP_TICK_DV of horizontal speed where the velocity change
    is mostly horizontal (a pad side or a wall; a landing or a ramp turns speed upward instead). Consecutive ticks = one hit."""
    P,V=_tick_arrays(native)
    if V is None or len(V)<2:return 0
    h=np.hypot(V[:,0],V[:,1]);dh=h[:-1]-h[1:]
    dvh=np.hypot(V[1:,0]-V[:-1,0],V[1:,1]-V[:-1,1]);dvz=np.abs(V[1:,2]-V[:-1,2])
    hit=(dh>BUMP_TICK_DV)&(dvh>dvz)
    return int(np.count_nonzero(hit[1:]&~hit[:-1])+int(bool(hit[0])))


def cluster_penalty(engine,dynamics,gem,rollout,label,pickup,after,item):
    """2026-10-23 (operator): extra cost (u) of a candidate for (1) side impacts, (2) dropping off the level of the tour's next gem
    after collecting this one, (3) a Super Speed spent on one gem inside a cluster; and the counts, for the journal."""
    native=rollout.get('native_states');P,V=_tick_arrays(native)
    if P is None or len(P)<2:return 0.,None
    lab=str(label or '');cost=0.;why={}
    if not ((BUMP_EXEMPT_WALL and 'wall' in lab) or (BUMP_EXEMPT_EDGE and lab.startswith('edge_hit'))):
        n=side_impacts(native)
        if n:cost+=BUMP_COST_U*n;why['side_impacts']=n
    if pickup is not None and after is not None and LEDGE_DROP_U:
        # the reference is the NEXT gem's level (the gem just taken may be caught mid-air on the way down): when the next gem is
        # not below this one and the marble is up at its level at the pickup, ending up LEDGE_DROP_DZ below it means climbing back
        zg=float(gem[2]);zn=float(np.asarray(after[1],dtype=float)[2])
        k=min(len(P)-1,max(0,int(pickup)*4))
        if zn>zg-1. and float(P[k,2])>zn-LEDGE_DROP_DZ:
            tail=P[k:,2]
            if float(tail.min())<zn-LEDGE_DROP_DZ:cost+=LEDGE_DROP_U;why['ledge_drop']=round(zn-float(tail.min()),2)
        if AWAY_U_PER_U_S and not _gem_demo.TIME_COST:   # TIME_COST prices the velocity away from the next gem in time_to_go
            j=min(len(P)-1,k+4*AWAY_AFTER_REPLIES);d=np.asarray(after[1],dtype=float)[:2]-P[j,:2];nd=float(np.linalg.norm(d))
            if nd>.5:
                away=-float(V[j,:2]@(d/nd))
                if away>1.:cost+=AWAY_U_PER_U_S*away;why['away_u_s']=round(away,1)
    elif pickup is None and LEDGE_DROP_U and LEDGE_TARGET_LEVEL:
        # a plan that does not reach its gem: on the gem's level now, it must not leave it downward (momentum off a ledge
        # between two pickups, then the climb back; the route to a gem on this level never needs a drop)
        zg=float(gem[2])
        if float(P[0,2])>zg-.5 and float(P[:,2].min())<zg-LEDGE_DROP_DZ:
            cost+=LEDGE_DROP_U;why['ledge_drop']=round(zg-float(P[:,2].min()),2)
    held=getattr(rollout['states'],'held',None)
    if held is None:held=np.asarray([int(s.get('held',0)) for s in rollout['states']])
    if SS_ONE_GEM_U and len(held)>1 and bool(np.any((held[:-1]==2)&(held[1:]==0))):
        gems=[i for i in dynamics.get('items',[]) if i.get('gem')]
        near=[i for i in gems if np.linalg.norm(np.asarray(i['position'],dtype=float)-np.asarray(gem[:3],dtype=float))<=SS_CLUSTER_U
              and (item is None or i.get('id')!=item.get('id'))]
        if len(near)>=SS_CLUSTER_GEMS:
            got=sum(1 for i in gems if _gem_demo.tick_pickup(native,i) is not None)
            if got<=1:cost+=SS_ONE_GEM_U;why['ss_one_gem']=got
    # the counts are always computed (journal: choice.cluster_penalty, both arms of a pair); the cost only when enabled
    return (cost if getattr(engine,'cluster_penalties',CLUSTER_PENALTIES) else 0.),(why or None)


def assess(engine,route,dynamics,gem,rollout,label,target_item=None,routed=True,_stop_checked=False):
    states=rollout['states'];end=states[-1]
    gem=np.asarray(gem[:3])
    item=None
    if routed:
        item=target_item or next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-gem)<.1),None)
    detail={}
    after=through_context(engine,route,gem,item) if routed else None
    next_item=None;next_rate=None
    pad_after=None
    if after is not None and getattr(engine,'pad_arrival',False):
        # 2026-10-21 pad arrival: the second-gem credit when the next gem is a low pad right behind this one
        from nav.marble_dynamics.pad_hops import pad_next,PAD_NEXT_U_PER_S
        try:pad_after=pad_next(engine,gem,after)
        except Exception:pad_after=None
        if pad_after is not None:next_rate=PAD_NEXT_U_PER_S
    if after is not None and (_gem_demo.NEXT_PICKUP_U_PER_S or next_rate or _gem_demo.TIME_COST):   # v57: the next gem's item (second pickup)
        next_item=next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-after[1])<.1),None)
    cost,close,pickup=candidate_cost(route,states,gem,routed,item,after,
                                     native=rollout.get('native_states'),detail=detail,next_item=next_item,next_rate=next_rate)
    presses=jump_presses(rollout.get('proposed') or [])
    if presses and JUMP_COST_U and not getattr(engine,'crossing_phase',False) and float(gem[2])-float(states[0]['position'][2])<JUMP_FREE_RISE_U:
        # v51 (operator: "jumping too much and stumbling around when collecting tight clusters"): a jump is charged
        # unless the gem is up high, so a jump plan wins only when it is clearly faster (v49 clusters: 73 % of the
        # jumps with the gem < 0.5 u above the marble; the end-speed penalty made a jump plan win every tie)
        cost+=JUMP_COST_U*presses
    penalty,penalty_why=cluster_penalty(engine,dynamics,gem,rollout,label,pickup,after,item) if routed else (0.,None)
    cost+=penalty
    if TRAVEL_TURN_U_PER_RAD and routed and pickup is None:
        # 2026-10-24: on a trip (the gem TRAVEL_MODE_FROM_U or more along the route) a snaking plan loses speed: charge its turning
        d0=float(route.distance(states[0]['position']))
        if math.isfinite(d0) and d0>TRAVEL_MODE_FROM_U:
            _P,_V=_tick_arrays(rollout.get('native_states'))
            if _V is not None and len(_V)>2:
                hs=np.hypot(_V[:,0],_V[:,1]);ok=hs>3.;ang=np.arctan2(_V[:,1],_V[:,0])
                d=np.abs((np.diff(ang)+np.pi)%(2*np.pi)-np.pi);d=d[ok[1:]&ok[:-1]]
                turn=float(d.sum());cost+=TRAVEL_TURN_U_PER_RAD*turn
                if penalty_why is None:penalty_why={}
                penalty_why['turn_rad']=round(turn,2)
    tail=states[-4:]
    supported,stable,normal_speed=terminal_support(engine.mesh,tail)
    if (SS_RIDE and not stable and supported and normal_speed is not None and normal_speed<2. and 'use' in str(label)
            and int(dynamics['native']['state']['held'])==2):
        stable=True   # 2026-10-24 SS_RIDE: a Super Speed plan may end above 18 u/s on the floor (the braking backup still checks edges)
    reason=None
    if rollout['failure']:reason='prediction_failure'
    elif rollout['ambiguous_intervals']:reason='contact_ambiguity'
    elif not stable:reason='no_supported_continuation'
    if not math.isfinite(cost):reason='nonfinite_goal_cost'
    if reason is None and _stop_checked and getattr(engine,'terminal_stop_replies',0) and np.linalg.norm(end['velocity'])>=1.:
        reason='stopping_speed_above_limit'
    held=getattr(states,'held',None)   # 2026-10-19: packed compiled forecasts (nav/native/forecast.py) need no state dicts here
    if held is not None:resources=int(((held[:-1]>0)&(held[1:]==0)).sum())
    else:resources=sum(int(a['held']>0 and b['held']==0) for a,b in zip(states,states[1:]))
    sec=getattr(rollout.get('controls'),'script_event_count',None)
    resources+=sec if sec is not None else sum(len(c.get('script_events',[])) for c in rollout.get('controls',[]))
    info=dict(label=label,cost=cost,rejected=reason,minimum_gem_distance_u=close,resources_consumed=resources,
                goal_item_id=item.get('id') if item else None,
                end_position=end['position'],terminal_support=supported,stable_continuation=stable,
                maximum_tail_normal_speed_u_per_s=normal_speed,
                continuation_s=min(3,len(states)-1)*.064,possible_pickup_interval=pickup,
                ambiguous_intervals=rollout['ambiguous_intervals'],failure=rollout['failure'])
    if 'post_u' in detail:info['through_post_u']=float(detail['post_u'])
    if 't_next' in detail:info['t_next_s']=round(float(detail['t_next']),3)   # TIME_COST: estimated time to the tour's next gem
    if penalty_why:info['cluster_penalty']=dict(penalty_why,u=round(penalty,2))
    stopping_replies=getattr(engine,'terminal_stop_replies',0)
    stopping_scope=getattr(engine,'terminal_stop_scope','all')
    needs_stop=stopping_scope=='all' or pickup is not None or not routed
    if reason is None and stopping_replies and needs_stop and not _stop_checked:
        from nav.marble_dynamics.viability import braking_viability
        check,continued=braking_viability(engine,dynamics,rollout,gem,stopping_replies)
        if not check['valid']:
            info['rejected']='no_stopping_continuation:'+check['rejected']
            info['terminal_stop_check']=check
            return info
        # Keep the verified backup controls in the actual proposed schedule,
        # including the bridge's two delayed commands. A native gem pickup can
        # preserve this stopping destination via the existing landing contract.
        rollout.clear();rollout.update(continued)
        info=assess(engine,route,dynamics,gem,rollout,label,target_item,routed,True)
        info['terminal_stop_check']=check
        info['forecast_s']=(len(rollout['states'])-1)*.064
        if routed and item and item.get('gem') and info['possible_pickup_interval'] is not None:
            info['landing_goal']=list(rollout['states'][-1]['position'])
            info['landing_after_reply']=info['possible_pickup_interval']
            p=np.asarray(info['landing_goal']);r=rollout['states'][-1]['radius']
            hit=engine.mesh.raycast(p+[0,0,.05],p-[0,0,r+.10],cull_backfaces=True)
            info['landing_triangle_id']=hit['triangle_id'] if hit else None
    return info


def blasts_fired(rollout):
    """Blast impulses in a forecast (script events of the regular Blast)."""
    n=getattr((rollout or {}).get('controls'),'script_event_count',None)   # compiled forecasts: every script event is a Blast impulse
    if n is not None:return n
    return sum(1 for c in (rollout or {}).get('controls',[]) for e in c.get('script_events',[]) if e.get('kind')=='blast_impulse')


def keep_blast_for(position,gem):
    """v48: the target needs the Blast to climb (above plain jump reach) and is not far away."""
    p=np.asarray(position[:3],dtype=float);g=np.asarray(gem[:3],dtype=float)
    return KEEP_BLAST and float(g[2]-p[2])>=KEEP_BLAST_RISE_U and float(np.linalg.norm(g[:2]-p[:2]))<=KEEP_BLAST_NEAR_U


CLUSTER_TWO_LEVEL=8        # 2026-10-17 (operator: clusters): number of best pickup plans re-ranked by pickup time + a follow-up
                           # forecast to the tour's next gem (0 = off)
CLUSTER_LEG_U=12.          # ... only when the next gem is within this walking distance of the target
CLUSTER_FOLLOW_REPLIES=32  # follow-up forecast length (replies)
CLUSTER_EXTRA_S=.8         # v69: the follow-ups may run this far past the search deadline (pools: the game waits)
TL_JUMP_FOLLOW=True        # ON by default since 2026-10-25 05:25 (pair pf4 +1.0 a round, TreasureBox +5 / +31, Fractured +16 / +9; sim clear -3.1 percent, smacks -28 percent); 2026-10-25 (cluster speed): follow-ups also jump (direct aim, at TL_JUMP_TIMES replies) when the next gem
TL_JUMP_TIMES=(2,6,10,14)  # is TL_JUMP_RISE_U above the follow-up start or on a raised pad (pf2 TreasureBox: 30-43 percent of the rankings
TL_JUMP_RISE_U=0.          # had no follow-up pickup with the rolling controllers alone); 0 = pad gems only (sim s4: the 0.2 u rise
                           # test fired on Cragmire / Duplex slopes: x1.163 / x1.134, 6 of 7 Cragmire starts slower with lp)


CLUSTER_LINES=True
_LINE_ALLOWED={}


def line_specs(engine,route,dynamics,gem,through,steps):
    """Racing-line plans (A,B) and (A,B,C) for a flat cluster."""
    from nav.marble_dynamics import racing_line as RL
    st=dynamics['native']['state'];p=np.asarray(st['position'],dtype=float);v=np.asarray(st['velocity'],dtype=float)
    A=np.asarray(gem[:3],dtype=float);B=np.asarray(through[1][:3],dtype=float)
    if abs(A[2]-p[2])>1. or abs(B[2]-A[2])>1. or st.get('mode',0)!=0:return []
    gems=[np.asarray(i['position'],dtype=float) for i in dynamics.get('items',[]) if i['gem']]
    rest=[g for g in gems if np.linalg.norm(g-A)>.1 and np.linalg.norm(g-B)>.1 and abs(g[2]-B[2])<1. and np.linalg.norm((g-B)[:2])<=12.]
    C=min(rest,key=lambda g:float(np.linalg.norm((g-B)[:2]))) if rest else None
    key=(id(route.terrain),round(float(p[2]),1))
    al=_LINE_ALLOWED.get(key)
    if al is None:al=_LINE_ALLOWED[key]=RL.Allowed(route.terrain,float(p[2]))
    rng=np.random.default_rng(0);out=[]
    for name,seq in (('line2',[A,B]),('line3',[A,B,C] if C is not None else None)):
        if seq is None:continue
        pts,speeds,gi,cost,tend=RL.optimise(p,v,[g[:2] for g in seq],al,rng)
        out.append((name,RL.TrackLine(pts,speeds,float(A[2]),steps)))
    return out


def follow_time(engine,nroute,dynamics,ngem,nitem):
    """Earliest game-tick pickup (replies) of the next gem from `dynamics` over a few compiled controllers, or None."""
    best=None;n=CLUSTER_FOLLOW_REPLIES
    bounces=()
    if WALL_BOUNCE:   # 2026-10-19: a pickup that leaves the marble heading away from the next gem can bank off a wall to it
        from nav.marble_dynamics.wall_bounce import bounce_specs
        bounces=tuple(spec for _,spec in bounce_specs(engine,nroute,dynamics,ngem,None,n))
    jumps=()
    if TL_JUMP_FOLLOW:
        z0=float(dynamics['native']['state']['position'][2]);raised=bool(TL_JUMP_RISE_U) and float(ngem[2])-z0>=TL_JUMP_RISE_U
        if not raised:
            from nav.marble_dynamics.pad_hops import pad_under
            try:raised=pad_under(engine,ngem) is not None
            except Exception:raised=False
        if raised:jumps=tuple(SequenceProposal(nroute,ngem,n,jump=j,direct=True) for j in TL_JUMP_TIMES if j<n)
    for spec in (SequenceProposal(nroute,ngem,n),SequenceProposal(nroute,ngem,n,direct=True),
                 SequenceProposal(nroute,ngem,n,offset=-.5,turn_after=12),SequenceProposal(nroute,ngem,n,offset=.5,turn_after=12),
                 SequenceProposal(nroute,ngem,n,scale=.7))+jumps+bounces:
        try:roll=predict_replies(engine,dynamics,spec)
        except ValueError:continue
        if roll['failure']:continue
        k=_gem_demo.possible_pickup(roll['states'],nitem,roll.get('native_states'))
        if k is not None and (best is None or k<best):best=k
    return best


import atexit as _atexit, sys as _sys
from collections import Counter as _Counter
TL_STATS=_Counter()
# TL_STATS: cluster re-ranking counters (searches, cluster, reranked, changed, line_pickup, line_chosen)


def apply_two_level(engine,route,dynamics,gem,pairs,target_item):
    """Cluster re-ranking (CLUSTER_TWO_LEVEL): see the constant."""
    TL_STATS['searches']+=1
    if not CLUSTER_TWO_LEVEL or getattr(engine,'_two_level_active',False) or getattr(engine,'crossing_phase',False):return
    item=target_item or next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-gem)<.1),None)
    after=through_context(engine,route,gem,item)
    if after is None:return
    nroute,ngem,leg=after
    if not leg<=CLUSTER_LEG_U:return
    TL_STATS['cluster']+=1
    nitem=next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-ngem)<.1),None)
    if nitem is None:return
    allc=sorted([(info,roll) for info,roll in pairs if info['rejected'] is None and roll is not None
                  and info.get('possible_pickup_interval') is not None],key=lambda p:p[0]['cost'])
    nre=3 if getattr(engine,'_light_search',False) else CLUSTER_TWO_LEVEL
    ss=SS_RIDE and int(dynamics['native']['state']['held'])==2   # 2026-10-24: Super Speed plans are always ranked by the time to both gems
    cands=allc[:nre]+[p for p in allc[nre:] if p[0]['label'].startswith(('line','bounce')) or (ss and 'use' in p[0]['label'])]
    TL_STATS['line_pickup']+=sum(1 for p in allc if p[0]['label'].startswith('line'))
    if len(cands)<2:return
    from nav.marble_dynamics.viability import resume_forecast
    saved=getattr(engine,'next_for',None);engine._two_level_active=True;engine.next_for={};results=[]
    saved_dl=getattr(engine,'planning_deadline',None)
    if saved_dl is not None:engine.planning_deadline=saved_dl+CLUSTER_EXTRA_S
    try:
        for info,roll in cands:
            k=int(info['possible_pickup_interval'])
            if k>len(roll['proposed']):continue
            if info['label'].startswith(('line','bounce')):   # these plans drive to the next gem themselves
                kb=_gem_demo.possible_pickup(roll['states'],nitem,roll.get('native_states'))
                if kb is not None:
                    results.append((info,max(k,kb),0));continue
            tr=dict(roll,states=roll['states'][:k+1],controls=roll['controls'][:k],proposed=roll['proposed'][:k],
                    failure=None,ambiguous_intervals=[])
            try:dyn=resume_forecast(dynamics,tr)
            except ValueError:continue
            dyn['items']=[i for i in dynamics.get('items',[]) if i is not item]
            results.append((info,k,follow_time(engine,nroute,dyn,ngem,nitem)))
    except PlanningBudgetExceeded:pass   # keep the plans already followed up
    finally:engine._two_level_active=False;engine.next_for=saved;engine.planning_deadline=saved_dl
    if len(results)<2:return
    TL_STATS['reranked']+=1;before=min(results,key=lambda r:r[0]['cost'])[0]['label']
    for info,k,k2 in results:
        total=k+(k2 if k2 is not None else CLUSTER_FOLLOW_REPLIES+8)
        info['two_level_replies']=total;info['two_level_next']=k2
        # below every unchecked plan (pickup plans cost about -100), ordered by the total time to both gems; a plan that
        # fires the Blast keeps its meter charge (v84: re-ranking dropped it, so Blasts that save nothing over the two
        # gems won ties: Sprawl v81 s1 decision 2008, 52 vs 51 replies, -136.7 vs -136.9 without the 4.6 u charge)
        info['cost']=-150.+total*.064*_gem_demo.PICKUP_TIME_WEIGHT+float(info.get('blast_charge_u',0.) or 0.)
    after_label=min(results,key=lambda r:r[0]['cost'])[0]['label']
    TL_STATS['changed']+=int(after_label!=before);TL_STATS['line_chosen']+=int(after_label.startswith('line'))


def useful_candidate(pairs,route,start_distance,preserve_resources=False,keep_blast=False):
    """Choose among checked pickup/progress futures, with optional resource cost.

    Preserve mode prefers a predicted collection, then fewer actual modeled
    resource consumptions, then the existing route/continuation cost. A useful
    walking future therefore keeps scarce inventory unless a pickup or an
    otherwise inaccessible useful future justifies using it. No reward changes.
    """
    useful=[]
    for info,rollout in pairs:
        if info['rejected'] is not None or rollout is None:continue
        progress=start_distance-route.distance(rollout['states'][-1]['position'])
        if info['possible_pickup_interval'] is not None or progress>.3:
            useful.append((info,rollout))
    if not useful:return None
    def rank(pair):
        info=pair[0]
        # v48: below a raised gem a plan that spends the Blast without collecting it comes after every plan that keeps it
        spent=int(keep_blast and info['possible_pickup_interval'] is None and blasts_fired(pair[1])>0)
        return (spent,int(info['possible_pickup_interval'] is None),info.get('resources_consumed',0),info['cost']) if preserve_resources else (spent,info['cost'])
    return min(useful,key=rank)


def search(engine,route,dynamics,raw,gem,navigator,horizon=40,warm=None,brake_replies=0,target_item=None,learned=True):
    engine._speculated=None
    gem=np.asarray(gem[:3]);route.navigate(dynamics['native']['state']['position'],gem)
    held=int(dynamics['native']['state']['held'])
    # Holding an unused Helicopter must not lengthen every rolling proposal or
    # defeat explicitly short fallbacks. Actual activation/active flight still
    # receives the longer landing horizon and all existing physical checks.
    steps=max(horizon,96) if dynamics['native']['state']['effects'][5][0] else horizon
    use_steps=max(steps,96) if held==5 else steps
    # A powerup's effect outlasts a short horizon: judged over 1.5 s a Super
    # Speed is still too fast to count as settled and loses to rolling. With
    # use_horizon set, use candidates are forecast over at least that many replies.
    use_floor=int(getattr(engine,'use_horizon',0) or 0)
    if use_floor and 1<=held<=4:use_steps=max(use_steps,use_floor)
    specs=[('route',SequenceProposal(route,gem,steps)),('direct',SequenceProposal(route,gem,steps,direct=True))]
    gem_item=target_item or next((i for i in dynamics.get('items',[]) if i['gem'] and np.linalg.norm(np.asarray(i['position'])-gem)<.1),None)
    through=through_context(engine,route,gem,gem_item)
    here=dynamics['native']['state']['position'];reach=float(route.distance(here))
    if not math.isfinite(reach):reach=float(np.linalg.norm(gem[:3]-np.asarray(here)))
    if through is not None:
        if reach<=THROUGH_REACH_U:
            specs+=[('through',SequenceProposal(route,gem,steps,through=through)),
                    ('direct_through',SequenceProposal(route,gem,steps,direct=True,through=through)),
                    ('through_0.7',SequenceProposal(route,gem,steps,scale=.7,through=through))]
            if CLUSTER_LINES and through[2]<=CLUSTER_LEG_U and not getattr(engine,'crossing_phase',False):
                specs+=line_specs(engine,route,dynamics,gem,through,steps)
            if THROUGH_STEER:
                # 2026-10-17 v54 (operator: tight clusters quicker): curved approaches, off the line first (offset for 6 or
                # 12 replies), then through the gem toward the next one, so the marble can leave a gem heading to the next
                for off in (-1.,-.5,.5,1.):
                    for turn in (6,12):
                        specs.append(('through_steer_'+str(off)+'_'+str(turn),SequenceProposal(route,gem,steps,offset=off,turn_after=turn,through=through)))
            arrive=corner_speed(route,dynamics['native']['state']['position'],gem,through) if CORNER_PROPOSALS else None
            if arrive is not None:
                specs+=[('through_corner',SequenceProposal(route,gem,steps,through=through,arrive=arrive)),
                        ('direct_through_corner',SequenceProposal(route,gem,steps,direct=True,through=through,arrive=arrive))]
                if CORNER_SLOW:
                    slow=max(1.5,CORNER_SLOW_SHARE*arrive)
                    specs+=[('through_corner_slow',SequenceProposal(route,gem,steps,through=through,arrive=slow)),
                            ('direct_through_corner_slow',SequenceProposal(route,gem,steps,direct=True,through=through,arrive=slow))]
            if getattr(engine,'pad_arrival',False):
                # 2026-10-21 pad arrival (operator: the marble smacks into the pad instead of jumping): when the gem after this
                # one sits on a low pad close behind it, arrive at the hop speed for that distance so the hop can launch here
                from nav.marble_dynamics.pad_hops import hop_arrival_speed,pad_next,ThroughHopProposal
                try:hv=hop_arrival_speed(engine,gem,through);_pn=pad_next(engine,gem,through)
                except Exception:hv=None;_pn=None
                if hv is not None and _pn is not None:
                    specs+=[('through_hop',ThroughHopProposal(route,gem,steps,through,hv,_pn[0])),
                            ('direct_through_hop',ThroughHopProposal(route,gem,steps,through,hv,_pn[0],direct=True))]
    for top in FAST_TOPS:
        fast=top if FAST_PROPOSALS else None
        specs+=[('fast_'+str(int(top)),SequenceProposal(route,gem,steps,fast=fast)),
                ('direct_fast_'+str(int(top)),SequenceProposal(route,gem,steps,direct=True,fast=fast))]
    travel=TRAVEL_SKIP_SLOW and reach>TRAVEL_MODE_FROM_U
    if not travel:
        for scale in (.4,.7):specs.append(('throttle_'+str(scale),SequenceProposal(route,gem,steps,scale=scale)))
    if WALL_BOUNCE and not getattr(engine,'crossing_phase',False):
        # 2026-10-19 (operator: use wall hits): moving fast away from the target, bank off a wall toward it (wall_bounce.py)
        from nav.marble_dynamics.wall_bounce import bounce_specs,STEPS as BOUNCE_STEPS
        specs+=bounce_specs(engine,route,dynamics,gem,through,max(steps,BOUNCE_STEPS))
    # Climb crossings (opt-in) try more jump times and Blast-first families; gap
    # crossings keep the v7 families.
    _cr=(getattr(engine,'crossing',None) or {})
    climbing=bool(getattr(engine,'crossing_phase',False) and (_cr.get('climb') or _cr.get('gap_route')))   # 2026-10-25 gap crossings: the climb families (jumps every 2 replies, Blast 2 replies after the jump)
    if (RAISED_BLAST_FIRST and not climbing and raised_near(dynamics['native']['state']['position'],gem)
            and blast_meter_value(dynamics)>=float(getattr(engine,'detour_blast_meter',0.5) or 0.5) and not getattr(engine,'blast_reserved',False)):
        # v39: below a nearby raised gem with the meter charged, the climb families (a jump every 2 replies, the Blast 2
        # replies after it while rising, direct aim, half-throttle variants) go first, as for a climb crossing
        climbing=True
    jump_times=(0,2,4,6,8,10,12) if climbing else (0,2,5,9)
    lip_times=()
    if climbing and _cr.get('gap_route'):
        # 2026-10-25 gap crossings: jumps timed to the lip (the run-up from the staging point takes ~14 replies; the climb times
        # stop at 12, so every plain jump took off ~2 u early and fell short: sim Fractured decision 643 oscillated 12 s at the edge)
        from nav.marble_dynamics.gap_links import ROLL_ACC
        _st0=dynamics['native']['state'];_u=np.asarray(_cr['direction'],dtype=float)
        _d=max(0.,float((np.asarray(_cr['launch'][:2],dtype=float)-np.asarray(_st0['position'][:2],dtype=float))@_u))+GAP_LIP_AHEAD_U
        _v0=max(0.,float(np.asarray(_st0['velocity'][:2],dtype=float)@_u))
        _r=int(round(((-_v0+math.sqrt(_v0*_v0+2.*ROLL_ACC*_d))/ROLL_ACC)/.064))
        lip_times=tuple(j for j in range(_r-2,_r+2) if 0<=j<steps)
        # a lean set (sim: the climb times x route/direct + two Blast plans each, all with 40-reply landings, ran past the
        # search budget at speed and handed the marble to A2, which drove it off the edge at 20 u/s)
        jump_times=tuple(sorted({0,4,8}|set(lip_times)))
    if getattr(engine,'pad_hops',False) or BLAST_GATE_ALWAYS:
        from nav.marble_dynamics.pad_hops import blast_gated as _bg
        try:engine._pad_gate_now=_bg(engine,gem,dynamics['native']['state'])
        except Exception:engine._pad_gate_now=False
    else:engine._pad_gate_now=False
    for jump in jump_times:
        specs.append(('jump_'+str(jump),SequenceProposal(route,gem,steps,jump=jump)))
        specs.append(('direct_jump_'+str(jump),SequenceProposal(route,gem,steps,jump=jump,direct=True)))
    if getattr(engine,'pad_arrival',False) and not getattr(engine,'crossing_phase',False):
        # 2026-10-21 pad arrival: at a pad gem itself, the timed hop (descend onto the gem) as a candidate
        from nav.marble_dynamics.pad_hops import hop_specs
        try:specs+=hop_specs(engine,route,gem,steps,through,dynamics['native']['state'])
        except Exception:pass
    if getattr(engine,'pad_hops',False) and not getattr(engine,'crossing_phase',False):
        # 2026-10-21 (operator: plan the whole spawn): chain plans through 3-4 gems of the target's spawn in several
        # orders (nearest, straight, aligned, sweep, top-down, rows) at steady speeds, hopping pad gems; scored by every
        # gem the forecast collects (spawn_plans.chain_score). nav/marble_dynamics/spawn_plans.py
        from nav.marble_dynamics.spawn_plans import chain_specs
        try:specs+=chain_specs(engine,route,dynamics,gem,gem_item,steps,through)
        except Exception:pass
    _raised=late_raised(dynamics['native']['state']['position'],gem)
    if LATE_JUMPS and not climbing and (_raised or late_pad(engine,dynamics['native']['state']['position'],gem)):
        # 2026-10-16 (operator: "it goes until it hits a wall and only then jumps and blasts"): toward a raised gem the
        # standard families jump within 0.6 s, so from a few metres back no plan climbs and 'drive at the gem' wins until
        # the wall stops the marble. Later jumps, alone or with a Blast 2-6 replies after (while rising), let the climb
        # start on the approach. Offline at wall-hit spots (Sprawl, Vortex, Cragmire): pickups 0.7-2.7 s sooner where
        # the standard families had no pickup plan at all.
        late_blast=_raised and blast_available(dynamics.get('blast')) and not getattr(engine,'blast_reserved',False) and not getattr(engine,'_pad_gate_now',False)
        for jump in LATE_JUMP_TIMES:
            if jump>=steps:break
            for direct in (False,True):
                pre='direct_' if direct else ''
                specs.append((pre+'jump_'+str(jump),SequenceProposal(route,gem,steps,jump=jump,direct=direct)))
                for after in (LATE_BLAST_AFTER if late_blast else ()):
                    if jump+after<steps:
                        specs.append((pre+'blast_'+str(jump+after)+'_jump_'+str(jump),
                                      SequenceProposal(route,gem,steps,blast=jump+after,jump=jump,direct=direct)))
    for offset in (-1.,-.5,.5,1.):
        specs.append(('steering_'+str(offset),SequenceProposal(route,gem,steps,offset=offset,turn_after=12)))
    if not travel:specs.append(('accelerate_then_brake',SequenceProposal(route,gem,steps,brake_after=12)))
    # 2026-10-16 (operator: efficient powerup use): a Super Speed is not spent on a short leg to the LAST gem of a group.
    # The next group spawns only when that gem is taken, at least 30 u away (spawnBlock) in a direction nobody knows
    # yet, so a long trip always follows; a kick into the last gem saves well under a second and arrives at ~20 u/s
    # in a random direction (Cube Isle 10-16: 11 u the wrong way, ~2.5 s lost, and no Super Speed for the 88 u trip).
    last_gem=bool(gem_item is not None and gem_item.get('gem') and not any(
        np.linalg.norm(np.asarray(g[:3])-gem)>.1 for g in visible_gems(raw)))
    keep_ss=(held==2 and last_gem and reach<SS_RESERVE_LAST_U and float(raw[RAW_TIME_LEFT])>SS_RESERVE_TIME_MS)
    # 2026-10-20: a held Mega (inventory held_type 6; the native 'held' is 0 for it) gets the same use families when
    # --mega-proposals is on; its forecasts run through the script-event model (bridge.mega_relevant).
    mega_held=bool(getattr(engine,'mega_enabled',False) and (dynamics.get('inventory') or {}).get('held_type')==6
                   and not (dynamics.get('inventory') or {}).get('permanent_mega'))
    if (1<=held<=5 and not keep_ss) or mega_held:
        for use,jump in ((0,-1),(3,-1),(7,-1),(3,2),(7,5)):
            specs.append(('use_'+str(use)+'_jump_'+str(jump),SequenceProposal(route,gem,use_steps,use=use,jump=jump)))
            direct_spec=SequenceProposal(route,gem,use_steps,use=use,jump=jump,direct=True)
            if DIRECT_LEAD and held==2:direct_spec.lead_k=POW_LAUNCH_LEAD_K
            specs.append(('direct_use_'+str(use)+'_jump_'+str(jump),direct_spec))
        if POW_LAUNCH and held in POW_LAUNCH_TYPES and not mega_held and reach>=POW_LAUNCH_MIN_LEG_U:
            # 2026-10-25 night take-off launches: the use pressed with or just before the jump, so the kick (2 decisions after the
            # use) fires at the take-off, at points along the way; each forecast runs through the flight and the landing. Two aims:
            # along the route at the kick, or the route to the take-off spot and then the jump and the kick at the gem (direct_from)
            for at in POW_LAUNCH_AT:
                for lead in POW_LAUNCH_LEADS:
                    use,jump=at,at+lead;n=max(use_steps,jump+POW_LAUNCH_FLIGHT)
                    specs.append(('use_launch_%d_%d'%(use,jump),SequenceProposal(route,gem,n,use=use,jump=jump)))
                    aim=SequenceProposal(route,gem,n,use=use,jump=jump);aim.direct_from=use
                    if held==2:aim.lead_k=POW_LAUNCH_LEAD_K
                    aim.land=POW_LAUNCH_LAND
                    if LAND_SHORT and held==2:aim.land_short=True   # Python per-reply forecast (forecast._controller declines it)
                    specs.append(('use_launch_aim_%d_%d'%(use,jump),aim))
                if POW_LAUNCH_ROLL and held==2:
                    # 2026-10-26 (under test): the kick without a jump, aimed (lead) at the gem, the terrain doing the launching (Sprawl
                    # human demo: the Super Speed fired before a ramp, 37 u/s up the ramp, a 65 u flight; a jump at the kick flies over it)
                    for at in POW_LAUNCH_AT:
                        roll=SequenceProposal(route,gem,max(use_steps,at+POW_LAUNCH_FLIGHT),use=at,jump=-1);roll.direct_from=at
                        roll.lead_k=POW_LAUNCH_LEAD_K;roll.land=POW_LAUNCH_LAND
                        if LAND_SHORT:roll.land_short=True
                        specs.append(('use_launch_roll_%d'%at,roll))
        if SS_RIDE and held==2:
            for use in SS_RIDE_USES:
                specs.append(('use_ride_'+str(use),RideProposal(route,gem,use_steps,use)))
                specs.append(('direct_use_ride_'+str(use),RideProposal(route,gem,use_steps,use,direct=True)))
                if through is not None:
                    specs.append(('use_ride_through_'+str(use),RideProposal(route,gem,use_steps,use,through=through)))
                    specs.append(('direct_use_ride_through_'+str(use),RideProposal(route,gem,use_steps,use,direct=True,through=through)))
    _pad_gate=False
    if getattr(engine,'pad_hops',False) or BLAST_GATE_ALWAYS:
        # 2026-10-21: no Blast for a low pad gem close by (a jump reaches it; 32 of 52 Blasts gained under 1 u)
        from nav.marble_dynamics.pad_hops import blast_gated
        try:_pad_gate=blast_gated(engine,gem,dynamics['native']['state'])
        except Exception:_pad_gate=False
    if blast_available(dynamics.get('blast')) and not getattr(engine,'blast_reserved',False) and not _pad_gate:
        if climbing:
            # Crossing: the launch may be several replies away; a Blast two replies
            # after the jump (while still rising) gives the highest flight. These
            # go first so the search deadline cannot starve them, direct aim only,
            # with a half-throttle variant for short ledges.
            blast_specs=[]
            if _cr.get('gap_route'):
                # 2026-10-25 gap crossings: a Blast only where the gap needs one, at the lip-timed jumps
                if float(_cr.get('needed_meter',0.) or 0.)>0.:
                    for jump in lip_times:
                        blast_specs.append(('direct_blast_'+str(jump+2)+'_jump_'+str(jump),SequenceProposal(route,gem,steps,blast=jump+2,jump=jump,direct=True)))
            else:
                for jump in jump_times:
                    blast_specs.append(('direct_blast_'+str(jump+2)+'_jump_'+str(jump),SequenceProposal(route,gem,steps,blast=jump+2,jump=jump,direct=True)))
                    blast_specs.append(('direct_half_blast_'+str(jump+2)+'_jump_'+str(jump),SequenceProposal(route,gem,steps,blast=jump+2,jump=jump,direct=True,scale=.5)))
                blast_specs.append(('direct_blast_0_jump_-1',SequenceProposal(route,gem,steps,blast=0,jump=-1,direct=True)))
            specs=blast_specs+specs
        else:
            for blast,jump in ((0,-1),(3,0),(7,0),(7,3),(11,5)):
                for direct in (False,True):
                    label=('direct_' if direct else '')+'blast_'+str(blast)+'_jump_'+str(jump)
                    specs.append((label,SequenceProposal(route,gem,steps,blast=blast,jump=jump,direct=direct)))
    other_gems=[g for g in visible_gems(raw) if np.linalg.norm(np.asarray(g[:3])-gem)>.1]
    nxt=min(other_gems,key=lambda g:np.linalg.norm(np.asarray(g[:3])-gem)) if other_gems else None
    next_goal=snap_to_walkable(route.terrain,*nxt[:3])[0] if nxt else None
    # learned=False (reliability_eval.pick, 2026-10-16 v43): no learned proposal for this gem (it runs per reply in Python)
    nav_off=bool(NAV_SPAWN_OFF and reach<=NAV_SPAWN_U and sum(1 for g in visible_gems(raw) if np.linalg.norm(np.asarray(g[:2])-gem[:2])<=12.)>=NAV_SPAWN_GEMS)
    navigator_spec=('frozen_navigator',NavigatorProposal(navigator,raw,route,gem,next_goal,use_steps)) if learned and navigator is not None and not nav_off else None
    if navigator_spec is not None:navigator_spec[1].native=getattr(engine,'native_sim',None)   # 2026-10-19: compiled routing (exact)
    # Under a short wall budget, checking this last can starve the competent
    # learned proposal entirely. Ordering changes no physical acceptance gate.
    if getattr(engine,'use_first',False) and 1<=held<=5:
        # Holding a powerup: judge its use candidates before the deadline can
        # cut them off; the ordinary families still follow (at most ~10 moved).
        uses=[s for s in specs if 'use' in s[0]];others=[s for s in specs if 'use' not in s[0]]
        specs=others[:2]+uses+others[2:]
    detour_first=None
    if held in (1,5) and float(gem[2])-float(dynamics['native']['state']['position'][2])>=float(getattr(engine,'use_first_rise_u',1.5) or 1.5):
        # v21 (2026-10-15): holding a Super Jump or Helicopter below a raised gem,
        # the use plans are the ones that can collect it; judge them first so the
        # search deadline cannot starve them (it did: decision 2237 of the v16
        # Sprawl s1 round chose 'steering' with the pickup plan never reached).
        uses=[s for s in specs if 'use' in s[0]];others=[s for s in specs if 'use' not in s[0]]
        specs=uses+others
        # and project their landings right after the family (the end-of-search
        # projection is skipped once the deadline has hit)
        detour_first=uses[-1][0] if uses else None
    ss_early=None;ss_names=('use_0_jump_-1','direct_use_0_jump_-1')
    if held==2 and reach>=SS_EARLY_U and not getattr(engine,'crossing_phase',False):
        # 2026-10-16 (operator: do not miss a safe Super Speed): on a long leg the two 'use now' plans go right after
        # route/direct and their landings are projected at once. Judged after the 17 rolling plans, with the landing
        # projection left to the end of the search, they were cut by the 2 s deadline on every search of a 55 u
        # straight trip on Cube Isle (10-16, 4 searches); unbudgeted, the same states fire the Super Speed.
        early=[s for s in specs if s[0] in ss_names]
        if early:
            others=[s for s in specs if s[0] not in ss_names]
            specs=others[:2]+early+others[2:];ss_early=early[-1][0]
    if getattr(engine,'crossing_phase',False):
        # Crossing a gap: only candidates that can leave the ground are worth the
        # budget; rolling candidates would drive over the edge and are rejected anyway.
        specs=[s for s in specs if 'jump' in s[0]]
    elif (detour_ratio(route,dynamics['native']['state']['position'],gem)>=float(getattr(engine,'detour_blast_ratio',1.5) or 1.5)
            or (RAISED_BLAST_FIRST and raised_near(dynamics['native']['state']['position'],gem)))             and blast_meter_value(dynamics)>=float(getattr(engine,'detour_blast_meter',0.5) or 0.5) and not getattr(engine,'blast_reserved',False):
        # The walking route is a long way round compared with the straight line
        # (a wall or a step between here and the gem) and the Blast meter is
        # charged: forecast the Blast+jump families first so the search deadline
        # cannot starve the one family that can go over instead of around. The
        # physics forecast still decides whether any of them is accepted.
        blasts=[s for s in specs if 'blast' in s[0]];others=[s for s in specs if 'blast' not in s[0]]
        specs=blasts+others
        if detour_first is None:detour_first=blasts[-1][0] if blasts else None
    if getattr(engine,'_light_search',False):
        # order-choice alternatives: a few basic plans and the racing lines (no navigator, Blast, powerup or landing families)
        keep={'route','direct','through','direct_through','through_0.7','steering_-0.5','steering_0.5','jump_2','direct_jump_2','line2','line3','through_hop','direct_through_hop'}   # 2026-10-21 pad arrival: the order choice sees the hop arrival too
        specs=[sp for sp in specs if sp[0] in keep];navigator_spec=None
    if getattr(engine,'interleave_candidates',False):specs=interleave_specs(specs)
    if navigator_spec is None:pass
    elif getattr(engine,'navigator_first',False):specs.insert(0,navigator_spec)
    else:specs.append(navigator_spec)
    if getattr(engine,'edge_hits',False):
        # v5 (2026-10-21): evaluated after the learned navigator so a search deadline trims these first, never the
        # navigator or the extra-jump rescue; lips up to 2.6 u (wall-hit climbs, see edge_hits.jump_replies).
        # 2026-10-20 edge hits (operator): a target that needs a climb -> aim at the lip of a ledge above on the way,
        # jump timed so the rising marble clips it, and let the bounce (compiled physics, verified on 118 natural hits)
        # carry it up; then on to the gem. v2: lips above only, timed jumps (see nav/marble_dynamics/edge_hits.py).
        from nav.marble_dynamics.edge_hits import candidates as lip_candidates,jump_replies as lip_jumps
        _st=dynamics['native']['state']
        try:lip_points=lip_candidates(engine,_st['position'],gem)
        except Exception:lip_points=[]
        if lip_points:
            nroute=cached_route(route.terrain,gem);_p=np.asarray(_st['position'],dtype=float);_v=np.asarray(_st['velocity'],dtype=float)
            recent=[(i,np.asarray(q,dtype=float)) for i,q in getattr(engine,'edge_recent',[]) if getattr(engine,'decision_index',0)-i<EDGE_RETRY_DECISIONS]
            for k,lip in enumerate(lip_points):
                lp=np.asarray(lip['point'],dtype=float);_d=lp[:2]-_p[:2];_n=float(np.linalg.norm(_d))
                if any(np.linalg.norm(lp-q)<2. for _,q in recent) and float(_p[2])<float(lp[2])-.3:
                    continue   # v9: tried this lip within the last 3 s and still below it: let the other families work
                speed=float(_v[:2]@_d)/_n if _n>1e-6 else 0.
                leg=float(nroute.distance(lp))
                if not math.isfinite(leg):leg=float(np.linalg.norm(np.asarray(gem[:2])-lp[:2]))
                ctx=(nroute,np.asarray(gem[:3],dtype=float),leg)
                # v4: the lip-aimed plan gets its own route (goal = the lip point, snapped to 0.25 u so the field cache
                # hits across decisions) so the compiled controller runs it (it requires route.key == goal), and a
                # through box over the lip: passing over or through the lip switches the plan to the gem even when
                # the jump clears the lip by more than the 0.6 u centre test (which left v3 aiming back at the lip).
                lroute=cached_route(route.terrain,lp)
                box=[float(lp[0])-1.,float(lp[1])-1.,float(lp[2])-.6,float(lp[0])+1.,float(lp[1])+1.,float(lp[2])+2.5]
                if lip['rise']>1.0:
                    # v11 wall hits: jump when the wall is a marble-length away, whatever the approach speed
                    from nav.marble_dynamics.edge_hits import wall_hit_proposal,WALL_TRIGGER_U
                    for trig in WALL_TRIGGER_U:
                        wsp=wall_hit_proposal(SequenceProposal,lroute,lp,steps,ctx,trig);wsp.box=box;wsp._edge_lip_z=float(lp[2]);wsp._edge_lip=[float(x) for x in lp]
                        specs.append(('edge_hit_%d_wall_%g'%(k,trig),wsp))
                for jump in lip_jumps(lip['ahead'],speed,lip['rise']):
                    if jump>=steps:continue
                    spec=SequenceProposal(lroute,lp,steps,jump=jump,direct=True,through=ctx);spec.box=box;spec._edge_lip_z=float(lp[2]);spec._edge_lip=[float(x) for x in lp]
                    specs.append(('edge_hit_%d_jump_%d'%(k,jump),spec))
                    # the same timing on the ordinary route and direct lines (the lip point is often on them already;
                    # aiming at it only bends the approach): the right-timed hop onto the step, then on to the gem
                    for lab,sp in (('edge_hit_%d_route_jump_%d'%(k,jump),SequenceProposal(route,gem,steps,jump=jump)),
                                   ('edge_hit_%d_direct_jump_%d'%(k,jump),SequenceProposal(route,gem,steps,jump=jump,direct=True))):
                        sp._edge_lip_z=float(lp[2]);sp._edge_lip=[float(x) for x in lp];specs.append((lab,sp))
    if getattr(engine,'edge_hits',False):
        # v21: after the lip candidates, so a search deadline trims walls first (wall7 Sprawl 4420117: the ON arm lost a
        # lip landing candidate to the deadline and the round, -19, with no wall plan chosen)
        # v12 wall rebounds (operator demo 2026-10-21): run into a wall with the target behind/beside, jump a marble-length
        # before contact, rebound back up onto the ledge. Wall faces from the mesh; the forecast and the climb gate judge.
        from nav.marble_dynamics.edge_hits import wall_candidates,wall_hit_proposal,WALL_TRIGGER_U,WALL_SPEEDS,WALL_FAST_ONLY_RISE_U,WALL_CLIMB_TRIGGER_U,WALL_CLIMB_SPEEDS
        _st=dynamics['native']['state']
        try:walls=wall_candidates(engine,_st['position'],gem,route=route,velocity=_st['velocity']) if getattr(engine,'wall_rebounds',True) else []
        except Exception:walls=[]
        if walls and not any(float(w.get('rise',0.))>=WALL_FAST_ONLY_RISE_U for w in walls):walls=walls[:1]   # 2026-10-22: one wall unless a climb is on offer
        if walls:
            nroute=cached_route(route.terrain,gem)
            _di=getattr(engine,'decision_index',0);_recent=[(i,np.asarray(q,dtype=float)) for i,q in getattr(engine,'edge_recent',[]) if _di-i<EDGE_RETRY_DECISIONS]
            for k,w in enumerate(walls):
                wp=np.asarray(w['point'],dtype=float)
                _first=[i for i,q in _recent if np.linalg.norm(wp-q)<2.]
                if _first and _di-min(_first)>WALL_RETRY_DECISIONS:
                    continue   # v16: first tried over 2 s ago and the marble is still here (wall2 Duplex: ten tries, 0 points)
                leg=float(nroute.distance(wp))
                if not math.isfinite(leg):leg=float(np.linalg.norm(np.asarray(gem[:2])-wp[:2]))
                ctx=(nroute,np.asarray(gem[:3],dtype=float),leg);wroute=cached_route(route.terrain,wp)
                box=[float(wp[0])-1.2,float(wp[1])-1.2,float(wp[2])-.8,float(wp[0])+1.2,float(wp[1])+1.2,float(wp[2])+3.5]
                _climb=float(w.get('rise',0.))>WALL_FAST_ONLY_RISE_U
                for spd in (WALL_CLIMB_SPEEDS if _climb else WALL_SPEEDS):   # 2026-10-22: a climb needs the fast run and the later launch
                  for trig in (WALL_CLIMB_TRIGGER_U if _climb else WALL_TRIGGER_U):
                    wsp=wall_hit_proposal(SequenceProposal,wroute,wp,steps,ctx,trig,fast=spd);wsp.box=box
                    wsp._edge_lip_z=float(gem[2]);wsp._edge_lip=[float(x) for x in wp]   # the climb gate: reach the target's height
                    specs.append(('edge_hit_wall%d_rebound_%g_s%g'%(k,trig,spd),wsp))
    if THROUGH_BOX and gem_item is not None and gem_item.get('bounds') is not None:
        box=[float(x) for x in gem_item['bounds']]
        for _label,_spec in specs:
            if isinstance(_spec,SequenceProposal) and _spec.through is not None:_spec.box=box
    if ARRIVAL_MIN_SPEED and getattr(engine,'arrival_fast',None)==goal_key(gem):
        # v62: the end of a long trip (pick() records each target's starting distance): keep speed at the pickup
        for _label,_spec in specs:
            if isinstance(_spec,SequenceProposal):_spec.min_speed=ARRIVAL_MIN_SPEED
    gap_run=bool(GAP_RUN and getattr(engine,'crossing_phase',False) and _cr.get('gap_route'))
    if gap_run:specs=gap_run_specs(engine,route,dynamics,gem,_cr)   # 2026-10-25: a gap route's cross phase runs only its run-and-jump plans
    climb_mix=False
    if getattr(engine,'crossing_phase',False) and _cr.get('climb2'):
        # 2026-10-25 climb skill (climbs.py): a staged climb's run phase forecasts its climb plans (each holds its whole flight)
        # ahead of the ordinary jump plans for the gem itself (CLIMB_MIX; the controller passes the gem, not the landing point);
        # without CLIMB_MIX only the climb plans, with no landing extensions, like a gap run
        from nav.marble_dynamics.climbs import climb_specs,CLIMB_MIX
        if CLIMB_MIX:
            # in the air the climb plans' run and take-off are past (the kept plan flies on): the ordinary plans only
            skill=engine.crossings.climb_skill()
            grounded=skill is not None and not skill.airborne(dynamics['native']['state']['position'])
            specs=(climb_specs(engine,route,dynamics,gem,_cr) if grounded else [])+specs;climb_mix=True
        else:specs=climb_specs(engine,route,dynamics,gem,_cr);gap_run=True
    landing_replies=int(getattr(engine,'landing_replies',0) or 0)
    if getattr(engine,'crossing_phase',False) and not climb_mix:
        # Crossing a gap to another walking component: airborne candidates must
        # be allowed to land on the far side before they are judged.
        # (Not a climb's mixed search: its ordinary plans keep the ordinary extensions. 40 replies for every airborne plan doubled
        # the cost and ran the search out of time before the plan in flight was re-checked: sim Vortex 2834, the actor took over.)
        landing_replies=max(landing_replies,int(getattr(engine,'crossing_landing_replies',40) or 40))
    pairs=[]
    exhausted=False
    ballistic=bool(getattr(engine,'ballistic_landing',False)) and not getattr(engine,'_light_search',False);airborne=[]
    if gap_run:landing_replies=0;ballistic=False   # a gap run's forecast holds its whole flight: no landing extensions or projections
    reserve=float(getattr(engine,'ballistic_reserve_s',0.) or 0.);deadline=getattr(engine,'planning_deadline',None)
    pool=getattr(engine,'parallel',None)
    if pool is not None and len(specs)>=2:
        # 2026-10-19 (operator: speed up the search with more cores, toward real time): the candidates are forecast and
        # scored by worker processes (nav/marble_dynamics/parallel_search.py) and merged here in the serial order; with
        # no deadline the result is exactly the serial loop's. A candidate whose result misses the deadline is left out.
        results=pool.evaluate(engine,route,dynamics,gem,specs,target_item,steps,landing_replies,ballistic,brake_replies,deadline=deadline)
        engine._speculated={}
        for (label,proposal),r in zip(specs,results):
            if r is None:
                exhausted=True;continue
            if r['kind']=='unsupported':pairs.append(r['pair']);continue
            for part in ('main','landing'):
                if r.get('proj_'+part) is not None:engine._speculated[id(r[part][0])]=(r[part][0],r['proj_'+part])
            pairs.append(r['main'])
            if r['air_main']:airborne.append((label,)+tuple(r['main']))
            if r['landing'] is not None:
                pairs.append(r['landing'])
                if r['air_landing']:airborne.append((label+'_landing',)+tuple(r['landing']))
            try:
                if detour_first is not None and label==detour_first and ballistic and airborne:
                    project_airborne(engine,route,dynamics,gem,target_item,airborne,pairs);airborne=[]
                if ss_early is not None and label==ss_early and ballistic and airborne:
                    mine=[x for x in airborne if x[0].replace('_landing','') in ss_names];ids={id(x) for x in mine}
                    if mine:
                        project_airborne(engine,route,dynamics,gem,target_item,mine,pairs)
                        airborne=[x for x in airborne if id(x) not in ids]
            except PlanningBudgetExceeded:
                if not getattr(engine,'use_completed_candidates_on_timeout',False):raise
                exhausted=True
            if r['brake'] is not None:pairs.append(r['brake'])
        if exhausted:
            for info,_ in pairs:info['search_stopped_at_deadline']=True
        specs=[]
    for label,proposal in specs:
        if ballistic and reserve and deadline and time.perf_counter()>deadline-reserve:
            # Keep part of the budget for the landing projections of the
            # candidates already forecast; later families would otherwise
            # consume it and the long flights would never be judged.
            for info,_ in pairs:info['search_stopped_at_reserve']=True
            break
        try:
            try:rollout=predict_replies(engine,dynamics,proposal,stop_on_ambiguity=True)
            except ValueError as exc:
                pairs.append((dict(label=label,rejected='unsupported_proposal',failure=str(exc),cost=math.inf),None));continue
            info=assess(engine,route,dynamics,gem,rollout,label,target_item)
            edge_climb(proposal,label,info,rollout)
            chain_score(proposal,label,info,rollout,dynamics)
            info.update(evaluated_replies=len(rollout['controls']),requested_replies=rollout['requested_replies'],
                        stopped_on_ambiguity=rollout['stopped_on_ambiguity'])
            if ('use' in label and rollout['requested_replies']>steps and info['rejected'] is None and info['possible_pickup_interval'] is None
                    and not _gem_demo.TIME_COST):   # TIME_COST: the forecast time is already in the cost
                # A powerup candidate forecast over a longer horizon covers more
                # ground by itself; charge the extra time at a nominal rolling
                # speed so it wins only when the powerup is genuinely faster.
                penalty=(rollout['requested_replies']-steps)*.064*float(getattr(engine,'ballistic_time_rate',8.) or 8.)
                info['cost']+=penalty;info['time_penalty_u']=penalty
            pairs.append((info,rollout))
            if ballistic and info['rejected']=='no_supported_continuation' and not label.startswith('edge_hit') and not gap_run:
                airborne.append((label,info,rollout))   # edge plans use the landing extension only (v5)
            pickup_landing=int(getattr(engine,'pickup_landing_replies',0) or 0)
            collects=info.get('possible_pickup_interval') is not None
            extend=(landing_replies or (pickup_landing and collects)) and info['rejected']=='no_supported_continuation' and not gap_run   # a gap run's forecast already holds its flight
            if (not extend and '_wall' in label and landing_replies and info['rejected'] is None
                    and abs(float(rollout['states'][-1]['velocity'][2]))>WALL_AIRBORNE_VZ):
                # 2026-10-22 wall climbs: a rebound still in the air at the horizon (the run-up to the wall takes most of
                # it) passes the support gate on the deck below it and was scored mid-flight, 2-4 u short of where it
                # lands: let it land like every other plan of a staged crossing.
                extend=True
            if (not extend and label.startswith('edge_hit') and pickup_landing and info['rejected']=='no_supported_continuation'):
                # 2026-10-21 edge hits: a timed jump onto a lip is still in flight or settling at the horizon far more often
                # than a rolling plan (223 of 562 candidates in the climb-leg probe); let it land and be judged like a
                # pickup plan would be, over the same pickup_landing replies.
                extend=True
            if extend:
                # Still airborne (or not yet settled) at the horizon: follow the
                # route onward and score the landing with the same gates. With
                # only pickup_landing_replies set, this applies solely to a
                # candidate whose forecast already collects the gem (a jump or
                # blast up to a gem must not lose to standing underneath it).
                replies=landing_replies or pickup_landing
                landed=predict_replies(engine,dynamics,LandingContinuation(rollout['proposed'],route,gem,replies),stop_on_ambiguity=True)
                checked=assess(engine,route,dynamics,gem,landed,label+'_landing',target_item)
                edge_climb(proposal,label,checked,landed)
                chain_score(proposal,label,checked,landed,dynamics)
                checked.update(proposal_replies=len(rollout['proposed']),landing_replies=replies,pickup_only_landing=not landing_replies,
                               evaluated_replies=len(landed['controls']),requested_replies=landed['requested_replies'],
                               stopped_on_ambiguity=landed['stopped_on_ambiguity'],
                               forecast_s=(len(landed['states'])-1)*.064)
                if ((getattr(engine,'time_price',False) or (getattr(engine,'crossing',None) or {}).get('wall')) and not _gem_demo.TIME_COST
                        and checked.get('possible_pickup_interval') is None and math.isfinite(checked.get('cost',math.inf))):
                    # 2026-10-22 step 2a: an extended forecast (landing continuation, crossing phase) without a pickup pays
                    # for its replies beyond the horizon at the time price, so a 5.6 s plan no longer beats a 3 s plan on
                    # end distance alone (first seen on a staged wall climb: WALL_TIME_RATE_U_S; now every such plan)
                    rate=_gem_demo.TIME_PRICE_U_S if getattr(engine,'time_price',False) else WALL_TIME_RATE_U_S
                    extra=max(0,len(landed['states'])-1-steps)*.064*rate
                    checked['cost']+=extra;checked['wall_time_penalty_u']=extra
                pairs.append((checked,landed))
                if ballistic and collects and checked['rejected']=='no_supported_continuation':
                    # v21 (2026-10-15): a plan that COLLECTS the gem and is still in
                    # the air after the continuation (a Super Jump up to a platform)
                    # gets the no-input landing projection instead of a discard.
                    airborne.append((label+'_landing',checked,landed))
            if detour_first is not None and label==detour_first and ballistic and airborne:
                # Detour mode: the Blast families were forecast first; project their
                # landings now, before the rolling families spend the budget (the
                # end-of-search projection would be skipped at the deadline).
                project_airborne(engine,route,dynamics,gem,target_item,airborne,pairs);airborne=[]
            if ss_early is not None and label==ss_early and ballistic and airborne:
                # the early Super Speed plans of a long leg: project their landings now, and only theirs
                # (projecting route/direct too made a roll off a ledge win: Cube sim leg 2289, +2.8 s)
                mine=[x for x in airborne if x[0].replace('_landing','') in ss_names];ids={id(x) for x in mine}
                if mine:
                    project_airborne(engine,route,dynamics,gem,target_item,mine,pairs)
                    airborne=[x for x in airborne if id(x) not in ids]
            if brake_replies and info['rejected']=='no_supported_continuation':
                extended=predict_replies(engine,dynamics,BrakeContinuation(rollout['proposed'],gem,brake_replies),stop_on_ambiguity=True)
                checked=assess(engine,route,dynamics,gem,extended,label+'_brake_tail',target_item)
                checked.update(proposal_replies=len(rollout['proposed']),braking_replies=brake_replies,
                               evaluated_replies=len(extended['controls']),requested_replies=extended['requested_replies'],
                               stopped_on_ambiguity=extended['stopped_on_ambiguity'],
                               forecast_s=(len(extended['states'])-1)*.064)
                pairs.append((checked,extended))
        except PlanningBudgetExceeded:
            if not getattr(engine,'use_completed_candidates_on_timeout',False):raise
            # A fully forecast and assessed candidate is not a partial future.
            # Keep only those already appended; the unfinished candidate is lost.
            exhausted=True
            for info,_ in pairs:info['search_stopped_at_deadline']=True
            break
    if ballistic and airborne and not exhausted:
        try:project_airborne(engine,route,dynamics,gem,target_item,airborne,pairs)
        except PlanningBudgetExceeded:
            if not getattr(engine,'use_completed_candidates_on_timeout',False):raise
            exhausted=True
            for info,_ in pairs:info['search_stopped_at_deadline']=True
    if (getattr(engine,'pickup_stop',False) and getattr(engine,'pickup_stop_allowed',True) and not gap_run
            and not any(p[0]['rejected'] is None for p in pairs)
            and float(np.linalg.norm(np.asarray(dynamics['native']['state']['velocity'][:2],dtype=float)))>=PICKUP_STOP_MIN_SPEED
            and edge_ahead(engine.mesh,dynamics['native']['state'],horizon_s=PICKUP_STOP_EDGE_S,depth=PICKUP_STOP_EDGE_DEPTH_U)):
        # v4 gate (15:45): the free roll along the current velocity leaves the floor within PICKUP_STOP_EDGE_S. True at
        # the start of all 23 fall fallback runs in the fresh baselines (speeds 2-17 u/s), but at only 87 of v3's 379
        # stop decisions: the other 292 (Cragmire 149 of 172) stopped where nothing was about to fall and the marble
        # waited for the actor instead of exploring (Cragmire -30 in v3).
        # 2026-10-20 --pickup-stop, last resort when every candidate fails the gates (Cube 7347019 at 1322: a 15.8 u/s
        # approach to a gem on a down-ramp 3 u from the edge; every family collects it at reply ~29 and is over the
        # edge at the horizon; the unchecked actor then drove off). Stop NOW (full brake: goal = own position) or
        # stop PICKUP_STOP_SHORT_U short of the gem; the next search plans a slow approach. Progress plans under the
        # ordinary gates, never a forced A2 handoff. v1 also re-forecast rejected pickup plans as 'prefix through the
        # pickup, then brake' (3 brake starts each): none passed here (the brake law stops AT the gem, the ramp adds
        # gravity), they beat valid full-speed plans elsewhere (~400 stop decisions a round, Cragmire -30) and their
        # per-reply forecasts broke the search budget. Removed 15:20.
        saved_dl=getattr(engine,'planning_deadline',None)
        if saved_dl is not None:engine.planning_deadline=max(saved_dl,time.perf_counter())+EXTRA_JUMP_S
        try:
            p0=np.asarray(dynamics['native']['state']['position'],dtype=float);to_gem=gem[:3]-p0;d=float(np.linalg.norm(to_gem[:2]))
            goals=[('stop_now',p0)]
            if d>PICKUP_STOP_SHORT_U+.5:goals.append(('stop_short',p0+to_gem*((d-PICKUP_STOP_SHORT_U)/d)))
            for label,goal in goals:
                try:rollout=predict_replies(engine,dynamics,Proposal(np.asarray(goal[:3],dtype=float),steps=steps,brake=True),stop_on_ambiguity=True)
                except ValueError as exc:
                    pairs.append((dict(label=label,rejected='unsupported_proposal',failure=str(exc),cost=math.inf),None));continue
                info=assess(engine,route,dynamics,gem,rollout,label,target_item)
                info.update(evaluated_replies=len(rollout['controls']),requested_replies=rollout['requested_replies'],
                            stopped_on_ambiguity=rollout['stopped_on_ambiguity'],pickup_stop_goal=list(map(float,goal[:3])))
                pairs.append((info,rollout))
        except PlanningBudgetExceeded:
            if not getattr(engine,'use_completed_candidates_on_timeout',False):raise
            for info,_ in pairs:info['search_stopped_at_deadline']=True
        finally:engine.planning_deadline=saved_dl
    if EXTRA_JUMP_TIMES and not gap_run and not any(p[0]['rejected'] is None for p in pairs):
        # v79: nothing passed the gates (the controller would hand the marble to the actor next): other jump times,
        # route and direct aims (v70 A2 states: every candidate 'no_supported_continuation'; Vortex's 1.7 u gap is
        # cleared by a route jump at reply 3 or 4, the standard times all fall)
        saved_dl=getattr(engine,'planning_deadline',None)
        if saved_dl is not None:engine.planning_deadline=max(saved_dl,time.perf_counter())+EXTRA_JUMP_S
        try:
            for jump in EXTRA_JUMP_TIMES:
                if jump in jump_times or jump>=steps:continue
                for direct in (False,True):
                    label=('direct_' if direct else '')+'jump_'+str(jump)+'_extra'
                    try:rollout=predict_replies(engine,dynamics,SequenceProposal(route,gem,steps,jump=jump,direct=direct),stop_on_ambiguity=True)
                    except ValueError:continue
                    info=assess(engine,route,dynamics,gem,rollout,label,target_item)
                    info.update(evaluated_replies=len(rollout['controls']),requested_replies=rollout['requested_replies'],
                                stopped_on_ambiguity=rollout['stopped_on_ambiguity'])
                    pairs.append((info,rollout))
        except PlanningBudgetExceeded:pass
        finally:engine.planning_deadline=saved_dl
    if warm and not exhausted:
        rollout=predict_replies(engine,dynamics,warm,stop_on_ambiguity=True)
        pairs.append((assess(engine,route,dynamics,gem,rollout,'remaining_previous_plan',target_item),rollout))
    if REFINE_CANDIDATES and not getattr(engine,'_two_level_active',False):
        from nav.marble_dynamics.phased_search import refine
        pairs += refine(engine,route,dynamics,gem,pairs,target_item,REFINE_CANDIDATES,REFINE_FAILURE_ONLY)
    apply_crossing_landing(pairs,engine)
    apply_blast_gain(pairs,target_item)
    apply_blast_value(pairs,blast_meter_value(dynamics),float(getattr(engine,'blast_value_u',10.) if getattr(engine,'blast_value_u',None) is not None else 10.),float(getattr(engine,'blast_floor_u',0.) if getattr(engine,'blast_floor_u',None) is not None else 0.))
    apply_use_away(pairs,USE_AWAY_U)
    apply_use_margin(pairs,0. if (SS_RIDE and held==2) else float(getattr(engine,'use_margin_u',0.) or 0.))
    if BRAKING_BACKUP:check_backups(engine,route,dynamics,gem,pairs,steps,target_item)
    apply_two_level(engine,route,dynamics,gem,pairs,target_item)
    valid=[p for p in pairs if p[0]['rejected'] is None]
    return min(valid,key=lambda p:p[0]['cost']) if valid else None,pairs


def braking_backup(engine,dynamics,rollout,gem,replies=None):
    """Brake from the plan's end; (False, detail) only when the marble clearly leaves the floor.

    The terminal gate checks that a plan ENDS supported, not that the marble can stop
    afterwards: v21 Cube s2 fell three times right after the horizon of a plan that
    collected an edge gem at 8 u/s. Contact ambiguity or an unbuildable continuation is
    unknown, not a fall, and passes.
    """
    from nav.marble_dynamics.viability import resume_forecast
    replies=int(replies or BACKUP_REPLIES)
    try:start=resume_forecast(dynamics,rollout)
    except ValueError:return True,dict(skipped='unresumable')
    saved=getattr(engine,'planning_deadline',None);engine.planning_deadline=None
    try:cont=predict_replies(engine,start,Proposal(np.asarray(gem[:3],dtype=float),steps=replies,brake=True),stop_on_ambiguity=True)
    except ValueError as exc:return True,dict(skipped=str(exc)[:60])
    finally:engine.planning_deadline=saved
    end=cont['states'][-1];z0=float(rollout['states'][-1]['position'][2]);z1=float(end['position'][2])
    supported=support(engine.mesh,end);fell=False
    if not supported:
        # Still in the air after braking: a drop onto a lower floor is not a fall,
        # nothing below within FALL_FLOOR_U is.
        p=np.asarray(end['position'],dtype=float)
        hit=engine.mesh.raycast(p+[0,0,.05],p-[0,0,FALL_FLOOR_U],cull_backfaces=True)
        fell=hit is None or hit['normal'][2]<=.4
    return not fell,dict(supported=supported,drop_u=z0-z1,replies=len(cont['controls']),ambiguous=bool(cont['ambiguous_intervals']))


def check_backups(engine,route,dynamics,gem,pairs,steps=24,target_item=None):
    """Prefer plans with a braking backup: check the best useful candidates in cost order.

    The first one that passes wins (every cheaper one has failed); in ordinary
    states that is the first one checked. A useful plan without a backup (failed,
    or unchecked behind BACKUP_CHECKS failures) is charged BACKUP_PENALTY_U, not
    rejected: v22 rejected them and on a Cube slope near an edge gem, where no plan
    has a backup, every search then failed and the unchecked actor drove off the
    edge 11 times in 60 s (v21 took the best plan there: 81 points, 1 fall). When
    the best ones all fail their backups,
    full braking from now / in 4 / in 8 replies ('stop_N') is forecast and checked
    too; a stop with a backup then beats every penalised plan (the v21 Cube s2 fall
    at decision 1768 becomes stop_4). Stops are not ordinary candidates: as such
    they won 9 of 60 ordinary states on end speed alone (offline audit 2026-10-15).
    """
    position=dynamics['native']['state']['position'];start=route.distance(position)
    useful=[p for p in pairs if p[0]['rejected'] is None and p[1] is not None and (p[0].get('possible_pickup_interval') is not None
            or start-route.distance(p[1]['states'][-1]['position'])>.3)]
    useful.sort(key=lambda p:p[0]['cost'])
    for info,rollout in useful[:BACKUP_CHECKS]:
        ok,detail=braking_backup(engine,dynamics,rollout,gem)
        info['braking_backup']=detail
        if ok:return
        info['cost']=float(info['cost'])+BACKUP_PENALTY_U;info['no_braking_backup']=True
    for info,_ in useful[BACKUP_CHECKS:]:
        info['cost']=float(info['cost'])+BACKUP_PENALTY_U;info['braking_backup_unchecked']=True
    # v22d: stops only after useful plans FAILED their backups. With nothing useful
    # at all (often just the search deadline on a curved surface) v22c braked in open
    # terrain (Bowl s1: stop_N chosen 102 times while driving at 8-17 u/s); the
    # controller's actor/safe-hold rule handles that case.
    if not useful:return
    if getattr(engine,'crossing_phase',False):return
    saved=getattr(engine,'planning_deadline',None);engine.planning_deadline=None
    try:
        for n in (0,4,8):
            try:rollout=predict_replies(engine,dynamics,SequenceProposal(route,gem,steps,brake_after=n),stop_on_ambiguity=True)
            except ValueError:continue
            info=assess(engine,route,dynamics,gem,rollout,'stop_'+str(n),target_item)
            if info['rejected'] is None:
                ok,detail=braking_backup(engine,dynamics,rollout,gem);info['braking_backup']=detail
                if not ok:info['cost']=float(info['cost'])+BACKUP_PENALTY_U;info['no_braking_backup']=True
            pairs.append((info,rollout))
    finally:engine.planning_deadline=saved


def safe_hold_choice(engine,dynamics,pairs,gem,limit=4):
    """Best gate-checked candidate (by cost) with a braking backup; else the best gate-checked one; else None.

    Without any backup the plan still passed every physics gate for 1.5 s, which
    beats the unchecked actor (the v22 Cube loop fell off the edge under it).
    """
    valid=sorted([p for p in pairs if p[0]['rejected'] is None and p[1] is not None],key=lambda p:p[0]['cost'])
    for info,rollout in valid[:limit]:
        if info.get('braking_backup') is not None and not info.get('no_braking_backup'):return info,rollout
        if info.get('no_braking_backup'):continue
        ok,detail=braking_backup(engine,dynamics,rollout,gem) if BRAKING_BACKUP else (True,None)
        info['braking_backup']=detail
        if ok:return info,rollout
    if valid:
        valid[0][0]['safe_hold_without_backup']=True
        return valid[0]
    return None


def actor_forecast_unsafe(engine,dynamics,pairs,gem):
    """True when the search forecast the learned actor's proposal and none of its variants is
    gate-valid with a braking backup (it would leave the floor); False when one is, or when the
    deadline cut the search before the actor's proposal was forecast (unknown: v21 behaviour).
    Narrower definitions lost: late forecasts of a cut proposal (v24, v25: parking on Cragmire) and
    'unsafe only on a fall into the void' (v29: 94.8 / 102.7 vs 109.5 / 101.2, falls 37 vs 20)."""
    nav=[p for p in pairs if p[0]['label'].startswith('frozen_navigator') and p[1] is not None]
    if not nav:return False
    for info,rollout in nav:
        if info['rejected'] is not None or info.get('no_braking_backup'):continue
        if info.get('braking_backup') is not None:return False
        ok,detail=braking_backup(engine,dynamics,rollout,gem) if BRAKING_BACKUP else (True,None)
        info['braking_backup']=detail
        if ok:return False
    return True


def project_exact(engine,route,dynamics,gem,target_item,label,rollout):
    """project_airborne's exact projection of one airborne candidate: ('ok', checked, landed), ('reason', reason) or
    ('unsupported', message). Shared with the parallel search workers, which run it ahead for every candidate whose
    coarse landing is floor-like (nav/marble_dynamics/parallel_search.py); project_airborne then reads those."""
    from nav.marble_dynamics.ballistic import project_landing,MAX_FLIGHT_REPLIES,HELICOPTER_FLIGHT_REPLIES
    cap=HELICOPTER_FLIGHT_REPLIES if rollout['states'][-1]['effects'][5][0] else MAX_FLIGHT_REPLIES
    try:landed,reason=project_landing(engine,dynamics,rollout,cap)
    except ValueError as exc:return ('unsupported','unsupported:'+str(exc)[:40])
    if reason is not None:return ('reason',reason)
    checked=assess(engine,route,dynamics,gem,landed,label+'_ballistic',target_item)
    # A longer forecast covers more ground by itself. Charge the extra
    # flight time at a nominal rolling speed so a jump only wins when
    # it is genuinely faster than rolling for the same time.
    extra=max(0,len(landed['states'])-len(rollout['states']))
    penalty=extra*.064*float(getattr(engine,'ballistic_time_rate',8.) or 8.)
    if checked['possible_pickup_interval'] is None and not _gem_demo.TIME_COST:checked['cost']+=penalty   # TIME_COST: flight time is in the cost
    checked['time_penalty_u']=penalty
    checked.update(proposal_replies=len(rollout['proposed']),ballistic_replies=len(landed['proposed'])-len(rollout['proposed']),
                   evaluated_replies=len(landed['controls']),requested_replies=landed['requested_replies'],
                   stopped_on_ambiguity=landed['stopped_on_ambiguity'],forecast_s=(len(landed['states'])-1)*.064)
    return ('ok',checked,landed)


def project_airborne(engine,route,dynamics,gem,target_item,airborne,pairs):
    """Operator-approved ballistic landing projection: finish airborne candidates
    with NO input. Coarse parabola filter for all of them, exact engine projection
    only for the few with the best coarse landing. Appends the landed candidates
    to `pairs`; raises PlanningBudgetExceeded at the deadline."""
    from nav.marble_dynamics.ballistic import coarse_landing
    bounds=getattr(engine,'mission_bounds',None);ranked=[];collecting=[]
    ahead=getattr(engine,'_speculated',None) or {}   # 2026-10-19: worker results (the same values, computed in parallel)
    for label,info,rollout in airborne:
        pre=ahead.get(id(info));pre=pre[1] if pre is not None and pre[0] is info else None
        coarse=pre['coarse'] if pre is not None else coarse_landing(engine.mesh,rollout['states'][-1],bounds)
        info['ballistic_coarse']=coarse
        if not (coarse['hit'] and coarse['floor_like']):continue
        # v21: a candidate whose forecast already collects the gem is projected
        # whatever its landing distance; the others compete for the few exact slots.
        if info.get('possible_pickup_interval') is not None:collecting.append((route.distance(coarse['point']),label,info,rollout))
        else:ranked.append((route.distance(coarse['point']),label,info,rollout))
    ranked.sort(key=lambda item:item[0]);collecting.sort(key=lambda item:item[0])
    for _,label,info,rollout in collecting+ranked[:int(getattr(engine,'ballistic_exact_limit',4) or 4)]:
        pre=ahead.get(id(info));pre=pre[1] if pre is not None and pre[0] is info else None
        result=pre['exact'] if pre is not None and pre.get('exact') is not None else project_exact(engine,route,dynamics,gem,target_item,label,rollout)
        if result[0]!='ok':info['ballistic']=result[1];continue
        pairs.append((result[1],result[2]))


KEEP_BLAST=False          # 2026-10-16 v48: keep the Blast for a raised gem (useful_candidate keep_blast, set by pick()); NOT adopted 20:15 (111.6 vs 118.7)
KEEP_BLAST_RISE_U=1.5     # the gem at least this far above the marble (a plain jump reaches ~1.4 u) ...
KEEP_BLAST_NEAR_U=40.     # ... and within this horizontal distance
JUMP_COST_U=0.            # 2026-10-17 v59 alone (on v58): 118.7 vs 121.6, not adopted; v51 (1.0 with PICKUP_TIME_WEIGHT 8: 118.0 vs 120.0, not adopted; jumps in clusters -23%, cluster time unchanged): cost per jump press (u; the time term is 4 u a second, so ~0.25 s) when the gem ...
JUMP_FREE_RISE_U=1.0      # ... is less than this far above the marble (higher gems need the jump: no charge)


def jump_presses(actions):
    """Jump presses (rising edges of the jump key) in a planned action list."""
    n=0;prev=0
    for a in actions:
        j=int(bool(a[4])) if len(a)>4 else 0
        n+=int(j and not prev);prev=j
    return n


THROUGH_STEER=False       # 2026-10-17 v54: curved pass-through plans in clusters (see search())
LATE_JUMPS=True           # 2026-10-16 v47: later jump / Blast timings toward a raised gem (see search())
LATE_JUMP_TIMES=(6,8,10,12,14,16,20,24)   # replies; the standard families jump at 0, 2, 5 or 9
GAP_LIP_AHEAD_U=.5        # 2026-10-25 gap crossings: the lip is taken this far past the launch cell's centre for the lip-timed jumps
LATE_BLAST_AFTER=(2,4,6)  # Blast this many replies after the jump (while still rising)
LATE_JUMP_RISE_U=1.0      # the gem at least this far above the marble ...
LATE_JUMP_NEAR_U=14.      # ... and within this horizontal distance
LATE_PAD_JUMPS=True       # ON by default since 2026-10-25 05:25 (pair pf4 +1.0 a round, TreasureBox +5 / +31, Fractured +16 / +9; sim clear -3.1 percent, smacks -28 percent); 2026-10-25 (cluster speed): the late jumps (no Blast variants) also toward a gem on a raised pad (pad_hops.pad_under)
LATE_PAD_NEAR_U=8.        # within this horizontal distance when the marble is not on that pad (TreasureBox 7347019 decision 1841:
                          # the marble still up on the higher pad of the gem just taken, B on a 0.5 u pad 2.7 u behind; every
                          # standard candidate stopped against B's pad side and the frozen navigator's 3.3 s loop won; with the
                          # late jumps direct_jump_16 collects at reply 26 against 52)


def late_pad(engine,position,gem):
    """LATE_PAD_JUMPS: the gem sits on a raised pad within LATE_PAD_NEAR_U and the marble is not on that pad."""
    if not LATE_PAD_JUMPS:return False
    p=np.asarray(position[:3],dtype=float);g=np.asarray(gem[:3],dtype=float);d=float(np.linalg.norm(g[:2]-p[:2]))
    if d>LATE_PAD_NEAR_U:return False
    from nav.marble_dynamics.pad_hops import pad_under
    try:pad=pad_under(engine,g)
    except Exception:return False
    return pad is not None and d>float(pad[1])+.3


def late_raised(position,gem):
    p=np.asarray(position[:3],dtype=float);g=np.asarray(gem[:3],dtype=float)
    return float(g[2]-p[2])>=LATE_JUMP_RISE_U and float(np.linalg.norm(g[:2]-p[:2]))<=LATE_JUMP_NEAR_U


RAISED_BLAST_FIRST=False  # v39 (--raised-blast-first, under test): Blast families first below a nearby raised gem
RAISED_BLAST_RISE_U=1.5   # the gem at least this far above the marble ...
RAISED_BLAST_NEAR_U=8.    # ... and within this horizontal distance


def raised_near(position,gem):
    """v39: a gem the Blast (+ jump) can reach from here: raised and nearly overhead."""
    p=np.asarray(position[:3],dtype=float);g=np.asarray(gem[:3],dtype=float)
    return float(g[2]-p[2])>=RAISED_BLAST_RISE_U and float(np.linalg.norm(g[:2]-p[:2]))<=RAISED_BLAST_NEAR_U


def detour_ratio(route,position,gem):
    """Walking-route length over straight-line distance to the gem (inf when no walk route)."""
    straight=float(np.linalg.norm(np.asarray(gem[:3])[:2]-np.asarray(position[:3])[:2]))
    if straight<4.:return 1.
    route.navigate(position,gem[:3]);d=float(route.distance(position))
    return d/straight if math.isfinite(d) else math.inf


def blast_meter_value(dynamics):
    blast=dynamics.get('blast') if isinstance(dynamics,dict) else None
    return float(blast['value']) if blast and blast.get('enabled',True) else 0.


def apply_blast_value(pairs,meter,value_u,floor_u=0.):
    """Charge a candidate that fires the Blast for the meter it spends.

    A Blast empties the meter (its power scales with the charge) and the meter
    takes 25 s to refill, so a half-meter Blast for one unit of progress costs
    the full-meter hop over the next wall. The charge is meter * value_u in
    route units, so a Blast is chosen only when it saves more walking than
    that; a predicted pickup keeps its -100 credit.
    """
    charge=meter*value_u+(floor_u if meter>0 else 0.)
    if charge<=0:return
    for info,rollout in pairs:
        if rollout is None or 'blast' not in info['label']:continue
        n=getattr(rollout.get('controls'),'script_event_count',None)
        fired=(n>0) if n is not None else any(c.get('script_events') for c in rollout.get('controls',[]))
        if not fired:continue
        # floor_u: a fixed charge per fire (v14, 2026-10-15). In the v12 journals
        # low-meter fires (< 0.5) gained a median 2.5 u in the 3 s after the fire
        # against 13 u before it; the per-meter charge alone is too small there.
        info['blast_charge_u']=charge;info['cost']=float(info['cost'])+charge


BLAST_PICKUP_GAIN=4       # 2026-10-18 v84 (operator: Blast only when necessary): replies sooner than the best plan without a
                          # Blast that a Blast plan must collect its target, else it is rejected (apply_blast_gain)


def apply_blast_gain(pairs,target_item=None):
    """v84: a plan that fires the Blast and collects its target must collect it BLAST_PICKUP_GAIN replies before the best
    plan without a Blast that also collects it; otherwise it is rejected ('blast_no_gain'). No effect when no plan
    without a Blast collects the target (the Blast is then the way up) or when no plan collects anything. Toward a
    powerup (an optional detour or a fetch) a Blast plan must collect it (v81/v82: 75 fires on powerup detours, the
    progress toward the powerup fell from 10.6 u in the 3 s before the fire to 2.3 u after)."""
    if not BLAST_PICKUP_GAIN:return
    if target_item is not None and not target_item.get('gem'):
        for info,r in pairs:
            if (info['rejected'] is None and r is not None and 'blast' in info['label'] and blasts_fired(r)
                    and info.get('possible_pickup_interval') is None):
                info['rejected']='blast_no_gain';info['blast_gain_replies']=None
    plain=[info['possible_pickup_interval'] for info,r in pairs if info['rejected'] is None and r is not None
           and info.get('possible_pickup_interval') is not None and not ('blast' in info['label'] and blasts_fired(r))]
    if not plain:return
    best=min(plain)
    for info,r in pairs:
        if info['rejected'] is not None or r is None or not ('blast' in info['label'] and blasts_fired(r)):continue
        k=info.get('possible_pickup_interval')
        if k is not None and k>best-BLAST_PICKUP_GAIN:
            info['rejected']='blast_no_gain';info['blast_gain_replies']=best-k


def apply_crossing_landing(pairs,engine):
    """In a crossing, a Blast or powerup candidate must end on the far side.

    Spending the meter to land short of a ledge, or back on the launch side,
    counts as nothing (rejected 'resource_without_landing'). Ordinary jumps keep
    their progress scoring so the marble can still creep to the edge and restage
    with a run-up, unless they end under the ledge.
    """
    crossing=getattr(engine,'crossing',None) if getattr(engine,'crossing_phase',False) else None
    crossings=getattr(engine,'crossings',None)
    if crossing is None or crossings is None:return
    land=np.asarray(crossing['land'],dtype=float)
    for info,rollout in pairs:
        # The staged goal is the landing cell, so a 'pickup' here means the flight
        # passed through that cell, not that a gem was collected: no exemption.
        if rollout is None or info['rejected'] is not None:continue
        end=np.asarray(rollout['states'][-1]['position'],dtype=float)
        if crossing.get('climb2'):
            # 2026-10-25 climb skill: a plan counts only when its forecast stands on the place up there, or (CLIMB_MIX: the search's
            # goal is then the gem itself) when it collects the gem
            from nav.marble_dynamics.climbs import landed_on_place,CLIMB_MIX
            if CLIMB_MIX and not crossing.get('route_climb') and info.get('possible_pickup_interval') is not None:continue   # (a route
            # climb's goal is a floor point up a ledge, not a gem: passing near it is no pickup)
            skill=crossings.climb_skill();i=landed_on_place(crossing,rollout,skill) if skill is not None else None
            if i is None:info['rejected']='climb_not_up'
            elif info.get('possible_pickup_interval') is None:info['possible_pickup_interval']=i
            continue
        if crossing.get('gap_route'):
            # 2026-10-25 gap crossings: both sides of a gap may be one walking component (joined by a bridge elsewhere), so the
            # far side is judged along the crossing: past the gap, not below the landing (the plan may go on up to the gem)
            lp=np.asarray(crossing['launch'][:2],dtype=float);u=np.asarray(crossing['direction'],dtype=float)
            along=float((end[:2]-lp)@u);land_along=float((land[:2]-lp)@u)
            landed=along>=land_along-1. and end[2]>=land[2]-1.5
            if landed and GAP_RUN and info.get('possible_pickup_interval') is None:
                # a run that lands counts as reaching the staged goal (the controller keeps a running plan over a new one
                # that collects nothing): the reply where it is first over the landing cell
                states=rollout['states']
                info['possible_pickup_interval']=next((i for i in range(1,len(states))
                                                       if float((np.asarray(states[i]['position'][:2],dtype=float)-lp)@u)>=land_along-1.),len(states)-1)
        else:landed=crossings.component_at(end)==crossing['target'] and abs(end[2]-land[2])<=1.5
        if landed:continue
        label=info['label']
        if crossing.get('gap_route') and GAP_RUN:
            info['rejected']='gap_not_crossed';continue   # a gap route's cross phase keeps only plans that land on the far side
        if 'blast' in label or 'use' in label:info['rejected']='resource_without_landing'
        elif not crossing.get('climb'):continue
        elif info.get('possible_pickup_interval') is not None:
            # Flying through the landing cell and coming down elsewhere is not a
            # pickup of anything.
            info['rejected']='phantom_landing_pickup'
        elif np.linalg.norm(end[:2]-land[:2])<3. and end[2]<land[2]-1.5:
            # Standing under the ledge is not progress toward a goal on top of it,
            # whatever the plan-view distance says.
            info['rejected']='under_the_ledge'


def apply_use_away(pairs,limit):
    """2026-10-16 (operator): a powerup use that collects the gem but leaves the marble far from the rest of the gems
    is rejected ('use_away'): through_post_u (how much farther from the tour's next gem than the collected gem the
    marble is, THROUGH_EVAL_REPLIES after the start) above `limit`. A use that turns back toward the gems within that
    time (a wall hit, a braking curve) passes. Without a next gem (the last of a group) there is no term."""
    if limit<=0:return
    for info,r in pairs:
        if info['rejected'] is None and r is not None and 'use' in info['label'] and info.get('through_post_u',-math.inf)>limit:
            info['rejected']='use_away'


def apply_use_margin(pairs,margin):
    """A powerup-use candidate must beat the best non-use candidate by `margin` units.

    Marks the use candidates that fail as rejected ('use_margin'). A use that
    predicts a pickup when no non-use candidate does is kept whatever its cost:
    the powerup is then the only way to the gem in this horizon. Audit of v6
    (10-14): Super Speed and Helicopter uses that won on a small cost edge
    rarely led to a gem; Super Jump uses mostly did.
    """
    if margin<=0:return
    valid=[(info,r) for info,r in pairs if info['rejected'] is None and r is not None]
    non_use=[info for info,_ in valid if 'use' not in info['label']]
    if not non_use:return
    best=min(info['cost'] for info in non_use)
    non_use_pickup=any(info.get('possible_pickup_interval') is not None for info in non_use)
    for info,_ in valid:
        if 'use' not in info['label']:continue
        if info.get('possible_pickup_interval') is not None and not non_use_pickup:continue
        if info['cost']>best-margin:
            info['rejected']='use_margin';info['use_margin_shortfall_u']=info['cost']-(best-margin)


def edge_ahead(mesh,state,horizon_s=1.0,step_s=0.1,depth=4.0):
    """True when the free roll along the current velocity reaches a point with no floor below.

    A conservative floor-continuity sample, used only to decide whether the
    unchecked actor fallback is safe; it never chooses a route.
    """
    p=np.asarray(state['position'],dtype=float);v=np.asarray(state['velocity'],dtype=float)
    speed=float(np.linalg.norm(v[:2]))
    if speed<1.:return False
    r=float(state['radius']);t=step_s
    while t<=horizon_s+1e-9:
        q=p+v*t
        hit=mesh.raycast(np.array([q[0],q[1],p[2]+r+.5]),np.array([q[0],q[1],p[2]-depth]),cull_backfaces=True)
        if hit is None:return True
        t+=step_s
    return False



WALL_AIRBORNE_VZ=.5      # 2026-10-22: a wall-climb plan ending with more vertical speed than this is still flying: extend it to its landing
WALL_TIME_RATE_U_S=8.    # 2026-10-22: time price (u per s) of replies beyond the horizon for plans toward a staged wall climb's landing
EDGE_CLIMB_SLACK_U=.35   # v6: an edge-hit plan must bring the marble centre to within this of the lip height
EDGE_RETRY_DECISIONS=48  # v9: a lip attempted this recently (3 s) while the marble is still below it is not offered again
WALL_RETRY_DECISIONS=32  # v16: a wall first attempted longer ago than this (2 s: approach + flight) is not offered again within the 3 s window
WALL_RETAIN_REPLIES=24   # v18: a wall plan is kept at most this long (1.5 s: approach + hit); then a fresh search
WALL_LOCK_MAX_REPLIES=30 # v21: the wall lock covers at most this many replies (approach + contact)
WALL_COST_MARGIN_U=2.    # v18: a wall plan must beat the field by this (wall4: 52 of 76 wall choices were one-decision ties, median 0.5 u)

def chain_score(proposal,label,info,rollout,dynamics):
    """2026-10-21 spawn plans: a ChainProposal is scored by every gem its forecast collects (spawn_plans.chain_score)."""
    if not label.startswith('chain_'):return
    from nav.marble_dynamics.spawn_plans import chain_score as _cs
    _cs(proposal,label,info,rollout,dynamics)


def edge_climb(proposal,label,info,rollout):
    """v6 (2026-10-21): an edge-hit candidate that never reaches its lip is a wasted jump, not a climb: reject it.

    v5 Cragmire 12091843: 11 edge runs, 9 gained 0.1-1.3 u toward 1.7-2.6 u lips and scored 0 in the next 10 s
    (-47 on the round); the landing extension made such jumps valid progress plans that won on 1-3 u cost margins.
    """
    lip_z=getattr(proposal,'_edge_lip_z',None)
    if lip_z is None or not label.startswith('edge_hit'):return
    info['edge_lip']=getattr(proposal,'_edge_lip',None)
    if '_wall' in label and info.get('cost') is not None and math.isfinite(info['cost']):info['cost']+=WALL_COST_MARGIN_U
    if info.get('rejected') is not None:return
    states=rollout['states'];pos=getattr(states,'positions',None)
    if '_wall' in label:
        # v12 wall rebounds must END up there (a rebound that flies past the ledge and lands back below is a wasted hit)
        top=float(pos[-1][2]) if pos is not None else float(states[-1]['position'][2])
        z0=float(pos[0][2]) if pos is not None else float(states[0]['position'][2])
        lip_z=min(lip_z,z0)   # v14: a reversal toward a far, higher target climbs later; it just must not lose its level
    else:top=float(max(p[2] for p in pos)) if pos is not None else float(max(s['position'][2] for s in states))
    if top<lip_z-EDGE_CLIMB_SLACK_U:info['rejected']='edge_no_climb';info['edge_top_u']=top;info['edge_lip_z']=lip_z


def backup_risk(info):
    """A nominally supported plan can still lack a checked stopping backup."""
    return bool(info.get('no_braking_backup') or info.get('braking_backup_unchecked'))


def recheck_retained_backup(engine,dynamics,gem,memory,info,forecast):
    """Do not silently upgrade a known risky plan when reusing its schedule.

    Recheck only previously failed/unchecked backups. Unknown continuations
    retain the warning; a new successful check can clear it. The action schedule
    and queued discrete inputs remain unchanged.
    """
    if not getattr(engine,'recheck_risky_plans',False) or not memory.backup_risk:return
    ok,detail=braking_backup(engine,dynamics,forecast,gem)
    info['braking_backup']=detail;info['retained_backup_rechecked']=True
    if not ok or detail.get('skipped') or detail.get('ambiguous'):
        info['no_braking_backup' if not ok else 'braking_backup_unchecked']=True
        info['cost']+=BACKUP_PENALTY_U
    memory.backup_risk=backup_risk(info)


def retained_candidate(engine,route,dynamics,gem,memory,minimum_replies=8,target_item=None):
    """Revalidate a useful remaining plan before spending a full search budget.

    The same visible target must remain selected. Recompute from the NEW observed
    state; changed inventory/geometry events clear memory in the execution loop.
    A short, unsafe, ambiguous or no-progress remainder goes back to full search.
    Keep the absolute action schedule rather than installing the remainder again.
    """
    if (len(memory.remaining)<minimum_replies or memory.gem is None
            or np.linalg.norm(np.asarray(memory.gem)-gem[:3])>=.1):return None
    if (str(memory.label or '').startswith('edge_hit_wall') and memory.executed>WALL_RETAIN_REPLIES
            and not memory.landing_mode):
        # v18 (2026-10-21): a wall plan's 88-reply schedule replayed open-loop past the hit held the marble dithering for
        # 5 s (wall4 Cragmire 3126-3204 at 1-3 u/s, TreasureBox 3003-3082 wandering 20 u); after the hit, search again
        return None
    position=dynamics['native']['state']['position'];route.navigate(position,gem[:3])
    start=route.distance(position);forecast=predict_replies(engine,dynamics,memory.remaining,stop_on_ambiguity=True)
    info=assess(engine,route,dynamics,np.asarray(gem[:3]),forecast,memory.label,target_item,
                _stop_checked=memory.stopping_verified)
    if info['rejected']:return None
    recheck_retained_backup(engine,dynamics,gem,memory,info,forecast)
    progress=start-route.distance(forecast['states'][-1]['position'])
    # a safe hold (v22) makes no progress by definition; it stays while it stays safe
    if info['possible_pickup_interval'] is None and progress<=.3 and memory.label!='safe_hold':return None
    info['revalidated_remainder']=True;info['predicted_progress_u']=progress
    return info,forecast


class PlanMemory:
    """Maintain an absolute schedule until its first jump/use reaches physics.

    A pickup, changed world or prediction error can always cancel it. Ordinary
    four-reply replanning cannot shift an already chosen event into the future.
    """
    def __init__(self):self.clear()

    def clear(self):
        self.actions=[];self.executed=0;self.lock_until=0;self.lock_from=0;self.label=None;self.gem=None
        self.landing_goal=None;self.landing_after_reply=0;self.goal_item_id=None;self.landing_mode=False;self.landing_triangle_id=None
        self.observed_tail=[]
        self.stopping_verified=False
        self.physical_recovery=False
        self.last_landing_rejection=None
        self.backup_risk=False

    def accept(self,rollout,label,gem,info=None):
        launch='_launch' in str(label or '') or (str(label or '').startswith('remaining_previous_plan') and '_launch' in str(self.label or ''))
        self.clear()
        self.actions=copy.deepcopy(rollout['proposed']);self.executed=0;self.label=label;self.gem=list(gem)
        events=[i for i,a in enumerate(self.actions[:13]) if a[4] or a[6] or a[8]]
        self.lock_until=min(len(self.actions),events[0]+3) if events else 0
        if launch:
            # take-off launches (a kept launch plan too): held from POW_LAUNCH_LOCK_LEAD replies before the first timed input through
            # the kick (2 decisions after the use, the jump up to a reply later); before that the receding horizon may move the launch
            # or its aim
            if not '_launch' in str(label or ''):self.label=label+'_launch'
            ev=next((i for i,a in enumerate(self.actions) if a[4] or a[6] or a[8]),None)
            if ev is not None:self.lock_from=max(0,ev-POW_LAUNCH_LOCK_LEAD);self.lock_until=min(len(self.actions),ev+4)
        if str(label or '').startswith('edge_hit_wall') and info and info.get('edge_lip') is not None:
            # v20 (2026-10-21): commit to the wall until the forecast contact (+3 replies). Five pairs: wall choices lasting
            # ONE decision (ties re-ranked a step later) nudged the marble and the round diverged (wall6 Cragmire 4420117 -58)
            wp=np.asarray(info['edge_lip'][:2],dtype=float);states=rollout['states']
            dists=[float(np.linalg.norm(np.asarray(st['position'][:2],dtype=float)-wp)) for st in states[:WALL_LOCK_MAX_REPLIES]]
            if dists:self.lock_until=max(self.lock_until,min(len(self.actions),dists.index(min(dists))+3))   # v21: closest approach, always defined
        self.stopping_verified=bool(info and info.get('terminal_stop_check',{}).get('valid'))
        self.backup_risk=bool(info and backup_risk(info))
        if info and info.get('landing_goal') is not None:
            self.landing_goal=list(info['landing_goal']);self.landing_after_reply=info['landing_after_reply']
            self.goal_item_id=info.get('goal_item_id');self.landing_triangle_id=info.get('landing_triangle_id')

    def observe_pickup(self,score_delta,items):
        """Keep a landing only after native score and selected-item disappearance."""
        if (score_delta>0 and self.landing_goal is not None and self.goal_item_id is not None
                and not any(i['id']==self.goal_item_id for i in items)):
            self.landing_mode=True;return True
        return False

    @property
    def locked(self):return (self.lock_from<=self.executed<self.lock_until or self.landing_goal is not None) and bool(self.remaining)

    @property
    def remaining(self):return self.actions[self.executed:]

    def advance(self,observed_state=None):
        self.executed+=1
        if observed_state is not None:
            self.observed_tail.append(copy.deepcopy(observed_state));self.observed_tail=self.observed_tail[-4:]
