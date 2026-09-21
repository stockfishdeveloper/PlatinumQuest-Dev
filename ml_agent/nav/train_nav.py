"""Navigator trainer: waypoint task on whatever hunt map the game instances are running.

    1. launch the games:  .\\run_game_loop.ps1 -Mission FlatIslands_Hunt      (N_INSTANCES built-engine
                          instances on ports 8888.., see run_game_loop.ps1)
    2. python -m nav.train_nav            (from ml_agent/)

One worker PROCESS per game instance (nav/vec_worker.py) owns the socket, the map, observation
building and the segment logic; this process owns the policy (a CPU copy for acting, the GPU
copy for PPO), the rollouts and the log. Each decision: one batched forward for all instances,
the actions go out to every worker, every worker steps its lockstepped game in parallel and
sends back the transition. With the built engine (marbleblast_mbx.exe, 2026-09-18) an instance
runs one 64 ms decision per observation and never sees a stale observation.
Logs go to logs/nav/, checkpoints to models/nav/ (nav_latest.pth is resumed automatically).
No command-line flags: settings are the constants below.
"""
import os
import sys
import time
import numpy as np
import torch
import subprocess
from multiprocessing.connection import Listener
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav.env import TRAINING_MODE, OBS_MS                                          # noqa: E402
from nav.obs import NAV_OBS_VERSION                                                # noqa: E402
from nav.model import NavActorCritic                                               # noqa: E402
from nav.waypoints import (ARRIVE_MIN_SEGMENTS, ARRIVE_TIGHTEN_AT, ARRIVE_STEP,     # noqa: E402
                           ARRIVE_R_FINAL, ARRIVE_DZ_FINAL)
from nav.ppo_recurrent import Rollout, ppo_update, LR                              # noqa: E402

N_INSTANCES = 8                # game instances (ports PORT0 .. PORT0+N-1); run_game_loop.ps1 -Instances must match
PORT0 = 8888
ROLLOUT_PER_INSTANCE = 1024    # 8 x 1024 = 8192 decisions per update, within 2 % of the previous
                               # 8 x 1024 = 8192, so the PPO batch, the ~250 sequences per update and
                               # the number of gradient steps are effectively unchanged. Raising
                               # instances buys wall-clock throughput, not a different batch.
                               # WHY 12 AND NOT 16: the 8 GB GPU, not system RAM, is the ceiling.
                               # Each game instance holds ~450 MiB of VRAM and the PPO update needs
                               # ~1.9 GB. 16 instances left 329 MiB free, the update spilled into
                               # shared system memory: >8 min instead of 3 s. 12 left 1.1 GB and the
                               # update still cost 15 s, wiping out the faster collection (2026-09-18).
PIPELINE_GROUPS = 2            # workers are stepped in this many interleaved groups so the policy
                               # forward for one group overlaps the other group's game stepping
                               # (lockstep means a game holds until its reply arrives, so without
                               # this the games idle through every forward and the trainer idles
                               # through every game step). 16 / 2 keeps the forward batch at 8,
                               # exactly what it was with 8 instances.
CHECKPOINT_EVERY = 25          # numbered checkpoint every N updates
LATEST_EVERY = 5               # nav_latest.pth (the resume point) every N updates
LOG_EVERY_SEGMENTS = 100
SEED = 1
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(HERE, 'logs', 'nav')
CKPT_DIR = os.path.join(HERE, 'models', 'nav')
RESUME = True
TRACE = True                   # per-decision CSV in logs/nav/trace_<stamp>.csv (cheap; keep on)


class Logger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, f'train_nav_{datetime.now():%Y%m%d_%H%M%S}.log')
        self.fh = open(self.path, 'a', encoding='utf-8')

    def __call__(self, s):
        line = f'[{datetime.now():%H:%M:%S}] {s}'
        print(line, flush=True); self.fh.write(line + '\n'); self.fh.flush()


def save_ckpt(model, opt, update, steps, stats, mission, numbered=True, arrive_r=1.5, arrive_dz=2.0):
    os.makedirs(CKPT_DIR, exist_ok=True)
    d = {'model': model.state_dict(), 'opt': opt.state_dict(), 'update': update, 'steps': steps,
         'obs_version': NAV_OBS_VERSION, 'stats': stats, 'mission': mission, 'saved': datetime.now().isoformat(),
         'arrive_r': arrive_r, 'arrive_dz': arrive_dz}
    latest = os.path.join(CKPT_DIR, 'nav_latest.pth')
    _save_atomic(d, latest)
    if not numbered:
        return latest
    p = os.path.join(CKPT_DIR, f'nav_{update:06d}.pth')
    _save_atomic(d, p)
    return p


def _save_atomic(d, path, tries=12):
    """torch.save to <path>.tmp then replace, retrying on PermissionError.

    models/ lives under OneDrive, which opens the checkpoint to upload it; while it holds that
    handle os.replace fails with WinError 5 and the trainer used to die mid-run (2026-09-19 02:48).
    A checkpoint is written every 5 updates (~100 s) so the collision is not rare. Retrying for a
    few seconds rides it out; if it still fails the exception propagates as before."""
    torch.save(d, path + '.tmp')
    for i in range(tries):
        try:
            os.replace(path + '.tmp', path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.5)


class WorkerProxy:
    """The trainer's side of one instance worker process."""

    def __init__(self, idx, port, seed, arrive, dev):
        self.idx = idx; self.port = port
        # a plain subprocess (see nav/vec_worker.py): multiprocessing's spawn would re-import this
        # module, and torch with it, in every worker (380 MB each on a 16 GB machine)
        self.listener = Listener(('127.0.0.1', 0), authkey=b'nav')
        self.proc = subprocess.Popen([sys.executable, '-m', 'nav.vec_worker', str(idx), str(port), str(seed),
                                      f'{arrive["r"]:.4f}', f'{arrive["dz"]:.4f}', str(self.listener.address[1])], cwd=HERE)
        self.conn = self.listener.accept()
        self.roll = Rollout(ROLLOUT_PER_INSTANCE, dev, slack=2)
        self.h = None; self.reset_flag = 1.0
        self.crop = None; self.vec = None
        self.mission = ''; self.stats = None; self.seg_count = 0; self.flips = 0

    def recv(self, log):
        msg = self.conn.recv()
        for line in msg.get('log', []):
            log(line)
        if 'error' in msg:
            raise RuntimeError(f'worker {self.idx} failed: {msg["error"]}')
        return msg

    def stop(self):
        try:
            self.conn.send(('stop',))
        except Exception:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.listener.close()


def pooled_stats(workers, mission=None):
    """SegmentManager.stats() pooled over the instances (each reports its own last 300 segments),
    optionally only those on one mission."""
    st = [w.stats for w in workers if w.stats and w.stats.get('segments', 0) > 0 and (mission is None or w.mission == mission)]
    if not st:
        return {'segments': 0, 'arrive_pct': 0.0, 'falls_per_100u': 0.0, 'speed': 0.0, 'timeout_pct': 0.0,
                'gems_pct': 0.0, 'gems_per_group': 0.0, 'pickup_speed': 0.0, 'carry_speed': 0.0,
                'turn_deg': 0.0}
    n = sum(s['segments'] for s in st)
    wavg = lambda k: sum(s[k] * s['segments'] for s in st) / n
    return {'segments': n, 'arrive_pct': wavg('arrive_pct'), 'falls_per_100u': wavg('falls_per_100u'),
            'speed': wavg('speed'), 'timeout_pct': wavg('timeout_pct'),
            'gems_pct': wavg('gems_pct'), 'gems_per_group': wavg('gems_per_group'),
            'pickup_speed': wavg('pickup_speed'), 'carry_speed': wavg('carry_speed'),
            'turn_deg': wavg('turn_deg')}


def main():
    log = Logger()
    torch.manual_seed(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)       # the game instances and workers need the other cores
    model = NavActorCritic().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    # Acting happens on a CPU copy: with eight game instances on the GPU a batch-8 forward took
    # 25 ms there (sync + contention) against 2.7 ms on the CPU. The GPU does the PPO updates.
    cdev = torch.device('cpu')
    act_model = NavActorCritic().to(cdev); act_model.eval()
    update, steps = 0, 0
    latest = os.path.join(CKPT_DIR, 'nav_latest.pth')
    ck = None
    if RESUME and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev)
        if ck.get('obs_version') != NAV_OBS_VERSION:
            raise SystemExit(f'checkpoint obs version {ck.get("obs_version")} != {NAV_OBS_VERSION}; move models/nav aside')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        update, steps = ck['update'], ck['steps']
        log(f'resumed {latest}: update {update}, steps {steps:,}, trained on {ck.get("mission")}')
    else:
        log(f'fresh model ({sum(p.numel() for p in model.parameters()):,} params), obs {NAV_OBS_VERSION}, device {dev}')
    act_model.load_state_dict(model.state_dict())
    arrive = {'r': 1.5, 'dz': 2.0}
    if ck is not None:
        arrive = {'r': float(ck.get('arrive_r', 1.5)), 'dz': float(ck.get('arrive_dz', 2.0))}

    workers = [WorkerProxy(i, PORT0 + i, SEED * 1000 + i, arrive, dev) for i in range(N_INSTANCES)]
    log(f'{N_INSTANCES} instance worker(s) on ports {PORT0}..{PORT0 + N_INSTANCES - 1}; training mode {TRAINING_MODE} '
        f'({OBS_MS if TRAINING_MODE else 16} ms per decision); arrive radius {arrive["r"]:.2f} u; device {dev}')
    try:
        for w in workers:
            msg = w.recv(log)
            w.crop, w.vec, w.mission = msg['crop'], msg['vec'], msg['mission']
            w.h = act_model.initial_state(1, cdev); w.reset_flag = 1.0
            log(f'[{w.idx}] ready on {w.port}, mission {w.mission}')

        trace = None; t_wall0 = time.perf_counter()
        if TRACE:
            trace = open(os.path.join(LOG_DIR, f'trace_{datetime.now():%Y%m%d_%H%M%S}.csv'), 'w')
            trace.write('wall,step,seg,time_left_s,x,y,z,vx,vy,vz,on_floor,fwd,back,left,right,jump,brake,oob,reward,done,outcome,goal_d,gx,gy,gp,inst\n')

        seg_total = 0; t_last = time.perf_counter(); steps_last = steps
        recent_rewards = []
        prof = {'fwd': 0.0, 'send': 0.0, 'recv': 0.0, 'handle': 0.0, 'w_game': 0.0, 'w_obs': 0.0, 'w_begin': 0.0}   # w_* = summed over workers

        def maybe_tighten_all():
            s = pooled_stats(workers)
            if s['segments'] >= ARRIVE_MIN_SEGMENTS and s['arrive_pct'] >= ARRIVE_TIGHTEN_AT and arrive['r'] > ARRIVE_R_FINAL + 1e-6:
                arrive['r'] = max(ARRIVE_R_FINAL, arrive['r'] - ARRIVE_STEP)
                arrive['dz'] = max(ARRIVE_DZ_FINAL, arrive['dz'] - ARRIVE_STEP)
                for w in workers:
                    w.conn.send(('arrive', arrive['r'], arrive['dz']))
                    w.stats = None                 # the stat must be re-earned at the new radius
                log(f'ARRIVE radius tightened to {arrive["r"]:.2f} u (dz {arrive["dz"]:.2f})')

        # ---- pipelined stepping ------------------------------------------------------
        # Split the workers into PIPELINE_GROUPS interleaved groups and keep one group's games
        # running while the other group's forward is computed. Each worker still receives exactly
        # one action per observation, computed from THAT observation with THAT worker's hidden
        # state; only the interleaving changes, so nothing the policy learns is affected.
        groups = [workers[k::PIPELINE_GROUPS] for k in range(PIPELINE_GROUPS)]

        def forward_group(g):
            """Policy forward for one group. Captures the exact inputs each action was based on."""
            tp = time.perf_counter()
            c_np = np.stack([w.crop for w in g]); v_np = np.stack([w.vec for w in g])
            c_t = torch.from_numpy(c_np); v_t = torch.from_numpy(v_np)
            h_t = torch.cat([w.h for w in g], dim=0)
            with torch.no_grad():
                out = act_model.act(c_t, v_t, h_t)
                gp = act_model.gap_prior(c_t, v_t).numpy() if TRACE else np.zeros(len(g))
            o = {'a_game': out['action_game'].numpy(), 'a_buf': out['action_buf'].numpy(),
                 'mean_dir': out['mean_dir'].numpy(),
                 'logp': out['logp'].numpy(), 'value': out['value'].numpy(),
                 'h_prev': h_t.numpy(), 'h_next': out['h_next'], 'gp': gp,
                 'crop': c_np, 'vec': v_np, 'reset': [w.reset_flag for w in g]}
            prof['fwd'] += time.perf_counter() - tp
            return o

        def send_group(g, o):
            tp = time.perf_counter()
            for i, w in enumerate(g):
                # the mean direction rides along as elements 5-6 so the worker can charge
                # TURN_COST on the policy's actual steering decision rather than on the sample
                w.conn.send(('act', np.concatenate([o['a_game'][i], o['mean_dir'][i]])))
            prof['send'] += time.perf_counter() - tp

        def collect_group(g, o):
            """Receive one transition per worker in the group and record it."""
            nonlocal steps, seg_total
            for i, w in enumerate(g):
                tp = time.perf_counter()
                msg = w.recv(log)
                prof['recv'] += time.perf_counter() - tp; tp = time.perf_counter()
                w.mission = msg['mission']; w.flips = msg['flips']; w.seg_count = msg['seg_count']
                for k, v in msg.get('prof', {}).items():
                    prof['w_' + k] += v
                if msg['skip']:
                    # no transition (frame flip / round end): drop the corrupted tail, fresh state
                    w.roll.truncate(msg['truncate'])
                else:
                    w.roll.add(o['crop'][i], o['vec'][i], o['a_buf'][i], float(o['logp'][i]),
                               float(o['value'][i]), msg['r'], float(msg['done']), o['reset'][i], o['h_prev'][i])
                    steps += 1
                    if trace is not None:
                        trace.write(f'{time.perf_counter() - t_wall0:.1f},{steps},{msg["trace"]},{o["gp"][i]:.0f},{w.idx}\n')
                        if steps % 500 == 0:
                            trace.flush()
                    if msg['done']:
                        seg_total += 1
                        recent_rewards.append(msg['ep_reward'])
                        if len(recent_rewards) > 300:
                            del recent_rewards[:-300]
                        w.stats = msg['stats']
                        if seg_total % LOG_EVERY_SEGMENTS == 0:
                            st_ = pooled_stats(workers)
                            log(f'SEG n={seg_total} last={msg["outcome"]} arrive={st_["arrive_pct"]:.0f}% falls100={st_["falls_per_100u"]:.2f} '
                                f'speed={st_["speed"]:.1f} timeout={st_["timeout_pct"]:.0f}% rew={np.mean(recent_rewards):.1f} '
                                f'dps={(steps - steps_last) / max(time.perf_counter() - t_last, 1e-6):.0f}')
                if msg['new_segment']:
                    w.h = act_model.initial_state(1, cdev); w.reset_flag = 1.0
                else:
                    w.h = o['h_next'][i:i + 1]; w.reset_flag = 0.0
                w.crop, w.vec = msg['crop'], msg['vec']
                prof['handle'] += time.perf_counter() - tp

        def bootstrap_values():
            """Value of each worker's current observation, for the GAE bootstrap. Per group, so the
            forward batch stays the same as during acting."""
            by_idx = {}
            for g in groups:
                c_t = torch.from_numpy(np.stack([w.crop for w in g]))
                v_t = torch.from_numpy(np.stack([w.vec for w in g]))
                h_t = torch.cat([w.h for w in g], dim=0)
                with torch.no_grad():
                    vv = act_model.act(c_t, v_t, h_t)['value'].numpy()
                for i, w in enumerate(g):
                    by_idx[w.idx] = float(vv[i])
            return [by_idx[w.idx] for w in workers]

        pend = [None] * PIPELINE_GROUPS
        for gi, g in enumerate(groups):          # prime the pipeline
            o = forward_group(g); send_group(g, o); pend[gi] = o

        while True:
            for gi, g in enumerate(groups):
                collect_group(g, pend[gi]); pend[gi] = None
                o = forward_group(g); send_group(g, o); pend[gi] = o
            # update once the rollouts are full (in lockstep the games simply wait meanwhile)
            if any(w.roll.full() for w in workers):
                # settle the pipeline first: no action may be outstanding while the weights change
                for gi, g in enumerate(groups):
                    if pend[gi] is not None:
                        collect_group(g, pend[gi]); pend[gi] = None
                last_v = bootstrap_values()
                t0 = time.perf_counter()
                st = ppo_update(model, opt, [w.roll for w in workers], last_v, log)
                act_model.load_state_dict(model.state_dict())
                for w in workers:
                    w.roll.clear()
                update += 1
                maybe_tighten_all()
                s = pooled_stats(workers)
                wall = time.perf_counter() - t_last
                flips = sum(w.flips for w in workers)
                missions = sorted(set(w.mission for w in workers))
                if len(missions) > 1:
                    # a map mix across instances: per-map arrival / fall numbers on their own line
                    log('MAPS ' + ' | '.join(f'{m}: n={sum(1 for w in workers if w.mission == m)} arrive={pooled_stats(workers, m)["arrive_pct"]:.0f}% '
                                             f'falls100={pooled_stats(workers, m)["falls_per_100u"]:.2f} speed={pooled_stats(workers, m)["speed"]:.1f} '
                                             f'gems={pooled_stats(workers, m)["gems_per_group"]:.1f}/grp pickup={pooled_stats(workers, m)["pickup_speed"]:.1f}' for m in missions))
                log(f'NAV upd={update} map={"+".join(missions)} r={arrive["r"]:.2f} steps={steps:,} segs={seg_total} arrive={s["arrive_pct"]:.0f}% '
                    f'falls100={s["falls_per_100u"]:.2f} speed={s["speed"]:.1f} gems={s["gems_per_group"]:.1f} pickup={s["pickup_speed"]:.1f} carry={s["carry_speed"]:.1f} turn={s["turn_deg"]:.1f} rew={np.mean(recent_rewards) if recent_rewards else 0:.1f} '
                    f'pl={st.get("pl", 0):.3f} vl={st.get("vl", 0):.3f} ent={st.get("ent", 0):.2f} entd={st.get("ent_d", 0):.2f} ec={st.get("ent_coef", 0):.4f} kl={st.get("kl", 0):.3f} '
                    f'clip={st.get("clipfrac", 0):.2f} gn={st.get("gn", 0):.2f} ep={st.get("epochs", 0)} '
                    f'dstd={model.log_std.clamp(model.LOG_STD_MIN, model.LOG_STD_MAX).exp().item():.2f} flips={flips} '
                    f'upd_s={time.perf_counter() - t0:.1f} wall_s={wall:.0f} inst={len(workers)} dps={(steps - steps_last) / max(wall, 1e-6):.0f} '
                    f'prof=' + ','.join(f'{k}:{v:.0f}' for k, v in prof.items()))
                for k in prof:
                    prof[k] = 0.0
                t_last = time.perf_counter(); steps_last = steps
                if update % CHECKPOINT_EVERY == 0:
                    pth = save_ckpt(model, opt, update, steps, s, missions[0] if missions else '', arrive_r=arrive['r'], arrive_dz=arrive['dz'])
                    log(f'saved {pth}')
                elif update % LATEST_EVERY == 0:
                    save_ckpt(model, opt, update, steps, s, missions[0] if missions else '', numbered=False, arrive_r=arrive['r'], arrive_dz=arrive['dz'])
                for gi, g in enumerate(groups):      # refill the pipeline with the new weights
                    o = forward_group(g); send_group(g, o); pend[gi] = o
    finally:
        for w in workers:
            w.stop()


if __name__ == '__main__':
    import traceback
    while True:
        try:
            main()
            break
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc()
            print('trainer crashed; restarting in 10 s (checkpoint resume)', flush=True)
            time.sleep(10)
