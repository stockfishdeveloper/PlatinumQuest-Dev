"""Navigator trainer: waypoint task on whatever hunt map the game is running.

    1. launch the game:  marbleblast.exe -autotrain FlatGemTraining_Hunt   (or run_game_loop.ps1 -Mission ...)
    2. python -m nav.train_nav            (from ml_agent/)

The game connects on port 8888; the trainer asks it which mission is loaded
(INFO), loads that map's height stack from terrain_maps/, and runs segments
(random start -> random reachable goal) at 3x game speed. Logs go to
logs/nav/, checkpoints to models/nav/ (nav_latest.pth is resumed
automatically). No command-line flags: settings are the constants below.
"""
import os
import sys
import time
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_obs import TerrainMap                                                  # noqa: E402
from nav.protocol import RAW_VEL                                                    # noqa: E402
from nav.env import HuntEnv                                                         # noqa: E402
from nav.terrain import TerrainGrid                                                 # noqa: E402
from nav.obs import ObsBuilder, NAV_OBS_VERSION                                     # noqa: E402
from nav.model import NavActorCritic, action_to_joystick, HIDDEN                    # noqa: E402
from nav.waypoints import SegmentManager, RoundOver                                 # noqa: E402
from nav.ppo_recurrent import Rollout, ppo_update, LR                               # noqa: E402

PORT = 8888
GAME_SPEED = 3
ROLLOUT = 2048                 # decisions per update (~45 s of wall time at 3x on one instance)
CHECKPOINT_EVERY = 25          # numbered checkpoint every N updates
LATEST_EVERY = 5               # nav_latest.pth (the resume point) every N updates
LOG_EVERY_SEGMENTS = 20
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


def save_ckpt(model, opt, update, steps, stats, mission, numbered=True):
    os.makedirs(CKPT_DIR, exist_ok=True)
    d = {'model': model.state_dict(), 'opt': opt.state_dict(), 'update': update, 'steps': steps,
         'obs_version': NAV_OBS_VERSION, 'stats': stats, 'mission': mission, 'saved': datetime.now().isoformat()}
    latest = os.path.join(CKPT_DIR, 'nav_latest.pth')
    torch.save(d, latest + '.tmp'); os.replace(latest + '.tmp', latest)
    if not numbered:
        return latest
    p = os.path.join(CKPT_DIR, f'nav_{update:06d}.pth')
    torch.save(d, p + '.tmp'); os.replace(p + '.tmp', p)
    return p


def main():
    log = Logger()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NavActorCritic().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    update, steps = 0, 0
    latest = os.path.join(CKPT_DIR, 'nav_latest.pth')
    if RESUME and os.path.exists(latest):
        ck = torch.load(latest, map_location=dev)
        if ck.get('obs_version') != NAV_OBS_VERSION:
            raise SystemExit(f'checkpoint obs version {ck.get("obs_version")} != {NAV_OBS_VERSION}; move models/nav aside')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        update, steps = ck['update'], ck['steps']
        log(f'resumed {latest}: update {update}, steps {steps:,}, trained on {ck.get("mission")}')
    else:
        log(f'fresh model ({sum(p.numel() for p in model.parameters()):,} params), obs {NAV_OBS_VERSION}, device {dev}')

    env = HuntEnv(PORT, speed=GAME_SPEED, log=log)
    env.connect()
    mission = env.info.get('mission', '')
    if not mission:
        raise SystemExit('the game did not answer INFO; is mlAgent.cs up to date (delete the .dso)?')
    def load_map(name):
        t = TerrainGrid(TerrainMap.resolve(name))
        log(f'mission {name}: terrain {t.W}x{t.H} @ {t.res} u, walk grid {t.wH}x{t.wW}, '
            f'walkable {int(t.walkable.sum())} cells, edge {int(t.edge.sum())}')
        return t, ObsBuilder(t), SegmentManager(t, rng, log)

    terrain, obs_b, segs = load_map(mission)
    roll = Rollout(ROLLOUT, dev)

    def sync_map():
        """After any reconnect / new round: if the game came back on another map (the game
        loop rotates missions), reload that map's terrain. Using the previous map's terrain
        put teleports into the void and made every position read as off-map (2026-09-17)."""
        nonlocal mission, terrain, obs_b, segs
        if not env.info.get('mission'):
            env.request_info()
        if env.info.get('mission') and env.info['mission'] != mission:
            mission = env.info['mission']
            terrain, obs_b, segs = load_map(mission)
            log(f'MAP {mission}')

    def start_segment():
        """Begin a segment; if the round ends or the game reconnects meanwhile, resync and retry."""
        while True:
            try:
                return segs.begin(env)
            except RoundOver:
                segs.abandon()
                if env.round_ended:
                    env.wait_new_round()
                else:
                    env.request_info(); env.set_speed(GAME_SPEED)
                sync_map()
                log(f'new round (game connection {env.connections})')

    trace = None; t_wall0 = time.perf_counter()
    if TRACE:
        trace = open(os.path.join(LOG_DIR, f'trace_{datetime.now():%Y%m%d_%H%M%S}.csv'), 'w')
        trace.write('wall,step,seg,time_left_s,x,y,z,vx,vy,vz,on_floor,fwd,back,left,right,jump,brake,oob,reward,done,outcome,goal_d\n')

    goal = start_segment(); obs_b.reset()
    h = model.initial_state(1, dev)
    reset_flag = 1.0
    crop, vec, on_floor = obs_b.build(env.msg.obs, goal)
    seg_count = 0; t_last = time.perf_counter(); env.rtf()
    ep_reward = 0.0; recent_rewards = []
    while True:
        c_t = torch.as_tensor(crop, device=dev).unsqueeze(0)
        v_t = torch.as_tensor(vec, device=dev).unsqueeze(0)
        out = model.act(c_t, v_t, h)
        a_game = out['action_game'][0].tolist()
        vel = env.msg.obs[RAW_VEL]
        js = action_to_joystick(a_game[0], a_game[1], a_game[2], a_game[3], a_game[4], float(vel[0]), float(vel[1]))
        msg, info = env.step(js)
        if info['round_ended'] or info['reconnected']:
            # Round over: the game shows the results, restarts the level and reconnects.
            # Close the segment without a verdict and wait for the new round.
            segs.abandon()
            if info['round_ended']:
                env.wait_new_round()
            else:
                env.request_info(); env.set_speed(GAME_SPEED)
            sync_map()
            log(f'new round (game connection {env.connections}); rtf {env.rtf():.2f}')
            goal = start_segment(); obs_b.reset(); h = model.initial_state(1, dev); reset_flag = 1.0
            ep_reward = 0.0
            crop, vec, on_floor = obs_b.build(env.msg.obs, goal)
            continue
        airborne = not on_floor
        r, done, outcome = segs.step(msg.obs[:3], info['fell'], airborne, info['round_ended'], env.time_left_s(),
                                     braked=a_game[4] > 0.5)
        ep_reward += r
        if trace is not None:
            o = msg.obs
            trace.write(f'{time.perf_counter() - t_wall0:.1f},{steps},{seg_count},{env.time_left_s():.2f},{o[0]:.2f},{o[1]:.2f},{o[2]:.2f},{o[3]:.2f},{o[4]:.2f},{o[5]:.2f},'
                        f'{int(on_floor)},{js[0]:.2f},{js[1]:.2f},{js[2]:.2f},{js[3]:.2f},{js[4]},{int(a_game[4] > 0.5)},{int(info["fell"])},'
                        f'{r:.2f},{int(done)},{outcome or ""},{segs.seg.prev_d:.1f}\n')
            if steps % 200 == 0:
                trace.flush()
        roll.add(crop, vec, out['action_buf'][0].cpu().numpy(), out['logp'].item(), out['value'].item(),
                 r, float(done), reset_flag, h[0].cpu().numpy())
        steps += 1
        h = out['h_next']; reset_flag = 0.0
        if done:
            seg_count += 1; recent_rewards.append(ep_reward); ep_reward = 0.0
            recent_rewards = recent_rewards[-300:]
            if seg_count % LOG_EVERY_SEGMENTS == 0:
                s = segs.stats()
                log(f'SEG n={seg_count} last={outcome} arrive={s["arrive_pct"]:.0f}% falls100={s["falls_per_100u"]:.2f} '
                    f'speed={s["speed"]:.1f} timeout={s["timeout_pct"]:.0f}% rew={np.mean(recent_rewards):.1f} rtf={env.rtf():.2f}')
            goal = start_segment(); obs_b.reset(); h = model.initial_state(1, dev); reset_flag = 1.0
        crop, vec, on_floor = obs_b.build(env.msg.obs, goal)
        if roll.full():
            with torch.no_grad():
                last_v = model.act(torch.as_tensor(crop, device=dev).unsqueeze(0),
                                   torch.as_tensor(vec, device=dev).unsqueeze(0), h)['value'].item()
            t0 = time.perf_counter()
            st = ppo_update(model, opt, roll, last_v, log)
            roll.clear(); update += 1
            s = segs.stats()
            log(f'NAV upd={update} map={mission} steps={steps:,} segs={seg_count} arrive={s["arrive_pct"]:.0f}% '
                f'falls100={s["falls_per_100u"]:.2f} speed={s["speed"]:.1f} rew={np.mean(recent_rewards) if recent_rewards else 0:.1f} '
                f'pl={st.get("pl", 0):.3f} vl={st.get("vl", 0):.3f} ent={st.get("ent", 0):.2f} kl={st.get("kl", 0):.3f} '
                f'clip={st.get("clipfrac", 0):.2f} gn={st.get("gn", 0):.2f} ep={st.get("epochs", 0)} '
                f'dstd={model.log_std.clamp(model.LOG_STD_MIN, model.LOG_STD_MAX).exp().item():.2f} '
                f'upd_s={time.perf_counter() - t0:.1f} wall_s={time.perf_counter() - t_last:.0f}')
            t_last = time.perf_counter()
            if update % CHECKPOINT_EVERY == 0:
                p = save_ckpt(model, opt, update, steps, s, mission)
                log(f'saved {p}')
            elif update % LATEST_EVERY == 0:
                save_ckpt(model, opt, update, steps, s, mission, numbered=False)


if __name__ == '__main__':
    import traceback
    while True:
        try:
            main()
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc()
            print('trainer crashed; restarting in 10 s (checkpoint resume)', flush=True)
            time.sleep(10)
