"""Finite matched recorded-powerup handling/spending probes; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--starts', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n-per-map', type=int, default=3)
    p.add_argument('--branches', default='no_use,handling,learned_det,learned_use_sample')
    p.add_argument('--port', type=int, default=9230)
    p.add_argument('--maps', default='', help='Optional comma-separated mission subset')
    p.add_argument('--resources', default='1,0', help='Raw held types: SJ=1, SS=2, Helicopter=5, charged blast=0')
    args = p.parse_args()
    resources = {int(x) for x in args.resources.split(',')}
    if not resources or not resources <= {0, 1, 2, 5}:
        p.error('supported resources are 0,1,2,5')
    branches = args.branches.split(',')
    if not set(branches) <= {'no_use', 'handling', 'learned_det', 'learned_use_sample'}:
        p.error('unknown branch')
    os.environ.update(NAV_DRILL_PLAN='0:4', NAV_DRILL_EVAL='1', NAV_LIVE_RECORD='0',
                      NAV_PRACTICE_MODE='judgment', NAV_REAL_GEMS='1', NAV_TOUR='walk',
                      NAV_DIRECT_GEM='0', NAV_ESCAPE_POWERUPS='1',   # use damping: the checkpoint's own (40.97)
                      NAV_USE_EPS='0', NAV_JUMP_FORCE_P='0', NAV_JUMP_EXPLORE_P='0',
                      NAV_SPEED='3', NAV_RENDER_EVERY='100')
    import torch
    from nav.model import NavActorCritic
    from nav.vec_worker import InstanceWorker
    from nav.replay import load_starts, paired_summary
    from nav.phase_r import ROOT, hidden_process, stop
    from nav.protocol import RAW_POW_HELD, RAW_POW_BLAST, BLAST_REQUIRED
    torch.set_num_threads(1)
    model = NavActorCritic()
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=False)['model'])
    model.eval()
    pools = {}
    for s in load_starts(args.starts):
        held = int(s['raw'][RAW_POW_HELD])
        if held in resources and (held > 0 or s['raw'][RAW_POW_BLAST] >= BLAST_REQUIRED):
            pools.setdefault(s['mission'], []).append(s)
    rows = []
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoint=args.ckpt, checkpoint_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
                    starts=args.starts, starts_sha256=hashlib.sha256(Path(args.starts).read_bytes()).hexdigest(),
                    use_exploration_floor=0, use_damp=float(model.use_damp), branches=branches,
                    resources=sorted(resources),
                    stochastic_method='sample use only, deterministic movement; fixed seed per fixture',
                    handling='engine activation during reset excluded from reward and actor actions')
    def save():
        output.write_text(json.dumps(dict(manifest, rows=rows, summary=paired_summary(rows)), indent=2))
    for mission, pool in pools.items():
        if args.maps and mission not in args.maps.split(','):
            continue
        # Prioritize scarce SJ fixtures, then near targets, without looking at outcomes.
        pool.sort(key=lambda s: (int(s['raw'][RAW_POW_HELD]) != 1,
                                sum((a-b)**2 for a,b in zip(s['world']['pose'][:3], s['goal']))))
        selected = pool[:args.n_per_map] if args.n_per_map else pool
        w = InstanceWorker(0, args.port, 12345, .65, .6)
        w.env.srv.settimeout(120)
        w.eval_starts = selected
        w.load_map(mission)
        game = None
        try:
            print(f'PROBE map={mission} fixtures={len(selected)}', flush=True)
            exe = ROOT.parent / 'Marble Blast Platinum/marbleblast_mbx.exe'
            game = hidden_process([str(exe), '-autotrain', mission, '-aiport', str(args.port),
                                   '-offline', '-aicompetitive', '0', '-aiseed', '10100'], cwd=exe.parent)
            w.connect()
            w.env.conn.settimeout(120)
            print(f'CONNECTED map={mission}', flush=True)
            for s in selected:
                for branch in branches:
                    try:
                        torch.manual_seed(int.from_bytes(hashlib.sha256(s['id'].encode()).digest()[:4], 'big'))
                        w.begin_replay(s, handling=branch == 'handling')
                        h = model.initial_state(1, 'cpu')
                        with torch.no_grad():
                            for c, v in w.warm or []:
                                h = model.core(torch.as_tensor(c)[None], torch.as_tensor(v)[None], h)
                        w.warm = None
                        probabilities = []; airborne = False; landings = 0
                        outside_grid = 0; outside_grid_steering = 0; recovery_without_oob = 0
                        peak_z = float(w.env.pos()[2])
                        while True:
                            with torch.no_grad():
                                o = model.act(torch.as_tensor(w.crop)[None], torch.as_tensor(w.vec)[None], h, deterministic=True)
                            a = o['action_game'][0].tolist() + o['mean_dir'][0].tolist()
                            if float(model.use_prior(torch.as_tensor(w.vec)[None])[0]) > .5:
                                probabilities.append(float(o['use_probability'][0]))
                            if branch in ('no_use', 'handling'):
                                a[5] = 0  # handling evaluates one reset maneuver without additional uses
                            elif branch == 'learned_use_sample':
                                a[5] = float(torch.rand(()) < o['use_probability'][0])
                            outside = not w.terrain.contains(*w.env.pos()[:2])
                            outside_grid += int(outside)
                            outside_grid_steering += int(outside and not w.recovery_pending)
                            recovery_without_oob += int(w.recovery_pending and not w.env.msg.oob)
                            rep = w.step(a); h = o['h_next']
                            if rep['skip']:
                                raise RuntimeError('unexpected reset inside named window')
                            peak_z = max(peak_z, float(w.env.pos()[2]))
                            if not w.on_floor:
                                airborne = True
                            elif airborne:
                                landings += 1; airborne = False
                            if rep['done']:
                                row = dict(w.last_drill_result)
                                if row['elapsed_ms'] != w.replay_window_ms:
                                    raise RuntimeError('window clock mismatch')
                                row.update(landings=landings, peak_z=peak_z,
                                           outside_grid_decisions=outside_grid,
                                           outside_grid_steering_decisions=outside_grid_steering,
                                           recovery_controls_without_oob=recovery_without_oob,
                                           eligible_decisions=len(probabilities),
                                           learned_use_probability_mean=sum(probabilities)/len(probabilities) if probabilities else None)
                                break
                    except (ValueError, RuntimeError) as e:
                        row = dict(id=s['id'], status='failed', reason=str(e))
                        if 'catalog' in str(e):
                            row.update(expected_catalog=s['catalog'], actual_catalog=w.env.replay_catalog,
                                       recent_lines=w.env.recent_lines)
                    row.update(branch=branch, mission=mission, split_group=s['split_group'])
                    rows.append(row); save()
                    print(json.dumps(row), flush=True)
        finally:
            w.env.close(); stop(game)
    save()
    print(json.dumps(paired_summary(rows), indent=2), flush=True)


if __name__ == '__main__':
    main()
