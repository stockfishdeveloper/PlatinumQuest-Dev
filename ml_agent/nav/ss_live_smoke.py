"""40.40 end-to-end smoke test of the live-start pipeline on ONE game (training stopped):
  1. a round instance (no drill), the policy SAMPLED so the prior's Super Speeds fire, for --steps decisions:
     the worker records every fire with its continuation (logs/nav/live_starts_<stamp>_0.jsonl)
  2. nav.ss_drill_starts --live on that file -> datasets/ss_drill/starts_live_{dev,eval}.json
  3. a stage 4 window drill run on the eval starts (warm-up, GIVEPOW, the 12 s window), --drills of them
Usage (from ml_agent; the game is started by the caller against --port, e.g. nav/learned_nav/run_one.ps1):
  python -m nav.ss_live_smoke --port 9975 --steps 3000 --drills 6
"""
import argparse, glob, json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=9975); ap.add_argument('--map', default='KingOfTheMarble_Hunt')
ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--drills', type=int, default=6)
ap.add_argument('--ckpt', default=os.path.join(HERE, 'models', 'nav', 'nav_v10_28897.pth'))
args = ap.parse_args()
os.environ['NAV_DRILL_PLAN'] = '0:0'; os.environ['NAV_LIVE_RECORD'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from nav.vec_worker import InstanceWorker                      # noqa: E402
from nav.model import NavActorCritic                           # noqa: E402

ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
model = NavActorCritic(); model.load_state_dict(ck['model']); model.eval()
w = InstanceWorker(0, args.port, 777, float(ck.get('arrive_r', 0.65)), float(ck.get('arrive_dz', 0.6)))
print(f'smoke: {os.path.basename(args.ckpt)} on port {args.port}; {args.steps} sampled round decisions', flush=True)
w.connect()
h = model.initial_state(1, 'cpu'); fires = 0; t0 = time.time()
for k in range(args.steps):
    with torch.no_grad():
        o = model.act(torch.as_tensor(w.crop).unsqueeze(0), torch.as_tensor(w.vec).unsqueeze(0), h, deterministic=False)
    rep = w.step(o['action_game'][0].tolist() + o['mean_dir'][0].tolist())
    h = model.initial_state(1, 'cpu') if rep.get('new_segment') else o['h_next']
    for line in rep.get('log', []):
        if 'GAME map' in line:
            print('  ' + line, flush=True)
    if (k + 1) % 500 == 0:
        print(f'  {k + 1} decisions, pending fires {len(w.live_pending)}, written {w.live_file is not None}', flush=True)
# flush the pending continuations by stepping on a little
for _ in range(200):
    with torch.no_grad():
        o = model.act(torch.as_tensor(w.crop).unsqueeze(0), torch.as_tensor(w.vec).unsqueeze(0), h, deterministic=False)
    rep = w.step(o['action_game'][0].tolist() + o['mean_dir'][0].tolist())
    h = model.initial_state(1, 'cpu') if rep.get('new_segment') else o['h_next']
if w.live_file is not None:
    w.live_file.flush()
files = sorted(glob.glob(os.path.join(HERE, 'logs', 'nav', 'live_starts_*_0.jsonl')), key=os.path.getmtime)
if not files:
    print('NO live starts file written', flush=True); w.env.close(); sys.exit(1)
f = files[-1]; n = sum(1 for _ in open(f, encoding='utf-8'))
print(f'live file {os.path.basename(f)}: {n} fires with a continuation in {time.time() - t0:.0f} s', flush=True)
w.env.close()
if n == 0:
    sys.exit(1)
r = subprocess.run([sys.executable, '-m', 'nav.ss_drill_starts', '--live', f], cwd=HERE, capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-500:], flush=True)
ev = json.load(open(os.path.join(HERE, 'datasets', 'ss_drill', 'starts_live_eval.json')))
dv = json.load(open(os.path.join(HERE, 'datasets', 'ss_drill', 'starts_live_dev.json')))
if not ev:
    ev = dv[:args.drills]; json.dump(ev, open(os.path.join(HERE, 'datasets', 'ss_drill', 'starts_live_eval.json'), 'w'))
print(f'eval starts {len(ev)} (dev {len(dv)}); first id {ev[0]["id"]} spin {np.round(ev[0]["pre"][6:9], 1).tolist()} chain {len(ev[0]["chain"])} hist {len(ev[0]["hist"])}', flush=True)
print('now run the window drill eval (a new game is needed on another port):', flush=True)
print(f'  python -m nav.ss_drill_eval --ckpt {args.ckpt} --stage 4 --n {args.drills} --port {args.port + 1} --tag smoke_learned', flush=True)
