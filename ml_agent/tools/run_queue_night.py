"""Night queue (2026-10-29): runs run_flag_native.py campaigns one at a time from a JSON list, each after the previous one's
status file reads complete or failed, with no campaign / sim python running, a HOLD file next to this script absent (or older
than 30 min) and 60 s of pause. usage (repo root, PYTHONPATH=ml_agent, detached):
  python -u ml_agent/tools/run_queue_night.py AFTER_STATUS QUEUE.json
QUEUE.json: [{"label": "...", "seeds": [s1, s2], "flag": "--x", "require": "module.ATTR=json" or "", "workers": 6,
              "always_confirm": true, "bench": "" or "ml_agent/tools/x_bench.py", "note": "no_spaces",
              "baseline_flag": "" or "--y", "fixed_lead": 16 (optional, 2026-10-29: deterministic budget), "maps": [...] (optional)},
             ...]"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path('ml_agent/dynamics_reports/overnight_20261026')   # HOLD file, log, status files and campaign results (gitignored)
HOLD = ROOT / 'HOLD'
LOG = ROOT / 'run_queue_night.log'


def log(text):
    line = time.strftime('%H:%M:%S ') + text
    print(line, flush=True)
    with LOG.open('a') as f: f.write(line + '\n')


def busy():
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
                          "'native_worker_bench|_bench.py|run_flag_native|run_(?!queue).*_native|sim_bench' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    try: return int(out or 0)
    except ValueError: return 0


def stage(status):
    try: return json.loads((ROOT / status).read_text()).get('stage')
    except Exception: return None


def wait_done(status):
    while stage(status) not in ('complete', 'failed'):
        time.sleep(30)
    log('%s: %s' % (status, stage(status)))


def main():
    after, queue = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
    log('queue start after %s: %s' % (after, ' '.join(item['label'] for item in queue)))
    wait_done(after)
    for item in queue:
        time.sleep(60)
        while (HOLD.exists() and time.time() - HOLD.stat().st_mtime < 1800) or busy():
            time.sleep(30)
        label = item['label']
        if (ROOT / ('%s_native_plan.json' % label)).exists():
            log('skip %s: plan file exists' % label); continue
        cmd = [sys.executable, '-u', str(Path(__file__).resolve().parent / 'run_flag_native.py'), '--label', label, '--seeds', *map(str, item['seeds']),
               '--candidate-flag=' + item['flag'], '--port-base', '9440', '--workers-per-game', str(item.get('workers', 6)),
               '--note', item.get('note', 'queued_by_run_queue_night')]
        if item.get('require'): cmd += ['--require-global', item['require']]
        if item.get('baseline_flag'): cmd += ['--baseline-flag=' + item['baseline_flag']]
        if item.get('always_confirm', True): cmd += ['--always-confirm']
        if item.get('bench'): cmd += ['--bench', item['bench']]
        if item.get('fixed_lead'): cmd += ['--fixed-lead', str(item['fixed_lead'])]
        if item.get('maps'): cmd += ['--maps', *item['maps']]
        env = dict(os.environ, PYTHONPATH='ml_agent')
        with (ROOT / ('%s_runner.log' % label)).open('w') as out, (ROOT / ('%s_runner.err' % label)).open('w') as err:
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err)
        log('launched %s pid %d' % (label, proc.pid))
        proc.wait()
        log('%s exited %s; status %s' % (label, proc.returncode, stage('%s_native_status.json' % label)))
    log('queue done')


if __name__ == '__main__':
    main()
