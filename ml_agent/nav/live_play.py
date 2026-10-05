"""Live play against people (production, 2026-10-04): the navigator drives the model's account in a real
multiplayer game at normal speed, as a joined player or as the host.

The training code is not changed. This module sets live defaults, switches nav.env to real time, prints the
bridge's live diagnostics, and then runs nav.real_run as it is.

    python -m nav.live_play          (from ml_agent; or ml_agent/play_live.ps1, which also starts the game)

Game side: the repo build launched with  marbleblast_mbx.exe -ailive . The flag makes client/init.cs load
client/scripts/ai/live/agentLive.cs on top of the training bridge. Log in, join or host a KOTM server and
start the round: the bridge connects when the game starts, plays the Ready/Set countdown and drives from GO.

What differs from training:
* nav.env.TRAINING_MODE False: no FIXEDSTEP / LOCKSTEP / RENDEREVERY / VIEWYAW words. The game runs in real
  time and each decision is held for ACTION_REPEAT 16 ms ticks (64 ms), the bridge as it was before
  2026-09-18. Lockstep or a fixed step on a server with other players would stall or change their game.
* SPEED 1.
* The bridge's DEBUG|live| lines are printed: why it waits (marble, game state, GO) and what the gem scan sees.
* The map is NAV_MAP. A joined client may not report its mission file, and the terrain map is keyed by it.
Known limit (2026-10-04): as a joined player the powerup state is not readable (the observer reads it from the
server's client list), so Super Speed is not used. Collecting gems and the stuck-breaker work as in evals.

Settings are the nav.real_run environment variables; the defaults below are the production ones.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {
    'NAV_CKPT': os.path.join(HERE, 'models', 'nav', 'nav_r2_stop_29665.pth'),   # end of the 10-04 manoeuvre run
    'NAV_PORT': '8888',                       # the bridge's default port (no -aiport needed)
    'NAV_SPEED': '1',
    'NAV_ROUNDS': '50',
    'NAV_MAP': 'KingOfTheMarble_Hunt',
    'NAV_TOUR': 'walk',                       # the whole-spawn gem order, as in every gate since 10-02
    'NAV_TAG': 'live',
    'NAV_TRACE': os.path.join(HERE, 'logs', 'nav', 'real_trace_live.csv'),
    'CUDA_VISIBLE_DEVICES': '-1',             # CPU: one game at 15 decisions a second needs no GPU
}
for _k, _v in DEFAULTS.items():
    os.environ.setdefault(_k, _v)

sys.path.insert(0, HERE)
import nav.env as E                           # noqa: E402

E.TRAINING_MODE = False

_parse = E.parse_message


def _parse_and_show(line):
    m = _parse(line)
    if m.kind == 'debug' and m.fields and m.fields[0] == 'live':
        print('[live] game: ' + '|'.join(m.fields[1:])[:400], flush=True)
    return m


E.parse_message = _parse_and_show

_request_info = E.HuntEnv.request_info


def _request_info_live(self, wait_ticks=30):
    _request_info(self, wait_ticks)
    if os.environ.get('NAV_MAP'):
        self.info['mission'] = os.environ['NAV_MAP']
    return self.info


E.HuntEnv.request_info = _request_info_live

if __name__ == '__main__':
    print(f'live play: {os.path.basename(os.environ["NAV_CKPT"])} on {os.environ["NAV_MAP"]}, port '
          f'{os.environ["NAV_PORT"]}; start the game with marbleblast_mbx.exe -ailive', flush=True)
    sys.argv = ['nav.real_run']
    runpy.run_module('nav.real_run', run_name='__main__')
