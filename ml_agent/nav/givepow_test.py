"""Check the GIVEPOW bridge word (log 40.27) on one game: grant a Super Speed, see it held in the observation, fire it
along a yaw and measure the kick, clear it with GIVEPOW none.
    python -m nav.givepow_test --port 9975    then start: marbleblast_mbx.exe -autotrain KingOfTheMarble_Hunt -aiport 9975
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav.env import HuntEnv                                    # noqa: E402
from nav.protocol import RAW_POW_HELD, NOOP_ACTION             # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=9975); args = ap.parse_args()
env = HuntEnv(args.port, speed=3, log=print)
env.connect()
for _ in range(120):                                           # through the countdown
    env.step(NOOP_ACTION, repeat=1)
    if env.time_left_s() < 179.0:
        break
def held():
    return int(env.msg.obs[RAW_POW_HELD])
print('held at start', held())
ok = True
for trial in range(3):
    env.control('GIVEPOW SuperSpeedItem_MBU')
    for _ in range(2):
        env.step(NOOP_ACTION, repeat=1)
    h1 = held()
    v0 = (float(env.msg.obs[3]), float(env.msg.obs[4]))
    yaw = [0.0, math.pi / 2, math.pi][trial]                    # forward = (sin yaw, cos yaw)
    env.step(NOOP_ACTION, use_pow=1, pow_yaw=yaw)
    for _ in range(3):
        env.step(NOOP_ACTION, repeat=1)
    v1 = (float(env.msg.obs[3]), float(env.msg.obs[4]))
    dv = (v1[0] - v0[0], v1[1] - v0[1])
    exp = (math.sin(yaw), math.cos(yaw))
    along = dv[0] * exp[0] + dv[1] * exp[1]
    print(f'trial {trial}: held after GIVEPOW {h1}, after the use {held()}; dv ({dv[0]:.1f}, {dv[1]:.1f}) = {along:.1f} along yaw {yaw:.2f}')
    ok &= (h1 == 2 and held() == 0 and along > 15.0)
    for _ in range(30):
        env.step(NOOP_ACTION, repeat=1)
env.control('GIVEPOW SuperSpeedItem_MBU'); env.step(NOOP_ACTION, repeat=1); env.step(NOOP_ACTION, repeat=1); g = held()
env.control('GIVEPOW none'); env.step(NOOP_ACTION, repeat=1); env.step(NOOP_ACTION, repeat=1); n = held()
print(f'granted {g}, after GIVEPOW none {n}')
ok &= (g == 2 and n == 0)
print('GIVEPOW OK' if ok else 'GIVEPOW FAILED')
env.close()
