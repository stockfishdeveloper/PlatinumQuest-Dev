import os; os.environ['NAV_OBS_MS']='16'
from nav.env import HuntEnv
from nav.protocol import NOOP_ACTION
from nav.joystick import action_to_joystick
env = HuntEnv(port=8920, speed=1, action_repeat=1); env.connect(); env.request_info(); env.set_speed(1)
for _ in range(5): env.step(NOOP_ACTION, repeat=1)
o = env.msg.obs; print('spawn pos', [round(float(v),2) for v in o[0:6]])
env.teleport(-46.0, 28.0, 22.03 + 0.6, 6.0, 0.0, 0.0, settle_ticks=1)
for k in range(50):
    o = env.msg.obs
    print(k, 'x %.2f z %.3f vx %.2f vz %.3f' % (float(o[0]), float(o[2]), float(o[3]), float(o[5])))
    js = action_to_joystick(1.0, 0.0, 1.0, 1 if k in (30, 31) else 0, 0, float(o[3]), float(o[4]))
    env.step(js, repeat=1)
env.close()
