import sys, time
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.learned_nav.geometry import Geometry
from nav.learned_nav.session import Session
from nav.protocol import NOOP_ACTION
port = int(sys.argv[1])
g = Geometry('KingOfTheMarble_Hunt')
s = Session(port, 'kotmjump_p0', g, log=lambda x: print(x, flush=True))
s.ready()
def spin(ms):
    t = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < t: pass
for d in (0, 2, 3, 5, 8, 12, 20, 50, 100):
    n = 40; t0 = time.perf_counter(); waits = []
    for _ in range(n):
        spin(d); t1 = time.perf_counter(); s.step(NOOP_ACTION); waits.append(time.perf_counter() - t1)
    print('reply delay %3d ms: decision %.1f ms, game wait median %.1f max %.1f ms' % (d, 1000*(time.perf_counter()-t0)/n, 1000*sorted(waits)[n//2], 1000*max(waits)), flush=True)
print('done', flush=True)
