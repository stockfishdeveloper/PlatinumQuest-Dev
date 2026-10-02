"""Hold 5 s (CPU check), then the reply-delay series with a blocking recv and with a spin-polling recv."""
import sys, time, select
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
from nav.learned_nav.geometry import Geometry
from nav.learned_nav.session import Session
from nav.protocol import NOOP_ACTION
from nav import env as ENV
port = int(sys.argv[1])
g = Geometry('KingOfTheMarble_Hunt')
s = Session(port, 'kotmjump_p0', g, log=lambda x: print(x, flush=True))
s.ready()
def spin(ms):
    t = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < t: pass
def series(label):
    for d in (0, 3, 8, 20, 100):
        n = 40; t0 = time.perf_counter(); waits = []
        for _ in range(n):
            spin(d); t1 = time.perf_counter(); s.step(NOOP_ACTION); waits.append(time.perf_counter() - t1)
        print('%s reply delay %3d ms: decision %.1f ms, game wait median %.1f p90 %.1f max %.1f ms' % (label, d, 1000*(time.perf_counter()-t0)/n, 1000*sorted(waits)[n//2], 1000*sorted(waits)[int(0.9*n)], 1000*max(waits)), flush=True)
print("holding 5 s", flush=True); spin(5000); s.step(NOOP_ACTION); print("held", flush=True)
series('blocking')
orig = ENV.HuntEnv._readline
def polling(self):
    self.conn.setblocking(False)
    try:
        while True:
            i = self.buf.find(b'\n')
            if i >= 0:
                line, self.buf = self.buf[:i + 1], self.buf[i + 1:]
                return line.decode('utf-8', 'replace')
            try:
                data = self.conn.recv(65536)
            except BlockingIOError:
                continue
            if not data:
                raise RuntimeError('disconnected')
            self.buf += data
    finally:
        self.conn.setblocking(True)
ENV.HuntEnv._readline = polling
series('polling ')
print('done', flush=True)
