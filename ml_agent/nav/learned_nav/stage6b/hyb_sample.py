"""Sampling profiler around hybrid.play: where does a decision spend its wall time?"""
import sys, threading, time, collections, os
sys.path.insert(0, 'C:/Users/doug/OneDrive/Documents/GitHub/PlatinumQuest-Dev/ml_agent')
port = int(sys.argv[1]); secs = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
from nav.learned_nav import hybrid as HY
main_id = threading.get_ident()
leaf = collections.Counter(); hyb = collections.Counter(); stacks = collections.Counter()
def sampler():
    time.sleep(40)      # skip startup
    t_end = time.time() + secs; n = 0
    while time.time() < t_end:
        fr = sys._current_frames().get(main_id)
        if fr is not None:
            chain = []
            f = fr
            while f is not None:
                chain.append('%s:%d %s' % (os.path.basename(f.f_code.co_filename), f.f_lineno, f.f_code.co_name))
                f = f.f_back
            leaf[chain[0]] += 1
            h = next((c for c in chain if c.startswith('hybrid.py')), '-')
            hyb[h] += 1
            stacks[' < '.join(chain[:4])] += 1
            n += 1
        time.sleep(0.02)
    print('==== samples', n, flush=True)
    for name, c in hyb.most_common(15):
        print('%5.1f%%  hybrid line %s' % (100.0 * c / n, name), flush=True)
    print('---- leaves', flush=True)
    for name, c in leaf.most_common(15):
        print('%5.1f%%  %s' % (100.0 * c / n, name), flush=True)
    print('---- stacks', flush=True)
    for name, c in stacks.most_common(10):
        print('%5.1f%%  %s' % (100.0 * c / n, name), flush=True)
    os._exit(0)
threading.Thread(target=sampler, daemon=True).start()
HY.play(port, 'KingOfTheMarble_Hunt', 1, 'current', 'prof', lambda m: print(m, flush=True), shortcuts=0, rescue=False)
