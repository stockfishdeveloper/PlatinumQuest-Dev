"""
Play a captured situation yourself (2026-10-29, operator: "i need a way where i can stop the 1x and have you set that
situation up for me so i can play it a different way than the model and can have you fix the model").

record_demos.py's recorder plus the moment set up again at the start of every round:
  Ready/Set  the situation's gems are shown (GEMSET) and its powerup is given (GIVEPOW)
  GO         the marble is placed where the planner's was and held there 1.5 s (TELEPORT every tick: look around, turn
             the camera), then released with the situation's velocity and spin, and you play on from there
Every new round sets the same moment up again, so it can be tried several times. The recording is record_demos.py's
(demos/demo_<stamp>.npz) with demos/demo_<stamp>.situation.json beside it naming the situation and the rounds.

    python record_situation.py situations/CubeIsle_Hunt_20261029_091200_d318.json      (from ml_agent)

play_situation.ps1 starts this and the game together. Not set up again: the round clock (a round starts full) and the
powerups lying on the map (all there at a round start).
"""
import argparse
import json
import os
import socket
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from record_demos import DemoRecorder, resolve_terrain_path
from nav.protocol import RAW_TIME_LEFT

TICK_MS = (1., 200.)       # a running round clock drops this much between two messages (16 ms a tick); a bigger
                           # jump is a mission change (the game loads another mission before the one asked for)
HOLD_TICKS = 94            # after GO the marble is held at the situation's spot this many ticks (1.5 s), then released
GEM_RESENDS = (20, 50)     # ticks after GO: the gems again, while none has been collected (the round's own first group
                           # can appear a moment after GO and replace them)


def gemset(sit):
    return 'GEMSET ' + ' '.join('%.3f %.3f %.3f' % tuple(g[:3]) for g in sit['gems'])


def teleport(sit, moving):
    p = sit['position']; v = sit['velocity'] if moving else [0., 0., 0.]
    w = (sit.get('spin') or [0., 0., 0.]) if moving else [0., 0., 0.]
    return 'TELEPORT ' + ' '.join('%.3f' % x for x in list(p[:3]) + list(v[:3]) + list(w[:3]))


class SituationSetup:
    """Chooses each reply: the setup words, one per game message, at the start of every round; RECORD otherwise."""

    def __init__(self, sit):
        self.sit = sit; self.rounds = 0; self.reset()

    def reset(self):
        self.phase = 'wait'; self.queue = []; self.prev_ms = None; self.running = 0; self.hold = 0; self.after_go = 0
        self.picked = False
        self.round_ms = float(self.sit['round_s']) * 1000. if self.sit.get('round_s') else None

    def reply(self, line, recording):
        parts = line.split('|')
        try:
            obs = json.loads(parts[0])
        except (ValueError, IndexError):
            return 'RECORD'
        if len(obs) == 0:                       # round end: []|-total|0|1 -> the next round starts over
            self.reset(); return 'RECORD'
        if not recording or len(obs) <= RAW_TIME_LEFT:
            return 'RECORD'                     # the recording handshake comes first
        left = float(obs[RAW_TIME_LEFT])
        if self.prev_ms is not None and left > self.prev_ms + 1000. and self.phase != 'wait':
            self.reset()                        # the round clock jumped back up: a restarted round sets the moment up again
        drop = None if self.prev_ms is None else self.prev_ms - left
        self.prev_ms = left
        ticking = drop is not None and TICK_MS[0] <= drop <= TICK_MS[1] and (self.round_ms is None or left < self.round_ms)
        self.running = self.running + 1 if ticking else 0
        if self.phase == 'wait':
            self.phase = 'ready'
            self.queue = ([gemset(self.sit)] if self.sit['gems'] else []) + ['GIVEPOW ' + (self.sit.get('held_item') or 'none')]
            print('  [%s] Ready/Set: gems and powerup set up' % datetime.now().strftime('%H:%M:%S'))
        elif self.phase == 'ready' and self.running >= 2:
            self.phase = 'hold'; self.hold = HOLD_TICKS; self.rounds += 1
            self.queue = ([gemset(self.sit)] if self.sit['gems'] else []) + (
                ['GIVEPOW ' + self.sit['held_item']] if self.sit.get('held_item') else [])
            print('  [%s] GO (round %d): marble held at the spot for %.1f s' % (
                datetime.now().strftime('%H:%M:%S'), self.rounds, HOLD_TICKS * .016))
        elif self.phase == 'hold':
            self.hold -= 1
            if self.hold <= 0:
                self.phase = 'go'
                self.queue = [teleport(self.sit, True)] + ([gemset(self.sit)] if self.sit['gems'] else []) + (
                    ['GIVEPOW ' + self.sit['held_item']] if self.sit.get('held_item') else [])
                print('  [%s] released at %.1f u/s' % (datetime.now().strftime('%H:%M:%S'),
                                                      (self.sit['velocity'][0] ** 2 + self.sit['velocity'][1] ** 2) ** .5))
            elif not self.queue:
                return teleport(self.sit, False)
        elif self.phase == 'go':
            self.after_go += 1
            try:
                self.picked = self.picked or float(parts[1]) > 0
            except (ValueError, IndexError):
                pass
            if self.after_go in GEM_RESENDS and not self.picked and self.sit['gems']:
                self.queue.append(gemset(self.sit))
            if self.after_go > max(GEM_RESENDS):
                self.phase = 'playing'
        return self.queue.pop(0) if self.queue else 'RECORD'


def main():
    ap = argparse.ArgumentParser(description='Play a captured situation yourself (recorded like record_demos.py)')
    ap.add_argument('situation')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8888)
    ap.add_argument('--out', default=os.path.join(HERE, 'demos'))
    args = ap.parse_args()
    with open(args.situation, encoding='utf-8') as f:
        sit = json.load(f)
    rec = DemoRecorder(resolve_terrain_path(sit['map'], False), args.out)
    setup = SituationSetup(sit)
    meta = rec.path[:-4] + '.situation.json'

    def save_meta():
        with open(meta, 'w', encoding='utf-8') as f:
            json.dump(dict(situation=os.path.abspath(args.situation), map=sit['map'], decision=sit['decision'],
                           rounds=setup.rounds, demo=rec.path), f, indent=1)

    print('Situation: %s decision %d, marble at (%.1f, %.1f, %.1f), holding %s, %d gems' % (
        sit['map'], sit['decision'], *sit['position'][:3], sit.get('held_item') or 'nothing', len(sit['gems'])))
    print('Recording to %s. Listening on %s:%d; start the game on %s. Ctrl+C to stop.' % (rec.path, args.host, args.port, sit['map']))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(1.0)
    srv.bind((args.host, args.port)); srv.listen(1)
    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            print('Game connected from %s' % (addr,))
            buf = ''
            setup.reset()
            try:
                while True:
                    data = conn.recv(8192).decode('utf-8')
                    if not data:
                        print('Game disconnected; waiting for it again (Ctrl+C to finish).')
                        break
                    buf += data
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('DEBUG|'):
                            print('  game: ' + line[6:])     # the setup words' acknowledgements (no reply: replies pair with observations)
                            continue
                        try:
                            rec.handle(line)
                        except Exception as e:
                            rec.malformed += 1
                            if rec.malformed <= 5:
                                print('  bad message (%s): %s' % (e, line[:120]))
                        conn.sendall((setup.reply(line, rec.recording_started) + '\n').encode('utf-8'))
            except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                print('Connection lost (%s); waiting for the game again (Ctrl+C to finish).' % e)
            finally:
                conn.close()
                rec.frame_history.clear()
                rec.recording_started = False
                rec.save(); save_meta()
    except KeyboardInterrupt:
        pass
    finally:
        rec.save(); save_meta()
        rec.report(final=True)
        srv.close()


if __name__ == '__main__':
    main()
