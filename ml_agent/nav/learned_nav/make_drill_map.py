"""Build the P0 drill map `kotmjump_p0` from `kotmjump` (KOTMJUMP_START_HERE.md, decision 1).

    python -m nav.learned_nav.make_drill_map            # writes .../hunt/custom/kotmjump_p0.mcs .. kotmjump_p3.mcs

The drill map is kotmjump with:
* one gem only: the floating gem of the first gem group (over the big hole at x -30.25, y 10.05);
* the other gem groups removed entirely (an empty group left in GemGroups could be picked by the
  game's random group choice and spawn nothing);
* every powerup removed (super speed, super jump, blast and mega marble change the marble's motion);
* a one-hour round, so trials never meet a round end.
The game respawns a group EXCLUDING the gem just collected, so after a pickup a one-gem map stays empty
until the recorder sends the GEMRESET control word (mlAgent.cs).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUSTOM = os.path.join(HERE, '..', 'Marble Blast Platinum', 'platinum', 'data', 'multiplayer', 'hunt', 'custom')
SRC = os.path.join(CUSTOM, 'kotmjump.mcs')
DST = os.path.join(CUSTOM, 'kotmjump_p0.mcs')
TARGET = '-30.25 10.05 21.7'
# stage 4 (2026-09-28): one drill map per floating gem of kotmjump
DRILLS = {'kotmjump_p0': '-30.25 10.05 21.7', 'kotmjump_p1': '-20.25 10.05 21.7',
          'kotmjump_p2': '-20.25 20.05 21.7', 'kotmjump_p3': '-30.25 20.05 21.7'}
POWERUPS = ('SuperSpeedItem_MBU', 'SuperJumpItem_MBU', 'BlastItem_MBU', 'MegaMarbleItem_MBU')


def block_end(text, start):
    """Index just past the '};' closing the brace block whose '{' is the first one at or after start."""
    i = text.index('{', start)
    depth = 0
    while True:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                j = text.index(';', i)
                return j + 1
        i += 1


def make(name, TARGET):
    DST = os.path.join(CUSTOM, name + '.mcs')
    t = open(SRC, encoding='utf-8').read()
    # the game looks the info function up as MP_PQ_<alphanumerics of the file name>_GetMissionInfo
    # (shared/mission.cs getMissionInfo), so the underscore of kotmjump_p0 is dropped
    t = t.replace('MP_PQ_kotmjump_', 'MP_PQ_%s_' % re.sub(r'[^A-Za-z0-9]', '', name))
    t = t.replace('name = "KOTM Jump";', 'name = "KOTM Jump %s drill";' % name.split('_')[-1].upper())
    t = re.sub(r'desc = "[^"]*";', 'desc = "Drill map for the jump physics prototype: one floating gem, no powerups, one-hour round.";', t, count=1)
    for key, val in (('Time', '3600000'), ('gems', '1'), ('gems1', '1'), ('gems2', '0'), ('maxScore', '1')):
        t, n = re.subn(r'(\n\t\t%s = )"[^"]*";' % key, r'\1"%s";' % val, t, count=1)
        assert n == 1, key

    # gem groups: keep one group, holding only the target gem
    g0 = t.index('new SimGroup(GemGroups)')
    g_end = block_end(t, g0)
    body_start = t.index('{', g0) + 1
    groups, pos = [], body_start
    while True:
        k = t.find('new SimGroup()', pos, g_end)
        if k < 0:
            break
        e = block_end(t, k)
        groups.append((k, e))
        pos = e
    keep = None
    for k, e in groups:
        if TARGET in t[k:e]:
            keep = (k, e)
    assert keep, 'target gem not found'
    k, e = keep
    grp = t[k:e]
    items, pos = [], 0
    while True:
        a = grp.find('new Item()', pos)
        if a < 0:
            break
        b = block_end(grp, a)
        items.append(grp[a:b])
        pos = b
    target_items = [it for it in items if TARGET in it]
    assert len(target_items) == 1
    new_group = 'new SimGroup() {\n\n\t\t\t' + target_items[0] + '\n\t\t};'
    new_gemgroups = 'new SimGroup(GemGroups) {\n\n\t\t' + new_group + '\n\t};'
    t = t[:g0] + new_gemgroups + t[g_end:]

    # powerups
    removed = 0
    while True:
        m = None
        for mm in re.finditer(r'new Item\(\)', t):
            e = block_end(t, mm.start())
            if any(p in t[mm.start():e] for p in POWERUPS):
                m = (mm.start(), e)
                break
        if not m:
            break
        a, e = m
        # drop the block and the whitespace before it
        a2 = a
        while a2 > 0 and t[a2 - 1] in ' \t\n':
            a2 -= 1
        t = t[:a2] + t[e:]
        removed += 1

    assert t.count('{') == t.count('}'), 'unbalanced braces'
    assert not any(p in t for p in POWERUPS)
    assert t.count('GemItem') == 1
    open(DST + '.tmp', 'w', encoding='utf-8').write(t)
    os.replace(DST + '.tmp', DST)
    if os.path.exists(DST + '.dso'):
        os.remove(DST + '.dso')
    print(f'wrote {os.path.normpath(DST)}: 1 gem at {TARGET}, {removed} powerups removed, 1 h round')


def main():
    for name, target in DRILLS.items():
        make(name, target)


if __name__ == '__main__':
    sys.exit(main())
