"""Stage 3 practice copies of the operator-approved maps (2026-09-27): physics data only, never PPO training.

    python -m nav.learned_nav.practice_maps            # writes hunt/custom/<Mission>_phys.<mcs|mis> for every map

A practice copy keeps the map's geometry (every InteriorInstance), its gems, spawn and bounds triggers and camera
path nodes, and removes what would change the marble's motion or is solid but not part of the exported geometry:
powerups and every other non-gem item, solid decorations (Duplex's glass panes, Gems Ahoy's graffiti, Marble Agility
Course's signs) and the physics-modifier zone of Gems Ahoy. The round is one hour. The info function of a .mcs is
renamed the way the game looks it up: MP_PQ_<alphanumerics of the file name>_GetMissionInfo.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUNT = os.path.join(HERE, '..', 'Marble Blast Platinum', 'platinum', 'data', 'multiplayer', 'hunt')
OUT_DIR = os.path.join(HUNT, 'custom')
SUFFIX = '_phys'

# operator approval 2026-09-27: 10 training, 2 held-out development, 3 reserves; plus KOTM (the base task)
TRAIN = ['intermediate/VortexEffect_Hunt.mcs', 'intermediate/GemsInTheRoad_Hunt.mcs', 'advanced/Tilo_Hunt.mcs',
         'expert/BasinHill_Hunt.mis', 'expert/ParkourPeaks_Hunt.mcs', 'intermediate/Duplex_Hunt.mis',
         'intermediate/Cragmire_Hunt.mis', 'beginner/Sprawl_Hunt.mcs', 'beginner/KingOfTheRing_Hunt.mis',
         'beginner/MaximoCenter_Hunt.mcs', 'beginner/KingOfTheMarble_Hunt.mcs']
DEV = ['beginner/GemsAhoy_Hunt.mcs', 'beginner/Acropolis2_Hunt.mis']
RESERVE = ['advanced/TreasureBox_Hunt.mis', 'beginner/MarbleAgilityCourse_Hunt.mcs', 'intermediate/Skatium_Hunt.mis']
KEEP_SHAPES = ('pathnode', 'astrolabe')          # camera path markers and a far-away sky decoration
DROP_TRIGGERS = ('marblephysmodtrigger',)


def alnum(s):
    return re.sub(r'[^A-Za-z0-9]', '', s)


def block_end(text, start):
    i = text.index('{', start)
    depth = 0
    while True:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text.index(';', i) + 1
        i += 1


def drop_blocks(t, kind, keep):
    """Remove every `new <kind>(...) {...};` block for which keep(datablock_lowercase) is False."""
    removed = 0
    pos = 0
    while True:
        m = re.compile(r'new %s\([^)]*\)' % kind).search(t, pos)
        if not m:
            break
        e = block_end(t, m.start())
        db = re.search(r'dataBlock\s*=\s*"([^"]*)"', t[m.start():e], re.I)
        name = db.group(1).lower() if db else ''
        if keep(name):
            pos = e
            continue
        a = m.start()
        while a > 0 and t[a - 1] in ' \t\r\n':
            a -= 1
        t = t[:a] + t[e:]
        removed += 1
        pos = a
    return t, removed


def make(rel):
    src = os.path.join(HUNT, rel)
    base, ext = os.path.splitext(os.path.basename(src))
    out_base = base + SUFFIX
    t = open(src, encoding='utf-8', errors='replace').read()
    if ext == '.mcs':
        old_fn = 'MP_PQ_%s_' % alnum(base)
        assert old_fn in t, f'{rel}: {old_fn} not found'
        t = t.replace(old_fn, 'MP_PQ_%s_' % alnum(out_base))
    # name and a one-hour round, in the info function (.mcs) and the MissionInfo object (both formats)
    t, n_name = re.subn(r'(\n\s*name\s*=\s*")([^"]*)(";)', lambda m: m.group(1) + m.group(2) + ' (physics practice)' + m.group(3), t, count=2)
    t, n_time = re.subn(r'(\n\s*time\s*=\s*")(\d+)(";)', r'\g<1>3600000\g<3>', t, flags=re.I)
    assert n_name >= 1 and n_time >= 1, f'{rel}: name {n_name} time {n_time}'
    t, n_items = drop_blocks(t, 'Item', lambda db: 'gem' in db)
    t, n_shapes = drop_blocks(t, 'StaticShape', lambda db: db in KEEP_SHAPES)
    t, n_trig = drop_blocks(t, 'Trigger', lambda db: db not in DROP_TRIGGERS)
    assert t.count('{') == t.count('}'), f'{rel}: unbalanced braces'
    n_int = len(re.findall(r'new InteriorInstance\(', t))
    n_gems = len(re.findall(r'new Item\(', t))
    dst = os.path.join(OUT_DIR, out_base + ext)
    open(dst + '.tmp', 'w', encoding='utf-8').write(t)
    os.replace(dst + '.tmp', dst)
    for stale in (dst + '.dso',):
        if os.path.exists(stale):
            os.remove(stale)
    return out_base, f'{n_int} interiors, {n_gems} gems kept, removed {n_items} items / {n_shapes} shapes / {n_trig} triggers'


def main():
    for rel in TRAIN + DEV + RESERVE:
        name, msg = make(rel)
        print(f'{name:36s} {msg}')


if __name__ == '__main__':
    sys.exit(main())
