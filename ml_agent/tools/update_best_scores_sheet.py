"""Push the repo's best-scores CSV into the canonical Google Sheet through its Apps Script web app.

Usage: python ml_agent/tools/update_best_scores_sheet.py <web app URL> [--check]
The URL comes from the Apps Script deployment (ml_agent/tools/best_scores_sheet_appsscript.gs).
Reads ml_agent/docs/BEST_REAL_SCORES_*.csv (newest), posts every map's best and a History row
for every date present, then reads the sheet back and prints it.
"""
import csv, glob, json, sys, urllib.request, collections
TOKEN = 'pq-best-scores-2026'
HISTORY_SEED = {  # bests as of each earlier date (from the handoff logs), carried forward by the script consumer
    '2026-10-06': {'KingOfTheMarble_Hunt': 175},
    '2026-10-13': {'KingOfTheMarble_Hunt': 175, 'Bowl_Hunt': 83, 'Duplex_Hunt': 118},
    '2026-10-14': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 140, 'CubeIsle_Hunt': 111, 'Bowl_Hunt': 83, 'Duplex_Hunt': 118,
                   'FracturedIslands_Hunt': 114, 'Cragmire_Hunt': 109, 'VortexEffect_Hunt': 105, 'TreasureBox_Hunt': 163, 'Concentric_Hunt': 90},
    '2026-10-15': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 152, 'CubeIsle_Hunt': 125, 'Bowl_Hunt': 103, 'Duplex_Hunt': 131,
                   'FracturedIslands_Hunt': 114, 'Cragmire_Hunt': 121, 'VortexEffect_Hunt': 109, 'TreasureBox_Hunt': 174, 'Concentric_Hunt': 92},
    '2026-10-16': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 157, 'CubeIsle_Hunt': 128, 'Bowl_Hunt': 113, 'Duplex_Hunt': 141,
                   'FracturedIslands_Hunt': 114, 'Cragmire_Hunt': 121, 'VortexEffect_Hunt': 111, 'TreasureBox_Hunt': 174, 'Concentric_Hunt': 109},
    '2026-10-17': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 178, 'CubeIsle_Hunt': 138, 'Bowl_Hunt': 123, 'Duplex_Hunt': 153,
                   'FracturedIslands_Hunt': 114, 'Cragmire_Hunt': 126, 'VortexEffect_Hunt': 114, 'TreasureBox_Hunt': 188, 'Concentric_Hunt': 124},
    '2026-10-18': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 191, 'CubeIsle_Hunt': 138, 'Bowl_Hunt': 125, 'Duplex_Hunt': 153,
                   'FracturedIslands_Hunt': 115, 'Cragmire_Hunt': 126, 'VortexEffect_Hunt': 117, 'TreasureBox_Hunt': 196, 'Concentric_Hunt': 124},
    '2026-10-19': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 191, 'CubeIsle_Hunt': 146, 'Bowl_Hunt': 125, 'Duplex_Hunt': 153,
                   'FracturedIslands_Hunt': 115, 'Cragmire_Hunt': 132, 'VortexEffect_Hunt': 117, 'TreasureBox_Hunt': 196, 'Concentric_Hunt': 124},
    '2026-10-20': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 191, 'CubeIsle_Hunt': 146, 'Bowl_Hunt': 133, 'Duplex_Hunt': 153,
                   'FracturedIslands_Hunt': 115, 'Cragmire_Hunt': 142, 'VortexEffect_Hunt': 118, 'TreasureBox_Hunt': 196, 'Concentric_Hunt': 124},
    '2026-10-21': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 192, 'CubeIsle_Hunt': 146, 'Bowl_Hunt': 143, 'Duplex_Hunt': 153,
                   'FracturedIslands_Hunt': 115, 'Cragmire_Hunt': 142, 'VortexEffect_Hunt': 118, 'TreasureBox_Hunt': 205, 'Concentric_Hunt': 131},
    '2026-10-22': {'KingOfTheMarble_Hunt': 175, 'Sprawl_Hunt': 192, 'CubeIsle_Hunt': 146, 'Bowl_Hunt': 143, 'Duplex_Hunt': 157,
                   'FracturedIslands_Hunt': 115, 'Cragmire_Hunt': 145, 'VortexEffect_Hunt': 125, 'TreasureBox_Hunt': 205, 'Concentric_Hunt': 131},
}

def load_csv():
    path = sorted(glob.glob('ml_agent/docs/BEST_REAL_SCORES_*.csv'))[-1]
    rows = list(csv.reader(open(path, encoding='utf-8')))[1:]
    best = []
    for r in rows:
        if not r or not r[0].strip():
            continue
        best.append([r[0], int(r[2]), int(r[3]), r[-2]])  # map, score, falls, date (unquoted commas inside: date is 2nd from last)
    return path, best

def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode()

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    url = sys.argv[1]
    if '--clear' in sys.argv:
        # --clear History!M1 History!A7 ...
        ranges = [a for a in sys.argv[2:] if '!' in a]
        clear = [dict(tab=r.split('!')[0], range=r.split('!')[1]) for r in ranges]
        print('clear:', post(url, dict(token=TOKEN, clear=clear))[:400]); return
    path, best = load_csv()
    history = {k: dict(v) for k, v in HISTORY_SEED.items()}
    today = max(r[3] for r in best)
    history.setdefault(today, {})
    for m, s, f, d in best:
        history[today][m] = s
    payload = json.dumps(dict(token=TOKEN, best=best, history=history)).encode()
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        print('write:', resp.read().decode()[:600])
    with urllib.request.urlopen(url, timeout=60) as resp:
        print('read back:', resp.read().decode()[:1500])

if __name__ == '__main__':
    main()
