"""Dashboard: POINTS PER TRAINING GAME as bars, from the workers' 'GAME map=.. points=..' log lines
(real-gem training, HANDOFF 28.25). Replaces the eval-based points chart."""
import os, ast
p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'nav', 'dashboard_nav.py')
s = open(p, encoding='utf-8').read()

def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)

rep("RESUME_RE = re.compile(r'resumed .*: update (\\d+)')\n",
    "RESUME_RE = re.compile(r'resumed .*: update (\\d+)')\n"
    "GAME_RE = re.compile(r'\\[(\\d\\d:\\d\\d:\\d\\d)\\] \\[(\\d+)\\] GAME map=(\\S+) points=([\\d.]+) gems=(\\d+) falls=(\\d+)')\n")
rep("    rtfs = []\n    current_map = ''\n    pending_maps = None",
    "    rtfs = []\n    games = []             # real-gem training rounds: (update, inst, map, points, gems, falls, time)\n    current_map = ''\n    pending_maps = None")
rep("            m = MAP_RE.search(line)\n            if m:\n                current_map = m.group(2)",
    "            m = GAME_RE.search(line)\n            if m:\n"
    "                games.append((last_update, int(m.group(2)), m.group(3), float(m.group(4)), int(m.group(5)), int(m.group(6)), m.group(1)))\n"
    "                continue\n"
    "            m = MAP_RE.search(line)\n            if m:\n                current_map = m.group(2)")
# on a revert, drop games past the resume point as well
rep("                    for k in [k for k in by_update if k > resume_at]:\n                        del by_update[k]\n                    last_update = resume_at",
    "                    for k in [k for k in by_update if k > resume_at]:\n                        del by_update[k]\n"
    "                    games = [g for g in games if g[0] <= resume_at]\n"
    "                    last_update = resume_at")
rep("    updates = [by_update[k] for k in sorted(by_update)]\n    return updates, events, counters, rtfs",
    "    updates = [by_update[k] for k in sorted(by_update)]\n    return updates, events, counters, rtfs, games")
rep("    updates, events, counters, rtfs = parse_logs()", "    updates, events, counters, rtfs, games = parse_logs()")
rep("        'pm': pm, 'human_sgem': HUMAN_S_PER_GEM, 'evals': eval_series(), 'human_points': HUMAN_POINTS,",
    "        'pm': pm, 'human_sgem': HUMAN_S_PER_GEM, 'evals': eval_series(), 'human_points': HUMAN_POINTS,\n"
    "        'games': [{'update': g[0], 'inst': g[1], 'map': g[2], 'points': g[3], 'gems': g[4], 'falls': g[5], 'time': g[6]} for g in games],")
rep('POINTS PER GAME, selected map (every real evaluation round as a dot, 8-round mean as the line, by checkpoint update; dashed = human)',
    'POINTS PER TRAINING GAME, selected map (one bar per finished round, real score from the game, by update; dashed = human; sampled policy, no stuck-breaker)')
old = s[s.index("  {\n    const emap = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;"):s.index("    Plotly.react('c-points', tr, darkLayout(Object.assign({yaxis:{title:'points', gridcolor:'#21262d', color:'#7d8590'}}, legend)), cfg);\n  }")]
new = ("  {\n"
       "    const emap = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;\n"
       "    const gs = (st.games || []).filter(g => g.map === emap && g.update >= LO);\n"
       "    const x = gs.map(g => g.update + g.inst * 0.4), y = gs.map(g => g.points);\n"
       "    const txt = gs.map(g => `upd ${g.update} inst ${g.inst} ${g.time}: ${g.points} pts, ${g.gems} gems, ${g.falls} falls`);\n"
       "    const tr = [{x, y, type:'bar', width: gs.map(() => 0.4), marker:{color:'#3fb950'}, text:txt, hovertemplate:'%{text}<extra></extra>', name:'training game'}];\n"
       "    if (gs.length >= 5) {\n"
       "      const k = 10, mx = [], my = [];\n"
       "      for (let i = k - 1; i < gs.length; i++) { let sum = 0; for (let j = i - k + 1; j <= i; j++) sum += gs[j].points; mx.push(gs[i].update); my.push(sum / k); }\n"
       "      tr.push({x:mx, y:my, type:'scatter', mode:'lines', line:{color:'#f0c040', width:2}, name:'10-game mean'});\n"
       "    }\n"
       "    const hp = (st.human_points || {})[emap];\n"
       "    if (hp && x.length) tr.push({x:[Math.min(...x), Math.max(...x)], y:[hp, hp], type:'scatter', mode:'lines', line:{dash:'dash', color:'#e6edf3', width:1}, name:'human'});\n")
rep(old, new)
ast.parse(s)
open(p, 'w', encoding='utf-8').write(s)
print('dashboard games patch applied')
