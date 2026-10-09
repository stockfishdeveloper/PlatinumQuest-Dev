
const darkLayout = (extra) => Object.assign({
  paper_bgcolor:'#161b22', plot_bgcolor:'#0d1117', font:{color:'#e6edf3', size:10, family:'Consolas, monospace'},
  margin:{l:50, r:12, t:8, b:32}, showlegend:false,
  xaxis:{gridcolor:'#21262d', color:'#7d8590', zeroline:false, title:'update'}, yaxis:{gridcolor:'#21262d', color:'#7d8590', zeroline:false},
}, extra || {});
const cfg = {responsive:true, displayModeBar:false};
const palette = ['#3fb950','#58a6ff','#f0c040','#bc8cff','#f85149','#d29922','#39d3c3','#ff7b72'];
const legend = {showlegend:true, legend:{x:0, y:1.15, orientation:'h', font:{size:9}}};
['c-sgem','c-points','c-jump1','c-jump2','c-heat','c-heatj','c-falls','c-speed','c-rew','c-ent','c-kl','c-loss','c-gn','c-wall'].forEach(id => Plotly.newPlot(id, [], darkLayout(legend), cfg));

let ST = null;   // latest state, so per-map traces can read the MAPS-line series (st.pm)
// Map selector: one map at a time (default KingOfTheMarble, the scored map) or the overlay of all.
let SEL = (function(){ try { return localStorage.getItem('nav_map_sel') || 'KingOfTheMarble_Hunt'; } catch (e) { return 'KingOfTheMarble_Hunt'; } })();
let WIN = (function(){ try { return parseInt(localStorage.getItem('nav_win_sel') || '500'); } catch (e) { return 500; } })();
let LO = -Infinity;   // first update shown; recomputed on every state push
const winEl = document.getElementById('win-sel'); winEl.value = String(WIN);
winEl.addEventListener('change', () => { WIN = parseInt(winEl.value); try { localStorage.setItem('nav_win_sel', String(WIN)); } catch (e) {} if (ST) update(ST); });
function windowed(series) {   // copy of the pooled series restricted to update >= LO
  const keep = series.update.map(u => u >= LO); const out = {};
  for (const k of Object.keys(series)) out[k] = series[k].filter((_, i) => keep[i]);
  return out;
}
const selEl = document.getElementById('map-sel');
selEl.addEventListener('change', () => { SEL = selEl.value; try { localStorage.setItem('nav_map_sel', SEL); } catch (e) {} if (ST) update(ST); loadHeatmap(); });
let HEAT_LOADED = '';   // "<map>@<trace_time>" currently drawn, so we only redraw when the trace changes
async function loadHeatmap() {
  const name = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;
  try {
    const h = await (await fetch('/heatmap?map=' + encodeURIComponent(name))).json();
    const title = document.getElementById('heat-title');
    if (h.error) { title.textContent = 'Marble speed heat map: ' + h.error; Plotly.react('c-heat', [], darkLayout(), cfg); HEAT_LOADED = ''; return; }
    const key = h.map + '@' + h.trace_time; if (key === HEAT_LOADED) return; HEAT_LOADED = key;
    title.textContent = 'Marble speed heat map, ' + h.map.replace('_Hunt','') + ' (' + h.rounds + ' real rounds, ' + h.decisions.toLocaleString() + ' floor decisions, trace ' + h.trace_time + '; green = fastest, red = slowest, 1 u cells)';
    const floor = {x:h.x, y:h.y, z:h.floor, type:'heatmap', colorscale:[[0,'#2a2f36'],[1,'#2a2f36']], showscale:false, hoverinfo:'skip', zmin:0, zmax:1};
    const speed = {x:h.x, y:h.y, z:h.z, type:'heatmap', zmin:2, zmax:11, colorscale:[[0,'#a50026'],[0.25,'#f46d43'],[0.5,'#fee08b'],[0.75,'#a6d96a'],[1,'#1a9850']],
                   colorbar:{title:{text:'u/s', font:{color:'#e6edf3'}}, tickfont:{color:'#e6edf3'}, thickness:10}, hovertemplate:'x %{x}, y %{y}: %{z} u/s<extra></extra>'};
    Plotly.react('c-heat', [floor, speed], darkLayout({margin:{l:40, r:10, t:8, b:32}, xaxis:{title:'x', gridcolor:'#21262d', color:'#7d8590', scaleanchor:'y', constrain:'domain'}, yaxis:{title:'y', gridcolor:'#21262d', color:'#7d8590'}}), cfg);
    if (h.jz) {
      document.getElementById('heatj-title').textContent = 'Jump heat map, ' + h.map.replace('_Hunt','') + ' (' + h.takeoffs + ' takeoffs in ' + h.rounds + ' real rounds; takeoffs as % of floor decisions per 1 u cell; green = jumps a lot, red = never)';
      const jump = {x:h.x, y:h.y, z:h.jz, type:'heatmap', zmin:0, zmax:5, colorscale:[[0,'#a50026'],[0.3,'#f46d43'],[0.6,'#fee08b'],[1,'#1a9850']],
                    colorbar:{title:{text:'% takeoff', font:{color:'#e6edf3'}}, tickfont:{color:'#e6edf3'}, thickness:10}, hovertemplate:'x %{x}, y %{y}: %{z}% takeoffs<extra></extra>'};
      Plotly.react('c-heatj', [floor, jump], darkLayout({margin:{l:40, r:10, t:8, b:32}, xaxis:{title:'x', gridcolor:'#21262d', color:'#7d8590', scaleanchor:'y', constrain:'domain'}, yaxis:{title:'y', gridcolor:'#21262d', color:'#7d8590'}}), cfg);
    }
  } catch (e) { console.error(e); }
}
loadHeatmap(); setInterval(loadHeatmap, 60000);
function syncSelector(st) {
  const names = st.pm ? Object.keys(st.pm) : [];
  const have = [...selEl.options].map(o => o.value);
  names.forEach(n => { if (!have.includes(n)) { const o = document.createElement('option'); o.value = n; o.textContent = n.replace('_Hunt',''); selEl.appendChild(o); } });
  if (![...selEl.options].some(o => o.value === SEL)) SEL = names.includes('KingOfTheMarble_Hunt') ? 'KingOfTheMarble_Hunt' : 'ALL';
  selEl.value = SEL;
}
function perMapTraces(s, key, mode) {
  const traces = [];
  if (ST && ST.pm && Object.keys(ST.pm).length) {
    const maps = Object.keys(ST.pm);
    if (!ST.pm[maps[0]][key]) {
      // pooled-only series (e.g. reward): one trace over the whole run
      const x = [], y = [];
      s.update.forEach((u, k) => { if (s[key][k] !== null && s[key][k] !== undefined) { x.push(u); y.push(s[key][k]); } });
      return [{x, y, type:'scatter', mode:'lines', line:{width:1.5, color:palette[0]}, name:'all maps (pooled)'}];
    }
    maps.forEach((mp, i) => {
      if (SEL !== 'ALL' && mp !== SEL) return;
      const d = ST.pm[mp]; if (!d[key]) return;
      const x = [], y = [];
      d.update.forEach((u, k) => { if (u >= LO && d[key][k] !== null && d[key][k] !== undefined) { x.push(u); y.push(d[key][k]); } });
      traces.push({x, y, type:'scatter', mode: mode || 'lines', line:{width:1.5, color:palette[i % palette.length]}, name: mp.replace('_Hunt','')});
    });
    return traces;
  }
  const maps = [...new Set(s.map)];
  maps.forEach((mp, i) => {
    const x = [], y = [];
    s.update.forEach((u, k) => { if (s.map[k] === mp && s[key][k] !== null) { x.push(u); y.push(s[key][k]); } });
    traces.push({x, y, type:'scatter', mode: mode || 'lines+markers', marker:{size:3}, line:{width:1.5, color:palette[i % palette.length]}, name: mp.replace('_Hunt','')});
  });
  return traces;
}
function fmt(v, d) { return (v === null || v === undefined) ? '--' : Number(v).toFixed(d === undefined ? 2 : d); }

function update(st) {
  ST = st;
  syncSelector(st);
  const allU = st.series.update; LO = (WIN > 0 && allU.length) ? allU[allU.length - 1] - WIN : -Infinity;
  const s = windowed(st.series), L = st.latest;
  const pmSel = SEL !== 'ALL' ? st.per_map[SEL] : null;   // the selected map's own numbers for the gauges
  const selName = SEL === 'ALL' ? 'all maps' : SEL.replace('_Hunt','');
  document.getElementById('g-sgem-label').textContent = 'Seconds between pickups (' + selName + ')';
  document.getElementById('g-sgem').textContent = (pmSel && pmSel.sgem) ? fmt(pmSel.sgem) + ' s' : (L && L.sgem ? fmt(L.sgem) + ' s' : '--');
  document.getElementById('g-sgem-sub').textContent = 'human 1.57 s (KOTM)' + ((pmSel && pmSel.best_sgem) ? ' | best ' + fmt(pmSel.best_sgem) : '');
  {
    const bmap = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;
    const all = (st.games || []).filter(g => g.map === bmap && g.gems <= 190);
    const best = all.length ? all.reduce((a, b) => (b.points > a.points ? b : a)) : null;
    document.getElementById('g-best-label').textContent = 'Best points per game (' + bmap.replace('_Hunt', '') + ')';
    document.getElementById('g-best').textContent = best ? best.points.toFixed(0) : '--';
    document.getElementById('g-best-sub').textContent = 'human 143.5 (KOTM)' + (best ? ' | upd ' + best.update + ' ' + best.time : '') + ' | ' + all.length + ' games';
  }
  {
    const tr = perMapTraces(s, 'sgem');
    if (s.update.length) tr.push({x:[s.update[0], s.update[s.update.length-1]], y:[st.human_sgem, st.human_sgem], type:'scatter', mode:'lines', line:{dash:'dash', color:'#e6edf3', width:1}, name:'human (KOTM demo)'});
    Plotly.react('c-sgem', tr, darkLayout(Object.assign({yaxis:{title:'s / pickup', gridcolor:'#21262d', color:'#7d8590'}}, legend)), cfg);
  }
  {
    const js = (st.jumps || []).filter(j => j.update >= LO);
    const x = js.map(j => j.update);
    const t1 = [{x, y: js.map(j => j.appr_per_min), type:'bar', marker:{color:'#58a6ff'}, name:'approved takeoffs / inst-min', yaxis:'y'},
                {x, y: js.map(j => j.appr_success), type:'scatter', mode:'lines+markers', line:{color:'#3fb950', width:2}, name:'success %', yaxis:'y2',
                 text: js.map(j => 'n=' + j.appr_n), hovertemplate:'%{y:.0f}% (%{text})<extra></extra>'}];
    Plotly.react('c-jump1', t1, darkLayout(Object.assign({yaxis:{title:'takeoffs / min', gridcolor:'#21262d', color:'#7d8590', rangemode:'tozero'},
      yaxis2:{title:'success %', overlaying:'y', side:'right', range:[0, 100], color:'#7d8590', showgrid:false}}, legend)), cfg);
    const t2 = [{x, y: js.map(j => j.cross_j_per_min), type:'scatter', mode:'lines', line:{color:'#3fb950', width:2}, name:'crossings with a jump / min'},
                {x, y: js.map(j => j.cross_nj_per_min), type:'scatter', mode:'lines', line:{color:'#7d8590', width:1.5}, name:'crossings without a jump / min'},
                {x, y: js.map(j => j.fall_appr_per_min), type:'scatter', mode:'lines', line:{color:'#f85149', width:2}, name:'falls after approved takeoff / min'}];
    Plotly.react('c-jump2', t2, darkLayout(Object.assign({yaxis:{title:'per inst-min', gridcolor:'#21262d', color:'#7d8590', rangemode:'tozero'}}, legend)), cfg);
  }
  {
    const emap = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;
    // one round cannot hold more than ~130 gems: larger entries are two rounds merged by a worker
    // that crossed a round end mid-respawn (fixed in vec_worker 17:00; older workers may still emit them)
    const gs = (st.games || []).filter(g => g.map === emap && g.update >= LO && g.gems <= 130);
    const x = gs.map((g, i) => i + 1), y = gs.map(g => g.points);
    let bi = -1; y.forEach((v, i) => { if (bi < 0 || v > y[bi]) bi = i; });
    const cols = y.map((v, i) => (i === bi ? '#3fb950' : '#f0883e'));
    const tr = [{x, y, type:'bar', marker:{color:cols}, hovertemplate:'%{y}<extra></extra>', name:'training game'}];
    if (gs.length >= 5) {
      const k = 10, mx = [], my = [];
      for (let i = k - 1; i < gs.length; i++) { let sum = 0; for (let j = i - k + 1; j <= i; j++) sum += gs[j].points; mx.push(i + 1); my.push(sum / k); }
      tr.push({x:mx, y:my, type:'scatter', mode:'lines', line:{color:'#58a6ff', width:2}, name:'10-game mean'});
    }
    const hp = (st.human_points || {})[emap];
    if (hp && x.length) tr.push({x:[Math.min(...x), Math.max(...x)], y:[hp, hp], type:'scatter', mode:'lines', line:{dash:'dash', color:'#e6edf3', width:1}, name:'human'});
    Plotly.react('c-points', tr, darkLayout(Object.assign({yaxis:{title:'points', gridcolor:'#21262d', color:'#7d8590'}, xaxis:{title:'training game (in order)', gridcolor:'#21262d', color:'#7d8590'}, bargap:0.15}, legend)), cfg);
  }
  {
    const emap = SEL === 'ALL' ? 'KingOfTheMarble_Hunt' : SEL;
    const gs = (st.games || []).filter(g => g.map === emap && g.update >= LO && g.gems <= 130 && g.pow);
    const x = gs.map((g, i) => i + 1);
    const labels = st.pow_labels || {};
    const colors = {f1:'#3fb950', f2:'#58a6ff', f3:'#f0c040', f4:'#bc8cff', f5:'#39d2c0', f6:'#f0883e', blast:'#f85149'};
    const keys = Object.keys(labels).filter(k => gs.some(g => (g.pow[k] || 0) > 0));
    const tr = keys.map(k => ({x, y: gs.map(g => g.pow[k] || 0), type:'bar', marker:{color:colors[k] || '#7d8590'}, name: labels[k] || k}));
    const tot = gs.map(g => keys.reduce((a, k) => a + (g.pow[k] || 0), 0));
    const uses = gs.map(g => Object.keys(g.pow).filter(k => k[0] === 'u' || k === 'blast').reduce((a, k) => a + g.pow[k], 0));
    const roll = (arr, k) => { const mx = [], my = []; for (let i = k - 1; i < arr.length; i++) { let sum = 0; for (let j = i - k + 1; j <= i; j++) sum += arr[j]; mx.push(i + 1); my.push(sum / k); } return [mx, my]; };
    if (gs.length >= 10) { const [mx, my] = roll(tot, 10); tr.push({x:mx, y:my, type:'scatter', mode:'lines', line:{color:'#e6edf3', width:2}, name:'10-game mean fires'}); }
    Plotly.react('c-pow1', tr, darkLayout(Object.assign({barmode:'stack', yaxis:{title:'fires / game', gridcolor:'#21262d', color:'#7d8590', rangemode:'tozero'}, xaxis:{title:'training game (in order)', gridcolor:'#21262d', color:'#7d8590'}, bargap:0.15}, legend)), cfg);
    // points with vs without a fire, 10-game means over the sequence (NaN where the window has no game of that kind)
    const k = 10, px = [], pw = [], pn = [];
    for (let i = k - 1; i < gs.length; i++) {
      let sw = 0, nw = 0, sn = 0, nn = 0;
      for (let j = i - k + 1; j <= i; j++) { if (tot[j] > 0) { sw += gs[j].points; nw++; } else { sn += gs[j].points; nn++; } }
      px.push(i + 1); pw.push(nw ? sw / nw : null); pn.push(nn ? sn / nn : null);
    }
    const t2 = [{x:px, y:pw, type:'scatter', mode:'lines', line:{color:'#3fb950', width:2}, name:'points, games with a fire', connectgaps:true},
                {x:px, y:pn, type:'scatter', mode:'lines', line:{color:'#7d8590', width:2}, name:'points, games without', connectgaps:true}];
    if (gs.length >= 10) { const [ux, uy] = roll(uses, 10); t2.push({x:ux, y:uy, type:'scatter', mode:'lines', line:{color:'#f0c040', width:1.5, dash:'dot'}, name:'use decisions / game', yaxis:'y2'}); }
    Plotly.react('c-pow2', t2, darkLayout(Object.assign({yaxis:{title:'points', gridcolor:'#21262d', color:'#7d8590'}, yaxis2:{title:'use decisions', overlaying:'y', side:'right', color:'#f0c040', showgrid:false, rangemode:'tozero'}, xaxis:{title:'training game (in order)', gridcolor:'#21262d', color:'#7d8590'}}, legend)), cfg);
  }
  document.getElementById('trainer-badge').textContent = 'Trainer: ' + (st.trainer_alive ? 'running' : 'stopped');
  document.getElementById('trainer-badge').className = 'badge badge-game' + (st.trainer_alive ? ' connected' : '');
  if (L) {
    document.getElementById('m-update').textContent = L.update; document.getElementById('m-steps').textContent = L.steps.toLocaleString();
    document.getElementById('m-segs').textContent = L.segs; document.getElementById('m-map').textContent = L.map.replace('_Hunt','');
    const G = pmSel || L;   // selected map's arrive / falls / speed, pooled when 'all maps'
    document.getElementById('g-falls').textContent = fmt(G.falls100);
    document.getElementById('g-speed').textContent = fmt(G.speed, 1); 
    document.getElementById('g-ent').textContent = fmt(L.ent); document.getElementById('g-dstd').textContent = 'dir std ' + fmt(L.dstd);

  }
  document.getElementById('m-dps').textContent = fmt(st.dec_per_s, 1); document.getElementById('m-rtf').textContent = st.rtf === null ? '--' : fmt(st.rtf);
  document.getElementById('m-ckpt').textContent = st.ckpt + ' (' + st.ckpt_time + ')';
  Plotly.react('c-falls', perMapTraces(s, 'falls100'), darkLayout(legend), cfg);
  Plotly.react('c-speed', perMapTraces(s, 'speed'), darkLayout(legend), cfg);
  Plotly.react('c-rew', perMapTraces(s, 'rew'), darkLayout(legend), cfg);
  Plotly.react('c-ent', [{x:s.update, y:s.ent, type:'scatter', mode:'lines', line:{color:'#bc8cff', width:1.5}, name:'entropy'},
                         {x:s.update, y:s.dstd, type:'scatter', mode:'lines', line:{color:'#58a6ff', width:1, dash:'dot'}, name:'dir std', yaxis:'y2'}],
               darkLayout(Object.assign({yaxis2:{overlaying:'y', side:'right', color:'#58a6ff', gridcolor:'#21262d'}}, legend)), cfg);
  Plotly.react('c-kl', [{x:s.update, y:s.kl, type:'scatter', mode:'lines', line:{color:'#f0c040', width:1.5}, name:'KL'},
                        {x:s.update, y:s.clip, type:'scatter', mode:'lines', line:{color:'#7d8590', width:1, dash:'dot'}, name:'clip frac', yaxis:'y2'}],
               darkLayout(Object.assign({yaxis2:{overlaying:'y', side:'right', color:'#7d8590', gridcolor:'#21262d', range:[0,1]}}, legend)), cfg);
  Plotly.react('c-loss', [{x:s.update, y:s.pl, type:'scatter', mode:'lines', line:{color:'#3fb950', width:1.5}, name:'policy loss'},
                          {x:s.update, y:s.vl, type:'scatter', mode:'lines', line:{color:'#f85149', width:1}, name:'value loss', yaxis:'y2'}],
               darkLayout(Object.assign({yaxis2:{overlaying:'y', side:'right', color:'#f85149', gridcolor:'#21262d', type:'log'}}, legend)), cfg);
  Plotly.react('c-gn', [{x:s.update, y:s.gn, type:'scatter', mode:'lines', line:{color:'#58a6ff', width:1.5}, name:'grad norm'}], darkLayout(), cfg);
  Plotly.react('c-wall', [{x:s.update, y:s.wall_s, type:'bar', marker:{color:'#7d8590'}, name:'wall s'}], darkLayout(), cfg);
  const tb = document.querySelector('#t-maps tbody'); tb.innerHTML = '';
  for (const [mp, v] of Object.entries(st.per_map)) {
    tb.innerHTML += `<tr><td>${mp.replace('_Hunt','')}</td><td class="num">${v.updates}</td><td class="num" style="color:#f0c040">${v.sgem ? v.sgem.toFixed(2) : '--'}</td><td class="num">${v.best_sgem ? v.best_sgem.toFixed(2) : '--'}</td><td class="num">${v.falls100.toFixed(2)}</td><td class="num" style="color:#f0c040">${v.best_falls100.toFixed(2)}</td><td class="num">${v.speed.toFixed(1)}</td><td class="num" style="color:#f0c040">${v.best_speed.toFixed(1)}</td></tr>`;
  }
  const te = document.querySelector('#t-events tbody'); te.innerHTML = '';
  st.events.slice().reverse().forEach(e => { te.innerHTML += `<tr><td class="num">upd ${e[0]}</td><td class="ev ${e[1]}">${e[2]}</td></tr>`; });
}
const source = new EventSource('/stream');
source.onmessage = (e) => { try { update(JSON.parse(e.data)); } catch (err) { console.error(err); showErr(err); } };
// 10-08 22:55: a script error used to leave the page silently blank; now it shows in the badge (and the console)
function showErr(err) { try { const b = document.getElementById('live-badge'); b.textContent = 'SCRIPT ERROR: ' + (err && err.message ? err.message : err) + ' @ ' + ((err && err.stack) ? err.stack.split(String.fromCharCode(10))[1] : ''); b.className = 'badge badge-reconnecting'; } catch (e2) {} }
window.onerror = (msg, src, line, col, err) => { showErr(err || new Error(msg + ' (line ' + line + ')')); };
source.onopen = () => { const b = document.getElementById('live-badge'); b.textContent = 'LIVE'; b.className = 'badge badge-live'; };
source.onerror = () => { const b = document.getElementById('live-badge'); b.textContent = 'RECONNECTING...'; b.className = 'badge badge-reconnecting'; };
