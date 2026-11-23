"""report.html for possession.video: one self-contained page, no network needed.

The charts use the two fixed series colours (blue for the lighter kit, orange for
the darker), not the shirt colours: a white kit would vanish on a light page and
two similar kits would be hard to tell apart. The shirt colour is shown in the table.
"""
from __future__ import annotations

import json
from pathlib import Path

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --on-fill: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926;
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root { background: var(--page); color: var(--text-primary); min-height: 100vh;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 32px 16px 48px; }
.wrap { max-width: 960px; margin: 0 auto; }
h1 { font-size: 26px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 16px; font-weight: 600; margin: 0 0 2px; }
.sub { color: var(--text-secondary); margin: 0 0 24px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px; margin: 0 0 16px; }
.note { color: var(--text-secondary); font-size: 13px; margin: 0 0 14px; }
.warn { border-left: 3px solid var(--axis); padding: 8px 12px; margin: 0 0 16px; color: var(--text-secondary); }
.split { display: flex; gap: 2px; height: 28px; margin: 14px 0 10px; }
.split .seg { border-radius: 4px; color: var(--on-fill); font-weight: 600; font-size: 14px;
  display: flex; align-items: center; padding-left: 8px; white-space: nowrap; outline-offset: 2px; }
.split .seg:hover, .split .seg:focus { filter: brightness(1.08); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--text-primary); font-size: 14px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
svg text { fill: var(--text-muted); font-size: 12px; font-variant-numeric: tabular-nums; }
svg .lane-label { fill: var(--text-secondary); font-size: 13px; font-variant-numeric: normal; }
.chart { position: relative; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 13px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  display: none; white-space: nowrap; z-index: 2; }
.tip b { font-weight: 600; }
.tip .row { display: flex; align-items: center; gap: 8px; color: var(--text-secondary); }
.tip .row b { color: var(--text-primary); }
.tip .key { width: 12px; height: 2px; border-radius: 1px; display: inline-block; }
.tip .when { color: var(--text-muted); margin-bottom: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 500; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.chip { width: 12px; height: 12px; border-radius: 50%; display: inline-block; vertical-align: -1px;
  box-shadow: 0 0 0 1px var(--axis); margin-right: 6px; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; }
details table { margin-top: 8px; }
.scroll { max-height: 320px; overflow: auto; }
.method p { margin: 0 0 8px; color: var(--text-secondary); font-size: 14px; }
.method ul { margin: 0; padding-left: 18px; color: var(--text-secondary); font-size: 14px; }
a { color: inherit; }
</style>
</head>
<body>
<div class="viz-root">
<div class="wrap">
  <h1>Who has the ball?</h1>
  <p class="sub" id="sub"></p>
  <div class="warn" id="warn" hidden></div>

  <section class="card">
    <h2>Possession</h2>
    <p class="note">Share of the time either team had the ball. Time with nobody on the ball is left out. Chart colours mark the teams; the shirt colours are in the table below.</p>
    <div class="split" id="split" role="img"></div>
    <div class="legend" id="legend"></div>
  </section>

  <section class="card">
    <h2>When each team had the ball</h2>
    <p class="note">Each block is one spell of possession. Gaps are moments with nobody on the ball, or the ball out of view.</p>
    <div class="chart" id="timeline" role="img"></div>
    <details><summary>Table view: every spell</summary><div class="scroll"><table id="spells-table"></table></div></details>
  </section>

  <section class="card" id="blocks-card" hidden>
    <h2>Possession over the match</h2>
    <p class="note" id="blocks-note"></p>
    <div class="legend" id="legend2" style="margin-bottom:8px"></div>
    <div class="chart" id="blocks" role="img"></div>
    <details><summary>Table view</summary><table id="blocks-table"></table></details>
  </section>

  <section class="card">
    <h2>Numbers</h2>
    <table id="stats"></table>
  </section>

  <section class="card method">
    <h2>How this was measured</h2>
    <p>Five times a second, a detector found the players and the ball in the picture. Players were split into two teams by shirt colour; the team whose nearest player was closest to the ball, measured in player heights, got that moment, and a hidden Markov model smoothed the sequence so a single frame cannot flip possession.</p>
    <ul id="quality"></ul>
    <p style="margin-top:8px">Limits: distances are measured in the picture, so a ball passing in front of a player can count as near him. Goalkeepers and referees are never given the ball; while a keeper holds it, possession stays with the last team seen on it. Replays and close-ups count as time with nobody on the ball only when the grass leaves the picture.</p>
  </section>
</div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const SERIES = ['var(--series-1)', 'var(--series-2)'];
  const STATES = ['team_a', 'team_b'];
  const T = D.teams;
  const t0 = D.spells.length ? D.spells[0].start : 0;
  const pct = v => (v * 100).toFixed(0) + '%';
  const fine = D.duration_seconds < 600;   // tenths of a second only for short clips
  const clock = s => { s = Math.max(0, fine ? Math.round(s * 10) / 10 : Math.round(s));
    const m = Math.floor(s / 60), r = s - m * 60;
    return String(m).padStart(2, '0') + ':' + (r < 10 ? '0' : '') + (fine && r % 1 ? r.toFixed(1) : r.toFixed(0)); };
  const el = (tag, attrs, text) => { const e = document.createElement(tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; };
  const svgEl = (tag, attrs) => { const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]); return e; };

  // Header
  const mins = D.duration_seconds / 60;
  document.getElementById('sub').textContent = D.video + ' · ' +
    (mins >= 1 ? mins.toFixed(0) + ' min' : D.duration_seconds.toFixed(0) + ' s') +
    ' of video, looked at ' + D.rate_hz + ' times a second';
  if (D.warning) { const w = document.getElementById('warn'); w.hidden = false; w.textContent = D.warning; }

  // Tooltip helper: values lead, labels follow; line keys
  function tooltip(host) {
    const tip = el('div', {class: 'tip'}); host.appendChild(tip);
    return {
      show(x, y, when, rows, below) {
        tip.replaceChildren();
        if (when) tip.appendChild(el('div', {class: 'when'}, when));
        for (const r of rows) {
          const row = el('div', {class: 'row'});
          if (r.colour) { const k = el('i', {class: 'key'}); k.style.background = r.colour; row.appendChild(k); }
          row.appendChild(el('b', {}, r.value)); row.appendChild(document.createTextNode(r.label));
          tip.appendChild(row);
        }
        tip.style.display = 'block';
        const hw = host.clientWidth, tw = tip.offsetWidth;
        tip.style.left = Math.min(Math.max(0, x + 12), hw - tw) + 'px';
        tip.style.top = (below ? y + 8 : Math.max(0, y - tip.offsetHeight - 10)) + 'px';
      },
      hide() { tip.style.display = 'none'; }
    };
  }

  // 1. Tall segmented bar
  const split = document.getElementById('split');
  split.setAttribute('aria-label', 'Possession: ' + T.map(t => t.name + ' ' + pct(t.possession)).join(', '));
  const splitTip = tooltip(split.parentElement);
  split.parentElement.style.position = 'relative';
  T.forEach((t, k) => {
    if (t.possession <= 0) return;
    const seg = el('div', {class: 'seg', tabindex: '0',
      'aria-label': t.name + ': ' + pct(t.possession) + ', ' + clock(t.seconds_on_ball) + ' on the ball'});
    seg.style.flex = String(t.possession); seg.style.background = SERIES[k];
    const label = el('span', {}, pct(t.possession)); seg.appendChild(label);
    split.appendChild(seg);
    const show = () => { const r = seg.getBoundingClientRect(), h = split.parentElement.getBoundingClientRect();
      splitTip.show(r.left - h.left, r.top - h.top, null,
        [{value: pct(t.possession), label: t.name + ', ' + clock(t.seconds_on_ball) + ' on the ball', colour: SERIES[k]}]); };
    seg.addEventListener('pointerenter', show); seg.addEventListener('focus', show);
    seg.addEventListener('pointerleave', splitTip.hide); seg.addEventListener('blur', splitTip.hide);
  });
  requestAnimationFrame(() => split.querySelectorAll('.seg').forEach(seg => {
    const span = seg.firstChild; if (span.offsetWidth + 16 > seg.clientWidth) span.remove(); }));
  for (const id of ['legend', 'legend2']) {
    const lg = document.getElementById(id);
    T.forEach((t, k) => { const s = el('span'); const i = el('i'); i.style.background = SERIES[k];
      s.appendChild(i); s.appendChild(document.createTextNode(t.name)); lg.appendChild(s); });
  }

  // Ticks on a time axis
  function ticks(a, b, maxN) {
    const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    const span = b - a, step = steps.find(s => span / s <= maxN) || 3600;
    const out = []; for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(v);
    return out;
  }

  // 2. Timeline: one lane per team
  const tl = document.getElementById('timeline');
  const tlTip = tooltip(tl);
  function drawTimeline() {
    tl.querySelectorAll('svg').forEach(s => s.remove());
    const W = tl.clientWidth, labelW = Math.min(150, Math.max(90, W * 0.18)), right = 8;
    const laneH = 18, laneGap = 12, top = 6, axisH = 24;
    const H = top + 2 * laneH + laneGap + axisH + 6;
    const a = t0, b = t0 + D.duration_seconds;
    const x = v => labelW + (v - a) / (b - a || 1) * (W - labelW - right);
    const svg = svgEl('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H});
    const axisY = top + 2 * laneH + laneGap + 6;
    for (const v of ticks(a, b, Math.max(3, Math.floor((W - labelW) / 90)))) {
      svg.appendChild(svgEl('line', {x1: x(v), x2: x(v), y1: top - 2, y2: axisY, stroke: 'var(--grid)', 'stroke-width': 1}));
      const anchor = x(v) > W - right - 24 ? 'end' : 'middle';
      const tx = svgEl('text', {x: anchor === 'end' ? W - right : x(v), y: axisY + 16, 'text-anchor': anchor});
      tx.textContent = clock(v); svg.appendChild(tx);
    }
    svg.appendChild(svgEl('line', {x1: labelW, x2: W - right, y1: axisY, y2: axisY, stroke: 'var(--axis)', 'stroke-width': 1}));
    T.forEach((t, k) => {
      const y = top + k * (laneH + laneGap);
      const lab = svgEl('text', {x: 0, y: y + laneH / 2 + 4, class: 'lane-label'});
      lab.textContent = t.name.length > 18 ? t.name.slice(0, 17) + '…' : t.name; svg.appendChild(lab);
      for (const s of D.spells) {
        if (s.state !== STATES[k]) continue;
        const x1 = x(s.start), w = Math.max(1, x(s.end) - x1);
        svg.appendChild(svgEl('rect', {x: x1, y: y, width: w, height: laneH, rx: Math.min(2, w / 2), fill: SERIES[k]}));
      }
    });
    const cross = svgEl('line', {y1: top - 2, y2: axisY, stroke: 'var(--text-muted)', 'stroke-width': 1, visibility: 'hidden'});
    svg.appendChild(cross);
    const hit = svgEl('rect', {x: labelW, y: 0, width: W - labelW - right, height: axisY, fill: 'transparent', tabindex: '0'});
    svg.appendChild(hit);
    const step = 1 / D.rate_hz;
    function at(v) {
      v = Math.round((v - a) / step) * step + a;
      const s = D.spells.find(s => v >= s.start && v < s.end) || D.spells[D.spells.length - 1];
      cross.setAttribute('x1', x(v)); cross.setAttribute('x2', x(v)); cross.setAttribute('visibility', 'visible');
      const k = STATES.indexOf(s.state);
      tlTip.show(x(v), axisY + 20, clock(v), k >= 0
        ? [{value: T[k].name, label: ' has the ball (' + s.seconds.toFixed(1) + ' s spell)', colour: SERIES[k]}]
        : [{value: 'Nobody', label: ' on the ball'}], true);
    }
    hit.addEventListener('pointermove', e => { const r = svg.getBoundingClientRect();
      at(a + (e.clientX - r.left - labelW) / (W - labelW - right) * (b - a)); });
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tlTip.hide(); });
    let kv = a;
    hit.addEventListener('focus', () => at(kv));
    hit.addEventListener('blur', () => { cross.setAttribute('visibility', 'hidden'); tlTip.hide(); });
    hit.addEventListener('keydown', e => {
      const jump = Math.max(step, (b - a) / 50);
      if (e.key === 'ArrowRight') { kv = Math.min(b - step, kv + jump); at(kv); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { kv = Math.max(a, kv - jump); at(kv); e.preventDefault(); }
    });
    tl.prepend(svg);
  }
  const held = T.map(t => t.spells + ' spells');
  tl.setAttribute('aria-label', 'Possession timeline: ' + T.map((t, k) => t.name + ' ' + held[k]).join(', '));
  const st = document.getElementById('spells-table');
  st.appendChild(el('tr')).append(el('th', {}, 'From'), el('th', {}, 'To'), el('th', {}, 'Ball'), el('th', {class: 'num'}, 'Seconds'));
  for (const s of D.spells) {
    const k = STATES.indexOf(s.state);
    st.appendChild(el('tr')).append(el('td', {}, clock(s.start)), el('td', {}, clock(s.end)),
      el('td', {}, k >= 0 ? T[k].name : 'Nobody'), el('td', {class: 'num'}, s.seconds.toFixed(1)));
  }

  // 3. Possession per block: 100% stacked columns (only when there are enough blocks)
  function roundTop(x0, top, w, bottom, r) {   // rounded data end on top, square at the baseline
    return 'M' + x0 + ',' + bottom + 'V' + (top + r) + 'Q' + x0 + ',' + top + ' ' + (x0 + r) + ',' + top +
      'H' + (x0 + w - r) + 'Q' + (x0 + w) + ',' + top + ' ' + (x0 + w) + ',' + (top + r) + 'V' + bottom + 'Z';
  }
  const blocks = D.blocks.filter(b => b.team_a !== null);
  const bc = document.getElementById('blocks');
  const bTip = tooltip(bc);
  function drawBlocks() {
    bc.querySelectorAll('svg').forEach(s => s.remove());
    const W = bc.clientWidth, left = 40, right = 8, top = 8, plotH = 160, axisH = 24, H = top + plotH + axisH;
    const svg = svgEl('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H});
    const y = v => top + (1 - v) * plotH;
    for (const v of [0, 0.5, 1]) {
      svg.appendChild(svgEl('line', {x1: left, x2: W - right, y1: y(v), y2: y(v),
        stroke: v === 0 ? 'var(--axis)' : 'var(--grid)', 'stroke-width': 1}));
      const tx = svgEl('text', {x: left - 6, y: y(v) + 4, 'text-anchor': 'end'}); tx.textContent = pct(v); svg.appendChild(tx);
    }
    const n = D.blocks.length, slot = (W - left - right) / n, bw = Math.min(24, slot * 0.6);
    const every = Math.ceil(n / Math.max(2, Math.floor((W - left) / 70)));
    D.blocks.forEach((blk, i) => {
      const cx = left + slot * (i + 0.5);
      if (i % every === 0) { const tx = svgEl('text', {x: cx, y: top + plotH + 16, 'text-anchor': 'middle'});
        tx.textContent = clock(blk.start); svg.appendChild(tx); }
      if (blk.team_a === null) return;
      const xa = cx - bw / 2, ya = y(blk.team_a), r = 4;
      const segs = [
        {k: 0, path: 'M' + xa + ',' + y(0) + 'V' + ya + 'H' + (xa + bw) + 'V' + y(0) + 'Z'},
      ];
      const yb = y(1), yb1 = Math.max(yb, ya - 2);
      if (blk.team_b > 0 && yb1 - yb > 0.5) {
        const rr = Math.min(r, (yb1 - yb) / 2);
        segs.push({k: 1, path: roundTop(xa, yb, bw, yb1, rr)});
      } else if (blk.team_a > 0) {
        const rr = Math.min(r, (y(0) - ya) / 2);
        segs[0].path = roundTop(xa, ya, bw, y(0), rr);
      }
      for (const s of segs) {
        if (s.k === 0 && blk.team_a <= 0) continue;
        const p = svgEl('path', {d: s.path, fill: SERIES[s.k], tabindex: '0'});
        const show = () => bTip.show(cx, s.k === 0 ? ya : yb, clock(blk.start) + '–' + clock(blk.end), [
          {value: pct(blk.team_a), label: ' ' + T[0].name, colour: SERIES[0]},
          {value: pct(blk.team_b), label: ' ' + T[1].name, colour: SERIES[1]}]);
        p.addEventListener('pointerenter', show); p.addEventListener('focus', show);
        p.addEventListener('pointerleave', bTip.hide); p.addEventListener('blur', bTip.hide);
        svg.appendChild(p);
      }
      const hitR = svgEl('rect', {x: cx - slot / 2, y: top, width: slot, height: plotH, fill: 'transparent'});
      hitR.addEventListener('pointerenter', () => bTip.show(cx, ya, clock(blk.start) + '–' + clock(blk.end), [
        {value: pct(blk.team_a), label: ' ' + T[0].name, colour: SERIES[0]},
        {value: pct(blk.team_b), label: ' ' + T[1].name, colour: SERIES[1]}]));
      hitR.addEventListener('pointerleave', bTip.hide);
      svg.insertBefore(hitR, svg.firstChild);
    });
    bc.prepend(svg);
  }
  if (blocks.length >= 4) {
    document.getElementById('blocks-card').hidden = false;
    const bs = D.block_seconds;
    document.getElementById('blocks-note').textContent = 'Possession split in each ' +
      (bs >= 60 ? (bs / 60) + '-minute' : bs + '-second') + ' block.';
    bc.setAttribute('aria-label', 'Possession by block of ' + bs + ' seconds; values in the table view');
    const bt = document.getElementById('blocks-table');
    bt.appendChild(el('tr')).append(el('th', {}, 'From'), el('th', {}, 'To'),
      el('th', {class: 'num'}, T[0].name), el('th', {class: 'num'}, T[1].name));
    for (const blk of D.blocks) bt.appendChild(el('tr')).append(el('td', {}, clock(blk.start)), el('td', {}, clock(blk.end)),
      el('td', {class: 'num'}, blk.team_a === null ? '–' : pct(blk.team_a)),
      el('td', {class: 'num'}, blk.team_b === null ? '–' : pct(blk.team_b)));
  }

  // 4. Numbers
  const tb = document.getElementById('stats');
  tb.appendChild(el('tr')).append(el('th', {}, 'Team'), el('th', {}, 'Shirts seen'), el('th', {class: 'num'}, 'Possession'),
    el('th', {class: 'num'}, 'On the ball'), el('th', {class: 'num'}, 'Spells'), el('th', {class: 'num'}, 'Longest spell'));
  T.forEach((t, k) => {
    const name = el('td'); const key = el('i', {class: 'chip'}); key.style.background = SERIES[k]; key.style.borderRadius = '3px';
    key.style.boxShadow = 'none'; name.append(key, document.createTextNode(t.name));
    const kit = el('td'); const chip = el('i', {class: 'chip'}); chip.style.background = t.colour; kit.append(chip, document.createTextNode(t.colour));
    tb.appendChild(el('tr')).append(name, kit, el('td', {class: 'num'}, pct(t.possession)),
      el('td', {class: 'num'}, clock(t.seconds_on_ball)), el('td', {class: 'num'}, String(t.spells)),
      el('td', {class: 'num'}, t.longest_spell_seconds.toFixed(1) + ' s'));
  });
  const q = document.getElementById('quality');
  [['Frames analysed', D.frames_analysed.toLocaleString()],
   ['Ball found by the detector', pct(D.ball_seen_share) + ' of frames'],
   ['Ball position filled in over short gaps', pct(D.ball_filled_share) + ' of frames'],
   ['No pitch in view (close-ups, crowd)', pct(D.no_pitch_view_share) + ' of frames'],
   ['Time with nobody on the ball', clock(D.seconds_no_possession)],
   ['Detector', D.detector]].forEach(([k, v]) => q.appendChild(el('li', {}, k + ': ' + v)));

  function draw() { drawTimeline(); if (blocks.length >= 4) drawBlocks(); }
  draw();
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
})();
</script>
</body>
</html>
"""


def write_report(path: Path, summary: dict) -> None:
    data = json.dumps(summary).replace("</", "<\\/")
    title = f"Who has the ball? {summary.get('video', '')}".replace("&", "&amp;").replace("<", "&lt;")
    Path(path).write_text(PAGE.replace("__TITLE__", title).replace("__DATA__", data), encoding="utf-8")
