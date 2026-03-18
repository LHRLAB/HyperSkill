"""
HyperSkill Training Monitor — real-time web dashboard.

Streams training metrics from outputs/train_log.jsonl via SSE.
Also serves a hypergraph skill visualization page.

Usage:
    python scripts/monitor.py          # opens at http://localhost:8888
    python scripts/monitor.py --port 8080
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

LOG_FILE      = Path("outputs/train_log.jsonl")
SKILL_LIB_DIR = Path("expr/skill_library")

app = FastAPI()


# ---------------------------------------------------------------------------
# Training dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HyperSkill Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; }

  nav { padding: 0 28px; border-bottom: 1px solid #1e2433;
        display: flex; align-items: center; gap: 0; }
  nav .brand { font-size: 1rem; font-weight: 600; padding: 14px 0; margin-right: 28px; }
  nav a { padding: 14px 16px; font-size: .82rem; color: #64748b; text-decoration: none;
          border-bottom: 2px solid transparent; }
  nav a:hover { color: #e2e8f0; }
  nav a.active { color: #60a5fa; border-bottom-color: #60a5fa; }
  #status { margin-left: auto; font-size: .72rem; padding: 3px 10px;
            border-radius: 999px; background: #1e2433; color: #94a3b8; }
  #status.live { background: #052e16; color: #4ade80; }

  main { padding: 20px 28px; }

  .overview { display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
              gap: 12px; margin-bottom: 22px; }
  .card { background: #1a1f2e; border: 1px solid #1e2433; border-radius: 10px;
          padding: 14px 16px; }
  .card .label { font-size: .68rem; color: #64748b; text-transform: uppercase;
                 letter-spacing: .07em; margin-bottom: 5px; }
  .card .value { font-size: 1.55rem; font-weight: 700; line-height: 1; }
  .card .sub   { font-size: .72rem; color: #64748b; margin-top: 4px;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card.highlight { border-color: #1d4ed8; }
  .card.highlight .value { color: #60a5fa; }

  /* Progress bar */
  .progress-wrap { margin-bottom: 20px; }
  .progress-label { font-size: .75rem; color: #64748b; margin-bottom: 6px; }
  .progress-bar { height: 6px; background: #1e2433; border-radius: 3px; overflow: hidden; }
  .progress-bar .fill { height: 100%; background: #3b82f6; border-radius: 3px;
                        transition: width .3s ease; }

  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
  .chart-box { background: #1a1f2e; border: 1px solid #1e2433; border-radius: 10px; padding: 16px; }
  .chart-box h2 { font-size: .8rem; color: #94a3b8; margin-bottom: 12px; }
  canvas { max-height: 220px; }

  .table-section { background: #1a1f2e; border: 1px solid #1e2433; border-radius: 10px;
                   padding: 16px; margin-bottom: 20px; }
  .table-section h2 { font-size: .8rem; color: #94a3b8; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: .8rem; }
  th { text-align: left; color: #475569; font-weight: 500; padding: 5px 10px 8px;
       border-bottom: 1px solid #1e2433; }
  th:not(:first-child) { text-align: right; }
  td { padding: 6px 10px; border-bottom: 1px solid #111827; }
  td:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: none; }
  .td-ds { color: #7dd3fc; font-weight: 500; }
  .td-good { color: #4ade80; }  .td-mid { color: #fbbf24; }  .td-bad { color: #f87171; }
  .active-row td { background: #0f172a; }
  .badge { display: inline-block; font-size: .62rem; padding: 1px 6px;
           border-radius: 999px; background: #1d4ed8; color: #bfdbfe; margin-left: 6px; }

  .log-section { background: #1a1f2e; border: 1px solid #1e2433; border-radius: 10px; padding: 16px; }
  .log-section h2 { font-size: .8rem; color: #94a3b8; margin-bottom: 10px; }
  #log { font-family: "Fira Mono", monospace; font-size: .73rem; line-height: 1.75;
         max-height: 220px; overflow-y: auto; color: #64748b; }
  #log .ts { color: #334155; } #log .ds { color: #7dd3fc; }
  #log .f1 { color: #4ade80; } #log .sk { color: #a78bfa; } #log .el { color: #475569; }
</style>
</head>
<body>
<nav>
  <span class="brand">⬡ HyperSkill</span>
  <a href="/" class="active">Training</a>
  <a href="/graph">Skill Graph</a>
  <span id="status">connecting…</span>
</nav>
<main>
  <div class="overview">
    <div class="card highlight">
      <div class="label">Active dataset</div>
      <div class="value" id="ov-active" style="font-size:1rem;padding-top:4px">—</div>
      <div class="sub" id="ov-active-epoch"></div>
    </div>
    <div class="card">
      <div class="label">Latest F1</div>
      <div class="value" id="ov-f1">—</div>
      <div class="sub" id="ov-f1-sub"></div>
    </div>
    <div class="card">
      <div class="label">Best F1</div>
      <div class="value" id="ov-best-f1">—</div>
      <div class="sub" id="ov-best-f1-sub"></div>
    </div>
    <div class="card">
      <div class="label">Skill nodes</div>
      <div class="value" id="ov-skills">—</div>
      <div class="sub" id="ov-skills-sub">latest dataset</div>
    </div>
    <div class="card">
      <div class="label">Datasets trained</div>
      <div class="value" id="ov-datasets">—</div>
      <div class="sub" id="ov-epochs-sub">— epochs total</div>
    </div>
    <div class="card">
      <div class="label">Batch elapsed</div>
      <div class="value" id="ov-elapsed" style="font-size:1.2rem;padding-top:4px">—</div>
      <div class="sub" id="ov-elapsed-sub"></div>
    </div>
  </div>

  <div class="progress-wrap" id="progress-wrap" style="display:none">
    <div class="progress-label" id="progress-label">Batch 0 / 0</div>
    <div class="progress-bar"><div class="fill" id="progress-fill" style="width:0%"></div></div>
  </div>

  <div class="charts">
    <div class="chart-box">
      <h2>F1 per epoch <span style="color:#475569;font-weight:400">(by dataset)</span></h2>
      <canvas id="chartF1"></canvas>
    </div>
    <div class="chart-box">
      <h2>Skill nodes per epoch <span style="color:#475569;font-weight:400">(by dataset)</span></h2>
      <canvas id="chartSkills"></canvas>
    </div>
  </div>

  <div class="table-section">
    <h2>Per-dataset summary</h2>
    <table>
      <thead>
        <tr><th>Dataset</th><th>Epochs</th><th>Best F1</th><th>Latest F1</th><th>Skill nodes</th><th>Last epoch</th></tr>
      </thead>
      <tbody id="ds-table-body">
        <tr><td colspan="6" style="color:#334155;text-align:center;padding:16px">No data yet</td></tr>
      </tbody>
    </table>
  </div>

  <div class="log-section">
    <h2>Event log</h2>
    <div id="log"></div>
  </div>
</main>

<script>
const COLORS = ['#60a5fa','#4ade80','#f59e0b','#f472b6','#a78bfa','#34d399',
                '#fb923c','#38bdf8','#e879f9','#a3e635','#facc15','#2dd4bf'];

const chartState = {};   // dsName -> { epochStep, color }
let colorIdx = 0;
const dsStats = {};      // dsName -> { epochs, bestF1, latestF1, skillNodes, lastElapsed }
let totalEpochs = 0, activeDs = null;

function getColor(ds) {
  if (!chartState[ds]) chartState[ds] = { epochStep: 0, color: COLORS[colorIdx++ % COLORS.length] };
  return chartState[ds].color;
}

const scatterOpts = (yLabel) => ({
  responsive: true, animation: false,
  plugins: {
    legend: { labels: { color: '#94a3b8', boxWidth: 10, font: { size: 10 },
      filter: item => !item.text.endsWith('·batch') } },
    tooltip: { callbacks: {
      title: items => items[0].dataset.label.replace('·batch',''),
      label: item => {
        const r = item.raw;
        const v = typeof r.y === 'number' ? r.y.toFixed(4) : r.y;
        return r.batch ? `batch ${r.batch}  →  ${v}` : `epoch ${Math.round(r.x)}  →  ${v}`;
      },
    }},
  },
  scales: {
    x: { type: 'linear', title: { display: true, text: 'epoch', color: '#475569', font: { size: 10 } },
         ticks: { color: '#475569', stepSize: 1, precision: 0 }, grid: { color: '#1a2030' } },
    y: { title: { display: true, text: yLabel, color: '#475569', font: { size: 10 } },
         ticks: { color: '#475569' }, grid: { color: '#1a2030' }, beginAtZero: true },
  },
});

const ctxF1 = document.getElementById('chartF1').getContext('2d');
const ctxSk = document.getElementById('chartSkills').getContext('2d');
const chartF1 = new Chart(ctxF1, { type: 'scatter', data: { datasets: [] }, options: scatterOpts('F1') });
const chartSk = new Chart(ctxSk, { type: 'scatter', data: { datasets: [] }, options: scatterOpts('skill nodes') });

function getOrAddDs(chart, ds, isBatch) {
  const key = isBatch ? ds + '·batch' : ds;
  let d = chart.data.datasets.find(d => d.label === key);
  if (!d) {
    const color = getColor(ds);
    d = isBatch
      ? { label: key, data: [], borderColor: color + '55', backgroundColor: color + '33',
          showLine: true, tension: 0, pointRadius: 2, borderWidth: 1,
          borderDash: [3, 3], pointStyle: 'circle' }
      : { label: ds, data: [], borderColor: color, backgroundColor: color + '33',
          showLine: true, tension: 0.3, pointRadius: 5, borderWidth: 2 };
    chart.data.datasets.push(d);
  }
  return d;
}

function fmtElapsed(s) {
  if (s == null) return '—';
  return s < 60 ? s.toFixed(0) + 's' : s < 3600 ? (s / 60).toFixed(1) + 'm' : (s / 3600).toFixed(1) + 'h';
}

function f1Class(v) { return v >= 0.6 ? 'td-good' : v >= 0.3 ? 'td-mid' : 'td-bad'; }

function handleBatchRecord(d) {
  const pct = d.total_batches > 0 ? (d.batch / d.total_batches * 100).toFixed(1) : 0;
  document.getElementById('progress-wrap').style.display = '';
  document.getElementById('progress-label').textContent =
    `${d.dataset}  epoch ${d.epoch}/${d.total_epochs}  —  batch ${d.batch} / ${d.total_batches}`;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('ov-active').textContent = d.dataset;
  document.getElementById('ov-active-epoch').textContent = `epoch ${d.epoch}/${d.total_epochs} · batch ${d.batch}/${d.total_batches}`;
  document.getElementById('ov-elapsed').textContent = fmtElapsed(d.elapsed);
  document.getElementById('ov-elapsed-sub').textContent = 'per batch';
  document.getElementById('ov-f1').textContent = d.avg_f1.toFixed(4);
  document.getElementById('ov-f1-sub').textContent = d.dataset + ' (batch)';
  document.getElementById('ov-skills').textContent = (d.skill_nodes || 0).toLocaleString();
  document.getElementById('ov-skills-sub').textContent = d.dataset;
  activeDs = d.dataset;

  // Add fractional point to F1 + Skill charts: x = (epoch-1) + batch/totalBatches
  const cs = chartState[d.dataset] || (getColor(d.dataset), chartState[d.dataset]);
  const xFrac = (d.epoch - 1) + (d.batch / (d.total_batches || 1));
  getOrAddDs(chartF1, d.dataset, true).data.push({ x: xFrac, y: d.avg_f1, batch: d.batch });
  chartF1.update('none');
  const sk = d.skill_nodes || 0;
  if (sk > 0) {
    getOrAddDs(chartSk, d.dataset, true).data.push({ x: xFrac, y: sk, batch: d.batch });
    chartSk.update('none');
  }

  const ts = new Date(d.time * 1000).toLocaleTimeString();
  const logEl = document.getElementById('log');
  logEl.innerHTML =
    `<div><span class="ts">[${ts}]</span> <span class="ds">${d.dataset}</span> ` +
    `b${d.batch}/${d.total_batches} e${d.epoch}/${d.total_epochs} · ` +
    `F1=<span class="f1">${d.avg_f1.toFixed(4)}</span> ` +
    `skills=<span class="sk">${(d.skill_nodes||0).toLocaleString()}</span> ` +
    `<span class="el">${fmtElapsed(d.elapsed)}</span></div>` + logEl.innerHTML;
}

function handleEpochRecord(d) {
  const ds = d.dataset, epoch = d.epoch, f1 = d.avg_f1, sk = d.skill_nodes || 0;
  if (!dsStats[ds]) dsStats[ds] = { epochs: 0, bestF1: 0, latestF1: 0, skillNodes: 0, lastElapsed: null };
  const stat = dsStats[ds];
  stat.epochs++; stat.latestF1 = f1; stat.bestF1 = Math.max(stat.bestF1, f1);
  stat.skillNodes = sk; stat.lastElapsed = d.elapsed;
  totalEpochs++;
  activeDs = ds;

  // Update charts — epoch points at integer x
  const cs = chartState[ds] || (getColor(ds), chartState[ds]);
  cs.epochStep++;
  getOrAddDs(chartF1, ds, false).data.push({ x: cs.epochStep, y: f1 }); chartF1.update();
  getOrAddDs(chartSk, ds, false).data.push({ x: cs.epochStep, y: sk }); chartSk.update();

  // Cards
  document.getElementById('ov-active').textContent = ds;
  document.getElementById('ov-active-epoch').textContent = `epoch ${epoch}/${d.total_epochs}`;
  document.getElementById('ov-f1').textContent = f1.toFixed(4);
  document.getElementById('ov-f1-sub').textContent = ds;
  let bestF1 = 0, bestDs = '';
  for (const [k, s] of Object.entries(dsStats)) if (s.bestF1 > bestF1) { bestF1 = s.bestF1; bestDs = k; }
  document.getElementById('ov-best-f1').textContent = bestF1.toFixed(4);
  document.getElementById('ov-best-f1-sub').textContent = bestDs;
  document.getElementById('ov-skills').textContent = sk.toLocaleString();
  document.getElementById('ov-skills-sub').textContent = ds;
  document.getElementById('ov-datasets').textContent = Object.keys(dsStats).length;
  document.getElementById('ov-epochs-sub').textContent = totalEpochs + ' epochs total';
  document.getElementById('ov-elapsed').textContent = fmtElapsed(d.elapsed);
  document.getElementById('ov-elapsed-sub').textContent = 'last epoch';

  // Hide batch progress bar when epoch completes
  document.getElementById('progress-wrap').style.display = 'none';

  rebuildTable();

  const ts = new Date(d.time * 1000).toLocaleTimeString();
  const logEl = document.getElementById('log');
  logEl.innerHTML =
    `<div><span class="ts">[${ts}]</span> <span class="ds">${ds}</span> ` +
    `✓ epoch ${epoch}/${d.total_epochs} · ` +
    `F1=<span class="f1">${f1.toFixed(4)}</span> ` +
    `skills=<span class="sk">${sk.toLocaleString()}</span> ` +
    `<span class="el">${fmtElapsed(d.elapsed)}</span></div>` + logEl.innerHTML;
}

function rebuildTable() {
  const tbody = document.getElementById('ds-table-body');
  const rows = Object.entries(dsStats).sort((a, b) => b[1].latestF1 - a[1].latestF1);
  if (!rows.length) { tbody.innerHTML = '<tr><td colspan="6" style="color:#334155;text-align:center;padding:16px">No data yet</td></tr>'; return; }
  const dot = ds => `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${getColor(ds)};margin-right:6px"></span>`;
  tbody.innerHTML = rows.map(([ds, s]) =>
    `<tr class="${ds === activeDs ? 'active-row' : ''}">
      <td class="td-ds">${dot(ds)}${ds}${ds === activeDs ? '<span class="badge">active</span>' : ''}</td>
      <td>${s.epochs}</td>
      <td class="${f1Class(s.bestF1)}">${s.bestF1.toFixed(4)}</td>
      <td class="${f1Class(s.latestF1)}">${s.latestF1.toFixed(4)}</td>
      <td>${s.skillNodes.toLocaleString()}</td>
      <td>${fmtElapsed(s.lastElapsed)}</td>
    </tr>`).join('');
}

function resetAll() {
  chartF1.data.datasets = []; chartSk.data.datasets = [];
  chartF1.update('none'); chartSk.update('none');
  Object.keys(chartState).forEach(k => delete chartState[k]);
  Object.keys(dsStats).forEach(k => delete dsStats[k]);
  colorIdx = 0; totalEpochs = 0; activeDs = null;
  document.getElementById('log').innerHTML = '';
  document.getElementById('ds-table-body').innerHTML = '<tr><td colspan="6" style="color:#334155;text-align:center;padding:16px">No data yet</td></tr>';
  document.getElementById('progress-wrap').style.display = 'none';
  ['ov-active','ov-f1','ov-f1-sub','ov-best-f1','ov-best-f1-sub','ov-skills','ov-datasets','ov-elapsed'].forEach(id => {
    const el = document.getElementById(id); if (el) el.textContent = '—';
  });
  document.getElementById('ov-epochs-sub').textContent = '— epochs total';
  document.getElementById('ov-active-epoch').textContent = '';
}

function connect() {
  const es = new EventSource('/stream');
  const statusEl = document.getElementById('status');
  es.addEventListener('open', () => { statusEl.textContent = 'live'; statusEl.className = 'live'; });
  es.addEventListener('record', e => {
    try {
      const d = JSON.parse(e.data);
      if (d.reset) { resetAll(); return; }
      if (d.phase === 'batch') handleBatchRecord(d);
      else handleEpochRecord(d);
    } catch {}
  });
  es.addEventListener('error', () => {
    statusEl.textContent = 'reconnecting…'; statusEl.className = '';
    es.close(); setTimeout(connect, 3000);
  });
}
connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Graph visualization HTML
# ---------------------------------------------------------------------------

GRAPH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HyperSkill — Skill Graph</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9/dist/vis-network.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/vis-network@9/dist/vis-network.min.css">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }

  nav { padding: 0 28px; border-bottom: 1px solid #1e2433; flex-shrink: 0;
        display: flex; align-items: center; gap: 0; }
  nav .brand { font-size: 1rem; font-weight: 600; padding: 14px 0; margin-right: 28px; }
  nav a { padding: 14px 16px; font-size: .82rem; color: #64748b; text-decoration: none;
          border-bottom: 2px solid transparent; }
  nav a:hover { color: #e2e8f0; }
  nav a.active { color: #60a5fa; border-bottom-color: #60a5fa; }

  .toolbar { padding: 12px 28px; border-bottom: 1px solid #1e2433; flex-shrink: 0;
             display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .toolbar label { font-size: .8rem; color: #94a3b8; }
  select { background: #1a1f2e; border: 1px solid #334155; color: #e2e8f0;
           padding: 5px 10px; border-radius: 6px; font-size: .82rem; }
  button { background: #1d4ed8; border: none; color: #e2e8f0; padding: 5px 14px;
           border-radius: 6px; font-size: .82rem; cursor: pointer; }
  button:hover { background: #2563eb; }
  .toolbar .sep { width: 1px; height: 22px; background: #1e2433; }
  .info { font-size: .78rem; color: #64748b; }
  #node-count { color: #94a3b8; }

  .main { display: flex; flex: 1; overflow: hidden; }
  #graph-container { flex: 1; background: #0d1117; }

  .sidebar { width: 280px; border-left: 1px solid #1e2433; overflow-y: auto;
             padding: 16px; flex-shrink: 0; }
  .sidebar h3 { font-size: .82rem; color: #64748b; text-transform: uppercase;
                letter-spacing: .06em; margin-bottom: 14px; }
  .detail-empty { font-size: .8rem; color: #334155; text-align: center; margin-top: 40px; }
  .detail-field { margin-bottom: 12px; }
  .detail-field .key { font-size: .68rem; color: #475569; text-transform: uppercase;
                       letter-spacing: .06em; margin-bottom: 3px; }
  .detail-field .val { font-size: .82rem; line-height: 1.5; word-break: break-word; }
  .tag { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: .72rem;
         margin: 2px 3px 2px 0; }
  .tag-skill { background: #1e3a5f; color: #93c5fd; }
  .tag-hyperedge { background: #2e1065; color: #c4b5fd; }
  .adv-bar-wrap { height: 6px; background: #1e2433; border-radius: 3px; overflow: hidden; margin-top: 4px; }
  .adv-bar { height: 100%; border-radius: 3px; }

  .legend { padding: 12px 28px; border-top: 1px solid #1e2433; flex-shrink: 0;
            display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: .75rem; color: #64748b; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .diamond { width: 10px; height: 10px; background: #a78bfa; transform: rotate(45deg); }
</style>
</head>
<body>
<nav>
  <span class="brand">⬡ HyperSkill</span>
  <a href="/">Training</a>
  <a href="/graph" class="active">Skill Graph</a>
</nav>

<div class="toolbar">
  <label>Dataset</label>
  <select id="ds-select"><option value="">Loading…</option></select>
  <button onclick="loadGraph()">Load</button>
  <div class="sep"></div>
  <label>Max nodes</label>
  <select id="max-nodes">
    <option value="100">100</option>
    <option value="300" selected>300</option>
    <option value="500">500</option>
    <option value="1000">1000</option>
  </select>
  <div class="sep"></div>
  <button onclick="network && network.fit()">Fit view</button>
  <div class="sep"></div>
  <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
    <input type="checkbox" id="auto-refresh" checked style="accent-color:#3b82f6">
    Auto-refresh
  </label>
  <span class="info" id="node-count"></span>
  <span class="info" id="last-updated" style="color:#334155"></span>
</div>

<div class="main">
  <div id="graph-container"></div>
  <div class="sidebar">
    <h3>Node details</h3>
    <div id="detail"><p class="detail-empty">Click a node to see details</p></div>
  </div>
</div>

<div class="legend">
  <div class="legend-item"><div class="dot" style="background:#60a5fa"></div> Skill node (high advantage)</div>
  <div class="legend-item"><div class="dot" style="background:#f87171"></div> Skill node (low advantage)</div>
  <div class="legend-item"><div class="diamond"></div> Hyperedge</div>
  <div class="legend-item" style="margin-left:auto;font-size:.72rem;color:#334155">Node size ∝ degree · Color ∝ advantage</div>
</div>

<script>
let network = null;
let allNodes = [], allEdges = [];

// Load available datasets
async function loadDatasets() {
  const res = await fetch('/api/datasets');
  const datasets = await res.json();
  const sel = document.getElementById('ds-select');
  if (!datasets.length) { sel.innerHTML = '<option value="">No skill libraries found</option>'; return; }
  sel.innerHTML = datasets.map(d => `<option value="${d}">${d}</option>`).join('');
  loadGraph();
}

async function loadGraph() {
  const ds = document.getElementById('ds-select').value;
  const maxNodes = parseInt(document.getElementById('max-nodes').value);
  if (!ds) return;

  document.getElementById('node-count').textContent = 'Loading…';
  try {
    const res = await fetch(`/api/graph/${encodeURIComponent(ds)}?max_nodes=${maxNodes}`);
    const data = await res.json();
    allNodes = data.nodes;
    allEdges = data.edges;
    renderGraph(allNodes, allEdges);
    document.getElementById('node-count').textContent =
      `${data.nodes.length} nodes · ${data.edges.length} edges`;
    document.getElementById('last-updated').textContent =
      `· updated ${new Date().toLocaleTimeString()}`;
  } catch(e) {
    document.getElementById('node-count').textContent = 'Error loading graph';
  }
}

function advantageColor(adv) {
  // adv in [-1, 1]: negative = red, zero = grey, positive = blue->green
  if (adv > 0.5) return '#4ade80';
  if (adv > 0.2) return '#60a5fa';
  if (adv > 0)   return '#94a3b8';
  if (adv > -0.3) return '#fbbf24';
  return '#f87171';
}

function renderGraph(nodes, edges) {
  const container = document.getElementById('graph-container');

  const visNodes = nodes.map(n => {
    const isEdge = n.node_type === 'hyperedge';
    const adv = n.advantage || 0;
    const color = isEdge ? '#a78bfa' : advantageColor(adv);
    const size = isEdge ? 12 : Math.max(8, Math.min(24, 8 + (n.degree || 0) * 1.5));
    return {
      id: n.id,
      label: n.label,
      title: n.description || n.label,
      color: { background: color, border: color, highlight: { background: '#fff', border: color } },
      size,
      shape: isEdge ? 'diamond' : 'dot',
      font: { color: '#94a3b8', size: 10 },
      _data: n,
    };
  });

  const visEdges = edges.map(e => ({
    from: e.from, to: e.to,
    color: { color: '#1e2433', highlight: '#475569' },
    width: Math.max(1, Math.min(4, (e.weight || 0.1) * 3)),
    arrows: { to: { enabled: false } },
  }));

  const visData = { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) };
  const options = {
    physics: {
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -30, centralGravity: 0.003,
                          springLength: 80, springConstant: 0.04, damping: 0.6 },
      stabilization: { iterations: 150, fit: true },
    },
    interaction: { hover: true, tooltipDelay: 100 },
    nodes: { borderWidth: 1 },
    edges: { smooth: { type: 'continuous' } },
  };

  if (network) network.destroy();
  network = new vis.Network(container, visData, options);

  network.on('click', params => {
    if (!params.nodes.length) { showDetail(null); return; }
    const nodeId = params.nodes[0];
    const node = nodes.find(n => n.id === nodeId);
    showDetail(node);
  });
}

function showDetail(node) {
  const el = document.getElementById('detail');
  if (!node) { el.innerHTML = '<p class="detail-empty">Click a node to see details</p>'; return; }

  const adv = node.advantage || 0;
  const advPct = Math.round((adv + 1) / 2 * 100);
  const advColor = advantageColor(adv);
  const typeTag = node.node_type === 'hyperedge'
    ? '<span class="tag tag-hyperedge">hyperedge</span>'
    : '<span class="tag tag-skill">skill</span>';

  el.innerHTML = `
    <div class="detail-field"><div class="key">Type</div><div class="val">${typeTag}</div></div>
    <div class="detail-field"><div class="key">Name</div><div class="val">${node.label}</div></div>
    <div class="detail-field">
      <div class="key">Advantage <span style="color:${advColor}">${adv.toFixed(3)}</span></div>
      <div class="adv-bar-wrap"><div class="adv-bar" style="width:${advPct}%;background:${advColor}"></div></div>
    </div>
    ${node.description ? `<div class="detail-field"><div class="key">Description</div><div class="val">${node.description}</div></div>` : ''}
    ${node.modality ? `<div class="detail-field"><div class="key">Modality</div><div class="val">${node.modality}</div></div>` : ''}
    ${node.source ? `<div class="detail-field"><div class="key">Source</div><div class="val">${node.source}</div></div>` : ''}
    ${node.examples && node.examples.length ? `<div class="detail-field"><div class="key">Examples</div><div class="val">${node.examples.map(e => `<div style="margin-bottom:4px;color:#94a3b8">· ${e}</div>`).join('')}</div></div>` : ''}
    <div class="detail-field"><div class="key">Degree</div><div class="val">${node.degree || 0}</div></div>
    <div class="detail-field"><div class="key">Node ID</div><div class="val" style="font-family:monospace;font-size:.72rem;color:#475569">${node.id}</div></div>
  `;
}

// Auto-refresh: reload graph data and dataset list periodically
let _refreshTimer = null;

function scheduleRefresh() {
  if (_refreshTimer) clearTimeout(_refreshTimer);
  if (!document.getElementById('auto-refresh').checked) return;
  _refreshTimer = setTimeout(async () => {
    // Re-fetch dataset list in case a new library appeared
    const res = await fetch('/api/datasets');
    const datasets = await res.json();
    const sel = document.getElementById('ds-select');
    const cur = sel.value;
    sel.innerHTML = datasets.length
      ? datasets.map(d => `<option value="${d}"${d===cur?' selected':''}>${d}</option>`).join('')
      : '<option value="">No skill libraries found</option>';
    // Reload graph silently if a dataset is selected
    if (sel.value) await loadGraph(true);
    scheduleRefresh();
  }, 30000);  // every 30 s
}

document.getElementById('auto-refresh').addEventListener('change', () => {
  if (document.getElementById('auto-refresh').checked) scheduleRefresh();
  else { clearTimeout(_refreshTimer); _refreshTimer = null; }
});

const _origLoadGraph = loadGraph;
loadGraph = async function(silent = false) {
  await _origLoadGraph();
  if (!silent) scheduleRefresh();
};

loadDatasets();
scheduleRefresh();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
async def list_datasets():
    """List available skill library datasets (including ones being trained)."""
    if not SKILL_LIB_DIR.exists():
        return []
    return sorted(d.name for d in SKILL_LIB_DIR.iterdir() if d.is_dir())


@app.get("/api/graph/{dataset_name}")
async def get_graph(dataset_name: str, max_nodes: int = 300):
    """
    Return vis.js-compatible node/edge data for the skill hypergraph.
    Nodes are sorted by degree (most connected first) and capped at max_nodes.
    """
    import networkx as nx

    lib_dir = SKILL_LIB_DIR / dataset_name
    graphml_path = lib_dir / "hypergraph.graphml"
    kv_path      = lib_dir / "kv_skills.json"

    if not graphml_path.exists():
        return {"nodes": [], "edges": [], "error": "Graph not found"}

    # Load graph
    G = nx.read_graphml(str(graphml_path))

    # Load skill metadata
    kv_skills = {}
    if kv_path.exists():
        with open(kv_path, encoding="utf-8") as f:
            kv_skills = json.load(f)

    # Sort nodes by degree, keep top max_nodes
    degree_map = dict(G.degree())
    top_nodes = sorted(degree_map, key=lambda n: -degree_map[n])[:max_nodes]
    top_set = set(top_nodes)

    nodes = []
    for node_id in top_nodes:
        attrs = dict(G.nodes[node_id])
        kv = kv_skills.get(node_id, {})
        name = kv.get("name") or attrs.get("name") or node_id[:20]
        nodes.append({
            "id":        node_id,
            "label":     name[:30],
            "node_type": attrs.get("node_type", "skill"),
            "advantage": float(attrs.get("advantage", 0.0)),
            "modality":  attrs.get("modality", ""),
            "source":    attrs.get("source", ""),
            "degree":    degree_map[node_id],
            "description": kv.get("description", attrs.get("description", "")),
            "examples":  kv.get("examples", []),
        })

    edges = []
    for src, dst, edata in G.edges(data=True):
        if src in top_set and dst in top_set:
            edges.append({
                "from":   src,
                "to":     dst,
                "weight": float(edata.get("weight", 0.1)),
            })

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

async def _event_stream():
    """Tail outputs/train_log.jsonl and stream new lines as SSE events."""
    offset = 0
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                yield f"event: record\ndata: {line}\n\n"
                await asyncio.sleep(0)
        offset = LOG_FILE.stat().st_size

    while True:
        await asyncio.sleep(1)
        if not LOG_FILE.exists():
            continue
        size = LOG_FILE.stat().st_size
        if size < offset:
            offset = 0
        if size <= offset:
            continue
        with open(LOG_FILE, encoding="utf-8") as f:
            f.seek(offset)
            new = f.read()
        offset = size
        for line in new.splitlines():
            line = line.strip()
            if line:
                yield f"event: record\ndata: {line}\n\n"


@app.get("/stream")
async def stream():
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history")
async def history():
    if not LOG_FILE.exists():
        return []
    records = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


@app.get("/graph", response_class=HTMLResponse)
async def graph_page():
    return GRAPH_HTML


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperSkill Training Monitor")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"[Monitor] Dashboard  → http://localhost:{args.port}")
    print(f"[Monitor] Skill Graph → http://localhost:{args.port}/graph")
    print(f"[Monitor] Log file   → {LOG_FILE.resolve()}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
