"""generate_wave_editor_v2.py — click-to-annotate wave editor.

Інтерфейс:
  - Ліворуч: cumulative + rate chart
  - Клік на графік → додає хвилю (червона лінія)
  - Клік на червону лінію → видаляє її
  - Праворуч: список хвиль + поля test_resp / notes
  - Кнопка Export CSV

Запуск:
    .venv/Scripts/python.exe research/benchmarks/generate_wave_editor_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[2]

GROUND_TRUTH = {
    "1GM-api8tg1DaVEE_NJ203b9K01TA4_uCW_3c2gkkh0c": {
        "waves_n": 1,
        "test_resp": 0,
        "notes": "одна агітація, природня ранкова хвиля наступного дня",
    },
    "15EZp01e49mqyyrZfncLNXe2fpbtoCh-muer_s568Eow": {
        "waves_n": 6,
        "test_resp": 2,
        "notes": "довго відкрита, 98% після 4 днів, 6 агітаційних хвиль",
    },
    "1_zS99FyaYl2eNPl9pG_4KrbRjbRJxc10b3Z3JLfeBzc": {
        "waves_n": 7,
        "test_resp": 0,
        "notes": "близькі хвилі",
    },
    "1p0ERtAe-_c4J_EL0H-f3Ykbbc6GBbWmqY4-SX1r9I3Y": {
        "waves_n": 5,
        "test_resp": 0,
        "notes": "5 хвиль, деякі важко детектити",
    },
    "1ci4V9v25Ifn2qojemQvXNgZH4i0SmG5wtmfCn15sVy4": {
        "waves_n": 8,
        "test_resp": 3,
        "notes": "8 хвиль, 3 тестові на початку",
    },
    "1IUps2ikeV37yMz9saxC7EZ-qlc5u9_XQSJhBXMoSHM8": {
        "waves_n": 4,
        "test_resp": 3,
        "notes": "4 хвилі, 3 тестові",
    },
    "1mTWRQ_TjLDkpPTRRFjZ0D5WUWo4F9js7k5-jyDv2YQA": {
        "waves_n": 2,
        "test_resp": 10,
        "notes": "2 хвилі, 10 тестових",
    },
    "1UB0wdRgRKeifVdqWOv8_uW4NExdLRxi7FZhp1e8Dfuw": {
        "waves_n": 6,
        "test_resp": 0,
        "notes": "6 хвиль",
    },
    "1tBbO1LJWg0D8gcsjKKOFwYVXsFyKjmqZX9q5NcqZy-s": {
        "waves_n": 2,
        "test_resp": 0,
        "notes": "2 чисті хвилі",
    },
    "1OGdtkgxm8pQIxgzN4Rk5qwKhu6cHsc6XJAzL4Jtyb3E": {"waves_n": 2, "test_resp": 1, "notes": ""},
    "1vGlYyxrvRbui3bp9FM2iyk0-c4i_rJGitTRXK_hHExI": {"waves_n": 2, "test_resp": 0, "notes": ""},
    "1q8Qla8GbBpvOecdeDrWPVFFKZjsZ2uoUu6tWs9c2e_U": {"waves_n": 2, "test_resp": 0, "notes": ""},
    "1a1kthX0P5wN14oDlviTpXHNHTiLKvcMctctrulwoRpo": {
        "waves_n": 3,
        "test_resp": 1,
        "notes": "3 хвилі",
    },
    "1Tw_K22VdkSmUGpDY9g4JD03or1of94y3-4hK_dkoH2c": {"waves_n": 5, "test_resp": 1, "notes": ""},
    "1asTOD6MbE4oCX8gexU0lwrP6xVr7JaIsout-kx58Qs8": {"waves_n": 3, "test_resp": 0, "notes": ""},
    "160qXOXUowJAvCVmRzBOROT5QCxr_NQKb4J5rqYzVYPE": {"waves_n": 2, "test_resp": 0, "notes": ""},
    "1mfL4VNf6XIwylD78AtTDANju6xevE-Zc9ifZ_Z7qPYs": {"waves_n": 4, "test_resp": 0, "notes": ""},
    "1qOYJsx8rtZcImp4ODnJSfIgPCbTojLgM_UlsdnzhlP8": {
        "waves_n": 4,
        "test_resp": 0,
        "notes": "вкрай складний, кілька відкриттів форми",
    },
    "1gX_YqOPw7oRrPgEvaehaiBWuz-pyNKkvuI-ez7kamZw": {
        "waves_n": None,
        "test_resp": 0,
        "notes": "exponential overall, many mini-log waves",
    },
    "1cJPzf6bMkNulhX16CPzt0yGQK8DUY2ZHkXSSMDT08fE": {
        "waves_n": None,
        "test_resp": 0,
        "notes": "з 92 починається друга хвиля",
    },
    "1uBn4wyNG5LlIfdymzXe_rpx5-qMmuXrCRbdzl_Y9rMs": {
        "waves_n": None,
        "test_resp": 0,
        "notes": "4+ хвилі, можливо закриття на час",
    },
    "13Lbv-OpFHnwMdkzlt0tO72wSqjCeWmFyuRIWJL5UJhw": {"waves_n": None, "test_resp": 0, "notes": ""},
}


def auto_detect_waves(ts_list, test_skip=0, prominence_frac=0.15, min_dist_h=0.5, smooth_w=2):
    ts = sorted(ts_list)[test_skip:]
    if len(ts) < 5:
        return [ts[0].isoformat()] if ts else []
    t0 = ts[0]
    span_h = (ts[-1] - t0).total_seconds() / 3600
    if span_h < 0.5:
        return [t0.isoformat()]
    slot_h = max(0.25, span_h / 60)
    nbins = max(5, int(span_h / slot_h))
    slot_secs = int(slot_h * 3600)
    counts = np.zeros(nbins)
    for t in ts:
        idx = min(int((t - t0).total_seconds() / slot_secs), nbins - 1)
        counts[idx] += 1
    if smooth_w > 1:
        kernel = np.ones(smooth_w) / smooth_w
        cs = np.convolve(counts, kernel, mode="same")
    else:
        cs = counts.copy()
    max_r = cs.max()
    if max_r == 0:
        return [t0.isoformat()]
    min_dist_slots = max(1, int(min_dist_h / slot_h))
    peaks, _ = find_peaks(cs, prominence=max_r * prominence_frac, distance=min_dist_slots)
    if len(peaks) == 0:
        return [t0.isoformat()]
    offsets = pd.date_range(t0, periods=nbins, freq=pd.Timedelta(seconds=slot_secs))
    return [offsets[p].isoformat() for p in peaks]


def build_series(ts_list, test_skip=0):
    ts = sorted(ts_list)
    ts_clean = ts[test_skip:] if test_skip < len(ts) else ts
    if not ts_clean:
        return {}, []
    t0 = ts_clean[0]
    span_h = (ts_clean[-1] - t0).total_seconds() / 3600 if len(ts_clean) > 1 else 1.0
    slot_h = max(0.25, span_h / 80)
    slot_secs = max(60, int(slot_h * 3600))
    nbins = max(5, int(span_h * 3600 / slot_secs))

    # Rate bins
    rate_t = [t0 + pd.Timedelta(seconds=i * slot_secs) for i in range(nbins)]
    rate_y = [0] * nbins
    for t in ts_clean:
        idx = min(int((t - t0).total_seconds() / slot_secs), nbins - 1)
        rate_y[idx] += 1

    # Cumulative — use actual timestamps
    cum_x = [t.isoformat() for t in ts_clean]
    cum_y = list(range(1, len(ts_clean) + 1))

    # Pre-agitation (test responses)
    pre_x = [t.isoformat() for t in ts[:test_skip]]
    pre_y = list(range(1, test_skip + 1))

    return {
        "cum_x": cum_x,
        "cum_y": cum_y,
        "rate_x": [t.isoformat() for t in rate_t],
        "rate_y": rate_y,
        "pre_x": pre_x,
        "pre_y": pre_y,
        "t_min": t0.isoformat(),
        "t_max": ts_clean[-1].isoformat(),
        "span_h": round(span_h, 1),
    }, ts_clean


def suspect_test(ts_list):
    if len(ts_list) < 10:
        return False
    ia = np.diff([t.timestamp() for t in sorted(ts_list)])
    ia_pos = ia[ia > 0]
    if len(ia_pos) < 5:
        return False
    later = np.median(ia_pos[3:])
    return bool(later > 0 and ia_pos[0] > 20 * later and ia_pos[0] > 3600)


def main():
    data_csv = REPO / "data" / "Form Timestamp Collection.csv"
    catalog_tsv = REPO / "data" / "Form Catalog.tsv"
    form_types_csv = REPO / "research" / "reports" / "figures" / "07_form_types.csv"
    shapes_csv = REPO / "research" / "reports" / "figures" / "01_per_form_features.csv"
    output_html = REPO / "research" / "reports" / "wave_editor_v2.html"

    df = pd.read_csv(data_csv)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    catalog = {}
    if catalog_tsv.exists():
        cat_df = pd.read_csv(catalog_tsv, sep="\t", dtype=str).fillna("")
        for _, row in cat_df.iterrows():
            catalog[row["form_id"]] = row.get("form_title", "")[:55]
    ftmap = {}
    if form_types_csv.exists():
        ft = pd.read_csv(form_types_csv)
        ftmap = dict(zip(ft["form_id"], ft["form_type"], strict=False))
    shmap = {}
    if shapes_csv.exists():
        sh = pd.read_csv(shapes_csv)
        shmap = dict(zip(sh["form_id"], sh["shape"], strict=False))

    form_sizes = df.groupby("FORM_ID").size().sort_values(ascending=False)
    forms_data = []
    for fid, n in form_sizes.items():
        if n < 5:
            continue
        ts_list = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().tolist()
        gt = GROUND_TRUTH.get(fid, {})
        test_skip = int(gt.get("test_resp") or 0)
        waves_auto = auto_detect_waves(ts_list, test_skip=test_skip)
        series, _ = build_series(ts_list, test_skip)
        forms_data.append(
            {
                "fid": fid,
                "title": catalog.get(fid, ""),
                "n": n,
                "form_type": ftmap.get(fid, "unknown"),
                "shape": shmap.get(fid, "unknown"),
                "span_h": series.get("span_h", 0),
                "suspect_test": suspect_test(ts_list),
                "waves_init": waves_auto,  # авто-детектовані (можна редагувати)
                "waves_n_gt": gt.get("waves_n"),  # GT від користувача
                "test_resp": test_skip,
                "notes": gt.get("notes", ""),
                "series": series,
            }
        )

    payload = json.dumps(forms_data, default=str, ensure_ascii=False)

    html = _build_html(payload, len(forms_data))
    output_html.write_text(html, encoding="utf-8")
    print(f"Editor v2: {output_html}")
    print(f"Forms: {len(forms_data)}")


def _build_html(payload: str, n_forms: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Wave Editor v2</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root{{--accent:#e74c3c;--accent2:#0984e3;--bg:#f4f6f8;--card:#fff;--border:#e0e0e0;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',sans-serif;background:var(--bg);display:flex;flex-direction:column;height:100vh;overflow:hidden;}}

/* ── Top bar ── */
#topbar{{background:#1a1a2e;color:#fff;padding:8px 16px;display:flex;gap:12px;align-items:center;flex-shrink:0;}}
#topbar h1{{font-size:14px;font-weight:600;}}
.tb-spacer{{flex:1;}}
#progress{{font-size:12px;color:#aaa;}}
#topbar select,#topbar input[type=text]{{padding:3px 8px;border-radius:4px;border:none;font-size:12px;}}
#topbar input[type=text]{{width:160px;}}
#topbar button{{padding:4px 12px;border-radius:4px;border:none;cursor:pointer;font-size:12px;font-weight:600;}}
#btn-export{{background:#00b894;color:#fff;}}
#btn-export:hover{{background:#00997a;}}

/* ── Main layout ── */
#layout{{display:flex;flex:1;overflow:hidden;}}

/* ── Left: form list ── */
#list{{width:260px;flex-shrink:0;border-right:1px solid var(--border);overflow-y:auto;background:#fff;}}
.list-item{{padding:8px 12px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .15s;}}
.list-item:hover{{background:#f0f4ff;}}
.list-item.active{{background:#e8f0ff;border-left:3px solid var(--accent2);}}
.list-item.done{{border-left:3px solid #00b894;}}
.li-title{{font-size:12px;font-weight:600;color:#2d3436;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.li-meta{{font-size:10px;color:#888;margin-top:2px;}}
.li-badge{{display:inline-block;font-size:9px;padding:1px 5px;border-radius:8px;}}
.lb-suspect{{background:#fab1a0;color:#c0392b;}}
.lb-gt{{background:#74b9ff;color:#0984e3;}}
.lb-done{{background:#55efc4;color:#00805a;}}

/* ── Right: editor ── */
#editor{{flex:1;display:flex;overflow:hidden;}}
#chart-area{{flex:1;overflow:hidden;display:flex;flex-direction:column;padding:12px 8px;gap:4px;}}
#chart-cum{{height:65%;min-height:200px;}}
#chart-rate{{height:28%;min-height:80px;}}
#chart-hint{{font-size:11px;color:#888;padding:2px 0;}}

/* ── Wave panel ── */
#wave-panel{{width:270px;flex-shrink:0;border-left:1px solid var(--border);display:flex;flex-direction:column;background:#fafafa;}}
#wp-header{{padding:10px 14px;background:#2d3436;color:#fff;font-size:13px;font-weight:600;}}
#wp-meta{{padding:8px 14px;border-bottom:1px solid var(--border);font-size:11px;color:#636e72;line-height:1.6;}}
#wave-list{{flex:1;overflow-y:auto;padding:8px;}}
.wave-row{{display:flex;align-items:center;gap:6px;padding:5px 8px;border-radius:4px;background:#fff;border:1px solid var(--border);margin-bottom:4px;transition:background .15s;cursor:pointer;}}
.wave-row:hover{{background:#fff0f0;border-color:var(--accent);}}
.wave-row.selected{{background:#ffeaea;border-color:var(--accent);}}
.wr-num{{width:20px;font-size:11px;font-weight:700;color:#888;}}
.wr-time{{flex:1;font-size:11px;font-family:monospace;}}
.wr-del{{background:none;border:none;color:#ccc;cursor:pointer;font-size:14px;line-height:1;padding:0 2px;}}
.wr-del:hover{{color:var(--accent);}}
#wp-fields{{padding:10px 14px;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:8px;}}
.wf-row{{display:flex;flex-direction:column;gap:3px;}}
.wf-row label{{font-size:11px;color:#636e72;}}
.wf-row input,.wf-row textarea{{padding:5px 8px;border:1px solid var(--border);border-radius:4px;font-size:12px;background:#fff;resize:vertical;}}
.wf-row input:focus,.wf-row textarea:focus{{outline:none;border-color:var(--accent2);}}
#btn-save-form{{margin:0 14px 12px;padding:6px;background:var(--accent2);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px;font-weight:600;}}
#btn-save-form:hover{{background:#0773c5;}}
#empty-state{{flex:1;display:flex;align-items:center;justify-content:center;color:#aaa;font-size:13px;text-align:center;line-height:1.8;}}
</style>
</head>
<body>

<div id="topbar">
  <h1>Wave Annotation Editor v2</h1>
  <select id="flt-status">
    <option value="all">Всі</option>
    <option value="todo">Ще не оброблено</option>
    <option value="done">Оброблено</option>
    <option value="suspect">Підозра тест</option>
    <option value="gt">Є GT</option>
  </select>
  <select id="flt-type">
    <option value="">Всі типи</option>
    <option value="survey">survey</option>
    <option value="event_registration">event_reg</option>
    <option value="recruitment">recruitment</option>
    <option value="service">service</option>
    <option value="volunteer_donor">volunteer</option>
    <option value="unknown">unknown</option>
  </select>
  <input type="text" id="search" placeholder="Пошук...">
  <span class="tb-spacer"></span>
  <span id="progress"></span>
  <button id="btn-export" onclick="exportCSV()">⬇ Експорт CSV</button>
</div>

<div id="layout">
  <div id="list" id="form-list"></div>

  <div id="editor">
    <div id="chart-area">
      <div id="chart-hint">← Оберіть форму зі списку</div>
      <div id="chart-cum"></div>
      <div id="chart-rate"></div>
    </div>

    <div id="wave-panel">
      <div id="wp-header">Хвилі агітації</div>
      <div id="wp-meta" style="display:none"></div>
      <div id="wave-list"><div id="empty-state">Оберіть форму<br>і кликайте на графік<br>щоб додати хвилі</div></div>
      <div id="wp-fields" style="display:none">
        <div class="wf-row">
          <label>Тестові відповіді (зрізати з початку)</label>
          <input type="number" id="inp-test" min="0" max="30" value="0" onchange="onTestChange(this.value)">
        </div>
        <div class="wf-row">
          <label>Notes</label>
          <textarea id="inp-notes" rows="3" placeholder="коментар..."></textarea>
        </div>
      </div>
      <button id="btn-save-form" style="display:none" onclick="saveCurrentForm()">✓ Зберегти форму</button>
    </div>
  </div>
</div>

<script>
const FORMS = {payload};

// ── State ──────────────────────────────────────────────────────────────────
const state = {{}};
FORMS.forEach(f => {{
  const gt = f.waves_n_gt;
  state[f.fid] = {{
    waves: [...(f.waves_init || [])],   // ISO strings
    test_resp: f.test_resp || 0,
    notes: f.notes || '',
    done: gt !== null && gt !== undefined,
  }};
}});

let currentFid = null;
let cumChartReady = false;
let rateChartReady = false;
// Pre-built sorted ms arrays per fid for fast binary search (built on first open)
const cumMsCache = {{}};
// Track which trace index is the wave-dots trace
const WAVE_TRACE_IDX = 2; // after cumulative(0), pre-test(1 optional), wave-dots(2)

// ── List rendering ──────────────────────────────────────────────────────────
function buildList() {{
  const flt = document.getElementById('flt-status').value;
  const fltType = document.getElementById('flt-type').value;
  const search = document.getElementById('search').value.toLowerCase();
  const el = document.getElementById('list');
  el.innerHTML = '';
  FORMS.forEach(f => {{
    const s = state[f.fid];
    if (flt === 'todo' && s.done) return;
    if (flt === 'done' && !s.done) return;
    if (flt === 'suspect' && !f.suspect_test) return;
    if (flt === 'gt' && f.waves_n_gt === null) return;
    if (fltType && f.form_type !== fltType) return;
    if (search && !f.title.toLowerCase().includes(search) &&
        !f.fid.toLowerCase().includes(search) &&
        !f.form_type.toLowerCase().includes(search)) return;

    const div = document.createElement('div');
    div.className = 'list-item' + (s.done ? ' done' : '') + (f.fid === currentFid ? ' active' : '');
    div.dataset.fid = f.fid;
    const badges = [
      f.suspect_test ? '<span class="li-badge lb-suspect">⚠test</span>' : '',
      f.waves_n_gt !== null ? '<span class="li-badge lb-gt">GT</span>' : '',
      s.done ? '<span class="li-badge lb-done">✓</span>' : '',
    ].filter(Boolean).join(' ');
    div.innerHTML = `<div class="li-title">${{f.title || f.fid.substring(0,20)+'…'}}</div>
      <div class="li-meta">${{f.form_type}} · N=${{f.n}} · ${{f.span_h}}h · waves=${{s.waves.length}} ${{badges}}</div>`;
    div.addEventListener('click', () => openForm(f.fid));
    el.appendChild(div);
  }});
  updateProgress();
}}

function updateProgress() {{
  const done = Object.values(state).filter(s => s.done).length;
  document.getElementById('progress').textContent = `${{done}} / ${{FORMS.length}} оброблено`;
}}

// ── Open form ───────────────────────────────────────────────────────────────
function openForm(fid) {{
  currentFid = fid;
  chartRendered = false;
  buildList();  // update active
  const f = FORMS.find(x => x.fid === fid);
  if (!f) return;
  const s = state[fid];

  // Update meta
  const meta = document.getElementById('wp-meta');
  meta.style.display = '';
  meta.innerHTML = `<b>${{f.title || fid.substring(0,24)}}</b><br>
    Тип: ${{f.form_type}} · Shape: ${{f.shape}}<br>
    N=${{f.n}} · Span=${{f.span_h}}h<br>
    ${{f.suspect_test ? '⚠ Підозра на тестові' : ''}}
    ${{f.waves_n_gt !== null ? `✓ GT: ${{f.waves_n_gt}} хвиль` : ''}}`;

  document.getElementById('inp-test').value = s.test_resp;
  document.getElementById('inp-notes').value = s.notes;
  document.getElementById('wp-fields').style.display = '';
  document.getElementById('btn-save-form').style.display = '';
  document.getElementById('chart-hint').textContent =
    'Клік на графік → додати хвилю  |  Клік на ● хвилі → видалити';

  // Auto-set first wave to first clean response (after test skip)
  ensureFirstWave();
  drawChart(f, s);
  renderWaveList();
}}

// ── Chart ────────────────────────────────────────────────────────────────────
// drawChart: called once per form open (full render)
// fastUpdateWaves: called on every wave add/remove (shapes + dot trace only)

function waveShapes(waves) {{
  return waves.map((wt, i) => ({{
    type:'line', xref:'x', yref:'paper',
    x0: wt, x1: wt, y0: 0, y1: 1,
    line: {{color: '#e74c3c', width: 1.5, dash: 'solid'}},
  }}));
}}

function waveDotTrace(c, waves) {{
  // O(n log n): for each wave, binary-search cum_x for nearest y
  const cache = cumMsCache[currentFid];
  const wX = [], wY = [], wTxt = [];
  waves.forEach((wt, i) => {{
    const wtMs = new Date(wt).getTime();
    let lo = 0, hi = c.cum_x.length - 1;
    const ms = cache ? cache.ms : c.cum_x.map(x => new Date(x).getTime());
    while (lo < hi) {{ const mid=(lo+hi)>>1; if(ms[mid]<wtMs) lo=mid+1; else hi=mid; }}
    if (lo > 0 && Math.abs(ms[lo-1]-wtMs) < Math.abs(ms[lo]-wtMs)) lo--;
    wX.push(wt); wY.push(c.cum_y[lo]); wTxt.push(`W${{i+1}}`);
  }});
  return {{ x:wX, y:wY, type:'scatter', mode:'markers+text',
    text:wTxt, textposition:'top center', textfont:{{size:10,color:'#e74c3c'}},
    marker:{{color:'#e74c3c',size:10,symbol:'circle'}},
    name:'хвилі', hovertemplate:'<b>%{{text}}</b><br>%{{x}}<extra></extra>' }};
}}

function fastUpdateWaves() {{
  if (!currentFid) return;
  const f = FORMS.find(x => x.fid === currentFid);
  const s = state[currentFid];
  const c = f.series;
  // Only update shapes + wave-dot trace — no full redraw
  Plotly.relayout('chart-cum', {{shapes: waveShapes(s.waves)}});
  const dot = waveDotTrace(c, s.waves);
  Plotly.restyle('chart-cum', {{x:[dot.x], y:[dot.y], text:[dot.text]}}, [2]);
}}

function drawChart(f, s) {{
  const c = f.series;
  if (!c || !c.cum_x) return;

  // ── Cumulative chart (top) ─────────────────────────────────────────────
  const cumTraces = [
    {{ x: c.cum_x, y: c.cum_y, type:'scatter', mode:'lines',
       name:'cumulative', line:{{color:'#0984e3',width:2}} }},
  ];
  if (c.pre_x && c.pre_x.length) {{
    cumTraces.push({{ x:c.pre_x, y:c.pre_y, type:'scatter', mode:'markers',
      name:'тест', marker:{{color:'#d63031',size:7,symbol:'x'}} }});
  }}
  // Placeholder wave-dots trace (index 2, always present so restyle works)
  cumTraces.push(waveDotTrace(c, s.waves));

  const cumLayout = {{
    autosize:true, margin:{{l:45,r:10,t:12,b:35}},
    showlegend:false, shapes: waveShapes(s.waves),
    yaxis:{{title:'відп.'}},
    paper_bgcolor:'#fff', plot_bgcolor:'#fafcff',
    hovermode:'closest',
  }};
  const cumEl = document.getElementById('chart-cum');
  Plotly.react(cumEl, cumTraces, cumLayout, {{displayModeBar:false, responsive:true}})
    .then(() => {{ cumEl.on('plotly_click', onChartClick); }});

  // ── Rate chart (bottom) ────────────────────────────────────────────────
  // Add wave vertical lines as shapes on rate chart too for alignment
  const rateShapes = s.waves.map(wt => ({{
    type:'line', xref:'x', yref:'paper',
    x0:wt, x1:wt, y0:0, y1:1,
    line:{{color:'#e74c3c', width:1, dash:'dot'}},
  }}));
  const rateTraces = [
    {{ x:c.rate_x, y:c.rate_y, type:'bar', name:'rate',
       marker:{{color:'#fdcb6e', opacity:0.8}}, hovertemplate:'%{{y}}<extra></extra>' }},
  ];
  const rateLayout = {{
    autosize:true, margin:{{l:45,r:10,t:4,b:40}},
    showlegend:false, shapes: rateShapes,
    yaxis:{{title:'rate'}},
    paper_bgcolor:'#fff', plot_bgcolor:'#fafcff',
    hovermode:'closest',
  }};
  Plotly.react('chart-rate', rateTraces, rateLayout, {{displayModeBar:false, responsive:true}});
  cumChartReady = true;
}}

// ── Utilities ────────────────────────────────────────────────────────────────

// Binary search: find nearest cum_x string to a given ms timestamp
function snapToCumXFast(fid, clickMs) {{
  if (!cumMsCache[fid]) {{
    const f = FORMS.find(x => x.fid === fid);
    if (!f || !f.series.cum_x) return null;
    // Build sorted ms array once
    cumMsCache[fid] = {{ strs: f.series.cum_x, ms: f.series.cum_x.map(s => new Date(s).getTime()) }};
  }}
  const cache = cumMsCache[fid];
  const ms = cache.ms;
  if (!ms.length) return null;
  let lo = 0, hi = ms.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (ms[mid] < clickMs) lo = mid + 1; else hi = mid;
  }}
  // Compare lo and lo-1
  if (lo > 0 && Math.abs(ms[lo-1] - clickMs) < Math.abs(ms[lo] - clickMs)) lo--;
  return cache.strs[lo];
}}

function onChartClick(data) {{
  if (!currentFid || !data.points.length) return;
  const pt = data.points[0];
  if (!pt.x) return;
  const s = state[currentFid];

  // Clicked on wave-dot marker → remove
  if (pt.data?.name === 'хвилі' && pt.pointIndex !== undefined) {{
    s.waves.splice(pt.pointIndex, 1);
    s.waves.sort();
    renderWaveList();
    fastUpdateWaves();
    return;
  }}

  // Snap to nearest real cum_x timestamp (O(log n), avoids timezone drift)
  const rawMs = typeof pt.x === 'number' ? pt.x : new Date(pt.x).getTime();
  const snapped = snapToCumXFast(currentFid, rawMs);
  if (!snapped) return;

  if (!s.waves.includes(snapped)) {{
    s.waves.push(snapped);
    s.waves.sort();
  }}
  renderWaveList();
  fastUpdateWaves();
}}

// ── Wave list panel ──────────────────────────────────────────────────────────
function renderWaveList() {{
  if (!currentFid) return;
  const s = state[currentFid];
  const el = document.getElementById('wave-list');
  if (s.waves.length === 0) {{
    el.innerHTML = '<div style="padding:12px;font-size:12px;color:#aaa;text-align:center">Хвиль немає.<br>Кликай на графік.</div>';
    return;
  }}
  el.innerHTML = s.waves.map((wt, i) => {{
    // Display the stored string directly — no JS Date conversion to avoid timezone drift
    // "2021-11-19T19:32:37" → "2021-11-19 19:32"
    const display = wt.replace('T',' ').substring(0,16);
    const inputVal = wt.substring(0,16).replace(' ','T'); // for datetime-local
    return `<div class="wave-row" id="wr-${{i}}">
      <span class="wr-num" style="color:#e74c3c;font-weight:700">W${{i+1}}</span>
      <input type="datetime-local" class="wr-input" value="${{inputVal}}"
        style="flex:1;font-size:11px;font-family:monospace;padding:2px 4px;border:1px solid #ddd;border-radius:3px;background:#fff;"
        onchange="editWave(${{i}}, this.value)"
        onclick="event.stopPropagation()">
      <button class="wr-del" title="Видалити хвилю"
        onclick="event.stopPropagation(); removeWave(event,${{i}})">✕</button>
    </div>`;
  }}).join('');
}}

function editWave(idx, newVal) {{
  if (!currentFid) return;
  const s = state[currentFid];
  s.waves[idx] = newVal + ':00';
  s.waves.sort();
  renderWaveList();
  fastUpdateWaves();
}}

function removeWave(e, idx) {{
  e.stopPropagation();
  if (!currentFid) return;
  state[currentFid].waves.splice(idx, 1);
  renderWaveList();
  fastUpdateWaves();
}}

// Auto-set first wave to first real (non-test) response.
// Always replaces the first wave — auto-detected peak is the rate maximum,
// but the wave START is the first response. Count stays the same.
function ensureFirstWave() {{
  if (!currentFid) return;
  const f = FORMS.find(x => x.fid === currentFid);
  const s = state[currentFid];
  const c = f && f.series;
  if (!c || !c.cum_x || !c.cum_x.length) return;
  const firstReal = c.cum_x[0]; // first response after test skip
  if (!s.waves.includes(firstReal)) {{
    if (s.waves.length > 0) {{
      // Replace first wave with actual start (keep count the same)
      s.waves[0] = firstReal;
    }} else {{
      s.waves.push(firstReal);
    }}
    s.waves.sort();
  }}
}}

function onTestChange(val) {{
  if (!currentFid) return;
  const n = parseInt(val) || 0;
  state[currentFid].test_resp = n;
  // Rebuild cache (test_skip changed → cum_x changes in series)
  // We need the Python-built series to have pre_x/cum_x split correctly.
  // Since series is pre-built at page load, we approximate:
  // Find the (n+1)-th response overall and treat it as new start
  const f = FORMS.find(x => x.fid === currentFid);
  if (!f) return;
  // Clear cache so next snap uses updated data
  delete cumMsCache[currentFid];
  // Ensure first wave starts at right place
  ensureFirstWave();
  renderWaveList();
  fastUpdateWaves();
}}

function saveCurrentForm() {{
  if (!currentFid) return;
  const s = state[currentFid];
  s.notes = document.getElementById('inp-notes').value;
  s.test_resp = parseInt(document.getElementById('inp-test').value) || 0;
  s.done = true;
  buildList();
}}

// ── Export ──────────────────────────────────────────────────────────────────
function exportCSV() {{
  const hdr = 'form_id,form_title,span_h,shape,form_type,waves_n,test_resp,waves_iso,suspect_test,notes\\n';
  const rows = FORMS.map(f => {{
    const s = state[f.fid];
    const esc = v => '"' + String(v||'').replace(/"/g,'""') + '"';
    return [f.fid, esc(f.title), f.span_h, f.shape, f.form_type,
            s.waves.length, s.test_resp,
            esc(s.waves.join('|')),
            f.suspect_test ? 1 : 0,
            esc(s.notes)].join(',');
  }}).join('\\n');
  const blob = new Blob([hdr+rows],{{type:'text/csv;charset=utf-8;'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'wave_annotations.csv';
  a.click();
}}

// ── Init ────────────────────────────────────────────────────────────────────
document.getElementById('flt-status').addEventListener('change', buildList);
document.getElementById('flt-type').addEventListener('change', buildList);
document.getElementById('search').addEventListener('input', buildList);

buildList();

// Keyboard nav
document.addEventListener('keydown', e => {{
  if (e.key === 'Enter' && currentFid) {{ saveCurrentForm(); }}
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
