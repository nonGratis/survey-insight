"""generate_wave_editor.py — interactive wave annotation editor.

Генерує standalone HTML де:
- Кожна форма: cumulative curve + hourly rate bars
- Авто-детектовані хвилі показані як вертикальні лінії
- Поля wave_count, test_resp, notes — редаговані inline
- Кнопка "Export CSV" зберігає всі анотації

Запуск:
    .venv/Scripts/python.exe research/benchmarks/generate_wave_editor.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[2]

# ── user's partial ground truth ─────────────────────────────────────────────
GROUND_TRUTH = {
    "1GM-api8tg1DaVEE_NJ203b9K01TA4_uCW_3c2gkkh0c": {"waves_n": 1, "test_resp": 0, "notes": "одна агітація, природня ранкова хвиля наступного дня"},
    "15EZp01e49mqyyrZfncLNXe2fpbtoCh-muer_s568Eow":  {"waves_n": 6, "test_resp": 2, "notes": "довго відкрита, 98% після 4 днів, 6 агітаційних хвиль"},
    "1_zS99FyaYl2eNPl9pG_4KrbRjbRJxc10b3Z3JLfeBzc": {"waves_n": 7, "test_resp": 0, "notes": "близькі хвилі"},
    "1p0ERtAe-_c4J_EL0H-f3Ykbbc6GBbWmqY4-SX1r9I3Y": {"waves_n": 5, "test_resp": 0, "notes": "5 хвиль, деякі важко детектити"},
    "1ci4V9v25Ifn2qojemQvXNgZH4i0SmG5wtmfCn15sVy4":  {"waves_n": 8, "test_resp": 3, "notes": "8 хвиль, 3 тестові на початку"},
    "1IUps2ikeV37yMz9saxC7EZ-qlc5u9_XQSJhBXMoSHM8": {"waves_n": 4, "test_resp": 3, "notes": "4 хвилі, 3 тестові"},
    "1mTWRQ_TjLDkpPTRRFjZ0D5WUWo4F9js7k5-jyDv2YQA": {"waves_n": 2, "test_resp": 10, "notes": "2 хвилі, 10 тестових"},
    "1UB0wdRgRKeifVdqWOv8_uW4NExdLRxi7FZhp1e8Dfuw":  {"waves_n": 6, "test_resp": 0, "notes": "6 хвиль"},
    "1tBbO1LJWg0D8gcsjKKOFwYVXsFyKjmqZX9q5NcqZy-s": {"waves_n": 2, "test_resp": 0, "notes": "2 чисті хвилі"},
    "1OGdtkgxm8pQIxgzN4Rk5qwKhu6cHsc6XJAzL4Jtyb3E": {"waves_n": 2, "test_resp": 1, "notes": ""},
    "1vGlYyxrvRbui3bp9FM2iyk0-c4i_rJGitTRXK_hHExI":  {"waves_n": 2, "test_resp": 0, "notes": ""},
    "1q8Qla8GbBpvOecdeDrWPVFFKZjsZ2uoUu6tWs9c2e_U":  {"waves_n": 2, "test_resp": 0, "notes": ""},
    "1a1kthX0P5wN14oDlviTpXHNHTiLKvcMctctrulwoRpo":  {"waves_n": 3, "test_resp": 1, "notes": "3 хвилі"},
    "1Tw_K22VdkSmUGpDY9g4JD03or1of94y3-4hK_dkoH2c":  {"waves_n": 5, "test_resp": 1, "notes": ""},
    "1asTOD6MbE4oCX8gexU0lwrP6xVr7JaIsout-kx58Qs8": {"waves_n": 3, "test_resp": 0, "notes": ""},
    "160qXOXUowJAvCVmRzBOROT5QCxr_NQKb4J5rqYzVYPE": {"waves_n": 2, "test_resp": 0, "notes": ""},
    "1mfL4VNf6XIwylD78AtTDANju6xevE-Zc9ifZ_Z7qPYs":  {"waves_n": 4, "test_resp": 0, "notes": ""},
    "1gX_YqOPw7oRrPgEvaehaiBWuz-pyNKkvuI-ez7kamZw":   {"waves_n": None, "test_resp": 0, "notes": "exponential overall, many mini-log waves"},
    "1cJPzf6bMkNulhX16CPzt0yGQK8DUY2ZHkXSSMDT08fE":  {"waves_n": None, "test_resp": 0, "notes": "з 92 починається друга хвиля"},
    "1qOYJsx8rtZcImp4ODnJSfIgPCbTojLgM_UlsdnzhlP8":  {"waves_n": 4, "test_resp": 0, "notes": "вкрай складний, кілька відкриттів форми"},
    "1uBn4wyNG5LlIfdymzXe_rpx5-qMmuXrCRbdzl_Y9rMs":  {"waves_n": None, "test_resp": 0, "notes": "4+ хвилі, можливо закриття на час"},
    "13Lbv-OpFHnwMdkzlt0tO72wSqjCeWmFyuRIWJL5UJhw":  {"waves_n": None, "test_resp": 0, "notes": ""},
}

# ── wave auto-detector ──────────────────────────────────────────────────────
def auto_detect_waves(ts_list: list, test_skip: int = 0,
                      prominence_frac: float = 0.15,
                      min_dist_h: float = 0.5, smooth_w: int = 2) -> list[str]:
    """Return list of ISO-format timestamps where waves start."""
    ts = sorted(ts_list)[test_skip:]
    if len(ts) < 5:
        return [ts[0].isoformat()] if ts else []
    t0 = ts[0]
    span_h = (ts[-1] - t0).total_seconds() / 3600
    if span_h < 0.5:
        return [t0.isoformat()]
    slot_h = max(0.25, span_h / 60)
    nbins = max(5, int(span_h / slot_h))
    counts = np.zeros(nbins)
    for t in ts:
        idx = min(int((t - t0).total_seconds() / (slot_h * 3600)), nbins - 1)
        counts[idx] += 1
    if smooth_w > 1:
        kernel = np.ones(smooth_w) / smooth_w
        counts_s = np.convolve(counts, kernel, mode="same")
    else:
        counts_s = counts.copy()
    max_r = counts_s.max()
    if max_r == 0:
        return [t0.isoformat()]
    min_dist_slots = max(1, int(min_dist_h / slot_h))
    peaks, _ = find_peaks(counts_s, prominence=max_r * prominence_frac,
                          distance=min_dist_slots)
    if len(peaks) == 0:
        return [t0.isoformat()]
    slot_secs = int(slot_h * 3600)
    offsets = pd.date_range(t0, periods=nbins, freq=pd.Timedelta(seconds=slot_secs))
    return [offsets[p].isoformat() for p in peaks]


def build_chart_data(fid: str, ts_list: list, test_skip: int) -> dict:
    ts = sorted(ts_list)
    ts_clean = ts[test_skip:] if test_skip < len(ts) else ts
    t0 = ts_clean[0] if ts_clean else ts[0]
    span_h = (ts_clean[-1] - t0).total_seconds() / 3600 if len(ts_clean) > 1 else 1
    slot_h = max(0.25, span_h / 60)
    nbins = max(5, int(span_h / slot_h))
    slot_secs = int(slot_h * 3600)
    edges = pd.date_range(t0, periods=nbins + 1, freq=pd.Timedelta(seconds=slot_secs))
    counts = np.zeros(nbins)
    for t in ts_clean:
        idx = min(int((t - t0).total_seconds() / (slot_h * 3600)), nbins - 1)
        counts[idx] += 1
    cum_x = [str(t) for t in ts_clean]
    cum_y = list(range(1, len(ts_clean) + 1))
    rate_x = [str(edges[i]) for i in range(nbins)]
    rate_y = counts.tolist()
    if test_skip > 0:
        pre_x = [str(t) for t in ts[:test_skip]]
        pre_y = list(range(1, test_skip + 1))
    else:
        pre_x, pre_y = [], []
    return {"cum_x": cum_x, "cum_y": cum_y, "rate_x": rate_x, "rate_y": rate_y,
            "pre_x": pre_x, "pre_y": pre_y, "n": len(ts), "span_h": round(span_h, 1)}


def main() -> None:
    data_csv = REPO / "data" / "Form Timestamp Collection.csv"
    catalog_tsv = REPO / "data" / "Form Catalog.tsv"
    form_types_csv = REPO / "research" / "reports" / "figures" / "07_form_types.csv"
    shapes_csv = REPO / "research" / "reports" / "figures" / "01_per_form_features.csv"
    output_html = REPO / "research" / "reports" / "wave_editor.html"

    df = pd.read_csv(data_csv)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    catalog: dict[str, str] = {}
    if catalog_tsv.exists():
        cat_df = pd.read_csv(catalog_tsv, sep="\t", dtype=str).fillna("")
        for _, row in cat_df.iterrows():
            catalog[row["form_id"]] = row.get("form_title", "")[:60]
    ftmap: dict[str, str] = {}
    if form_types_csv.exists():
        ft = pd.read_csv(form_types_csv)
        ftmap = dict(zip(ft["form_id"], ft["form_type"]))
    shmap: dict[str, str] = {}
    if shapes_csv.exists():
        sh = pd.read_csv(shapes_csv)
        shmap = dict(zip(sh["form_id"], sh["shape"]))

    form_sizes = df.groupby("FORM_ID").size().sort_values(ascending=False)
    forms_data: list[dict] = []
    for fid, n in form_sizes.items():
        if n < 5:
            continue
        ts_list = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().tolist()
        gt = GROUND_TRUTH.get(fid, {})
        test_skip = int(gt.get("test_resp") or 0)
        waves_auto = auto_detect_waves(ts_list, test_skip=test_skip)
        chart = build_chart_data(fid, ts_list, test_skip)
        # Suspect test responses
        ia = np.diff([t.timestamp() for t in sorted(ts_list)])
        ia_pos = ia[ia > 0]
        suspect = bool(len(ia_pos) > 4 and ia_pos[0] > 20 * np.median(ia_pos[3:]) and ia_pos[0] > 3600)
        forms_data.append({
            "fid": fid, "title": catalog.get(fid, ""), "n": n,
            "form_type": ftmap.get(fid, "unknown"), "shape": shmap.get(fid, "unknown"),
            "span_h": chart["span_h"], "suspect_test": suspect,
            "waves_auto": waves_auto, "waves_auto_n": len(waves_auto),
            "waves_n": gt.get("waves_n"), "test_resp": test_skip,
            "notes": gt.get("notes", ""),
            "chart": chart,
        })

    # Embed as JSON, render with Plotly.js + vanilla JS editor
    forms_json = json.dumps(forms_data, default=str)

    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Wave Annotation Editor</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
#header { background: #1a1a2e; color: white; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
#header h1 { margin: 0; font-size: 16px; }
#controls { display: flex; gap: 10px; align-items: center; }
#controls select, #controls input { padding: 4px 8px; border-radius: 4px; border: none; font-size: 13px; }
#export-btn { background: #00b894; color: white; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: bold; }
#export-btn:hover { background: #00a381; }
#progress { color: #aaa; font-size: 12px; }
#container { display: grid; grid-template-columns: 1fr; gap: 0; }
.form-card { background: white; border-bottom: 1px solid #e0e0e0; padding: 12px 20px; }
.form-card.annotated { border-left: 4px solid #00b894; }
.form-card.has-gt { border-left: 4px solid #0984e3; }
.card-header { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
.form-meta { flex: 1; min-width: 200px; }
.form-id { font-family: monospace; font-size: 11px; color: #888; }
.form-title { font-size: 14px; font-weight: 600; color: #1a1a2e; }
.badges { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }
.badge { font-size: 10px; padding: 1px 6px; border-radius: 10px; background: #dfe6e9; color: #2d3436; }
.badge.suspect { background: #fab1a0; color: #d63031; }
.badge.has-gt { background: #74b9ff; color: #0984e3; }
.fields { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.field-group { display: flex; flex-direction: column; }
.field-group label { font-size: 11px; color: #636e72; margin-bottom: 2px; }
.field-group input, .field-group textarea {
  padding: 4px 8px; border: 1px solid #ddd; border-radius: 4px;
  font-size: 13px; background: #fafafa; transition: border 0.2s;
}
.field-group input:focus, .field-group textarea:focus {
  outline: none; border-color: #0984e3; background: white;
}
.field-group input.changed { border-color: #00b894; background: #f0fff8; }
.chart-container { margin-top: 6px; }
.auto-note { font-size: 11px; color: #636e72; margin-top: 4px; }
</style>
</head>
<body>
<div id="header">
  <h1>Wave Annotation Editor — Survey Insight</h1>
  <div id="controls">
    <span id="progress"></span>
    <select id="filter">
      <option value="all">All forms</option>
      <option value="unannotated">Unannotated only</option>
      <option value="suspect">Suspect test resp.</option>
      <option value="annotated">Annotated</option>
    </select>
    <input type="text" id="search" placeholder="Search title / ID..." style="width:180px">
    <button id="export-btn" onclick="exportCSV()">⬇ Export CSV</button>
  </div>
</div>
<div id="container"></div>

<script>
const FORMS = """ + forms_json + r""";

// State: per-form edits
const state = {};
FORMS.forEach(f => {
  state[f.fid] = {
    waves_n: f.waves_n !== null ? f.waves_n : f.waves_auto_n,
    test_resp: f.test_resp,
    notes: f.notes,
    changed: f.waves_n !== null,
  };
});

function updateProgress() {
  const done = Object.values(state).filter(s => s.changed).length;
  document.getElementById('progress').textContent = `${done}/${FORMS.length} annotated`;
}

function renderCard(f, idx) {
  const s = state[f.fid];
  const isGT = f.waves_n !== null;
  const cls = s.changed ? (isGT ? 'form-card has-gt' : 'form-card annotated') : 'form-card';
  const suspBadge = f.suspect_test ? '<span class="badge suspect">⚠ тест?</span>' : '';
  const gtBadge = isGT ? '<span class="badge has-gt">✓ GT</span>' : '';

  return `
  <div class="${cls}" id="card-${idx}" data-fid="${f.fid}" data-title="${f.title.toLowerCase()}" data-type="${f.form_type}">
    <div class="card-header">
      <div class="form-meta">
        <div class="form-id">${f.fid.substring(0,24)}…</div>
        <div class="form-title">${f.title || '(без назви)'}</div>
        <div class="badges">
          <span class="badge">${f.form_type}</span>
          <span class="badge">${f.shape}</span>
          <span class="badge">N=${f.n}</span>
          <span class="badge">${f.span_h}h</span>
          ${suspBadge}${gtBadge}
        </div>
      </div>
      <div class="fields">
        <div class="field-group">
          <label>Хвиль (waves_n)</label>
          <input type="number" min="0" max="50" value="${s.waves_n ?? ''}" style="width:70px"
            oninput="onWavesChange('${f.fid}', ${idx}, this.value)" class="${s.changed ? 'changed' : ''}">
        </div>
        <div class="field-group">
          <label>Тестових resp</label>
          <input type="number" min="0" max="20" value="${s.test_resp}" style="width:60px"
            oninput="onTestChange('${f.fid}', ${idx}, this.value)">
        </div>
        <div class="field-group">
          <label>Notes</label>
          <input type="text" value="${(s.notes||'').replace(/"/g,'&quot;')}" style="width:300px"
            oninput="onNotesChange('${f.fid}', this.value)">
        </div>
      </div>
    </div>
    <div class="auto-note">Авто: ${f.waves_auto_n} хвиль детектовано</div>
    <div class="chart-container" id="chart-${idx}"></div>
  </div>`;
}

function drawChart(f, idx) {
  const c = f.chart;
  const autoWaves = f.waves_auto.map(w => new Date(w));
  const shapes = autoWaves.map(w => ({
    type: 'line', xref: 'x', yref: 'paper',
    x0: w, x1: w, y0: 0, y1: 1,
    line: { color: '#e17055', width: 1.5, dash: 'dash' }
  }));
  const traces = [
    { x: c.cum_x, y: c.cum_y, type: 'scatter', mode: 'lines', name: 'cumulative',
      line: { color: '#0984e3', width: 2 }, xaxis: 'x', yaxis: 'y' },
    { x: c.rate_x, y: c.rate_y, type: 'bar', name: 'rate/slot',
      marker: { color: '#fdcb6e', opacity: 0.8 }, xaxis: 'x2', yaxis: 'y2' },
  ];
  if (c.pre_x.length) {
    traces.push({ x: c.pre_x, y: c.pre_y, type: 'scatter', mode: 'markers',
      name: 'тестові', marker: { color: '#d63031', size: 6, symbol: 'x' },
      xaxis: 'x', yaxis: 'y' });
  }
  const layout = {
    height: 200, margin: { l: 40, r: 10, t: 10, b: 30 },
    showlegend: false,
    grid: { rows: 1, columns: 2, subplots: [['xy','x2y2']] },
    shapes: shapes,
    xaxis: { domain: [0, 0.58] },
    xaxis2: { domain: [0.62, 1] },
    yaxis: { title: 'cum' },
    yaxis2: { title: 'rate' },
  };
  Plotly.newPlot(`chart-${idx}`, traces, layout, { displayModeBar: false, responsive: true });
}

function onWavesChange(fid, idx, val) {
  state[fid].waves_n = val === '' ? null : parseInt(val);
  state[fid].changed = true;
  const card = document.getElementById(`card-${idx}`);
  card.classList.remove('form-card');
  card.classList.add('form-card', 'annotated');
  const inp = card.querySelector('input[type="number"]');
  inp.classList.add('changed');
  updateProgress();
}
function onTestChange(fid, idx, val) {
  state[fid].test_resp = val === '' ? 0 : parseInt(val);
  state[fid].changed = true;
  updateProgress();
}
function onNotesChange(fid, val) {
  state[fid].notes = val;
}

function exportCSV() {
  const header = 'form_id,form_title,span_h,shape,form_type,waves_n,test_resp,waves_auto_n,suspect_test,notes\n';
  const rows = FORMS.map(f => {
    const s = state[f.fid];
    const esc = v => '"' + String(v || '').replace(/"/g, '""') + '"';
    return [f.fid, esc(f.title), f.span_h, f.shape, f.form_type,
            s.waves_n ?? '', s.test_resp, f.waves_auto_n,
            f.suspect_test ? 1 : 0, esc(s.notes)].join(',');
  }).join('\n');
  const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'wave_annotations.csv'; a.click();
}

function applyFilter() {
  const fv = document.getElementById('filter').value;
  const sv = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.form-card').forEach(card => {
    const fid = card.dataset.fid;
    const s = state[fid];
    const title = card.dataset.title || '';
    const type = card.dataset.type || '';
    let show = true;
    if (fv === 'unannotated') show = !s.changed;
    else if (fv === 'annotated') show = s.changed;
    else if (fv === 'suspect') show = card.querySelector('.badge.suspect') !== null;
    if (sv && !title.includes(sv) && !fid.toLowerCase().includes(sv) && !type.includes(sv)) show = false;
    card.style.display = show ? '' : 'none';
  });
}
document.getElementById('filter').addEventListener('change', applyFilter);
document.getElementById('search').addEventListener('input', applyFilter);

// Render all cards
const container = document.getElementById('container');
container.innerHTML = FORMS.map((f, i) => renderCard(f, i)).join('');

// Draw charts (with IntersectionObserver for lazy load)
const drawn = new Set();
const obs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting && !drawn.has(e.target.id)) {
      drawn.add(e.target.id);
      const idx = parseInt(e.target.id.replace('chart-', ''));
      drawChart(FORMS[idx], idx);
    }
  });
}, { rootMargin: '200px' });
FORMS.forEach((_, i) => {
  const el = document.getElementById(`chart-${i}`);
  if (el) obs.observe(el);
});

updateProgress();
</script>
</body>
</html>""".replace("const FORMS = ;", f"const FORMS = {forms_json};")

    output_html.write_text(html, encoding="utf-8")
    print(f"Editor written: {output_html}")
    print(f"Forms: {len(forms_data)}, with ground truth: {len(GROUND_TRUTH)}")


if __name__ == "__main__":
    main()
