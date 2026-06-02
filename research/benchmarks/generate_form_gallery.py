"""generate_form_gallery.py — HTML-галерея всіх форм для ручної розмітки.

Генерує один HTML-файл зі всіма ~177 формами у вигляді:
- cumulative curve (кількість відповідей з часом)
- intra-day rate bar chart (відповіді погодинно)
- inter-arrival histogram

Для кожної форми показано:
- ID форми (скорочений), назва, тип, shape
- N відповідей, span, перша відповідь
- Підозра на тестові відповіді (*)

Мета: ручна розмітка, скільки хвиль агітації, форма кривої, тестові відповіді тощо.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/generate_form_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[2]


def _suspect_test_responses(ts: list) -> bool:
    """Чи є підозра на тестові відповіді (ізольований ранній відгук)."""
    if len(ts) < 10:
        return False
    secs = [(t - ts[0]).total_seconds() for t in ts]
    gaps = np.diff(secs)
    if len(gaps) < 5:
        return False
    later_med = np.median(gaps[3:]) if len(gaps) > 3 else np.median(gaps)
    if later_med <= 0:
        return False
    return any(
        gaps[k] > 20 * later_med and gaps[k] > 3600
        for k in range(min(3, len(gaps)))
    )


def _build_form_card(fid: str, ts_list: list, title: str, form_type: str, shape: str) -> str:
    """Один HTML-блок з двома Plotly-графіками для форми."""
    ts = sorted(ts_list)
    n = len(ts)
    t0 = ts[0]
    span_h = (ts[-1] - t0).total_seconds() / 3600.0
    suspect = _suspect_test_responses(ts)

    # Cumulative curve
    cum_x = [t.isoformat() for t in ts]
    cum_y = list(range(1, n + 1))

    # Rate per 1-hour buckets
    edges = pd.date_range(t0, ts[-1] + pd.Timedelta(hours=1), freq="h")
    counts = pd.cut(pd.Series(ts), bins=edges).value_counts().sort_index()
    rate_x = [str(i.left) for i in counts.index]
    rate_y = counts.values.tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Cumulative", "Hourly rate"],
        column_widths=[0.6, 0.4],
    )
    fig.add_trace(
        go.Scatter(x=cum_x, y=cum_y, mode="lines+markers",
                   marker=dict(size=3), line=dict(color="#1f77b4")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=rate_x, y=rate_y, marker_color="#ff7f0e"),
        row=1, col=2,
    )
    fig.update_layout(
        height=260, margin=dict(l=40, r=20, t=30, b=40),
        showlegend=False,
    )
    plot_html = fig.to_html(include_plotlyjs=False, full_html=False,
                            config={"displayModeBar": False})

    suspect_badge = '<span style="background:#e74c3c;color:white;padding:2px 6px;border-radius:3px;font-size:11px">⚠ тест?</span>' if suspect else ""
    short_id = fid[:24] + "…"

    card = f"""
<div style="border:1px solid #ddd;border-radius:6px;padding:12px;margin-bottom:16px;background:#fafafa">
  <div style="font-size:13px;margin-bottom:4px">
    <code style="font-size:11px;color:#555">{short_id}</code> {suspect_badge}
    &nbsp;|&nbsp; <b>{title[:60]}</b>
    &nbsp;|&nbsp; type:<b>{form_type}</b>
    &nbsp;|&nbsp; shape:<b>{shape}</b>
    &nbsp;|&nbsp; N={n} &nbsp;|&nbsp; span={span_h:.0f}h
    &nbsp;|&nbsp; t0={t0.strftime('%Y-%m-%d %H:%M')}
  </div>
  <div style="margin-top:4px;font-size:11px;color:#888">
    <b>Мітки (заповни вручну):</b>
    хвиль=_____ &nbsp; форма=_____ &nbsp; коментар=_____
  </div>
  {plot_html}
</div>
"""
    return card


def main() -> None:
    data_csv = REPO / "data" / "Form Timestamp Collection.csv"
    catalog_tsv = REPO / "data" / "Form Catalog.tsv"
    form_types_csv = REPO / "research" / "reports" / "figures" / "07_form_types.csv"
    shapes_csv = REPO / "research" / "reports" / "figures" / "01_per_form_features.csv"
    output_html = REPO / "research" / "reports" / "form_gallery.html"

    df = pd.read_csv(data_csv)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    catalog = {}
    if catalog_tsv.exists():
        cat_df = pd.read_csv(catalog_tsv, sep="\t", dtype=str).fillna("")
        for _, row in cat_df.iterrows():
            catalog[row["form_id"]] = row.get("form_title", "")

    ftmap = {}
    if form_types_csv.exists():
        ft = pd.read_csv(form_types_csv)
        ftmap = dict(zip(ft["form_id"], ft["form_type"], strict=False))

    shmap = {}
    if shapes_csv.exists():
        sh = pd.read_csv(shapes_csv)
        shmap = dict(zip(sh["form_id"], sh["shape"], strict=False))

    # Sort by N descending so big forms come first
    form_sizes = df.groupby("FORM_ID").size().sort_values(ascending=False)

    cards = []
    for fid, _n in form_sizes.items():
        ts_list = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().tolist()
        title = catalog.get(fid, "")
        ftype = ftmap.get(fid, "unknown")
        shape = shmap.get(fid, "unknown")
        try:
            card = _build_form_card(fid, ts_list, title, ftype, shape)
            cards.append(card)
        except Exception as exc:
            cards.append(f"<div style='color:red'>ERROR {fid[:20]}: {exc}</div>")

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Form Gallery — ручна розмітка хвиль</title>
{plotly_cdn}
<style>
  body {{ font-family: sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 16px; }}
  h1 {{ font-size: 20px; }}
  p.legend {{ font-size: 13px; color: #555; background: #fff8e1;
              padding: 8px 12px; border-radius: 4px; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>Галерея форм — {len(cards)} форм, відсортовано за N</h1>
<p class="legend">
  <b>Як заповнювати:</b>
  хвиль = кількість агітаційних хвиль (1, 2, 3, …);
  форма = burst / log / flat / multi / unclear;
  коментар = тест?, ранкова хвиля?, front-load?, plateaued? тощо
  <br><b>⚠ тест?</b> = підозра на тестові відповіді (ізольований ранній відгук перед основним потоком)
</p>
{"".join(cards)}
</body>
</html>"""

    output_html.write_text(html, encoding="utf-8")
    print(f"Gallery written: {output_html}")
    print(f"Forms: {len(cards)}")


if __name__ == "__main__":
    main()
