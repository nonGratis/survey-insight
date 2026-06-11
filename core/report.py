"""Генератор PDF-звітів — універсальний шар рендерингу (без доменних знань).

Призначення (SRP): перетворити структурований опис звіту (`Report` із блоків)
на байти PDF. Модуль НЕ знає нічого про зважування, прогноз чи крос-таби — він
лише рендерить заголовки, абзаци, сітки показників і таблиці. Доменні дані
перетворюють на `Report` окремі білдери (`core.reports`), а викликає рендер
відповідна сторінка UI — так само, як локальні CSV-кнопки (SoC).

DRY: єдиний рендер для всіх сторінок. Кирилиця — через вбудований шрифт
Liberation Sans (OFL), що додається в образ разом із кодом (assets/fonts),
тож працює і локально, і в slim-контейнері без системних шрифтів.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import PageBreak as _RLPageBreak

# --- шрифти (вбудовані, OFL Liberation Sans) --------------------------------
_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT = "LiberationSans"
FONT_BOLD = "LiberationSans-Bold"

# --- палітра/розміри (названі, не «магічні») --------------------------------
_HEADER_BG = colors.HexColor("#DCE6F1")
_GRID = colors.HexColor("#9AA7B8")
_MUTED = colors.HexColor("#666666")
_PAGE_MARGIN = 18 * mm
_HEADING_SIZE = {1: 16, 2: 13, 3: 11.5}


def _ensure_fonts() -> None:
    """Зареєструвати шрифти один раз (ідемпотентно)."""
    if FONT in pdfmetrics.getRegisteredFontNames():
        return
    pdfmetrics.registerFont(TTFont(FONT, str(_FONTS_DIR / "LiberationSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(_FONTS_DIR / "LiberationSans-Bold.ttf")))
    pdfmetrics.registerFontFamily(
        FONT, normal=FONT, bold=FONT_BOLD, italic=FONT, boldItalic=FONT_BOLD
    )


# --- модель контенту (доменно-нейтральна) -----------------------------------
@dataclass(frozen=True)
class Heading:
    """Заголовок рівня 1–3."""

    text: str
    level: int = 1


@dataclass(frozen=True)
class Para:
    """Абзац тексту (підтримує <b>…</b>)."""

    text: str


@dataclass(frozen=True)
class Metric:
    """Один показник для сітки BAN-ів."""

    label: str
    value: str


@dataclass(frozen=True)
class Metrics:
    """Сітка показників у `columns` стовпців."""

    items: Sequence[Metric]
    columns: int = 4


@dataclass(frozen=True)
class TableBlock:
    """Таблиця: шапка + рядки (усі значення вже відформатовані як рядки)."""

    headers: Sequence[str]
    rows: Sequence[Sequence[str]]
    col_widths: Sequence[float] | None = None  # частки 0..1; None = рівні


@dataclass(frozen=True)
class BarChart:
    """Горизонтальна стовпчаста діаграма: мітки → значення (векторна)."""

    labels: Sequence[str]
    values: Sequence[float]
    max_label_chars: int = 32


@dataclass(frozen=True)
class PageBreak:
    """Розрив сторінки (наступний блок — з нової сторінки)."""


@dataclass(frozen=True)
class Report:
    """Повний опис звіту для рендерингу."""

    title: str
    subtitle: str = ""
    blocks: list[object] = field(default_factory=list)
    footer: str = "Survey Insight"


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName=FONT, fontSize=10, leading=14, alignment=TA_LEFT)
    return {
        "body": base,
        "muted": ParagraphStyle("muted", parent=base, textColor=_MUTED, fontSize=9),
        "metric_label": ParagraphStyle("ml", parent=base, textColor=_MUTED, fontSize=8),
        "metric_value": ParagraphStyle("mv", fontName=FONT_BOLD, fontSize=13, leading=15),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11),
        "cell_h": ParagraphStyle("cellh", fontName=FONT_BOLD, fontSize=8.5, leading=11),
        **{
            f"h{lvl}": ParagraphStyle(
                f"h{lvl}",
                fontName=FONT_BOLD,
                fontSize=sz,
                leading=sz + 4,
                spaceBefore=8,
                spaceAfter=4,
            )
            for lvl, sz in _HEADING_SIZE.items()
        },
    }


def _content_width() -> float:
    return A4[0] - 2 * _PAGE_MARGIN


def _metrics_flowable(block: Metrics, styles: dict) -> Flowable:
    cols = max(1, block.columns)
    cells = [
        [Paragraph(m.label, styles["metric_label"]), Paragraph(m.value, styles["metric_value"])]
        for m in block.items
    ]
    # Пакуємо вертикальні (label/value) міні-таблиці в сітку cols×rows.
    minis = [
        Table([[lbl], [val]], style=TableStyle([("TOPPADDING", (0, 0), (-1, -1), 1)]))
        for lbl, val in cells
    ]
    rows = [minis[i : i + cols] for i in range(0, len(minis), cols)]
    for r in rows:  # доповнити останній рядок порожніми клітинками
        r += [""] * (cols - len(r))
    width = _content_width() / cols
    grid = Table(rows, colWidths=[width] * cols)
    grid.setStyle(
        TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)])
    )
    return grid


def _table_flowable(block: TableBlock, styles: dict) -> Flowable:
    header = [Paragraph(str(h), styles["cell_h"]) for h in block.headers]
    body = [[Paragraph(str(c), styles["cell"]) for c in row] for row in block.rows]
    ncols = len(block.headers)
    if block.col_widths:
        widths = [w * _content_width() for w in block.col_widths]
    else:
        widths = [_content_width() / ncols] * ncols
    table = Table([header, *body], colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, _GRID),
                ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


_BAR_COLOR = colors.HexColor("#5A78C0")
_BAR_ROW_H = 16  # висота на один стовпець, пт


def _barchart_flowable(block: BarChart) -> Flowable:
    """Горизонтальна діаграма засобами reportlab.graphics (без зовнішніх
    залежностей, векторно — друкується чітко на будь-якому масштабі)."""
    labels = [
        (str(s)[: block.max_label_chars] + "…") if len(str(s)) > block.max_label_chars else str(s)
        for s in block.labels
    ]
    values = [float(v) for v in block.values]
    n = max(len(values), 1)
    height = max(60.0, _BAR_ROW_H * n + 28)
    width = _content_width()
    drawing = Drawing(width, height)
    chart = HorizontalBarChart()
    chart.x, chart.y = 120, 14
    chart.width = width - 140
    chart.height = height - 24
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = FONT
    chart.categoryAxis.labels.fontSize = 7.5
    chart.valueAxis.labels.fontName = FONT
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = _BAR_COLOR
    chart.bars[0].strokeColor = None
    chart.barLabels.fontName = FONT
    chart.barLabels.fontSize = 7
    chart.barLabels.dx = 3
    chart.barLabelFormat = "%d"
    chart.barLabels.boxAnchor = "w"
    drawing.add(chart)
    return drawing


def _to_flowables(report: Report, styles: dict) -> list[Flowable]:
    flow: list[Flowable] = [Paragraph(report.title, styles["h1"])]
    if report.subtitle:
        flow.append(Paragraph(report.subtitle, styles["muted"]))
    flow.append(Spacer(1, 6))
    for block in report.blocks:
        if isinstance(block, Heading):
            flow.append(Paragraph(block.text, styles[f"h{min(max(block.level, 1), 3)}"]))
        elif isinstance(block, Para):
            flow.append(Paragraph(block.text, styles["body"]))
            flow.append(Spacer(1, 4))
        elif isinstance(block, Metrics):
            flow.append(_metrics_flowable(block, styles))
        elif isinstance(block, TableBlock):
            flow.append(_table_flowable(block, styles))
            flow.append(Spacer(1, 6))
        elif isinstance(block, BarChart):
            flow.append(_barchart_flowable(block))
            flow.append(Spacer(1, 6))
        elif isinstance(block, PageBreak):
            flow.append(_RLPageBreak())
        else:  # pragma: no cover - захист від невідомого блоку
            raise TypeError(f"невідомий блок звіту: {type(block).__name__}")
    return flow


def _footer_painter(footer_text: str):
    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")

    def paint(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(FONT, 8)
        canvas.setFillColor(_MUTED)
        canvas.drawString(_PAGE_MARGIN, 10 * mm, f"{footer_text} · {stamp}")
        canvas.drawRightString(A4[0] - _PAGE_MARGIN, 10 * mm, f"с. {doc.page}")
        canvas.restoreState()

    return paint


def render_pdf(report: Report) -> bytes:
    """Відрендерити `Report` у байти PDF (A4, кирилиця, нумерація сторінок)."""
    _ensure_fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=_PAGE_MARGIN,
        rightMargin=_PAGE_MARGIN,
        topMargin=_PAGE_MARGIN,
        bottomMargin=18 * mm,
        title=report.title,
    )
    painter = _footer_painter(report.footer)
    doc.build(_to_flowables(report, _styles()), onFirstPage=painter, onLaterPages=painter)
    return buf.getvalue()
