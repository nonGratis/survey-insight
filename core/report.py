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
from itertools import pairwise
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
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


@dataclass(frozen=True)
class ReportTheme:
    """Unified visual theme for all PDF report primitives."""

    primary: str = "#1d4ed8"
    primary_dark: str = "#1e3a8a"
    text: str = "#111827"
    muted: str = "#64748b"
    border: str = "#dbe4ef"
    grid: str = "#d8e2ef"
    surface: str = "#f8fafc"
    surface_alt: str = "#f1f5f9"
    title_bg: str = "#eff6ff"
    title_border: str = "#bfdbfe"
    section_bg: str = "#f8fafc"
    metric_bg: str = "#ffffff"
    metric_border: str = "#dbeafe"
    table_header_bg: str = "#1e3a8a"
    table_header_text: str = "#ffffff"
    table_row_alt: str = "#f8fafc"
    chart_bar: str = "#2563eb"
    chart_grid: str = "#e2e8f0"
    footer_text: str = "#64748b"
    footer_rule: str = "#e2e8f0"
    page_margin: float = 18 * mm
    frame_padding: float = 6


DEFAULT_REPORT_THEME = ReportTheme()

# --- шрифти (вбудовані, OFL Liberation Sans) --------------------------------
_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT = "LiberationSans"
FONT_BOLD = "LiberationSans-Bold"

# --- палітра/розміри (названі, не «магічні») --------------------------------
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
    value_labels: Sequence[str] | None = None
    max_label_chars: int = 32
    max_label_lines: int = 3


@dataclass(frozen=True)
class FlowChartNode:
    """Вузол компактної карти переходів."""

    id: str
    label: str
    kind: str = "section"
    fill_color: str = "#f8fafc"
    stroke_color: str = "#cbd5e1"
    font_color: str = "#334155"
    dashed: bool = False


@dataclass(frozen=True)
class FlowChartEdge:
    """Ребро компактної карти переходів."""

    source: str
    target: str
    label: str = ""
    kind: str = "default"
    color: str = "#cbd5e1"
    font_color: str = "#475569"
    pen_width: float = 1.1
    dashed: bool = True


@dataclass(frozen=True)
class FlowChart:
    """Компактна векторна карта переходів: вузли + ребра."""

    nodes: Sequence[FlowChartNode]
    edges: Sequence[FlowChartEdge]
    fit_page: bool = False


@dataclass(frozen=True)
class PageBreak:
    """Розрив сторінки (наступний блок — з нової сторінки)."""


class _ScaledDrawingFlowable(Flowable):
    """ReportLab Flowable that scales a graphics Drawing into the available frame."""

    def __init__(self, drawing: Drawing, max_width: float, max_height: float) -> None:
        super().__init__()
        self.drawing = drawing
        self.max_width = max_width
        self.max_height = max_height
        self.scale = 1.0
        self.width = drawing.width
        self.height = drawing.height

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        max_width = min(self.max_width, avail_width)
        max_height = min(self.max_height, avail_height)
        self.scale = min(
            1.0,
            max_width / max(self.drawing.width, 1.0),
            max_height / max(self.drawing.height, 1.0),
        )
        self.width = self.drawing.width * self.scale
        self.height = self.drawing.height * self.scale
        return self.width, self.height

    def draw(self) -> None:
        self.canv.saveState()
        self.canv.scale(self.scale, self.scale)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


@dataclass(frozen=True)
class Report:
    """Повний опис звіту для рендерингу."""

    title: str
    subtitle: str = ""
    blocks: list[object] = field(default_factory=list)
    footer: str = "Survey Insight"
    theme: ReportTheme = DEFAULT_REPORT_THEME


def _styles(theme: ReportTheme = DEFAULT_REPORT_THEME) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "body",
        fontName=FONT,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.HexColor(theme.text),
    )
    return {
        "body": base,
        "title": ParagraphStyle(
            "title",
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(theme.primary_dark),
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base,
            textColor=colors.HexColor(theme.muted),
            fontSize=9.5,
            leading=12,
        ),
        "muted": ParagraphStyle(
            "muted", parent=base, textColor=colors.HexColor(theme.muted), fontSize=9
        ),
        "metric_label": ParagraphStyle(
            "ml",
            parent=base,
            textColor=colors.HexColor(theme.muted),
            fontSize=7.8,
            leading=10,
        ),
        "metric_value": ParagraphStyle(
            "mv",
            fontName=FONT_BOLD,
            fontSize=14,
            leading=16,
            textColor=colors.HexColor(theme.primary_dark),
        ),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11),
        "cell_h": ParagraphStyle(
            "cellh",
            fontName=FONT_BOLD,
            fontSize=8.3,
            leading=10.5,
            textColor=colors.HexColor(theme.table_header_text),
        ),
        **{
            f"h{lvl}": ParagraphStyle(
                f"h{lvl}",
                fontName=FONT_BOLD,
                fontSize=sz,
                leading=sz + 4,
                spaceBefore=10 if lvl == 2 else 7,
                spaceAfter=6 if lvl == 2 else 4,
                textColor=colors.HexColor(theme.primary_dark if lvl <= 2 else theme.text),
            )
            for lvl, sz in _HEADING_SIZE.items()
        },
    }


def _content_width(theme: ReportTheme = DEFAULT_REPORT_THEME) -> float:
    return A4[0] - 2 * theme.page_margin - 2 * theme.frame_padding


def _content_height(theme: ReportTheme = DEFAULT_REPORT_THEME) -> float:
    return A4[1] - 2 * theme.page_margin - 2 * theme.frame_padding


def _metrics_flowable(
    block: Metrics, styles: dict, theme: ReportTheme = DEFAULT_REPORT_THEME
) -> Flowable:
    cols = max(1, block.columns)
    width = _content_width(theme) / cols
    cells = [
        [Paragraph(m.label, styles["metric_label"]), Paragraph(m.value, styles["metric_value"])]
        for m in block.items
    ]
    # Пакуємо вертикальні (label/value) міні-таблиці в сітку cols×rows.
    minis = [
        Table(
            [[lbl], [val]],
            colWidths=[width - 8],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(theme.metric_bg)),
                    ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor(theme.metric_border)),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.0, colors.HexColor(theme.primary)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, 0), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                    ("TOPPADDING", (0, 1), (-1, 1), 0),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
        for lbl, val in cells
    ]
    rows = [minis[i : i + cols] for i in range(0, len(minis), cols)]
    for r in rows:  # доповнити останній рядок порожніми клітинками
        r += [""] * (cols - len(r))
    grid = Table(rows, colWidths=[width] * cols)
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return grid


def _table_flowable(
    block: TableBlock, styles: dict, theme: ReportTheme = DEFAULT_REPORT_THEME
) -> Flowable:
    header = [Paragraph(str(h), styles["cell_h"]) for h in block.headers]
    body = [[Paragraph(str(c), styles["cell"]) for c in row] for row in block.rows]
    ncols = len(block.headers)
    if block.col_widths:
        widths = [w * _content_width(theme) for w in block.col_widths]
    else:
        widths = [_content_width(theme) / ncols] * ncols
    table = Table([header, *body], colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(theme.table_header_bg)),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor(theme.table_row_alt)],
                ),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(theme.primary_dark)),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor(theme.grid)),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor(theme.border)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("TOPPADDING", (0, 1), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
            ]
        )
    )
    return table


_BAR_ROW_H = 16  # висота на один стовпець, пт
_BAR_V_PAD = 28
_FLOW_NODE_W = 118
_FLOW_NODE_H = 38
_FLOW_X_GAP = 32
_FLOW_Y_GAP = 34
_FLOW_MARGIN = 8
_FLOW_PAGE_MAX_H_REDUCTION = 48


def _wrap_lines(value: str, max_chars: int, max_lines: int) -> list[str]:
    clean = " ".join(str(value).split())
    if len(clean) <= max_chars:
        return [clean]
    lines: list[str] = []
    current = ""
    for word in clean.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word[:max_chars]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(clean.split())) > len(" ".join(lines)):
        lines[-1] = lines[-1].rstrip("…") + "…"
    return lines[:max_lines] or [""]


def _flow_label_lines(value: str, max_chars: int = 24, max_lines: int = 6) -> list[str]:
    lines: list[str] = []
    for part in str(value).splitlines() or [""]:
        lines.extend(_wrap_lines(part, max_chars, 2))
        if len(lines) >= max_lines:
            break
    return lines[:max_lines] or [""]


def _barchart_drawing(block: BarChart, theme: ReportTheme = DEFAULT_REPORT_THEME) -> Flowable:
    """Горизонтальна діаграма засобами reportlab.graphics (без зовнішніх
    залежностей, векторно — друкується чітко на будь-якому масштабі)."""
    wrapped_labels = [
        _wrap_lines(str(label), block.max_label_chars, block.max_label_lines)
        for label in block.labels
    ]
    labels = ["\n".join(lines) for lines in wrapped_labels]
    values = [float(v) for v in block.values]
    n = max(len(values), 1)
    max_lines = max((len(lines) for lines in wrapped_labels), default=1)
    row_height = _BAR_ROW_H + max(0, max_lines - 1) * 8
    height = max(60.0, row_height * n + _BAR_V_PAD)
    width = _content_width(theme)
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
    chart.valueAxis.strokeColor = colors.HexColor(theme.grid)
    chart.valueAxis.labels.fillColor = colors.HexColor(theme.muted)
    chart.categoryAxis.strokeColor = colors.HexColor(theme.grid)
    chart.categoryAxis.labels.fillColor = colors.HexColor(theme.muted)
    chart.bars[0].fillColor = colors.HexColor(theme.chart_bar)
    chart.bars[0].strokeColor = None
    chart.barLabels.fontName = FONT
    chart.barLabels.fontSize = 7
    chart.barLabels.fillColor = colors.HexColor(theme.text)
    chart.barLabels.dx = 3
    if block.value_labels:
        chart.barLabelArray = [list(block.value_labels)]
        chart.barLabelFormat = "values"
    else:
        chart.barLabelFormat = "%.1f" if any(abs(v - round(v)) > 1e-9 for v in values) else "%d"
    chart.barLabels.boxAnchor = "w"
    drawing.add(chart)
    return drawing


def _barchart_flowables(
    block: BarChart, theme: ReportTheme = DEFAULT_REPORT_THEME
) -> list[Flowable]:
    """Split large charts so one ReportLab Drawing never exceeds a page frame."""
    wrapped_labels = [
        _wrap_lines(str(label), block.max_label_chars, block.max_label_lines)
        for label in block.labels
    ]
    max_lines = max((len(lines) for lines in wrapped_labels), default=1)
    row_height = _BAR_ROW_H + max(0, max_lines - 1) * 8
    max_items = max(1, int((_content_height(theme) - _BAR_V_PAD) // row_height))
    if len(block.values) <= max_items:
        return [_barchart_drawing(block, theme)]

    out: list[Flowable] = []
    labels = list(block.labels)
    values = list(block.values)
    value_labels = list(block.value_labels) if block.value_labels else None
    for start in range(0, len(values), max_items):
        end = start + max_items
        out.append(
            _barchart_drawing(
                BarChart(
                    labels=labels[start:end],
                    values=values[start:end],
                    value_labels=value_labels[start:end] if value_labels else None,
                    max_label_chars=block.max_label_chars,
                    max_label_lines=block.max_label_lines,
                ),
                theme,
            )
        )
    return out


def _draw_arrow(
    drawing: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: colors.Color,
    pen_width: float,
    dashed: bool,
) -> None:
    line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=pen_width)
    if dashed:
        line.strokeDashArray = [3, 2]
    drawing.add(line)
    # Невеликий наконечник стрілки. Працює і для діагоналей, достатньо для PDF-огляду.
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 4.2
    drawing.add(
        Polygon(
            [
                x2,
                y2,
                x2 - ux * size + px * size * 0.55,
                y2 - uy * size + py * size * 0.55,
                x2 - ux * size - px * size * 0.55,
                y2 - uy * size - py * size * 0.55,
            ],
            fillColor=color,
            strokeColor=color,
        )
    )


def _draw_polyline_arrow(
    drawing: Drawing,
    points: list[tuple[float, float]],
    color: colors.Color,
    pen_width: float,
    dashed: bool,
) -> None:
    if len(points) < 2:
        return
    for (x1, y1), (x2, y2) in pairwise(points):
        line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=pen_width)
        if dashed:
            line.strokeDashArray = [3, 2]
        drawing.add(line)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 4.2
    drawing.add(
        Polygon(
            [
                x2,
                y2,
                x2 - ux * size + px * size * 0.55,
                y2 - uy * size + py * size * 0.55,
                x2 - ux * size - px * size * 0.55,
                y2 - uy * size - py * size * 0.55,
            ],
            fillColor=color,
            strokeColor=color,
        )
    )


def _flowchart_page_flowable(
    block: FlowChart, theme: ReportTheme = DEFAULT_REPORT_THEME
) -> Flowable:
    nodes = list(block.nodes)
    content_width = _content_width(theme)
    if not nodes:
        return Drawing(content_width, 1)

    node_w = _FLOW_NODE_W
    x_gap = 46
    margin = 12
    lane_gap = 18
    label_lines_by_id = {
        node.id: _flow_label_lines(node.label, max_chars=24, max_lines=6) for node in nodes
    }
    node_heights = {
        node.id: max(_FLOW_NODE_H, 16 + 8.6 * len(label_lines_by_id[node.id])) for node in nodes
    }
    max_node_h = max(node_heights.values(), default=_FLOW_NODE_H)
    node_by_id = {node.id: node for node in nodes}
    order_by_id = {node.id: index for index, node in enumerate(nodes)}

    routed_edges: list[tuple[FlowChartEdge, str, int]] = []
    top_lanes = 0
    bottom_lanes = 0
    for edge in block.edges:
        if edge.source not in order_by_id or edge.target not in order_by_id:
            continue
        source_index = order_by_id[edge.source]
        target_index = order_by_id[edge.target]
        target_is_submit = node_by_id[edge.target].kind == "submit"
        is_adjacent_default = (
            edge.kind == "default" and target_index == source_index + 1 and not target_is_submit
        )
        if is_adjacent_default:
            routed_edges.append((edge, "straight", 0))
        elif target_is_submit or edge.dashed or target_index <= source_index:
            routed_edges.append((edge, "bottom", bottom_lanes))
            bottom_lanes += 1
        else:
            routed_edges.append((edge, "top", top_lanes))
            top_lanes += 1

    node_y = margin + bottom_lanes * lane_gap + (8 if bottom_lanes else 0)
    width = 2 * margin + len(nodes) * node_w + max(0, len(nodes) - 1) * x_gap
    height = node_y + max_node_h + top_lanes * lane_gap + margin + (20 if top_lanes else 0)
    drawing = Drawing(width, height)
    positions = {
        node.id: (margin + index * (node_w + x_gap), node_y, node_heights[node.id])
        for index, node in enumerate(nodes)
    }

    def draw_edge_label(x: float, y: float, label: str, color: str) -> None:
        if not label:
            return
        drawing.add(
            String(
                x,
                y,
                _wrap_lines(label, 18, 1)[0],
                fontName=FONT,
                fontSize=6.5,
                fillColor=colors.HexColor(color),
                textAnchor="middle",
            )
        )

    for edge, route, lane in routed_edges:
        sx, sy, sh = positions[edge.source]
        tx, ty, th = positions[edge.target]
        color = colors.HexColor(edge.color)
        if route == "straight":
            x1 = sx + node_w
            y1 = sy + sh / 2
            x2 = tx
            y2 = ty + th / 2
            _draw_arrow(drawing, x1, y1, x2, y2, color, edge.pen_width, edge.dashed)
            draw_edge_label((x1 + x2) / 2, (y1 + y2) / 2 + 4, edge.label, edge.font_color)
            continue

        if route == "top":
            lane_y = node_y + max_node_h + 16 + lane * lane_gap
            x1 = sx + node_w
            y1 = sy + sh / 2
            x2 = tx
            y2 = ty + th / 2
            left_bend = x1 + x_gap / 3
            right_bend = x2 - x_gap / 3
            _draw_polyline_arrow(
                drawing,
                [
                    (x1, y1),
                    (left_bend, y1),
                    (left_bend, lane_y),
                    (right_bend, lane_y),
                    (right_bend, y2),
                    (x2, y2),
                ],
                color,
                edge.pen_width,
                edge.dashed,
            )
            draw_edge_label((left_bend + right_bend) / 2, lane_y + 4, edge.label, edge.font_color)
            continue

        lane_y = margin + (bottom_lanes - lane - 0.5) * lane_gap
        x1 = sx + node_w / 2
        y1 = sy
        x2 = tx + node_w / 2
        y2 = ty
        _draw_polyline_arrow(
            drawing,
            [(x1, y1), (x1, lane_y), (x2, lane_y), (x2, y2)],
            color,
            edge.pen_width,
            edge.dashed,
        )
        draw_edge_label((x1 + x2) / 2, lane_y + 4, edge.label, edge.font_color)

    for node in nodes:
        x, y, node_height = positions[node.id]
        rect = Rect(
            x,
            y,
            node_w,
            node_height,
            rx=4,
            ry=4,
            fillColor=colors.HexColor(node.fill_color),
            strokeColor=colors.HexColor(node.stroke_color),
        )
        if node.dashed:
            rect.strokeDashArray = [3, 2]
        drawing.add(rect)
        lines = label_lines_by_id[node.id]
        start_y = y + node_height / 2 + (len(lines) - 1) * 4.3
        for line_index, line in enumerate(lines):
            drawing.add(
                String(
                    x + node_w / 2,
                    start_y - line_index * 8.6,
                    line,
                    fontName=FONT_BOLD if line_index == 0 else FONT,
                    fontSize=7.3,
                    fillColor=colors.HexColor(node.font_color),
                    textAnchor="middle",
                )
            )

    max_height = max(120.0, _content_height(theme) - _FLOW_PAGE_MAX_H_REDUCTION)
    return _ScaledDrawingFlowable(drawing, content_width, max_height)


def _flowchart_flowable(block: FlowChart, theme: ReportTheme = DEFAULT_REPORT_THEME) -> Flowable:
    nodes = list(block.nodes)
    if block.fit_page:
        return _flowchart_page_flowable(block, theme)

    content_width = _content_width(theme)
    node_w = _FLOW_NODE_W
    node_h = _FLOW_NODE_H
    x_gap = _FLOW_X_GAP
    y_gap = _FLOW_Y_GAP
    margin = _FLOW_MARGIN
    label_chars = 24
    width = content_width
    if not nodes:
        return Drawing(width, 1)

    cols = max(
        1,
        min(
            len(nodes),
            int((width - 2 * margin + x_gap) // (node_w + x_gap)),
        ),
    )
    rows = (len(nodes) + cols - 1) // cols
    label_lines_by_id = {
        node.id: _flow_label_lines(node.label, max_chars=label_chars) for node in nodes
    }
    node_heights = {
        node.id: max(node_h, 16 + 8.6 * len(label_lines_by_id[node.id])) for node in nodes
    }
    row_heights = [
        max(
            node_heights[nodes[index].id]
            for index in range(row * cols, min((row + 1) * cols, len(nodes)))
        )
        for row in range(rows)
    ]
    height = sum(row_heights) + max(0, rows - 1) * y_gap + 2 * margin
    drawing = Drawing(width, height)

    positions: dict[str, tuple[float, float, float]] = {}
    row_width = cols * node_w + max(0, cols - 1) * x_gap
    x0 = max(margin, (width - row_width) / 2)
    y_cursor = height - margin
    for row in range(rows):
        row_height = row_heights[row]
        y = y_cursor - row_height
        for col in range(cols):
            index = row * cols + col
            if index >= len(nodes):
                continue
            node = nodes[index]
            x = x0 + col * (node_w + x_gap)
            positions[node.id] = (x, y, node_heights[node.id])
        y_cursor = y - y_gap

    for edge in block.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        sx, sy, sh = positions[edge.source]
        tx, ty, th = positions[edge.target]
        color = colors.HexColor(edge.color)
        x1 = sx + node_w
        y1 = sy + sh / 2
        x2 = tx
        y2 = ty + th / 2
        if tx < sx:
            x1 = sx + node_w / 2
            y1 = sy
            x2 = tx + node_w / 2
            y2 = ty + th
        _draw_arrow(
            drawing,
            x1,
            y1,
            x2,
            y2,
            color,
            pen_width=edge.pen_width,
            dashed=edge.dashed,
        )
        if edge.label:
            label = _wrap_lines(edge.label, 18, 1)[0]
            drawing.add(
                String(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2 + 4,
                    label,
                    fontName=FONT,
                    fontSize=6.7,
                    fillColor=colors.HexColor(edge.font_color),
                    textAnchor="middle",
                )
            )

    for node in nodes:
        x, y, node_height = positions[node.id]
        fill = colors.HexColor(node.fill_color)
        stroke = colors.HexColor(node.stroke_color)
        font = colors.HexColor(node.font_color)
        rect = Rect(x, y, node_w, node_height, rx=4, ry=4, fillColor=fill, strokeColor=stroke)
        if node.dashed:
            rect.strokeDashArray = [3, 2]
        drawing.add(rect)
        lines = label_lines_by_id[node.id]
        start_y = y + node_height / 2 + (len(lines) - 1) * 4.3
        for line_index, line in enumerate(lines):
            drawing.add(
                String(
                    x + node_w / 2,
                    start_y - line_index * 8.6,
                    line,
                    fontName=FONT_BOLD if line_index == 0 else FONT,
                    fontSize=7.3,
                    fillColor=font,
                    textAnchor="middle",
                )
            )
    return drawing


def _title_flowables(report: Report, styles: dict, theme: ReportTheme) -> list[Flowable]:
    rows: list[list[Flowable]] = [[Paragraph(report.title, styles["title"])]]
    if report.subtitle:
        rows.append([Paragraph(report.subtitle, styles["subtitle"])])
    title = Table(rows, colWidths=[_content_width(theme)])
    title.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(theme.title_bg)),
                ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor(theme.title_border)),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, colors.HexColor(theme.primary)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return [title, Spacer(1, 12)]


def _heading_flowable(block: Heading, styles: dict, theme: ReportTheme) -> Flowable:
    level = min(max(block.level, 1), 3)
    paragraph = Paragraph(block.text, styles[f"h{level}"])
    if level != 2:
        return paragraph
    table = Table([[paragraph]], colWidths=[_content_width(theme)])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(theme.section_bg)),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(theme.primary)),
                ("LINEBELOW", (0, 0), (-1, -1), 0.45, colors.HexColor(theme.border)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _to_flowables(report: Report, styles: dict, theme: ReportTheme) -> list[Flowable]:
    flow: list[Flowable] = _title_flowables(report, styles, theme)
    for block in report.blocks:
        if isinstance(block, Heading):
            flow.append(_heading_flowable(block, styles, theme))
            flow.append(Spacer(1, 5))
        elif isinstance(block, Para):
            flow.append(Paragraph(block.text, styles["body"]))
            flow.append(Spacer(1, 4))
        elif isinstance(block, Metrics):
            flow.append(_metrics_flowable(block, styles, theme))
        elif isinstance(block, TableBlock):
            flow.append(_table_flowable(block, styles, theme))
            flow.append(Spacer(1, 8))
        elif isinstance(block, BarChart):
            for chart in _barchart_flowables(block, theme):
                flow.append(chart)
                flow.append(Spacer(1, 6))
        elif isinstance(block, FlowChart):
            flow.append(_flowchart_flowable(block, theme))
            flow.append(Spacer(1, 6))
        elif isinstance(block, PageBreak):
            flow.append(_RLPageBreak())
        else:  # pragma: no cover - захист від невідомого блоку
            raise TypeError(f"невідомий блок звіту: {type(block).__name__}")
    return flow


def _footer_painter(footer_text: str, theme: ReportTheme):
    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")

    def paint(canvas, doc) -> None:
        canvas.saveState()
        footer_y = 14 * mm
        canvas.setStrokeColor(colors.HexColor(theme.footer_rule))
        canvas.setLineWidth(0.35)
        canvas.line(theme.page_margin, footer_y + 4, A4[0] - theme.page_margin, footer_y + 4)
        canvas.setFont(FONT, 8)
        canvas.setFillColor(colors.HexColor(theme.footer_text))
        canvas.drawString(theme.page_margin, 10 * mm, f"{footer_text} · {stamp}")
        canvas.drawRightString(A4[0] - theme.page_margin, 10 * mm, f"с. {doc.page}")
        canvas.restoreState()

    return paint


def render_pdf(report: Report) -> bytes:
    """Відрендерити `Report` у байти PDF (A4, кирилиця, нумерація сторінок)."""
    _ensure_fonts()
    theme = report.theme
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=theme.page_margin,
        rightMargin=theme.page_margin,
        topMargin=theme.page_margin,
        bottomMargin=18 * mm,
        title=report.title,
    )
    painter = _footer_painter(report.footer, theme)
    doc.build(
        _to_flowables(report, _styles(theme), theme), onFirstPage=painter, onLaterPages=painter
    )
    return buf.getvalue()
