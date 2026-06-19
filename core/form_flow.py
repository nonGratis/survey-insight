"""Form section flow parser and compact Graphviz renderer."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Literal

START_ID = "__start__"
SUBMIT_ID = "__submit__"

NodeKind = Literal["start", "section", "submit"]
EdgeKind = Literal["default", "conditional"]


@dataclass(frozen=True)
class FlowNode:
    id: str
    title: str
    kind: NodeKind
    index: int
    detail_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlowEdge:
    source: str
    target: str
    label: str
    kind: EdgeKind


@dataclass(frozen=True)
class FormFlow:
    nodes: list[FlowNode]
    edges: list[FlowEdge]
    unreachable_section_ids: list[str]
    terminal_section_ids: list[str]
    has_cycles: bool

    @property
    def section_count(self) -> int:
        return sum(node.kind in ("start", "section") for node in self.nodes)

    @property
    def conditional_edge_count(self) -> int:
        return sum(edge.kind == "conditional" for edge in self.edges)


def parse_form_flow(form: dict[str, Any]) -> FormFlow:
    """Parse Google Forms structure into a compact section-transition graph."""
    sections = _sections(form)
    submit = FlowNode(SUBMIT_ID, "Надіслати", "submit", len(sections))
    nodes = [*sections, submit]
    section_ids = [node.id for node in sections]
    section_by_id = {node.id: node for node in sections}
    current_section = START_ID
    edges_by_section: dict[str, list[FlowEdge]] = {node.id: [] for node in sections}
    needs_default_by_section = {node.id: False for node in sections}
    detail_lines_by_section: dict[str, list[str]] = {node.id: [] for node in sections}

    for item in form.get("items", []):
        if "pageBreakItem" in item:
            current_section = _item_id(item) or current_section
            continue
        conditional, needs_default, question_title = _conditional_edges(
            item, current_section, section_ids, section_by_id
        )
        if conditional:
            edges_by_section[current_section].extend(conditional)
            needs_default_by_section[current_section] = needs_default
            if question_title and question_title not in detail_lines_by_section[current_section]:
                detail_lines_by_section[current_section].append(question_title)

    edges: list[FlowEdge] = []
    for index, node in enumerate(sections):
        if len(sections) == 1 or needs_default_by_section[node.id]:
            target = section_ids[index + 1] if index + 1 < len(section_ids) else SUBMIT_ID
            edges.append(FlowEdge(node.id, target, "далі", "default"))
        edges.extend(edges_by_section[node.id])
    edges = _dedupe_edges(edges)
    reachable = _reachable_section_ids(edges)
    edges = _dedupe_edges([*edges, *_terminal_edges(edges, reachable)])
    reachable = _reachable_section_ids(edges)
    unreachable = [node.id for node in sections if node.id not in reachable]
    terminal = sorted(
        {
            edge.source
            for edge in edges
            if edge.target == SUBMIT_ID and edge.source in section_by_id
        },
        key=section_ids.index,
    )
    nodes = [
        *[
            FlowNode(
                node.id,
                node.title,
                node.kind,
                node.index,
                tuple(detail_lines_by_section[node.id]),
            )
            for node in sections
        ],
        submit,
    ]
    return FormFlow(
        nodes=nodes,
        edges=edges,
        unreachable_section_ids=unreachable,
        terminal_section_ids=terminal,
        has_cycles=_has_cycle(edges, section_ids),
    )


def _terminal_edges(edges: list[FlowEdge], reachable: set[str]) -> list[FlowEdge]:
    existing_sources = {edge.source for edge in edges}
    return [
        FlowEdge(section_id, SUBMIT_ID, "надіслати", "default")
        for section_id in sorted(reachable)
        if section_id not in existing_sources
    ]


def flow_to_dot(flow: FormFlow) -> str:
    """Render flow as compact DOT for `st.graphviz_chart`."""
    lines = [
        "digraph FormFlow {",
        '  graph [rankdir=LR, bgcolor="transparent", margin=0, nodesep=0.18, ranksep=0.34, pad=0.02];',
        '  node [shape=rect, style="rounded,filled", fontname="Arial", fontsize=10, margin="0.08,0.045", height=0.32];',
        '  edge [fontname="Arial", fontsize=8, arrowsize=0.62, penwidth=1.1, color="#94a3b8", fontcolor="#475569"];',
    ]
    for node in flow.nodes:
        fill, stroke, font, style = _node_style(node, flow)
        label = _node_label(node)
        lines.append(
            f'  "{_dot_escape(node.id)}" [label="{_dot_escape(label)}", style="{style}", fillcolor="{fill}", color="{stroke}", fontcolor="{font}"];'
        )
    for edge in flow.edges:
        color = "#2563eb" if edge.kind == "conditional" else "#cbd5e1"
        style = "solid" if edge.kind == "conditional" else "dashed"
        label = _wrap_label(edge.label, max_chars=22)
        lines.append(
            f'  "{_dot_escape(edge.source)}" -> "{_dot_escape(edge.target)}" [label="{_dot_escape(label)}", color="{color}", style="{style}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _sections(form: dict[str, Any]) -> list[FlowNode]:
    sections = [FlowNode(START_ID, "Старт", "start", 0)]
    for item in form.get("items", []):
        if "pageBreakItem" not in item:
            continue
        section_id = _item_id(item) or f"section_{len(sections)}"
        title = str(item.get("title") or f"Секція {len(sections)}")
        sections.append(FlowNode(section_id, title, "section", len(sections)))
    return sections


def _conditional_edges(
    item: dict[str, Any],
    current_section: str,
    section_ids: list[str],
    section_by_id: dict[str, FlowNode],
) -> tuple[list[FlowEdge], bool, str | None]:
    question = item.get("questionItem", {}).get("question", {})
    if not question:
        return [], True, None
    choice = question.get("choiceQuestion", {})
    options = choice.get("options", [])
    if not options:
        return [], True, None
    out: list[FlowEdge] = []
    option_without_target = False
    question_title = str(item.get("title") or "Питання")
    for option in options:
        target = _option_target(option, current_section, section_ids, section_by_id)
        if not target:
            option_without_target = True
            continue
        option_label = str(option.get("value") or "варіант")
        out.append(FlowEdge(current_section, target, option_label, "conditional"))
    if not out:
        return [], True, None
    return out, option_without_target, question_title


def _option_target(
    option: dict[str, Any],
    current_section: str,
    section_ids: list[str],
    section_by_id: dict[str, FlowNode],
) -> str | None:
    section_id = option.get("goToSectionId")
    if section_id in section_by_id:
        return str(section_id)
    action = option.get("goToAction")
    if action == "SUBMIT_FORM":
        return SUBMIT_ID
    if action == "RESTART_FORM":
        return START_ID
    if action == "NEXT_SECTION":
        try:
            index = section_ids.index(current_section)
        except ValueError:
            return None
        return section_ids[index + 1] if index + 1 < len(section_ids) else SUBMIT_ID
    return None


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("itemId") or item.get("item_id") or "")


def _dedupe_edges(edges: list[FlowEdge]) -> list[FlowEdge]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[FlowEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.label, edge.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _reachable_section_ids(edges: list[FlowEdge]) -> set[str]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source].append(edge.target)
    seen = {START_ID}
    queue = deque([START_ID])
    while queue:
        node = queue.popleft()
        for target in adjacency[node]:
            if target == SUBMIT_ID or target in seen:
                continue
            seen.add(target)
            queue.append(target)
    return seen


def _has_cycle(edges: list[FlowEdge], section_ids: list[str]) -> bool:
    section_set = set(section_ids)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.source in section_set and edge.target in section_set:
            adjacency[edge.source].append(edge.target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in adjacency[node]:
            if visit(target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in section_ids)


def _node_style(node: FlowNode, flow: FormFlow) -> tuple[str, str, str, str]:
    if node.id in flow.unreachable_section_ids:
        return "#fff1f2", "#fb7185", "#9f1239", "rounded,filled,dashed"
    if node.kind == "start":
        return "#eef2ff", "#6366f1", "#3730a3", "rounded,filled"
    if node.kind == "submit":
        return "#ecfdf5", "#10b981", "#065f46", "rounded,filled"
    return "#f8fafc", "#cbd5e1", "#334155", "rounded,filled"


def _node_label(node: FlowNode) -> str:
    lines = [_wrap_label(node.title, max_chars=24)]
    for detail in node.detail_lines[:2]:
        lines.append(_wrap_label(detail, max_chars=30))
    return "\n".join(lines)


def _wrap_label(value: str, max_chars: int) -> str:
    clean = " ".join(str(value).split())
    if len(clean) <= max_chars:
        return clean
    words = clean.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:max_chars]
    if current:
        lines.append(current)
    return "\n".join(lines[:2])


def _dot_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )
