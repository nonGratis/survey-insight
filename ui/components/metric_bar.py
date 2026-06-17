"""Shared BAN-metric bar for Streamlit pages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import streamlit as st

DeltaColor = Literal["normal", "inverse", "off"]


class _MetricColumn(Protocol):
    def metric(
        self,
        label: str,
        value: object,
        delta: object | None = None,
        *,
        delta_color: DeltaColor = "normal",
        help: str | None = None,
    ) -> object: ...


class _MetricContainer(Protocol):
    def columns(self, spec: int | Sequence[float], gap: str = "small") -> Sequence[_MetricColumn]: ...


@dataclass(frozen=True)
class MetricItem:
    """One metric in a top-level BAN row."""

    label: str
    value: object
    delta: object | None = None
    help: str | None = None
    delta_color: DeltaColor = "normal"


def render_metric_bar(
    items: Sequence[MetricItem],
    *,
    columns: int | None = None,
    gap: str = "small",
    container: _MetricContainer = st,
) -> None:
    """Render metrics in a stable horizontal row.

    Empty `items` are a no-op, which keeps call sites simple when metrics are
    optional.
    """
    if not items:
        return
    count = max(1, columns or len(items))
    cols = list(container.columns(count, gap=gap))
    for index, item in enumerate(items):
        col = cols[index % count]
        col.metric(
            item.label,
            item.value,
            delta=item.delta,
            delta_color=item.delta_color,
            help=item.help,
        )
