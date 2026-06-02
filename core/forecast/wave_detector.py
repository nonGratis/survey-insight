"""CUSUM-based agitation wave detector.

Axioms (verified on 627 manually annotated waves from 161 forms):
- A wave lasts ≤ 3 hours; first hour carries ~50% of wave responses.
- After per-type silence threshold (1-24h), a rate spike = exogenous shock
  (new agitation), not natural continuation of prior wave.
- Detection criterion: responses in last 1h > mult × background rate
  AND silence since last detected wave ≥ min_silence_h.

Per-type thresholds calibrated from wave_priors.json.

Reference: Hawkes (1971) self-exciting point processes; CUSUM (Page 1954).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_PRIORS_PATH = Path(__file__).parent / "wave_priors.json"
_DEFAULT_PARAMS = {"detector_min_silence_h": 6.0, "detector_mult": 5.0}

_CACHE: dict | None = None


def _priors() -> dict:
    global _CACHE
    if _CACHE is None and _PRIORS_PATH.exists():
        _CACHE = json.loads(_PRIORS_PATH.read_text(encoding="utf-8"))
    return _CACHE or {}


def _get_params(form_type: str | None) -> tuple[float, float]:
    """Return (min_silence_h, rate_mult) for given form_type."""
    p = _priors()
    if form_type and "per_type" in p and form_type in p["per_type"]:
        pt = p["per_type"][form_type]
        return float(pt["detector_min_silence_h"]), float(pt["detector_mult"])
    d = _DEFAULT_PARAMS
    return float(d["detector_min_silence_h"]), float(d["detector_mult"])


@dataclass
class WaveStart:
    """Detected start of an agitation wave."""

    timestamp: pd.Timestamp
    wave_index: int  # 0-based
    n_since_last: int  # responses since previous wave start
    silence_before_h: float  # hours since last wave


def detect_wave_starts(
    timestamps: list[pd.Timestamp] | pd.Series,
    form_type: str | None = None,
    test_skip: int = 0,
    baseline_window_h: float = 6.0,
) -> list[WaveStart]:
    """Detect agitation wave starts in a stream of response timestamps.

    Algorithm (CUSUM-style):
    1. Skip first `test_skip` responses (likely creator test responses).
    2. First clean response always marks wave #0.
    3. At each subsequent response t_i:
       - n_recent = responses in [t_i - 1h, t_i]
       - background = avg rate in [t_i - (baseline_window_h+1)h, t_i - 1h]
       - silence_h = time since last detected wave
       - is_wave = (n_recent >= MIN_RESP and
                    (background == 0 or n_recent > mult * background) and
                    silence_h >= min_silence_h)

    Args:
        timestamps: sorted (or unsorted) response timestamps.
        form_type: catalog form type for per-type thresholds.
        test_skip: strip this many initial responses before analysis.
        baseline_window_h: background rate window in hours.

    Returns:
        List of WaveStart objects, always ≥ 1 (first response is wave #0).
    """
    ts = pd.Series(pd.to_datetime(timestamps)).sort_values().reset_index(drop=True)
    ts = ts.iloc[test_skip:].reset_index(drop=True)
    if len(ts) == 0:
        return []

    min_sil, mult = _get_params(form_type)
    min_resp = 3  # minimum responses in 1h to qualify as spike

    secs = np.array([(t - ts.iloc[0]).total_seconds() for t in ts])

    waves: list[WaveStart] = [WaveStart(ts.iloc[0], 0, 1, 0.0)]
    last_wave_sec = secs[0]
    last_wave_idx = 0

    for i in range(1, len(secs)):
        t_cur = secs[i]
        t_1h_ago = t_cur - 3600.0
        t_base_end = t_cur - 3600.0
        t_base_start = t_cur - (baseline_window_h + 1.0) * 3600.0

        n_recent = int(((secs > t_1h_ago) & (secs <= t_cur)).sum())
        n_base = int(((secs > t_base_start) & (secs <= t_base_end)).sum())
        base_rate_per_h = n_base / max(baseline_window_h, 1.0)
        silence_h = (t_cur - last_wave_sec) / 3600.0

        is_spike = (
            n_recent >= min_resp
            and (base_rate_per_h == 0 or n_recent > mult * base_rate_per_h)
            and silence_h >= min_sil
        )
        if is_spike:
            n_since = i - last_wave_idx
            waves.append(WaveStart(ts.iloc[i], len(waves), n_since, silence_h))
            last_wave_sec = t_cur
            last_wave_idx = i

    return waves


def current_wave_timestamps(
    timestamps: list[pd.Timestamp] | pd.Series,
    form_type: str | None = None,
    test_skip: int = 0,
) -> tuple[pd.Timestamp, pd.Series]:
    """Return (wave_start_ts, timestamps_within_current_wave).

    Detects all waves and returns the LAST detected wave start plus
    only the responses from that wave start to the most recent response.
    This is the natural input for within-wave curve fitting.
    """
    ts = pd.Series(pd.to_datetime(timestamps)).sort_values().reset_index(drop=True)
    waves = detect_wave_starts(ts, form_type=form_type, test_skip=test_skip)
    if not waves:
        return ts.iloc[0], ts
    last_wave = waves[-1]
    wave_ts = ts[ts >= last_wave.timestamp].reset_index(drop=True)
    return last_wave.timestamp, wave_ts
