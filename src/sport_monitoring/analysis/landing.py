"""Phase 2: jump-landing detection + an automated LESS-subset score.

A jump is detected from an **upward spike in the player's body trajectory**: the
body centre (mid-hip) rises sharply off its grounded baseline to an apex, then
falls back down. The landing's initial contact (IC) is then pinned by the
**foot-height** returning to the ground after that apex. At each landing we score a
subset of the Landing Error Scoring System (LESS) items that are derivable from
2D body-17 keypoints, at initial contact (IC) and over the absorption phase
(IC -> peak knee flexion).

This is an automated **subset** of the 17-item clinical LESS (no ankle/foot items,
single-camera 2D) -- a relative screening proxy, not a diagnosis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import config
from .biomechanics import (
    JointAngles,
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
    _pt,
)

Point = tuple[int, int, float]

LESS_ITEMS = (
    "knee_flex_ic",
    "hip_flex_ic",
    "trunk_flex_ic",
    "lateral_trunk_ic",
    "knee_valgus_ic",
    "stance_width",
    "asymmetric",
    "knee_flex_disp",
    "knee_valgus_disp",
)


@dataclass
class LandingSample:
    """Per-frame data for the analysed player needed for landing detection."""

    frame: int
    foot_y: float | None
    center_y: float | None
    scale: float | None
    knee_flex: float | None
    angles: JointAngles
    ankle_dy: float | None
    stance_ratio: float | None


@dataclass
class Landing:
    """A detected landing with its LESS-subset breakdown."""

    index: int
    ic_frame: int
    apex_frame: int
    maxflex_frame: int
    items: dict[str, bool] = field(default_factory=dict)
    score: int = 0


def sample_from_points(
    frame: int, points: list[Point] | None, angles: JointAngles, min_score: float
) -> LandingSample:
    """Build a :class:`LandingSample` from one frame's keypoints + angles."""
    if points is None:
        return LandingSample(frame, None, None, None, None, angles, None, None)
    la, ra = _pt(points, L_ANKLE, min_score), _pt(points, R_ANKLE, min_score)
    lh, rh = _pt(points, L_HIP, min_score), _pt(points, R_HIP, min_score)
    ls, rs = _pt(points, L_SHOULDER, min_score), _pt(points, R_SHOULDER, min_score)

    foot_y = None
    if la and ra:
        foot_y = max(la[1], ra[1])
    elif la or ra:
        foot_y = (la or ra)[1]

    hip_ys = [p[1] for p in (lh, rh) if p]
    sh_ys = [p[1] for p in (ls, rs) if p]
    center_y = (
        sum(hip_ys) / len(hip_ys) if hip_ys
        else (sum(sh_ys) / len(sh_ys) if sh_ys else None)
    )

    scale = None
    if lh and la:
        scale = math.hypot(lh[0] - la[0], lh[1] - la[1])
    elif rh and ra:
        scale = math.hypot(rh[0] - ra[0], rh[1] - ra[1])

    knees = [v for v in (angles.knee_flex_l, angles.knee_flex_r) if v is not None]
    knee_flex = sum(knees) / len(knees) if knees else None
    ankle_dy = abs(la[1] - ra[1]) if (la and ra) else None
    stance_ratio = None
    if la and ra and ls and rs:
        sh = abs(ls[0] - rs[0])
        if sh > 1:
            stance_ratio = abs(la[0] - ra[0]) / sh
    return LandingSample(
        frame, foot_y, center_y, scale, knee_flex, angles, ankle_dy, stance_ratio
    )


def _smooth_signal(samples: list[LandingSample], attr: str) -> np.ndarray | None:
    """Gap-filled, moving-averaged time series for a per-sample y attribute."""
    n = len(samples)
    raw = np.array(
        [getattr(s, attr) if getattr(s, attr) is not None else np.nan for s in samples]
    )
    if np.all(np.isnan(raw)):
        return None
    idx = np.arange(n)
    good = ~np.isnan(raw)
    filled = np.interp(idx, idx[good], raw[good])
    k = max(1, config.LAND_SMOOTH)
    if k == 1:
        return filled
    pad = k // 2
    padded = np.pad(filled, pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def _local_maxima(sig: np.ndarray) -> list[int]:
    """Indices of local maxima of ``sig`` (plateau -> its centre)."""
    n = len(sig)
    peaks: list[int] = []
    i = 1
    while i < n - 1:
        if sig[i] > sig[i - 1] and sig[i] >= sig[i + 1]:
            j = i
            while j < n - 1 and sig[j + 1] == sig[i]:
                j += 1
            if j == n - 1 or sig[j + 1] < sig[i]:
                peaks.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return peaks


def detect_landings(samples: list[LandingSample]) -> list[Landing]:
    """Find jumps as up-and-down spikes in the body trajectory.

    For each peak we look ``LAND_MAX_AIR_FRAMES`` frames **before** and **after** it
    and require the body to have risen from the ground on the way up *and* dropped
    back to the ground on the way down -- i.e. an explicit spike up *and* down. Both
    sides must really exist (the peak can't sit at the clip edge), so a one-sided
    rise (standing up, or a clip cut mid-jump) is rejected. No global baseline.
    """
    samples = [s for s in samples if s.center_y is not None and s.scale is not None]
    n = len(samples)
    if n < 5:
        return []
    scales = [s.scale for s in samples if s.scale is not None]
    cen = _smooth_signal(samples, "center_y")
    foot = _smooth_signal(samples, "foot_y")
    if not scales or cen is None:
        return []
    scale = float(np.median(scales))
    if foot is None:
        foot = cen

    min_amp = config.LAND_MIN_JUMP_RATIO * scale
    min_air = config.LAND_MIN_AIR_RATIO * scale
    feet_visible = foot is not cen
    max_air = int(round(config.LAND_MAX_AIR_FRAMES))
    min_side = int(round(config.LAND_MIN_SIDE_FRAMES))
    absorb = int(round(config.LAND_ABSORB_FRAMES))

    height = -cen
    fr = [s.frame for s in samples]

    landings: list[Landing] = []
    last_end = -1
    for apex in _local_maxima(height):
        if apex <= last_end:
            continue
        lo = apex
        while lo > 0 and fr[apex] - fr[lo - 1] <= max_air:
            lo -= 1
        hi = apex
        while hi < n - 1 and fr[hi + 1] - fr[apex] <= max_air:
            hi += 1
        hi += 1
        if (apex - lo) < min_side or (hi - 1 - apex) < min_side:
            continue
        left_ground = float(np.min(height[lo:apex + 1]))
        right_ground = float(np.min(height[apex:hi]))
        rise_up = float(height[apex]) - left_ground
        rise_down = float(height[apex]) - right_ground
        if rise_up < min_amp or rise_down < min_amp:
            continue
        if feet_visible:
            foot_grounded = float(np.max(foot[lo:hi]))
            foot_peak = float(np.min(foot[lo:hi]))
            if (foot_grounded - foot_peak) < min_air:
                continue
        foot_ground = float(np.max(foot[apex:hi]))
        ic = apex + 1
        while ic < hi - 1 and foot[ic] < foot_ground - 0.3 * min_amp:
            ic += 1
        seg = [
            (samples[t2].knee_flex, t2)
            for t2 in range(ic, min(n, ic + absorb + 1))
            if samples[t2].knee_flex is not None
        ]
        maxflex = max(seg)[1] if seg else ic
        landings.append(
            Landing(
                index=len(landings) + 1,
                ic_frame=samples[ic].frame,
                apex_frame=samples[apex].frame,
                maxflex_frame=samples[maxflex].frame,
            )
        )
        _score(landings[-1], samples, ic, maxflex)
        last_end = ic
    return landings


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _score(landing: Landing, samples: list[LandingSample], ic: int, mf: int) -> None:
    """Score the LESS-subset items for one landing (each failed item = 1 point)."""
    s_ic, s_mf = samples[ic], samples[mf]
    a = s_ic.angles
    items: dict[str, bool] = {}

    def low(v: float | None, thr: float) -> bool:
        return v is not None and v < thr

    def high(v: float | None, thr: float) -> bool:
        return v is not None and v > thr

    hip_ic = _mean([a.hip_flex_l, a.hip_flex_r])
    valg_ic = _mean([a.knee_valgus_l, a.knee_valgus_r])
    valg_mf = _mean([s_mf.angles.knee_valgus_l, s_mf.angles.knee_valgus_r])

    items["knee_flex_ic"] = low(s_ic.knee_flex, config.LESS_KNEE_FLEX_IC_MIN)
    items["hip_flex_ic"] = low(hip_ic, config.LESS_HIP_FLEX_IC_MIN)
    items["trunk_flex_ic"] = low(a.trunk_lean, config.LESS_TRUNK_FLEX_IC_MIN)
    items["lateral_trunk_ic"] = high(a.trunk_lean, config.LESS_TRUNK_LEAN_MAX)
    items["knee_valgus_ic"] = high(valg_ic, config.LESS_KNEE_VALGUS_IC_MAX)
    items["stance_width"] = s_ic.stance_ratio is not None and (
        s_ic.stance_ratio < config.LESS_STANCE_NARROW
        or s_ic.stance_ratio > config.LESS_STANCE_WIDE
    )
    items["asymmetric"] = (
        s_ic.ankle_dy is not None
        and s_ic.scale is not None
        and (s_ic.ankle_dy / s_ic.scale) > config.LESS_SYMM_MAX
    )
    disp = (
        s_mf.knee_flex - s_ic.knee_flex
        if (s_mf.knee_flex is not None and s_ic.knee_flex is not None)
        else None
    )
    items["knee_flex_disp"] = low(disp, config.LESS_KNEE_FLEX_DISP_MIN)
    vdisp = (
        valg_mf - valg_ic if (valg_mf is not None and valg_ic is not None) else None
    )
    items["knee_valgus_disp"] = high(vdisp, config.LESS_KNEE_VALGUS_DISP_MAX)

    landing.items = items
    landing.score = sum(1 for v in items.values() if v)
