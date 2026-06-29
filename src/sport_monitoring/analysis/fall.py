"""Fall / on-ground detection from the player's head-to-toe height over time.

A fall makes a person **shorter**: when they go down, the head, hips and feet
collapse toward one height, so the body's standing height shrinks far below
normal. We measure that directly:

* **Head-to-toe height** (primary) -- the vertical (gravity-axis) span from the
  head down to the feet, made distance-invariant by dividing by the skeletal body
  length (``height = (ankle_y - head_y) / body_length``). A player keeps this near
  ~1.0 while upright; it drops toward 0 as they fall flat. We compare each frame to
  the player's **own normal** height (a high percentile of their height series), so
  a fall = ``height < FALL_HEIGHT_DROP_RATIO * normal``.
* **Upper-body drop** (temporal) -- how far the head+hip centre dropped over the
  previous ``FALL_DROP_WINDOW`` frames; tags the event *sudden*.
* **Cloud orientation** (secondary) -- PCA major-axis angle + bounding-box aspect,
  a fallback that catches a body lying diagonally when the height can't be measured.

Consecutive on-ground frames are grouped and debounced into events.
This is a 2D single-camera heuristic -- a relative flag, not a certified detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import config
from .biomechanics import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    _mid,
    _pt,
    _trunk_lean,
)

Point = tuple[int, int, float]
XY = tuple[float, float]

_AXIS_KEYPOINTS = (0, 5, 6, 11, 12, 13, 14, 15, 16)


@dataclass
class FallMetrics:
    """Per-frame body-part positions + alignment for the analysed player."""

    frame: int
    head_y: float | None = None
    hip_y: float | None = None
    knee_y: float | None = None
    ankle_y: float | None = None
    upper_y: float | None = None
    scale: float | None = None
    height: float | None = None
    axis_angle: float | None = None
    aspect: float | None = None
    horizontal: bool = False
    fallen: bool = False
    confidence: float = 0.0


@dataclass
class Fall:
    """A confirmed fall / on-ground event for the analysed player."""

    index: int
    start_frame: int
    end_frame: int
    lowest_frame: int
    duration_frames: int
    max_confidence: float
    sudden_drop: bool


def _body_scale(points: list[Point], min_score: float) -> float | None:
    """Body length (px): shoulder->hip + hip->ankle, with graceful fallbacks."""
    mid_sh = _mid(_pt(points, L_SHOULDER, min_score), _pt(points, R_SHOULDER, min_score))
    mid_hip = _mid(_pt(points, L_HIP, min_score), _pt(points, R_HIP, min_score))
    mid_ank = _mid(_pt(points, L_ANKLE, min_score), _pt(points, R_ANKLE, min_score))
    if mid_sh and mid_hip and mid_ank:
        return (math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
                + math.hypot(mid_hip[0] - mid_ank[0], mid_hip[1] - mid_ank[1]))
    if mid_sh and mid_hip:
        return 2.0 * math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
    if mid_hip and mid_ank:
        return 2.0 * math.hypot(mid_hip[0] - mid_ank[0], mid_hip[1] - mid_ank[1])
    return None


def _mean_y(*pts: XY | None) -> float | None:
    ys = [p[1] for p in pts if p is not None]
    return sum(ys) / len(ys) if ys else None


def fall_metrics(
    frame: int, points: list[Point] | None, min_score: float = config.JOINT_MIN_SCORE
) -> FallMetrics:
    """Compute per-frame body-part heights + alignment for fall detection."""
    m = FallMetrics(frame=frame)
    if not points:
        return m

    nose = _pt(points, 0, min_score)
    mid_sh = _mid(_pt(points, L_SHOULDER, min_score), _pt(points, R_SHOULDER, min_score))
    mid_hip = _mid(_pt(points, L_HIP, min_score), _pt(points, R_HIP, min_score))
    mid_kn = _mid(_pt(points, L_KNEE, min_score), _pt(points, R_KNEE, min_score))
    mid_ank = _mid(_pt(points, L_ANKLE, min_score), _pt(points, R_ANKLE, min_score))

    m.head_y = nose[1] if nose is not None else (mid_sh[1] if mid_sh else None)
    m.hip_y = mid_hip[1] if mid_hip else None
    m.knee_y = mid_kn[1] if mid_kn else None
    m.ankle_y = mid_ank[1] if mid_ank else None
    m.upper_y = _mean_y(
        (0.0, m.head_y) if m.head_y is not None else None,
        (0.0, m.hip_y) if m.hip_y is not None else None,
    )
    m.scale = _body_scale(points, min_score)

    if m.head_y is not None and m.ankle_y is not None and m.scale and m.scale > 1:
        m.height = (m.ankle_y - m.head_y) / m.scale

    trunk = _trunk_lean(points, min_score)
    elong = None
    cloud = [
        (float(points[i][0]), float(points[i][1]))
        for i in _AXIS_KEYPOINTS
        if i < len(points) and points[i][2] >= min_score
    ]
    if len(cloud) >= config.FALL_MIN_KEYPOINTS:
        arr = np.array(cloud, dtype=np.float64)
        xs, ys = arr[:, 0], arr[:, 1]
        w, h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
        m.aspect = (h / w) if w > 1.0 else None
        cov = np.cov((arr - arr.mean(axis=0)).T)
        evals, evecs = np.linalg.eigh(cov)
        lam_minor, lam_major = float(evals[0]), float(evals[1])
        major = evecs[:, 1]
        m.axis_angle = math.degrees(math.atan2(abs(major[0]), abs(major[1])))
        elong = math.sqrt(lam_major / lam_minor) if lam_minor > 1e-6 else float("inf")

    compact = m.aspect is not None and m.aspect <= config.FALL_ASPECT_MAX
    m.horizontal = bool(
        compact and (
            (m.axis_angle is not None and elong is not None
             and m.axis_angle >= config.FALL_AXIS_ANGLE_MIN
             and elong >= config.FALL_MIN_ELONGATION)
            or (trunk is not None and trunk >= config.FALL_TRUNK_ANGLE_MIN)
        )
    )
    return m


def detect_falls(metrics: list[FallMetrics]) -> list[Fall]:
    """Group per-frame on-ground postures into confirmed, debounced fall events."""
    n = len(metrics)
    if n == 0:
        return []
    scales = [m.scale for m in metrics if m.scale is not None]
    body_len = float(np.median(scales)) if scales else None

    heights = [m.height for m in metrics if m.height is not None]
    normal = float(np.percentile(heights, config.FALL_BASELINE_PCTL)) if heights else None
    threshold = (
        config.FALL_HEIGHT_DROP_RATIO * normal
        if normal is not None else config.FALL_VERTICALITY_MAX
    )
    for m in metrics:
        if m.height is not None:
            m.fallen = m.height < threshold
            m.confidence = max(0.0, min(1.0, 1.0 - m.height / normal)) if normal else 0.0
        else:
            m.fallen = m.horizontal
            m.confidence = 0.6 if m.horizontal else 0.0

    falls: list[Fall] = []
    i = 0
    while i < n:
        if not metrics[i].fallen:
            i += 1
            continue
        last_fallen = i
        k = i + 1
        while k < n and (metrics[k].fallen or (k - last_fallen) <= config.FALL_GAP_FRAMES):
            if metrics[k].fallen:
                last_fallen = k
            k += 1

        run = metrics[i:last_fallen + 1]
        if sum(1 for m in run if m.fallen) >= config.FALL_MIN_FRAMES:
            lowest = max(
                (m for m in run if m.upper_y is not None),
                key=lambda m: m.upper_y,
                default=run[0],
            )
            falls.append(Fall(
                index=len(falls) + 1,
                start_frame=metrics[i].frame,
                end_frame=metrics[last_fallen].frame,
                lowest_frame=lowest.frame,
                duration_frames=metrics[last_fallen].frame - metrics[i].frame + 1,
                max_confidence=max(m.confidence for m in run),
                sudden_drop=_had_sudden_drop(metrics, i, body_len),
            ))
        i = last_fallen + 1
    return falls


def _had_sudden_drop(
    metrics: list[FallMetrics], start: int, body_len: float | None
) -> bool:
    """True if the upper body dropped sharply in the frames before ``start``."""
    if body_len is None or body_len <= 0:
        return False
    lo = max(0, start - config.FALL_DROP_WINDOW)
    pre = [m.upper_y for m in metrics[lo:start] if m.upper_y is not None]
    here = metrics[start].upper_y
    if not pre or here is None:
        return False
    drop = here - min(pre)
    return drop >= config.FALL_DROP_RATIO * body_len
