"""2D joint-angle biomechanics for basketball injury-risk screening.

Angles are computed from the RTMPose COCO-17 keypoints and the risk thresholds
are inspired by the Landing Error Scoring System (LESS), the validated screening
tool for non-contact ACL injury. These are **2D single-camera estimates** -- good
for trends, relative comparison and flagging, not clinical-grade measurement.

Tracked angles (per leg L/R where relevant):
* knee flexion  -- low flexion (stiff landing) raises ACL / patellar load
* knee valgus   -- medial knee collapse, the principal ACL risk factor
* hip flexion   -- low flexion = stiff landing
* trunk lean    -- excessive lean (from vertical) raises knee load

Limitation: the body-17 model has no foot keypoints, so true ankle dorsiflexion
is not available (would need RTMPose's feet/wholebody model).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .. import config

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

Point = tuple[int, int, float]
XY = tuple[float, float]


def _pt(points: list[Point], idx: int, min_score: float) -> XY | None:
    if idx < len(points):
        x, y, score = points[idx]
        if score >= min_score:
            return float(x), float(y)
    return None


def _mid(a: XY | None, b: XY | None) -> XY | None:
    if a is None or b is None:
        return None
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _interior_angle(a: XY, b: XY, c: XY) -> float | None:
    """Interior angle at ``b`` (degrees) between segments b->a and b->c."""
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
    if nu == 0 or nv == 0:
        return None
    cos = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(cos))


def _flexion(a: XY | None, b: XY | None, c: XY | None) -> float | None:
    """Joint flexion (0 = straight, larger = more bent)."""
    if a is None or b is None or c is None:
        return None
    interior = _interior_angle(a, b, c)
    return None if interior is None else 180.0 - interior


def _knee_valgus(
    hip: XY | None, knee: XY | None, ankle: XY | None, opp_hip: XY | None
) -> float | None:
    """Medial knee deviation from the hip->ankle line, in degrees (+ = valgus)."""
    if hip is None or knee is None or ankle is None:
        return None
    dy = ankle[1] - hip[1]
    leg = math.hypot(ankle[0] - hip[0], ankle[1] - hip[1])
    if abs(dy) < 1.0 or leg == 0:
        return None
    t = (knee[1] - hip[1]) / dy
    line_x = hip[0] + t * (ankle[0] - hip[0])
    offset = knee[0] - line_x
    medial = 1.0
    if opp_hip is not None:
        medial = 1.0 if (opp_hip[0] - hip[0]) >= 0 else -1.0
    return math.degrees(math.atan((offset * medial) / leg))


def _trunk_lean(points: list[Point], min_score: float) -> float | None:
    """Trunk lean from vertical (degrees); mix of forward/lateral in 2D."""
    mid_sh = _mid(_pt(points, L_SHOULDER, min_score), _pt(points, R_SHOULDER, min_score))
    mid_hip = _mid(_pt(points, L_HIP, min_score), _pt(points, R_HIP, min_score))
    if mid_sh is None or mid_hip is None:
        return None
    dx, dy = mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))


@dataclass
class JointAngles:
    """Per-frame joint angles for one player (degrees; ``None`` if unavailable)."""

    knee_flex_l: float | None = None
    knee_flex_r: float | None = None
    knee_valgus_l: float | None = None
    knee_valgus_r: float | None = None
    hip_flex_l: float | None = None
    hip_flex_r: float | None = None
    trunk_lean: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def joint_angles(
    points: list[Point], min_score: float = config.JOINT_MIN_SCORE
) -> JointAngles:
    """Compute the LESS-inspired joint angles for one player's keypoints."""
    lsh = _pt(points, L_SHOULDER, min_score)
    rsh = _pt(points, R_SHOULDER, min_score)
    lhip = _pt(points, L_HIP, min_score)
    rhip = _pt(points, R_HIP, min_score)
    lkn = _pt(points, L_KNEE, min_score)
    rkn = _pt(points, R_KNEE, min_score)
    lank = _pt(points, L_ANKLE, min_score)
    rank = _pt(points, R_ANKLE, min_score)
    return JointAngles(
        knee_flex_l=_flexion(lhip, lkn, lank),
        knee_flex_r=_flexion(rhip, rkn, rank),
        knee_valgus_l=_knee_valgus(lhip, lkn, lank, rhip),
        knee_valgus_r=_knee_valgus(rhip, rkn, rank, lhip),
        hip_flex_l=_flexion(lsh, lhip, lkn),
        hip_flex_r=_flexion(rsh, rhip, rkn),
        trunk_lean=_trunk_lean(points, min_score),
    )


RISK_RULES: tuple[tuple[str, str, float], ...] = (
    ("knee_valgus_l", "high", config.RISK_KNEE_VALGUS_MAX),
    ("knee_valgus_r", "high", config.RISK_KNEE_VALGUS_MAX),
    ("trunk_lean", "high", config.RISK_TRUNK_LEAN_MAX),
)


def risk_flags(angles: JointAngles) -> dict[str, bool]:
    """Per-angle boolean risk flags using the configured thresholds."""
    values = angles.as_dict()
    flags: dict[str, bool] = {}
    for field, kind, threshold in RISK_RULES:
        v = values[field]
        if v is None:
            flags[field] = False
        elif kind == "low":
            flags[field] = v < threshold
        else:
            flags[field] = v > threshold
    return flags
