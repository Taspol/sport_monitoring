"""Biomechanics, landing/fall detection and injury-risk scoring.

Turns COCO-17 keypoints into joint angles (``biomechanics``), detects jumps and
scores landings with a LESS-inspired rubric (``landing``), flags falls
(``fall``), and aggregates everything into a per-player risk report (``risk``).
"""

from .biomechanics import JointAngles, joint_angles, risk_flags
from .fall import Fall, detect_falls, fall_metrics
from .landing import LESS_ITEMS, Landing, detect_landings, sample_from_points
from .risk import RiskSummary, assess_player_risk, format_risk_report

__all__ = [
    "JointAngles",
    "joint_angles",
    "risk_flags",
    "Fall",
    "detect_falls",
    "fall_metrics",
    "LESS_ITEMS",
    "Landing",
    "detect_landings",
    "sample_from_points",
    "RiskSummary",
    "assess_player_risk",
    "format_risk_report",
]
