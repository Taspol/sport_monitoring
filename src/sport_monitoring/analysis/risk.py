"""Per-player ACL injury-risk assessment for the analysed basketball player.

This aggregates the per-frame joint-angle flags and the per-landing LESS-subset
scores into a single, evidence-graded risk rating, grounded in the prospective
basketball / jump-landing literature:

* **Hewett et al., 2005** (Am J Sports Med, JUMP study) -- in female basketball /
  soccer / volleyball players, increased **dynamic knee valgus** (knee abduction
  angle/moment at landing) prospectively predicted non-contact ACL injury
  (78% sensitivity, 73% specificity).  Valgus is the principal *modifiable*
  factor, so sustained valgus exposure dominates this rating.
* **Leppanen et al., 2017** (Am J Sports Med) -- in 171 female **basketball /
  floorball** players, **stiff landings** (lower peak knee flexion; HR 0.55 per
  +10 deg of knee flexion) were associated with greater ACL-injury risk.  Low
  knee flexion at landing is the sport-specific red flag.
* **Padua et al., 2009** (Am J Sports Med, JUMP-ACL) -- the **Landing Error
  Scoring System (LESS)** is a validated 2D clinical screen; a total LESS
  score >= 5 of 17 marks a high-risk lander.  Our automated 9-item single-camera
  subset uses a scaled cut (see ``config.LESS_SUBSET_HIGH_SCORE``).

IMPORTANT: these are **2D single-camera screening proxies**, useful for relative
flagging and trend-spotting -- NOT a clinical diagnosis or a calibrated ACL
probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from .biomechanics import JointAngles
from .landing import Landing

_VALGUS_FLAGS = ("knee_valgus_l", "knee_valgus_r")
_STIFF_ITEMS = ("knee_flex_ic", "knee_flex_disp", "hip_flex_ic")
_VALGUS_ITEMS = ("knee_valgus_ic", "knee_valgus_disp")

LOW, MODERATE, HIGH = "LOW", "MODERATE", "HIGH"
_ORDER = {LOW: 0, MODERATE: 1, HIGH: 2}


@dataclass
class RiskSummary:
    """Aggregated injury-risk picture for one analysed player."""

    rating: str = LOW
    composite: int = 0
    n_frames: int = 0
    valgus_frame_rate: float = 0.0
    trunk_frame_rate: float = 0.0
    n_landings: int = 0
    mean_less: float | None = None
    max_less: int | None = None
    high_landings: int = 0
    stiff_landings: int = 0
    valgus_landings: int = 0
    n_falls: int = 0
    reasons: list[str] = field(default_factory=list)


def _rate(value: float, denom: float) -> float:
    return min(1.0, value / denom) if denom > 0 else 0.0


def assess_player_risk(
    rows: list[tuple[int, JointAngles, dict[str, bool]]],
    landings: list[Landing],
) -> RiskSummary:
    """Combine per-frame flags + per-landing LESS scores into a RiskSummary.

    Args:
        rows: ``(frame, JointAngles, risk_flags)`` for every frame the analysed
            player was tracked (as collected in the analyze pipeline).
        landings: detected landings with their scored LESS-subset items.
    """
    s = RiskSummary(n_frames=len(rows), n_landings=len(landings))
    if rows:
        valgus = sum(
            1 for _, _, fl in rows if any(fl.get(k) for k in _VALGUS_FLAGS)
        )
        trunk = sum(1 for _, _, fl in rows if fl.get("trunk_lean"))
        s.valgus_frame_rate = valgus / len(rows)
        s.trunk_frame_rate = trunk / len(rows)

    if landings:
        scores = [L.score for L in landings]
        s.mean_less = sum(scores) / len(scores)
        s.max_less = max(scores)
        s.high_landings = sum(
            1 for L in landings if L.score >= config.LESS_SUBSET_HIGH_SCORE
        )
        s.stiff_landings = sum(
            1 for L in landings if any(L.items.get(k) for k in _STIFF_ITEMS)
        )
        s.valgus_landings = sum(
            1 for L in landings if any(L.items.get(k) for k in _VALGUS_ITEMS)
        )

    s_valgus = _rate(s.valgus_frame_rate, config.RISK_VALGUS_FRAME_HIGH)
    s_trunk = _rate(s.trunk_frame_rate, 0.30)
    s_landing = _rate(s.mean_less or 0.0, 6.0)
    stiff_rate = (s.stiff_landings / s.n_landings) if s.n_landings else 0.0
    valgus_rate = (s.valgus_landings / s.n_landings) if s.n_landings else 0.0
    s.composite = round(
        100 * (0.40 * s_valgus + 0.25 * s_landing
               + 0.20 * stiff_rate + 0.15 * s_trunk)
    )

    reasons: list[str] = []
    rating = LOW

    def bump(level: str, why: str) -> None:
        nonlocal rating
        if _ORDER[level] > _ORDER[rating]:
            rating = level
        reasons.append(why)

    if s.valgus_frame_rate >= config.RISK_VALGUS_FRAME_HIGH:
        bump(HIGH, f"Dynamic knee valgus in {s.valgus_frame_rate*100:.0f}% of "
                   f"frames (Hewett 2005: valgus is the principal ACL predictor)")
    elif s.valgus_frame_rate >= config.RISK_VALGUS_FRAME_MOD:
        bump(MODERATE, f"Dynamic knee valgus in {s.valgus_frame_rate*100:.0f}% "
                       f"of frames")

    if s.high_landings:
        bump(HIGH, f"{s.high_landings} high-risk landing(s) "
                   f"(LESS-subset >= {config.LESS_SUBSET_HIGH_SCORE}/9; "
                   f"Padua 2009: LESS >= 5/17 = high risk)")
    elif (s.mean_less or 0) >= config.LESS_SUBSET_MOD_SCORE:
        bump(MODERATE, f"Mean landing LESS-subset {s.mean_less:.1f}/9")

    if s.n_landings and stiff_rate >= config.RISK_STIFF_LANDING_RATE:
        bump(HIGH, f"{s.stiff_landings}/{s.n_landings} landings are stiff / low "
                   f"knee flexion (Leppanen 2017: stiff landing raises ACL risk)")
    if s.n_landings and valgus_rate >= config.RISK_VALGUS_LANDING_RATE:
        bump(HIGH, f"{s.valgus_landings}/{s.n_landings} landings show knee valgus "
                   f"collapse")

    if s.trunk_frame_rate >= config.RISK_TRUNK_FRAME_MOD:
        bump(MODERATE, f"Excessive trunk lean in {s.trunk_frame_rate*100:.0f}% "
                       f"of frames")

    if s.composite >= config.RISK_COMPOSITE_HIGH:
        bump(HIGH, f"High composite exposure score ({s.composite}/100)")
    elif s.composite >= config.RISK_COMPOSITE_MOD:
        bump(MODERATE, f"Moderate composite exposure score ({s.composite}/100)")

    if not reasons:
        reasons.append("No sustained valgus, stiff landings, or high-LESS "
                       "landings detected in this clip.")

    s.rating = rating
    s.reasons = reasons
    return s


def format_risk_report(summary: RiskSummary, player_label: str) -> list[str]:
    """Render the RiskSummary as printable / file-writable text lines."""
    s = summary
    ml = "--" if s.mean_less is None else f"{s.mean_less:.1f}"
    mx = "--" if s.max_less is None else f"{s.max_less}"
    lines = [
        f"==== INJURY-RISK SUMMARY -- {player_label} ====",
        f"Overall risk: {s.rating}   (composite exposure {s.composite}/100)",
        f"  Tracked frames analysed     : {s.n_frames}",
        f"  Dynamic knee valgus exposure: {s.valgus_frame_rate*100:.0f}% of frames"
        "   [Hewett 2005]",
        f"  Excessive trunk-lean exposure: {s.trunk_frame_rate*100:.0f}% of frames",
        f"  Landings detected           : {s.n_landings}"
        f"  (mean LESS-subset {ml}/9, max {mx}/9)   [Padua 2009]",
        f"  High-risk landings (>= {config.LESS_SUBSET_HIGH_SCORE}/9): "
        f"{s.high_landings}",
        f"  Stiff / low-flexion landings : {s.stiff_landings}/{s.n_landings}"
        "   [Leppanen 2017]",
        f"  Knee-valgus landings         : {s.valgus_landings}/{s.n_landings}",
        f"  Fall / on-ground events      : {s.n_falls}"
        "   (acute -- keypoint-alignment detector)",
        "Why:",
    ]
    lines += [f"  - {r}" for r in s.reasons]
    lines.append(
        "NOTE: 2D single-camera screening proxy (relative flagging / trends), "
        "NOT a clinical diagnosis."
    )
    return lines
