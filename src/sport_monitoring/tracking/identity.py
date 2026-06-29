"""Persistent player identities on top of the raw ByteTrack ids.

ByteTrack matches on motion only, so when a player is occluded or leaves the
frame and comes back, it usually assigns a brand-new id. This layer keeps state
for every player -- their jersey-colour histogram and a predicted position -- and
when the raw tracker produces a fresh id, it tries to re-link that detection to a
recently-lost player instead of inventing a new identity.

Stable ids are what the rest of the pipeline (pose, colours) sees.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import color_id
from .. import config
from ..perception.detector import Detection

Box = tuple[int, int, int, int]


def _appearance_descriptor(frame: np.ndarray, box: Box) -> np.ndarray:
    """HS colour histogram of a player's torso region (captures jersey colour)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return np.zeros((16, 16), dtype=np.float32).flatten()
    tx1, tx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    ty1, ty2 = y1 + int(0.2 * h), y1 + int(0.55 * h)
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((16, 16), dtype=np.float32).flatten()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _centroid(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Identity:
    stable_id: int
    box: Box
    centroid: tuple[float, float]
    velocity: tuple[float, float]
    descriptor: np.ndarray
    raw_id: int
    last_seen: int

    def predict(self, frame_idx: int) -> tuple[float, float]:
        """Extrapolate position to ``frame_idx`` using last known velocity."""
        dt = frame_idx - self.last_seen
        return (
            self.centroid[0] + self.velocity[0] * dt,
            self.centroid[1] + self.velocity[1] * dt,
        )


@dataclass
class IDStabilizer:
    """Map raw tracker ids to persistent stable ids using appearance + motion."""

    max_lost: int = config.STABILIZER_MAX_LOST_FRAMES
    min_appearance: float = config.STABILIZER_MIN_APPEARANCE
    max_dist_ratio: float = config.STABILIZER_MAX_DIST_RATIO
    w_app: float = config.STABILIZER_APPEARANCE_WEIGHT
    w_pos: float = config.STABILIZER_POSITION_WEIGHT
    max_cost: float = config.STABILIZER_MAX_COST
    app_alpha: float = config.STABILIZER_APPEARANCE_ALPHA

    _identities: dict[int, _Identity] = field(default_factory=dict)
    _raw_to_stable: dict[int, int] = field(default_factory=dict)
    _next_id: int = 1
    _frame_idx: int = -1

    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[tuple[int, Detection]]:
        """Return ``(stable_id, detection)`` for every detection this frame."""
        self._frame_idx += 1
        max_dist = self.max_dist_ratio * float(np.hypot(*frame.shape[:2]))

        assigned: list[tuple[int, Detection]] = []
        used: set[int] = set()
        fresh: list[Detection] = []

        for det in detections:
            sid = self._raw_to_stable.get(det.track_id)
            if sid is not None and sid in self._identities:
                self._touch(sid, det, frame)
                used.add(sid)
                assigned.append((sid, det))
            else:
                fresh.append(det)

        for det in fresh:
            sid = self._rematch(det, frame, max_dist, exclude=used)
            if sid is None:
                sid = self._create(det, frame)
            else:
                self._raw_to_stable[det.track_id] = sid
                self._touch(sid, det, frame)
            used.add(sid)
            assigned.append((sid, det))

        self._expire()
        return assigned

    @property
    def active_ids(self) -> set[int]:
        return set(self._identities)

    def _rematch(
        self, det: Detection, frame: np.ndarray, max_dist: float, exclude: set[int]
    ) -> int | None:
        """Best lost identity for ``det`` under the cost gate, or ``None``."""
        desc = _appearance_descriptor(frame, det.box)
        cen = _centroid(det.box)

        best_sid: int | None = None
        best_cost = self.max_cost
        for sid, ident in self._identities.items():
            if sid in exclude:
                continue
            dist = float(np.hypot(*np.subtract(cen, ident.predict(self._frame_idx))))
            if dist > max_dist:
                continue
            appsim = float(
                cv2.compareHist(desc, ident.descriptor, cv2.HISTCMP_CORREL)
            )
            if appsim < self.min_appearance:
                continue
            cost = self.w_pos * (dist / max_dist) + self.w_app * (1.0 - appsim)
            if cost < best_cost:
                best_cost, best_sid = cost, sid
        return best_sid

    def _touch(self, sid: int, det: Detection, frame: np.ndarray) -> None:
        ident = self._identities[sid]
        cen = _centroid(det.box)
        ident.velocity = (cen[0] - ident.centroid[0], cen[1] - ident.centroid[1])
        ident.centroid = cen
        ident.box = det.box
        ident.raw_id = det.track_id
        ident.last_seen = self._frame_idx
        new_desc = _appearance_descriptor(frame, det.box)
        ident.descriptor = (
            self.app_alpha * new_desc + (1 - self.app_alpha) * ident.descriptor
        )

    def _create(self, det: Detection, frame: np.ndarray) -> int:
        sid = self._next_id
        self._next_id += 1
        self._identities[sid] = _Identity(
            stable_id=sid,
            box=det.box,
            centroid=_centroid(det.box),
            velocity=(0.0, 0.0),
            descriptor=_appearance_descriptor(frame, det.box),
            raw_id=det.track_id,
            last_seen=self._frame_idx,
        )
        self._raw_to_stable[det.track_id] = sid
        return sid

    def _expire(self) -> None:
        for sid in list(self._identities):
            if self._frame_idx - self._identities[sid].last_seen > self.max_lost:
                del self._identities[sid]
        self._raw_to_stable = {
            raw: sid
            for raw, sid in self._raw_to_stable.items()
            if sid in self._identities
        }




@dataclass
class _TargetState:
    stable_id: int
    centroid: tuple[float, float]
    velocity: tuple[float, float]
    box: Box
    last_seen: int
    raw_id: int | None = None
    acquired: bool = False
    missing: int = 0
    contest: int = 0
    color: np.ndarray | None = None


@dataclass
class MatchResult:
    """Outcome of matching one frame's detections to the selected players."""

    selected: list[tuple[int, Detection]]
    others: list[Detection]


class SelectedPlayerTracker:
    """Map ByteTrack tracklets to selected ids by jersey-colour identity.

    Hybrid of ByteTrack (continuity) and the colour registry (identity):

    * **Continuity** -- while a tracklet keeps its ByteTrack id and its colour
      still matches the id it's bound to, the id rides that tracklet (no per-frame
      re-decision -> few swaps).
    * **Re-verify** -- if the tracklet's colour diverges from the id's reference
      (a ByteTrack merge/swap), the binding is dropped and the id is freed.
    * **Assign** -- free ids are given the unbound tracklet whose colour is most
      similar (colour-primary, with a small position tie-breaker).

    Colour decides *which* id a tracklet is; ByteTrack decides *continuity*.
    Colour still cannot separate same-jersey teammates -- the tie-breaker does.
    """

    def __init__(self, targets) -> None:
        self._frame_idx = -1
        self._colors: dict[int, np.ndarray | None] = {}
        self._targets: list[_TargetState] = [
            _TargetState(
                stable_id=t.stable_id,
                centroid=_centroid(t.box),
                velocity=(0.0, 0.0),
                box=t.box,
                last_seen=0,
                raw_id=None,
                color=getattr(t, "color", None),
            )
            for t in targets
        ]

    def reference_color(self, stable_id: int) -> np.ndarray | None:
        """The current EMA reference jersey colour (HSV) for a tracked player."""
        for target in self._targets:
            if target.stable_id == stable_id:
                return target.color
        return None

    def registry(self) -> dict[int, np.ndarray]:
        """The saved ID -> identity (reference HSV colour) lookup table.

        This is what :meth:`update` matches each frame's detections against to
        decide which id to assign. Ids whose colour wasn't captured yet are
        omitted (they are bootstrapped spatially on their first frame).
        """
        return {t.stable_id: t.color for t in self._targets if t.color is not None}

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        colors: dict[int, np.ndarray | None] | None = None,
    ) -> MatchResult:
        self._frame_idx += 1
        self._colors = colors or {}
        by_tracklet = {det.track_id: det for det in detections}
        diag = float(np.hypot(*frame.shape[:2]))
        claimed: set[int] = set()

        for target in self._targets:
            det = by_tracklet.get(target.raw_id) if target.raw_id is not None else None
            if det is not None:
                target.missing = 0
                sim = self._colour_similarity(target, det)
                if sim is not None and self._contested(target, det, sim):
                    target.contest += 1
                    if target.contest >= config.SELECT_CONTEST_FRAMES:
                        target.raw_id = None
                        target.contest = 0
                    else:
                        self._touch(target, det, update_color=False)
                        claimed.add(det.track_id)
                else:
                    target.contest = 0
                    confident = sim is not None and sim >= config.SELECT_COLOR_MIN_SIM
                    self._touch(target, det, update_color=confident)
                    claimed.add(det.track_id)
            elif target.raw_id is not None:
                target.missing += 1
                if target.missing >= config.SELECT_TRACKLET_GRACE:
                    target.raw_id = None
                    target.missing = 0

        reassign_gate = config.SELECT_REASSIGN_DIST_RATIO * diag
        free = [
            (ti, t)
            for ti, t in enumerate(self._targets)
            if t.raw_id is None and t.color is not None
        ]
        scored: list[tuple[float, int, Detection]] = []
        for ti, target in free:
            pred = self._predict(target)
            for det in detections:
                if det.track_id in claimed:
                    continue
                cur = self._colors.get(det.track_id)
                if cur is None:
                    continue
                sim = color_id.similarity(target.color, cur)
                if sim < config.SELECT_COLOR_MIN_SIM:
                    continue
                cx, cy = _centroid(det.box)
                dist = float(np.hypot(cx - pred[0], cy - pred[1]))
                if target.acquired and dist > reassign_gate:
                    continue
                scored.append(
                    (sim - config.SELECT_COLOR_POS_TIEBREAK * (dist / diag), ti, det)
                )

        used_t: set[int] = set()
        for _, ti, det in sorted(scored, key=lambda s: s[0], reverse=True):
            if ti in used_t or det.track_id in claimed:
                continue
            used_t.add(ti)
            claimed.add(det.track_id)
            target = self._targets[ti]
            target.raw_id = det.track_id
            target.acquired = True
            self._touch(target, det)

        for target in self._targets:
            if target.raw_id is None and target.color is None:
                det = self._acquire(target, detections, claimed)
                if det is not None:
                    target.raw_id = det.track_id
                    target.acquired = True
                    self._touch(target, det)
                    claimed.add(det.track_id)

        selected = [
            (t.stable_id, by_tracklet[t.raw_id])
            for t in self._targets
            if t.raw_id is not None and t.raw_id in by_tracklet
        ]
        others = [d for d in detections if d.track_id not in claimed]
        return MatchResult(selected=selected, others=others)

    def _colour_similarity(
        self, target: _TargetState, det: Detection
    ) -> float | None:
        """Similarity of the detection's colour to the id's reference, or None.

        ``None`` means it can't be judged this frame (no reference yet, or the
        detection's colour wasn't sampled).
        """
        if target.color is None:
            return None
        cur = self._colors.get(det.track_id)
        if cur is None:
            return None
        return color_id.similarity(target.color, cur)

    def _contested(self, target: _TargetState, det: Detection, sim: float) -> bool:
        """True if another id's colour fits this tracklet clearly better than ``sim``."""
        cur = self._colors.get(det.track_id)
        if cur is None:
            return False
        for other in self._targets:
            if other is target or other.color is None:
                continue
            if color_id.similarity(other.color, cur) >= sim + config.SELECT_CONTEST_MARGIN:
                return True
        return False

    def _best_iou(
        self,
        target: _TargetState,
        detections: list[Detection],
        claimed: set[int],
        threshold: float,
    ) -> Detection | None:
        """Unclaimed detection most overlapping the target's predicted box."""
        pbox = self._predicted_box(target)
        best: Detection | None = None
        best_iou = threshold
        for det in detections:
            if det.track_id in claimed:
                continue
            overlap = _iou(pbox, det.box)
            if overlap > best_iou:
                best_iou = overlap
                best = det
        return best

    def _acquire(
        self, target: _TargetState, detections: list[Detection], claimed: set[int]
    ) -> Detection | None:
        """Initial lock: best overlap if any, else the nearest unclaimed box."""
        det = self._best_iou(target, detections, claimed, 0.05)
        if det is not None:
            return det
        point = target.centroid
        best: Detection | None = None
        best_dist = float("inf")
        for cand in detections:
            if cand.track_id in claimed:
                continue
            cx, cy = _centroid(cand.box)
            dist = float(np.hypot(cx - point[0], cy - point[1]))
            if dist < best_dist:
                best_dist = dist
                best = cand
        return best

    def _dt(self, target: _TargetState) -> int:
        """Frames since last seen, capped so the prediction can't drift away."""
        return min(self._frame_idx - target.last_seen, config.SELECT_PREDICT_MAX_DT)

    def _predicted_box(self, target: _TargetState) -> Box:
        dt = self._dt(target)
        x1, y1, x2, y2 = target.box
        vx, vy = target.velocity
        return (
            int(x1 + vx * dt),
            int(y1 + vy * dt),
            int(x2 + vx * dt),
            int(y2 + vy * dt),
        )

    def _predict(self, target: _TargetState) -> tuple[float, float]:
        dt = self._dt(target)
        return (
            target.centroid[0] + target.velocity[0] * dt,
            target.centroid[1] + target.velocity[1] * dt,
        )

    def _touch(
        self, target: _TargetState, det: Detection, update_color: bool = True
    ) -> None:
        """Update a target's motion state to the detection it is following.

        ``update_color`` is False while riding a tracklet through a colour-mismatch
        frame, so the reference isn't corrupted by a noisy / wrong colour.
        """
        centroid = _centroid(det.box)
        dt = max(1, self._frame_idx - target.last_seen)
        target.velocity = (
            (centroid[0] - target.centroid[0]) / dt,
            (centroid[1] - target.centroid[1]) / dt,
        )
        target.centroid = centroid
        target.box = det.box
        target.last_seen = self._frame_idx
        target.missing = 0
        if not update_color:
            return
        cand = self._colors.get(det.track_id)
        if cand is not None:
            target.color = (
                cand
                if target.color is None
                else color_id.ema(target.color, cand, config.COLOR_ID_EMA_ALPHA)
            )


class IdVoteSmoother:
    """Display-level smoothing: show the id a tracklet has held most recently.

    The tracker can briefly relabel a tracklet (a 1-frame colour flip); voting
    over a short window suppresses that so the on-screen id doesn't blink.
    """

    def __init__(self, window: int = config.SELECT_VOTE_WINDOW) -> None:
        self._window = window
        self._hist: dict[int, deque[int]] = {}

    def smooth(
        self, selected: list[tuple[int, Detection]]
    ) -> list[tuple[int, Detection]]:
        """Return ``selected`` with each id replaced by its per-tracklet majority."""
        out: list[tuple[int, Detection]] = []
        seen: set[int] = set()
        for sid, det in selected:
            hist = self._hist.setdefault(det.track_id, deque(maxlen=self._window))
            hist.append(sid)
            voted = Counter(hist).most_common(1)[0][0]
            out.append((voted, det))
            seen.add(det.track_id)
        for tid in [t for t in self._hist if t not in seen]:
            del self._hist[tid]
        return out
