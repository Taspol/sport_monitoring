"""Fused multi-cue player tracker.

Combines three orthogonal signals to assign stable ids to ByteTrack detections:

  1. **ByteTrack continuity** – while a tracklet keeps its raw id we trust it
     (near-zero cost).  This is the strongest signal inside an uninterrupted run.
  2. **Colour identity** – HS histogram of the torso region (pose-guided when
     keypoints are available, bounding-box crop otherwise).  Separates players
     who get different raw ids after occlusion.
  3. **Trajectory (Kalman)** – a constant-velocity Kalman filter per stable id
     predicts the next centroid; detections close to the prediction are cheap.
     This anchors re-identification to *where the player should be*, so two
     same-coloured teammates can still be told apart by position.

Assignment is done with the **Hungarian algorithm** (scipy.optimize.linear_sum_assignment)
over a cost matrix that blends the three cues.  Costs outside the acceptance
gates are clamped to a large value so they are never chosen.

The trajectory buffer stored per id is also exposed for downstream overlay
drawing (``TrajectoryStore``).

Config knobs are in ``config.py`` under the ``FUSED_*`` prefix.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .. import config
from ..perception.detector import Detection

if TYPE_CHECKING:
    pass

Box = tuple[int, int, int, int]
Point = tuple[int, int, float]



class _Kalman:
    """Constant-velocity Kalman filter tracking the centroid of one player."""

    def __init__(self, cx: float, cy: float) -> None:
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([50.0, 50.0, 100.0, 100.0])
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        q = config.FUSED_KALMAN_PROCESS_NOISE
        self.Q = np.diag([q, q, q * 4, q * 4])
        r = config.FUSED_KALMAN_MEAS_NOISE
        self.R = np.diag([r, r])

    def predict(self) -> tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float) -> None:
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])



@dataclass
class _Track:
    stable_id: int
    raw_id: int | None
    kalman: _Kalman
    color_hist: np.ndarray | None
    box: Box
    last_seen: int
    lost_frames: int = 0
    hits: int = 0
    confirmed: bool = False
    color_suspect_frames: int = 0
    trajectory: deque = field(default_factory=lambda: deque(
        maxlen=config.FUSED_TRAJ_LEN))

    def predict_centroid(self) -> tuple[float, float]:
        return self.kalman.predict()

    def update_centroid(self, cx: float, cy: float) -> None:
        self.kalman.update(cx, cy)



def _hungarian(cost: np.ndarray, threshold: float
               ) -> list[tuple[int, int]]:
    """Run scipy Hungarian on ``cost`` and return pairs below ``threshold``."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        pairs: list[tuple[int, int]] = []
        used_c: set[int] = set()
        for r in range(cost.shape[0]):
            best_c = int(np.argmin(cost[r]))
            if best_c not in used_c and cost[r, best_c] < threshold:
                pairs.append((r, best_c))
                used_c.add(best_c)
        return pairs

    row_ind, col_ind = linear_sum_assignment(cost)
    return [
        (int(r), int(c))
        for r, c in zip(row_ind, col_ind)
        if cost[r, c] < threshold
    ]



def _centroid(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _appearance_from_box(frame: np.ndarray, box: Box) -> np.ndarray:
    """HS histogram of the torso sub-region of the bounding box.

    Uses the same bin configuration as :func:`color_id.keypoint_color`
    (``config.COLOR_ID_HIST_BINS``) so that all histograms in the system
    have a consistent size and can be compared with ``cv2.compareHist``.
    """
    bins = list(config.COLOR_ID_HIST_BINS)
    n_bins = bins[0] * bins[1]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    tx1 = x1 + int(0.2 * w)
    tx2 = x2 - int(0.2 * w)
    ty1 = y1 + int(0.15 * h)
    ty2 = y1 + int(0.60 * h)
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    hist = cv2.GaussianBlur(hist, (3, 3), 0)
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _color_cost(ref: np.ndarray | None, cur: np.ndarray | None) -> float:
    """Colour distance in [0, 1]; 0 = identical, 1 = orthogonal.

    Returns ``FUSED_COLOR_UNKNOWN_COST`` when either histogram is missing or
    the two arrays have different sizes (avoids the OpenCV assertion crash).
    """
    if ref is None or cur is None:
        return config.FUSED_COLOR_UNKNOWN_COST
    a = np.asarray(ref, dtype=np.float32).ravel()
    b = np.asarray(cur, dtype=np.float32).ravel()
    if a.shape != b.shape:
        return config.FUSED_COLOR_UNKNOWN_COST
    corr = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    return 1.0 - max(0.0, corr)


def _position_cost(pred: tuple[float, float], cx: float, cy: float,
                   diag: float) -> float:
    """Normalised Euclidean distance to Kalman prediction, capped at 1.0.

    Capping prevents a drifted Kalman (player off-screen a long time) from
    making the position term so large that a colour-good re-link is rejected.
    """
    dist = math.hypot(pred[0] - cx, pred[1] - cy)
    return min(1.0, dist / max(1.0, diag))



@dataclass
class MatchResult:
    selected: list[tuple[int, Detection]]
    others: list[tuple[int, Detection]]



class TrajectoryStore:
    """Holds the centroid history + latest velocity for every stable id."""

    def __init__(self) -> None:
        self._traj: dict[int, deque] = {}
        self._vel: dict[int, tuple[float, float]] = {}

    def push(self, stable_id: int, cx: float, cy: float,
             vx: float = 0.0, vy: float = 0.0) -> None:
        if stable_id not in self._traj:
            self._traj[stable_id] = deque(maxlen=config.FUSED_TRAJ_LEN)
        self._traj[stable_id].append((cx, cy))
        self._vel[stable_id] = (vx, vy)

    def get(self, stable_id: int) -> list[tuple[float, float]]:
        return list(self._traj.get(stable_id, []))

    def velocity(self, stable_id: int) -> tuple[float, float]:
        """Latest Kalman velocity (vx, vy) in pixels/frame for ``stable_id``."""
        return self._vel.get(stable_id, (0.0, 0.0))

    def all(self) -> dict[int, list[tuple[float, float]]]:
        return {sid: list(q) for sid, q in self._traj.items()}



class FusedTracker:
    """Assign stable ids to ByteTrack detections with a two-stage cascade.

    Each frame:

      **Stage A -- ByteTrack continuity (trusted, hysteretic).**  For every track
      that still has a bound raw id, take the detection carrying that same raw id
      and keep it -- with *no* re-decision -- unless it (a) jumped farther than
      ``FUSED_CONTINUITY_POS_GATE`` from the Kalman prediction (a ByteTrack id
      swap) or (b) turned a clearly different jersey colour.  This handles the
      stable common case deterministically, which is what removes per-frame
      jitter and stops a ByteTrack swap from dragging an id onto the wrong player.

      **Stage B -- Hungarian re-association (leftovers only).**  Tracks that lost
      their raw id and detections with a new raw id are matched by colour + Kalman
      position via the Hungarian algorithm, behind a hard colour gate and a
      position gate, so a re-link can never grab a different-jersey / far player.

      **Confirmation.**  A detection that matches nothing creates a *tentative*
      track.  Tentative tracks are not returned (not drawn, no shown id) until
      they have been matched ``FUSED_CONFIRM_FRAMES`` consecutive frames -- this
      removes the flood of 1-frame ghost ids from flickering detections.

    Registered players (``immortal_ids``) are confirmed from creation and never
    expire; while lost they are re-acquired in Stage B through the wider
    ``FUSED_RELINK_POS_GATE`` window.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._frame_idx = -1
        self.trajectories = TrajectoryStore()
        self.immortal_ids: set[int] = set()


    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        keypoint_colors: dict[int, np.ndarray | None] | None = None,
    ) -> list[tuple[int, Detection]]:
        """Match detections to stable ids.  Returns ``(stable_id, detection)``
        for CONFIRMED tracks only (tentative ghost ids are withheld)."""
        self._frame_idx += 1
        diag = float(np.hypot(*frame.shape[:2]))
        kp_colors = keypoint_colors or {}

        if not detections:
            self._age_tracks()
            self._expire_tracks()
            return []

        det_colors: dict[int, np.ndarray | None] = {}
        for det in detections:
            kp_hist = kp_colors.get(det.track_id)
            det_colors[det.track_id] = (
                kp_hist if kp_hist is not None
                else _appearance_from_box(frame, det.box)
            )

        active = [
            t for t in self._tracks.values()
            if t.lost_frames <= config.FUSED_MAX_LOST_FRAMES
            or t.stable_id in self.immortal_ids
        ]
        pred = {t.stable_id: t.predict_centroid() for t in active}
        det_by_raw: dict[int, list[Detection]] = {}
        for det in detections:
            det_by_raw.setdefault(det.track_id, []).append(det)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        matches: list[tuple[_Track, Detection]] = []

        for t in active:
            if t.raw_id is None:
                continue
            cands = det_by_raw.get(t.raw_id)
            if not cands:
                continue
            det = cands[0]
            if det.track_id in matched_dets:
                continue
            cx, cy = _centroid(det.box)
            c_traj = _position_cost(pred[t.stable_id], cx, cy, diag)
            if c_traj > config.FUSED_CONTINUITY_POS_GATE:
                continue
            if t.lost_frames < config.FUSED_LOST_RELAX_AFTER:
                c_color = _color_cost(t.color_hist, det_colors.get(det.track_id))
                if t.color_hist is not None and c_color > config.FUSED_COLOR_HARD_GATE:
                    continue
            matches.append((t, det))
            matched_tracks.add(t.stable_id)
            matched_dets.add(det.track_id)

        self._resolve_swaps(matches, pred, det_colors, diag)

        rem_tracks = [t for t in active if t.stable_id not in matched_tracks]
        rem_dets = [d for d in detections if d.track_id not in matched_dets]
        if rem_tracks and rem_dets:
            wc, wt = config.FUSED_W_COLOR, config.FUSED_W_TRAJ
            wsum = wc + wt
            cost = np.full((len(rem_tracks), len(rem_dets)), 1e6, dtype=np.float64)
            for ti, t in enumerate(rem_tracks):
                relax = (
                    t.stable_id in self.immortal_ids
                    and t.lost_frames >= config.FUSED_LOST_RELAX_AFTER
                )
                pos_gate = (
                    config.FUSED_RELINK_POS_GATE if relax
                    else config.FUSED_REASSOC_POS_GATE
                )
                for di, d in enumerate(rem_dets):
                    cx, cy = _centroid(d.box)
                    c_traj = _position_cost(pred[t.stable_id], cx, cy, diag)
                    if c_traj > pos_gate:
                        continue
                    det_color = det_colors.get(d.track_id)
                    c_color = _color_cost(t.color_hist, det_color)
                    if (t.color_hist is not None and det_color is not None
                            and c_color > config.FUSED_COLOR_HARD_GATE):
                        continue
                    cost[ti, di] = (wc * c_color + wt * c_traj) / wsum
            for ti, di in _hungarian(cost, config.FUSED_REASSOC_MAX_COST):
                t, d = rem_tracks[ti], rem_dets[di]
                matches.append((t, d))
                matched_tracks.add(t.stable_id)
                matched_dets.add(d.track_id)

        for t, det in matches:
            self._update_track(t, det, det_colors, frame)
        for t in active:
            if t.stable_id not in matched_tracks:
                t.lost_frames += 1
                t.hits = 0
                t.color_suspect_frames = 0

        new_tracks: list[tuple[_Track, Detection]] = []
        for det in detections:
            if det.track_id in matched_dets:
                continue
            if (det.confidence is not None
                    and det.confidence < config.FUSED_NEW_TRACK_MIN_CONF):
                continue
            sid = self._create_track(det, det_colors, frame)
            new_tracks.append((self._tracks[sid], det))

        self._expire_tracks()

        return [
            (t.stable_id, det)
            for t, det in (matches + new_tracks)
            if t.confirmed or t.stable_id in self.immortal_ids
        ]

    @property
    def active_ids(self) -> set[int]:
        return {
            sid for sid, t in self._tracks.items()
            if t.lost_frames == 0
        }


    def _update_track(
        self,
        track: _Track,
        det: Detection,
        det_colors: dict[int, np.ndarray | None],
        frame: np.ndarray,
    ) -> None:
        cx, cy = _centroid(det.box)
        track.update_centroid(cx, cy)
        track.raw_id = det.track_id
        track.box = det.box
        track.last_seen = self._frame_idx
        track.lost_frames = 0
        track.trajectory.append((cx, cy))
        vx, vy = track.kalman.velocity
        self.trajectories.push(track.stable_id, cx, cy, vx, vy)

        track.hits += 1
        if not track.confirmed and (
            track.hits >= config.FUSED_CONFIRM_FRAMES
            or track.stable_id in self.immortal_ids
        ):
            track.confirmed = True

        new_color = det_colors.get(det.track_id)
        if track.color_hist is not None and new_color is not None:
            sim = 1.0 - _color_cost(track.color_hist, new_color)
            suspect = sim < config.FUSED_COLOR_WATCHDOG_MIN
        else:
            sim = None
            suspect = False

        if suspect:
            track.color_suspect_frames += 1
        else:
            track.color_suspect_frames = 0
            if new_color is not None:
                if track.color_hist is None:
                    track.color_hist = new_color
                else:
                    alpha = config.FUSED_COLOR_EMA_ALPHA
                    track.color_hist = (
                        (1 - alpha) * track.color_hist + alpha * new_color
                    ).astype(np.float32)


    def _cross_color_ok(
        self, track: _Track, det: Detection,
        det_colors: dict[int, np.ndarray | None],
    ) -> bool:
        """A crossed re-assignment must not violate the hard colour gate."""
        if track.color_hist is None:
            return True
        dc = det_colors.get(det.track_id)
        if dc is None:
            return True
        return _color_cost(track.color_hist, dc) <= config.FUSED_COLOR_HARD_GATE

    def _resolve_swaps(
        self,
        matches: list[tuple[_Track, Detection]],
        pred: dict[int, tuple[float, float]],
        det_colors: dict[int, np.ndarray | None],
        diag: float,
    ) -> None:
        """Undo crossed continuity matches using the Kalman trajectory.

        For every pair of Stage-A matches, if swapping their detections fits both
        tracks' *predicted* positions better by at least ``FUSED_SWAP_MARGIN`` (and
        the swap doesn't break the colour gate), swap them.  This keeps a
        still-tracked id on its own player when ByteTrack swaps two crossing ids.
        """
        n = len(matches)
        if n < 2:
            return
        for _ in range(3):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    ti, di = matches[i]
                    tj, dj = matches[j]
                    pi, pj = pred[ti.stable_id], pred[tj.stable_id]
                    ci, cj = _centroid(di.box), _centroid(dj.box)
                    direct = (_position_cost(pi, ci[0], ci[1], diag)
                              + _position_cost(pj, cj[0], cj[1], diag))
                    crossed = (_position_cost(pi, cj[0], cj[1], diag)
                               + _position_cost(pj, ci[0], ci[1], diag))
                    if (crossed + config.FUSED_SWAP_MARGIN < direct
                            and self._cross_color_ok(ti, dj, det_colors)
                            and self._cross_color_ok(tj, di, det_colors)):
                        matches[i] = (ti, dj)
                        matches[j] = (tj, di)
                        changed = True
            if not changed:
                return

    def _create_track(
        self,
        det: Detection,
        det_colors: dict[int, np.ndarray | None],
        frame: np.ndarray,
    ) -> int:
        cx, cy = _centroid(det.box)
        sid = self._next_id
        self._next_id += 1
        traj: deque = deque(maxlen=config.FUSED_TRAJ_LEN)
        traj.append((cx, cy))
        self._tracks[sid] = _Track(
            stable_id=sid,
            raw_id=det.track_id,
            kalman=_Kalman(cx, cy),
            color_hist=det_colors.get(det.track_id),
            box=det.box,
            last_seen=self._frame_idx,
            hits=1,
            trajectory=traj,
        )
        self.trajectories.push(sid, cx, cy)
        return sid

    def _age_tracks(self) -> None:
        for t in self._tracks.values():
            t.kalman.predict()
            t.lost_frames += 1
            if not t.confirmed:
                t.hits = 0

    def _expire_tracks(self) -> None:
        """Remove tracks that have been lost too long.

        Tracks in ``immortal_ids`` (registered players) are never expired; they
        are re-linked the moment the player reappears.  A *tentative* (still
        unconfirmed) track is dropped quickly -- it was probably detection noise.
        """
        dead = []
        for sid, t in self._tracks.items():
            if sid in self.immortal_ids:
                continue
            limit = (
                config.FUSED_MAX_LOST_FRAMES if t.confirmed
                else config.FUSED_TENTATIVE_MAX_LOST
            )
            if t.lost_frames > limit:
                dead.append(sid)
        for sid in dead:
            del self._tracks[sid]



@dataclass
class _RegisteredTarget:
    """A player registered at the reference frame."""
    stable_id: int
    box: Box
    color: np.ndarray | None = None


class FusedSelectedTracker:
    """Interactive-mode tracker backed by FusedTracker.

    Wraps ``FusedTracker`` so that:
    * Only the registered stable ids are in 'selected'; all others go to 'others'.
    * The reference colour from the registration frame seeds the Kalman track's
      ``color_hist`` before any frames are processed.
    """

    def __init__(self, targets) -> None:
        self._fused = FusedTracker()
        self._registered_ids: set[int] = set()
        for t in targets:
            cx, cy = _centroid(t.box)
            sid = self._fused._next_id
            self._fused._next_id += 1
            traj: deque = deque(maxlen=config.FUSED_TRAJ_LEN)
            traj.append((cx, cy))
            color = getattr(t, "color", None)
            self._fused._tracks[sid] = _Track(
                stable_id=sid,
                raw_id=None,
                kalman=_Kalman(cx, cy),
                color_hist=color,
                box=t.box,
                last_seen=0,
                hits=config.FUSED_CONFIRM_FRAMES,
                confirmed=True,
                trajectory=traj,
            )
            self._registered_ids.add(sid)
        track_list = list(self._fused._tracks.values())
        for i, t in enumerate(targets):
            old_sid = track_list[i].stable_id
            track_list[i].stable_id = t.stable_id
            self._fused._tracks.pop(old_sid)
            self._fused._tracks[t.stable_id] = track_list[i]
            if old_sid in self._registered_ids:
                self._registered_ids.discard(old_sid)
                self._registered_ids.add(t.stable_id)
        if self._fused._tracks:
            self._fused._next_id = max(self._fused._tracks) + 1
        self._fused.immortal_ids = set(self._registered_ids)

    @property
    def trajectories(self) -> TrajectoryStore:
        return self._fused.trajectories

    def reference_color(self, stable_id: int) -> np.ndarray | None:
        t = self._fused._tracks.get(stable_id)
        return t.color_hist if t is not None else None

    def registry(self) -> dict[int, np.ndarray]:
        return {
            sid: t.color_hist
            for sid, t in self._fused._tracks.items()
            if sid in self._registered_ids and t.color_hist is not None
        }

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        colors: dict[int, np.ndarray | None] | None = None,
    ) -> MatchResult:
        assigned = self._fused.update(detections, frame, colors)
        selected_ids = self._registered_ids
        selected: list[tuple[int, Detection]] = []
        others: list[tuple[int, Detection]] = []
        for sid, det in assigned:
            if sid in selected_ids:
                selected.append((sid, det))
            else:
                others.append((sid, det))
        return MatchResult(selected=selected, others=others)
