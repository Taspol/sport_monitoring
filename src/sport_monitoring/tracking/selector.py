"""Reference-frame registration.

Instead of asking the user to click, we take a reference frame, detect every
person, and register each one as a target: a stable id plus their pose-guided
colour identity. The tracker then follows all of them through the video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import config
from .color_id import keypoint_color
from ..perception.detector import PersonDetector
from ..perception.rtm_pose import RTMPoseEstimator

Box = tuple[int, int, int, int]


@dataclass
class SelectedTarget:
    """A registered player: a stable id, reference box and colour identity."""

    stable_id: int
    box: Box
    color: np.ndarray | None = None


def _identity_for(
    pose: RTMPoseEstimator, frame: np.ndarray, box: Box
) -> np.ndarray | None:
    """Pose-sample the reference jersey-colour histogram for a single box."""
    keypoints, scores = pose.estimate(frame, [box])
    if len(keypoints) == 0:
        return None
    points = [
        (int(x), int(y), float(s)) for (x, y), s in zip(keypoints[0], scores[0])
    ]
    return keypoint_color(frame, points)


def _read_frame(source: Path, index: int) -> tuple[int, np.ndarray | None]:
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = max(0, min(index, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return index, (frame if ok else None)


def _point_in_box(px: int, py: int, box: Box) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


_ONE_WINDOW = "Select ONE player to analyse  [click=pick  n/p=frame  ENTER=ok  ESC=cancel]"


def select_one_player(
    source: Path,
    person_confidence: float = config.DEFAULT_PERSON_CONFIDENCE,
    start_frame: int = 0,
) -> tuple[Box | None, int]:
    """Pop up the reference frame; user clicks one player to analyse.

    Returns ``(box, frame_index)`` for the chosen player, or ``(None, frame)`` if
    cancelled. Raises ``RuntimeError`` if no display is available.
    """
    frame_index, frame = _read_frame(source, start_frame)
    if frame is None:
        raise RuntimeError(f"Could not read frame {start_frame} from {source}")
    detector = PersonDetector(confidence=person_confidence)
    state = {"boxes": detector.detect_boxes(frame), "picked": None, "frame": frame}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for i, box in enumerate(state["boxes"]):
            if _point_in_box(x, y, box):
                state["picked"] = i
                break

    try:
        cv2.namedWindow(_ONE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_ONE_WINDOW, on_mouse)
    except cv2.error as exc:
        raise RuntimeError("Player selection needs a display.") from exc

    confirmed = False
    while True:
        canvas = state["frame"].copy()
        for i, (x1, y1, x2, y2) in enumerate(state["boxes"]):
            picked = i == state["picked"]
            col = (0, 255, 0) if picked else (180, 180, 180)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3 if picked else 1)
            cv2.putText(canvas, str(i), (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        msg = (
            f"picked {state['picked']} - ENTER to analyse"
            if state["picked"] is not None
            else "click the player to analyse"
        )
        cv2.putText(canvas, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(_ONE_WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and state["picked"] is not None:
            confirmed = True
            break
        if key == 27:
            break
        if key in (ord("n"), ord("p")):
            frame_index, nf = _read_frame(source, frame_index + (1 if key == ord("n") else -1))
            if nf is not None:
                state["frame"] = nf
                state["boxes"] = detector.detect_boxes(nf)
                state["picked"] = None

    cv2.destroyWindow(_ONE_WINDOW)
    if not confirmed or state["picked"] is None:
        return None, frame_index
    return state["boxes"][state["picked"]], frame_index


def auto_select_player(
    source: Path,
    person_confidence: float = config.DEFAULT_PERSON_CONFIDENCE,
    start_frame: int = 0,
) -> tuple[Box | None, int]:
    """Pick the jumping athlete with no UI: the most *vertically mobile* person.

    A headless stand-in for :func:`select_one_player` for batch / benchmark runs.
    Picking the largest box fails on real game footage -- a stationary foreground
    spectator is often the biggest. The athlete this tool cares about is the one who
    *jumps*, so we run a cheap detection-only ByteTrack pass and choose the track
    whose foot line (box bottom) sweeps the widest vertical range -- the ballistic
    signature of a jump-and-land. The reference frame returned is that track's first
    appearance (>= ``start_frame``), so processing starts before the jump.

    Returns ``(box, frame)`` for the chosen player, or ``(None, start_frame)`` if no
    one is detected.
    """
    from .. import video

    detector = PersonDetector(confidence=person_confidence)
    detector.reset()
    tracks: dict[int, list[tuple[int, Box]]] = {}
    for idx, frame in video.read_frames(source, start_frame=start_frame):
        for d in detector.detect(frame):
            tracks.setdefault(d.track_id, []).append((idx, d.box))
    if not tracks:
        return None, start_frame

    longest = max(len(v) for v in tracks.values())
    min_frames = max(15, int(0.1 * longest))

    def foot_excursion(items: list[tuple[int, Box]]) -> float:
        bottoms = [b[3] for _, b in items]
        return max(bottoms) - min(bottoms)

    eligible = {t: v for t, v in tracks.items() if len(v) >= min_frames}
    pool = eligible or tracks
    best_id = max(pool, key=lambda t: foot_excursion(pool[t]))
    ref_idx, ref_box = min(pool[best_id], key=lambda it: it[0])
    return ref_box, ref_idx


def register_all_players(
    source: Path,
    person_confidence: float = config.DEFAULT_PERSON_CONFIDENCE,
    start_frame: int = 0,
) -> tuple[list[SelectedTarget], int]:
    """Register every person detected in the reference frame.

    Returns ``(targets, frame_index)``: one :class:`SelectedTarget` per detected
    person (id assigned in detection order, with their captured colour identity),
    and the reference frame index. Processing should start at that frame so each
    target's box matches a real detection straight away.
    """
    frame_index, frame = _read_frame(source, start_frame)
    if frame is None:
        raise RuntimeError(f"Could not read frame {start_frame} from {source}")

    detector = PersonDetector(confidence=person_confidence)
    pose = RTMPoseEstimator()
    boxes = detector.detect_boxes(frame)

    targets = [
        SelectedTarget(
            stable_id=stable_id,
            box=box,
            color=_identity_for(pose, frame, box),
        )
        for stable_id, box in enumerate(boxes, start=1)
    ]
    return targets, frame_index
