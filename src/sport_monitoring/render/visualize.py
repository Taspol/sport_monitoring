"""Draw multi-person pose skeletons onto frames using OpenCV.

RTMPose returns 17 COCO keypoints per person, which we render as a skeleton
coloured by the player's stable track id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .. import config

if TYPE_CHECKING:
    from ..tracking.tracker import TrajectoryStore



@dataclass
class PersonPose:
    """One tracked player: a stable id, bounding box, and absolute keypoints.

    ``points`` are ``(x, y, score)`` in full-frame pixel coordinates;
    ``None`` means pose estimation produced nothing for that player this frame.
    """

    track_id: int
    box: tuple[int, int, int, int]
    points: list[tuple[int, int, float]] | None
    ref_color: tuple[int, int, int] | None = None
    cur_color: tuple[int, int, int] | None = None
    similarity: float | None = None
    raw_id: int | None = None
    suspect_frames: int = 0
    confidence: float | None = None

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

_PERSON_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
]


def _person_color(index: int) -> tuple[int, int, int]:
    return _PERSON_COLORS[index % len(_PERSON_COLORS)]


_REF_HEIGHT = 720.0


def _ui_scale(frame: np.ndarray) -> float:
    """Overlay scale: proportional to frame height, times a manual multiplier."""
    return config.OVERLAY_SCALE * max(0.6, frame.shape[0] / _REF_HEIGHT)


def _th(base: float, scale: float) -> int:
    """Scaled line/text thickness, never below 1px."""
    return max(1, round(base * scale))


def _draw_identity_card(frame: np.ndarray, person: "PersonPose") -> None:
    """Below the box: present colour swatches (ref + current) and similarity %."""
    if person.ref_color is None and person.cur_color is None:
        return
    s = _ui_scale(frame)
    side = round(14 * s)
    gap = round(16 * s)
    x1, _, _, y2 = person.box
    y = min(y2 + round(4 * s), frame.shape[0] - side - round(4 * s))
    cx = x1

    def chip(cx: int, colour: tuple[int, int, int]) -> int:
        cv2.rectangle(frame, (cx, y), (cx + side, y + side), colour, -1)
        cv2.rectangle(frame, (cx, y), (cx + side, y + side), (255, 255, 255), _th(1, s))
        return cx + gap

    if person.ref_color is not None:
        cx = chip(cx, person.ref_color)
    if person.cur_color is not None:
        cx = chip(cx, person.cur_color)
    if person.similarity is not None:
        cv2.putText(
            frame, f"{int(person.similarity * 100)}%", (cx + round(2 * s), y + side - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (255, 255, 255), _th(1, s), cv2.LINE_AA,
        )


def _draw_debug_label(
    frame: np.ndarray,
    person: "PersonPose",
    dim: bool,
) -> None:
    """Draw a compact debug info panel anchored to the top-left of the box.

    Shows for every detected person:
      • Stable ID  (or raw ByteTrack ID if no stable id)
      • Raw ByteTrack tracklet ID
      • Colour-similarity score (ref ↔ current frame)
      • Watchdog suspect-frame counter (orange when > 0, red when ≥ threshold)
      • YOLO detection confidence
    """
    s = _ui_scale(frame)
    x1, y1, x2, y2 = person.box
    line_h = round(14 * s)
    pad = round(3 * s)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.38 * s
    fth = max(1, round(0.8 * s))

    lines: list[tuple[str, tuple[int, int, int]]] = []

    sid_str = f"sID:{person.track_id}"
    raw_str = f" raw:{person.raw_id}" if person.raw_id is not None else ""
    lines.append((sid_str + raw_str, (220, 220, 220) if dim else (255, 255, 255)))

    if person.similarity is not None:
        sim_pct = int(person.similarity * 100)
        if sim_pct >= 60:
            sim_col: tuple[int, int, int] = (100, 255, 100)
        elif sim_pct >= 35:
            sim_col = (0, 200, 255)
        else:
            sim_col = (0, 80, 255)
        lines.append((f"sim:{sim_pct}%", sim_col))
    else:
        lines.append(("sim:--", (160, 160, 160)))

    watchdog_str = ""
    if person.suspect_frames > 0:
        w_col: tuple[int, int, int] = (
            (0, 60, 255) if person.suspect_frames >= config.FUSED_COLOR_WATCHDOG_FRAMES
            else (0, 165, 255)
        )
        watchdog_str = f" ⚠{person.suspect_frames}"
    else:
        w_col = (180, 180, 180)
    conf_str = f"conf:{int((person.confidence or 0)*100)}%" if person.confidence else ""
    lines.append(((conf_str + watchdog_str).strip() or "watchdog:ok", w_col))

    panel_w = round(90 * s)
    panel_h = len(lines) * line_h + pad * 2
    x0 = x1
    y0 = max(0, y1 - panel_h - pad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for li, (text, col) in enumerate(lines):
        ty = y0 + pad + (li + 1) * line_h - round(2 * s)
        cv2.putText(frame, text, (x0 + pad, ty), font, fscale, col, fth, cv2.LINE_AA)


_KP_NOSE, _KP_LSHO, _KP_RSHO, _KP_LHIP, _KP_RHIP = 0, 5, 6, 11, 12


def _body_yaw_depth(
    box: tuple[int, int, int, int],
    pts: list[tuple[int, int, float]] | None,
    min_visibility: float,
) -> tuple[float, float]:
    """Estimate body **yaw** and **sagittal depth** for a person from 2D keypoints.

    Lightweight monocular 3D cue (no model): a human's shoulder breadth is a roughly
    fixed fraction of stature, so the *foreshortening* of the shoulder line tells us
    how far the torso is turned away from the camera --
    ``cos(yaw) = observed_shoulder_width / expected_frontal_width``. The turn
    direction (sign) is taken from the nose offset relative to the shoulder centre.
    Hips are used as a fallback when the shoulders are not both visible.

    Returns ``(yaw_radians, depth_px)``. When no usable joints exist (or the player
    faces the camera) yaw is ``0`` and the box degrades to a plain axis-aligned
    extrusion -- the previous behaviour.
    """
    x1, y1, x2, y2 = box
    height = max(1, y2 - y1)
    depth = max(6.0, config.BOX3D_DEPTH_RATIO * height)

    def vis(i: int) -> bool:
        return pts is not None and i < len(pts) and pts[i][2] >= min_visibility

    if vis(_KP_LSHO) and vis(_KP_RSHO):
        a, b = pts[_KP_LSHO], pts[_KP_RSHO]
    elif vis(_KP_LHIP) and vis(_KP_RHIP):
        a, b = pts[_KP_LHIP], pts[_KP_RHIP]
    else:
        return 0.0, depth

    width_obs = math.hypot(b[0] - a[0], b[1] - a[1])
    if width_obs < 1.0:
        return 0.0, depth
    width_max = max(width_obs, config.BOX3D_SHOULDER_RATIO * height)
    yaw = math.acos(min(1.0, width_obs / width_max))
    if vis(_KP_NOSE) and pts[_KP_NOSE][0] < (a[0] + b[0]) / 2:
        yaw = -yaw
    return yaw, depth


def _draw_cuboid(
    frame: np.ndarray, box: tuple[int, int, int, int],
    col: tuple[int, int, int], th: int, yaw: float = 0.0,
    depth: float | None = None,
) -> None:
    """Draw an orientation-aware pseudo-3D cuboid around a 2D person box (``--3d``).

    The cuboid is the 2D box extruded back by ``depth`` pixels along a fixed
    weak-perspective camera axis (up-and-right), then **yaw-rotated** about the
    vertical so the box turns with the player's body (see :func:`_body_yaw_depth`).
    The front face is bright, the back face + depth edges dimmed, and the top face
    filled translucently so the volume reads as solid. With ``yaw == 0`` this is the
    plain axis-aligned extrusion, so missing-keypoint cases degrade gracefully.
    """
    x1, y1, x2, y2 = box
    if depth is None:
        depth = max(6.0, config.BOX3D_DEPTH_RATIO * (y2 - y1))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    dx, dy = config.BOX3D_DEPTH_DX, config.BOX3D_DEPTH_DY

    def proj(px: float, py: float, pz: float) -> tuple[int, int]:
        px_r = px * cos_y + pz * sin_y
        pz_r = -px * sin_y + pz * cos_y
        return int(round(cx + px_r + pz_r * dx)), int(round(cy - py - pz_r * dy))

    hw, hh = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    front = np.array([proj(-hw, hh, 0), proj(hw, hh, 0),
                      proj(hw, -hh, 0), proj(-hw, -hh, 0)], dtype=np.int32)
    back = np.array([proj(-hw, hh, depth), proj(hw, hh, depth),
                     proj(hw, -hh, depth), proj(-hw, -hh, depth)], dtype=np.int32)
    back_col = tuple(int(c * 0.45) for c in col)
    back_th = max(1, th - 1)

    cv2.polylines(frame, [back], True, back_col, back_th, cv2.LINE_AA)
    for (fx, fy), (bx, by) in zip(front, back):
        cv2.line(frame, (int(fx), int(fy)), (int(bx), int(by)),
                 back_col, back_th, cv2.LINE_AA)
    top = np.array([front[0], front[1], back[1], back[0]], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, top, col, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [front], True, col, th, cv2.LINE_AA)


def _draw_live_arrow(
    frame: np.ndarray, person: "PersonPose", store, color: tuple[int, int, int],
) -> None:
    """Real-time move arrow for one person, anchored to their box centre.

    Direction/length come from this person's own frame-to-frame centroid step
    (the last two trajectory points), scaled by ``TRAJ_ARROW_3D_GAIN`` -- so the
    arrow follows each detected person live, turning and growing with their motion.
    """
    if store is None:
        return
    pts = store.get(person.track_id)
    if len(pts) < 2:
        return
    dx = pts[-1][0] - pts[-2][0]
    dy = pts[-1][1] - pts[-2][1]
    if math.hypot(dx, dy) < config.TRAJ_MIN_SPEED:
        return
    s = _ui_scale(frame)
    x1, y1, x2, y2 = person.box
    ax, ay = (x1 + x2) // 2, (y1 + y2) // 2
    ex = int(ax + dx * config.TRAJ_ARROW_3D_GAIN)
    ey = int(ay + dy * config.TRAJ_ARROW_3D_GAIN)
    cv2.arrowedLine(frame, (ax, ay), (ex, ey), color, _th(2, s),
                    cv2.LINE_AA, tipLength=0.35)


def _draw_person(
    frame: np.ndarray, person: "PersonPose", min_visibility: float,
    dim: bool, debug: bool = False, three_d: bool = False, store=None,
) -> None:
    """Draw one player. ``dim`` = non-selected (grey, thin, no id).

    When ``three_d`` is set, the flat 2D box is replaced by a pseudo-3D cuboid and
    a live, per-person move arrow (from ``store``) is drawn instead of the global
    trajectory overlay.
    """
    s = _ui_scale(frame)
    x1, y1, x2, y2 = person.box
    base = 1 if dim else 2
    if dim:
        box_col = skel_col = (150, 150, 150)
    else:
        box_col = skel_col = _person_color(person.track_id)
    box_th = line_th = _th(base, s)
    joint_r = _th(base, s)

    if three_d:
        yaw, depth = _body_yaw_depth((x1, y1, x2, y2), person.points, min_visibility)
        _draw_cuboid(frame, (x1, y1, x2, y2), box_col, box_th, yaw, depth)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, box_th, cv2.LINE_AA)
    if not debug:
        lbl_col = (210, 210, 210) if dim else box_col
        lbl_scale = 0.4 if dim else 0.5
        cv2.putText(
            frame, f"ID {person.track_id}", (x1, max(0, y1 - round(6 * s))),
            cv2.FONT_HERSHEY_SIMPLEX, lbl_scale * s, lbl_col,
            _th(1 if dim else 2, s), cv2.LINE_AA,
        )
    if debug:
        _draw_debug_label(frame, person, dim)
    elif not three_d:
        _draw_identity_card(frame, person)

    if person.points:
        pts = person.points
        for start, end in POSE_CONNECTIONS:
            if start < len(pts) and end < len(pts):
                sx, sy, sv = pts[start]
                ex, ey, ev = pts[end]
                if sv >= min_visibility and ev >= min_visibility:
                    cv2.line(frame, (sx, sy), (ex, ey), skel_col, line_th, cv2.LINE_AA)
        joint_col = (200, 200, 200) if dim else (255, 255, 255)
        for x, y, vis in pts:
            if vis >= min_visibility:
                cv2.circle(frame, (x, y), joint_r, joint_col, -1, cv2.LINE_AA)

    if three_d:
        _draw_live_arrow(frame, person, store, box_col)


def _draw_header(frame: np.ndarray, num_people: int) -> None:
    s = _ui_scale(frame)
    cv2.putText(
        frame,
        f"People tracked: {num_people}",
        (round(12 * s), round(28 * s)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * s,
        (255, 255, 255),
        _th(2, s),
        cv2.LINE_AA,
    )


def draw_tracked_poses(
    frame_bgr: np.ndarray,
    people: list[PersonPose],
    min_visibility: float,
    others: list[PersonPose] | None = None,
    debug: bool = False,
    three_d: bool = False,
    store=None,
) -> np.ndarray:
    """Annotate the frame with selected players and (dimly) the non-selected ones.

    ``people`` are the selected players: a coloured box keyed to their id, an
    ``ID`` label, a skeleton, and an identity card (reference + current colour and
    similarity). ``others`` are non-selected people, drawn in grey with their
    skeleton and current colour swatch but no id.

    When ``debug=True`` every person (selected and others) gets a compact
    debug panel: stable ID, raw ByteTrack ID, colour-sim %, watchdog count,
    and YOLO confidence.
    """
    annotated = frame_bgr.copy()

    for person in others or []:
        _draw_person(annotated, person, min_visibility, dim=True, debug=debug,
                     three_d=three_d, store=store)
    for person in people:
        _draw_person(annotated, person, min_visibility, dim=False, debug=debug,
                     three_d=three_d, store=store)

    _draw_header(annotated, len(people))
    return annotated


def _corner_brackets(
    frame: np.ndarray, box: tuple[int, int, int, int],
    col: tuple[int, int, int], s: float,
) -> None:
    """Draw reticle-style corner brackets around a box (the 'tracked' marker)."""
    x1, y1, x2, y2 = box
    length = max(round(8 * s), int(min(x2 - x1, y2 - y1) * 0.28))
    t = _th(4, s)
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1),
    ):
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), col, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), col, t, cv2.LINE_AA)


def _draw_clean_person(
    frame: np.ndarray, person: "PersonPose", min_visibility: float, highlight: bool,
) -> None:
    """Box + skeleton + ID label only (no colour/identity card). Tracked = emphasised."""
    s = _ui_scale(frame)
    x1, y1, x2, y2 = person.box
    col = (0, 255, 255) if highlight else _person_color(person.track_id)
    box_th = _th(3 if highlight else 2, s)
    line_th = _th(3 if highlight else 2, s)
    joint_r = _th(3 if highlight else 2, s)

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, box_th, cv2.LINE_AA)
    if highlight:
        _corner_brackets(frame, person.box, col, s)

    label = f"ID {person.track_id}" + ("  TRACKED" if highlight else "")
    font, fscale, fth = cv2.FONT_HERSHEY_SIMPLEX, (0.55 if highlight else 0.5) * s, _th(2, s)
    (tw, tht), _ = cv2.getTextSize(label, font, fscale, fth)
    pad = round(3 * s)
    ly = max(tht + 2 * pad, y1 - round(6 * s))
    if highlight:
        cv2.rectangle(frame, (x1, ly - tht - 2 * pad), (x1 + tw + 2 * pad, ly), col, -1)
        cv2.putText(frame, label, (x1 + pad, ly - pad), font, fscale,
                    (0, 0, 0), fth, cv2.LINE_AA)
    else:
        cv2.putText(frame, label, (x1, ly - pad), font, fscale, col, fth, cv2.LINE_AA)

    if person.points:
        pts = person.points
        for start, end in POSE_CONNECTIONS:
            if start < len(pts) and end < len(pts):
                sx, sy, sv = pts[start]
                ex, ey, ev = pts[end]
                if sv >= min_visibility and ev >= min_visibility:
                    cv2.line(frame, (sx, sy), (ex, ey), col, line_th, cv2.LINE_AA)
        for x, y, vis in pts:
            if vis >= min_visibility:
                cv2.circle(frame, (x, y), joint_r, (255, 255, 255), -1, cv2.LINE_AA)


def draw_clean_poses(
    frame_bgr: np.ndarray,
    people: list[PersonPose],
    others: list[PersonPose] | None,
    min_visibility: float,
    highlight_id: int | None = None,
) -> np.ndarray:
    """Clean overlay: only bounding boxes, skeletons and ID labels for every player.

    No colour-value labels or identity cards. The tracked player (``highlight_id``)
    is emphasised with a bright box, corner-bracket reticle and a filled ID tag.
    Returns a new annotated frame (does not mutate the input).
    """
    annotated = frame_bgr.copy()
    for person in others or []:
        _draw_clean_person(annotated, person, min_visibility,
                           highlight=person.track_id == highlight_id)
    for person in people:
        _draw_clean_person(annotated, person, min_visibility,
                           highlight=person.track_id == highlight_id)
    return annotated


def draw_trajectories(
    frame: np.ndarray,
    store: "TrajectoryStore",
    min_points: int = 2,
    thickness: int = 2,
    fade: bool = True,
    three_d: bool = False,
) -> None:
    """Draw fading polyline trails for every stable id in ``store``.

    Each player gets a colour derived from their stable id (same palette as
    their bounding box).  Older points are drawn with lower opacity to give a
    motion-trail effect.  The overlay is drawn **in-place**.

    Args:
        frame: BGR image to annotate in place.
        store: :class:`~sport_monitoring.tracker.TrajectoryStore` from the
            :class:`~sport_monitoring.tracker.FusedTracker`.
        min_points: Skip players with fewer than this many recorded positions.
        thickness: Pixel width of the trail line.
        fade: If True, alpha-blend older segments to look like a fading trail.
        three_d: In ``--3d`` cuboid mode, suppress the trail path entirely and draw
            only a short, thin move arrow (the cuboid already conveys orientation).
    """
    if three_d:
        return
    s = _ui_scale(frame)
    th = max(1, round(thickness * s))

    for sid, pts in store.all().items():
        if len(pts) < min_points:
            continue
        color = _person_color(sid)
        n = len(pts)
        if fade and n > 1:
            overlay = frame.copy()
            for i in range(1, n):
                alpha = 0.2 + 0.8 * (i / (n - 1))
                x0, y0 = int(pts[i - 1][0]), int(pts[i - 1][1])
                x1, y1 = int(pts[i][0]), int(pts[i][1])
                cv2.line(overlay, (x0, y0), (x1, y1), color, th, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                overlay[:] = frame
        else:
            ipts = np.array(
                [[int(x), int(y)] for x, y in pts], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.polylines(frame, [ipts], isClosed=False, color=color,
                          thickness=th, lineType=cv2.LINE_AA)

        last_x, last_y = int(pts[-1][0]), int(pts[-1][1])
        cv2.circle(frame, (last_x, last_y), max(2, round(3 * s)),
                   color, -1, cv2.LINE_AA)

        vx, vy = store.velocity(sid)
        speed = math.hypot(vx, vy)
        if speed >= config.TRAJ_MIN_SPEED:
            ex = int(last_x + vx * config.TRAJ_ARROW_FRAMES)
            ey = int(last_y + vy * config.TRAJ_ARROW_FRAMES)
            cv2.arrowedLine(frame, (last_x, last_y), (ex, ey), color,
                            max(2, th + 1), cv2.LINE_AA, tipLength=0.3)
            cv2.putText(frame, f"{speed:.0f}px/f", (ex + round(4 * s), ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, color,
                        _th(1, s), cv2.LINE_AA)


def _viridis(t: float) -> tuple[int, int, int]:
    """Map ``t`` in [0, 1] to a BGR colour on the viridis ramp (dark -> bright).

    A small brightness floor keeps the oldest (t=0) skeleton visible against
    the dimmed background instead of fading to near-black.
    """
    inten = int(round((0.25 + 0.75 * max(0.0, min(1.0, t))) * 255))
    bgr = cv2.applyColorMap(
        np.array([[inten]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS
    )[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_pose_trail_legend(
    canvas: np.ndarray, title: str, s: float, show_risk: bool = False,
    show_ramp: bool = True,
) -> None:
    """Title plus a small 'old -> new' viridis colour bar (bottom-left)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (round(14 * s), round(34 * s)),
                font, 0.8 * s, (255, 255, 255), _th(2, s), cv2.LINE_AA)
    h = canvas.shape[0]
    bar_w, bar_h = round(220 * s), round(12 * s)
    x0, y0 = round(14 * s), h - round(34 * s)
    if show_ramp:
        for i in range(bar_w):
            col = _viridis(i / max(1, bar_w - 1))
            cv2.line(canvas, (x0 + i, y0), (x0 + i, y0 + bar_h), col, 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h),
                      (255, 255, 255), _th(1, s))
        cv2.putText(canvas, "old", (x0, y0 - round(5 * s)),
                    font, 0.45 * s, (220, 220, 220), _th(1, s), cv2.LINE_AA)
        cv2.putText(canvas, "new", (x0 + bar_w - round(28 * s), y0 - round(5 * s)),
                    font, 0.45 * s, (220, 220, 220), _th(1, s), cv2.LINE_AA)
    if show_risk:
        ry = y0 - round(22 * s)
        cv2.rectangle(canvas, (x0, ry), (x0 + bar_h, ry + bar_h),
                      (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "= injury-risk flagged",
                    (x0 + bar_h + round(8 * s), ry + bar_h),
                    font, 0.45 * s, (220, 220, 220), _th(1, s), cv2.LINE_AA)


def _pose_centroid(
    pts: list[tuple[int, int, float]], min_visibility: float
) -> tuple[int, int] | None:
    """Mean of a pose's visible keypoints (its on-screen anchor)."""
    xs = [x for x, _, v in pts if v >= min_visibility]
    ys = [y for _, y, v in pts if v >= min_visibility]
    if not xs:
        return None
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def _risk_clusters(
    risk_poses: list[tuple[int, list[tuple[int, int, float]]]], gap: int
) -> list[tuple[int, list[tuple[int, int, float]]]]:
    """Collapse runs of consecutive risk frames into ONE representative pose each.

    ``risk_poses`` is time-ordered ``(frame, points)`` for the flagged frames only.
    Frames within ``gap`` of each other belong to the same risk moment (cluster);
    the middle pose of each cluster is kept, so the trail shows one skeleton per
    distinct flagged moment instead of a dozen over-plotted copies of one moment.
    """
    if not risk_poses:
        return []
    clusters: list[list[tuple[int, list[tuple[int, int, float]]]]] = [[risk_poses[0]]]
    for fp in risk_poses[1:]:
        if fp[0] - clusters[-1][-1][0] <= gap:
            clusters[-1].append(fp)
        else:
            clusters.append([fp])
    return [cluster[len(cluster) // 2] for cluster in clusters]


def draw_pose_trajectory(
    background: np.ndarray,
    poses: list[tuple[int, list[tuple[int, int, float]]]],
    min_visibility: float,
    max_samples: int = 14,
    label: str = "",
    risk_frames: set[int] | None = None,
    risk_only: bool = False,
) -> np.ndarray:
    """Render a stroboscopic skeleton trail of one player onto a new canvas.

    ``poses`` is a time-ordered list of ``(frame_index, points)`` for the player.
    Up to ``max_samples`` poses are sampled evenly across the clip and drawn on a
    dimmed copy of ``background``, oldest first, coloured along the viridis ramp
    (dark = oldest, bright yellow = newest) so the whole motion reads as one still
    image. Poses whose frame is in ``risk_frames`` (an injury-risk flag was raised)
    are drawn **red** instead, and a small **move-trend arrow** is drawn at each
    pose pointing toward the next sampled pose. Returns the annotated canvas.

    When ``risk_only`` is set, ONLY the risk-flagged poses are kept and consecutive
    flagged frames are collapsed into clusters (see :func:`_risk_clusters`) -- one
    representative skeleton per distinct risk moment -- so the image shows the
    player's risk trajectory alone without over-plotting a single moment. Falls back
    to the full trail if no risk frames exist (so the output is never blank).
    """
    canvas = (background.astype(np.float32) * 0.32).astype(np.uint8)
    if not poses:
        return canvas
    s = _ui_scale(background)
    line_th = _th(2, s)
    joint_r = _th(2, s)
    risk_frames = risk_frames or set()
    red = (0, 0, 255)

    only_risk = risk_only and any(f in risk_frames for f, _ in poses)
    if only_risk:
        risk_poses = [(f, p) for f, p in poses if f in risk_frames]
        poses = _risk_clusters(risk_poses, config.RISK_CLUSTER_GAP)

    n = len(poses)
    k = min(max_samples, n)
    if k > 1:
        sample_idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    else:
        sample_idx = [0]

    centroids = [_pose_centroid(poses[j][1], min_visibility) for j in sample_idx]

    any_risk = False
    for order, j in enumerate(sample_idx):
        t = order / (k - 1) if k > 1 else 1.0
        frame_idx, pts = poses[j]
        risky = frame_idx in risk_frames
        any_risk = any_risk or risky
        col = red if risky else _viridis(t)
        if not pts:
            continue
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                ax, ay, av = pts[a]
                bx, by, bv = pts[b]
                if av >= min_visibility and bv >= min_visibility:
                    cv2.line(canvas, (ax, ay), (bx, by), col, line_th, cv2.LINE_AA)
        for x, y, vis in pts:
            if vis >= min_visibility:
                cv2.circle(canvas, (x, y), joint_r, col, -1, cv2.LINE_AA)

        here = centroids[order]
        nxt = next((centroids[m] for m in range(order + 1, k) if centroids[m]), None)
        if here and nxt:
            dx, dy = nxt[0] - here[0], nxt[1] - here[1]
            norm = math.hypot(dx, dy)
            if norm > 1.0:
                alen = round(20 * s)
                tip = (int(here[0] + dx / norm * alen), int(here[1] + dy / norm * alen))
                cv2.arrowedLine(canvas, here, tip, (0, 165, 255), _th(2, s),
                                cv2.LINE_AA, tipLength=0.45)

    suffix = "(risk only)" if only_risk else ""
    title = f"POSE TRAJECTORY {label} {suffix}".strip()
    _draw_pose_trail_legend(canvas, title, s, show_risk=any_risk, show_ramp=not only_risk)
    return canvas


def draw_trajectory_map(
    background: np.ndarray,
    paths: dict[int, list[tuple[float, float]]],
    label: str = "",
) -> np.ndarray:
    """Render the analysed player's movement trajectory as a still PNG.

    Each trajectory is a polyline of the player's centroid with a start dot, a
    direction **arrow** at its head, and the player's ``ID`` label. Returns a new
    canvas (does not mutate ``background``).
    """
    canvas = (background.astype(np.float32) * 0.32).astype(np.uint8)
    s = _ui_scale(background)
    th = _th(2, s)

    for sid, pts in sorted(paths.items()):
        if len(pts) < 2:
            continue
        color = _person_color(sid)
        ipts = np.array([[int(x), int(y)] for x, y in pts],
                        dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [ipts], isClosed=False, color=color,
                      thickness=th, lineType=cv2.LINE_AA)
        sx, sy = int(pts[0][0]), int(pts[0][1])
        cv2.circle(canvas, (sx, sy), max(2, round(3 * s)), color, -1, cv2.LINE_AA)
        ex, ey = int(pts[-1][0]), int(pts[-1][1])
        tail = pts[-6:]
        dx, dy = tail[-1][0] - tail[0][0], tail[-1][1] - tail[0][1]
        norm = math.hypot(dx, dy)
        if norm > 1.0:
            alen = round(24 * s)
            ax = int(ex + dx / norm * alen)
            ay = int(ey + dy / norm * alen)
            cv2.arrowedLine(canvas, (ex, ey), (ax, ay), color, th + 1,
                            cv2.LINE_AA, tipLength=0.4)
        cv2.putText(canvas, f"ID {sid}", (ex + round(6 * s), ey - round(6 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, color, _th(1, s), cv2.LINE_AA)

    title = f"TRAJECTORY MAP {label}".strip()
    cv2.putText(canvas, title, (round(14 * s), round(34 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), _th(2, s), cv2.LINE_AA)
    return canvas


_INJURY_ROWS = (
    ("Knee flex", "knee_flex_l", "knee_flex_r"),
    ("Knee valgus", "knee_valgus_l", "knee_valgus_r"),
    ("Hip flex", "hip_flex_l", "hip_flex_r"),
    ("Trunk lean", "trunk_lean", None),
)


def _fmt(v: float | None) -> str:
    return "--" if v is None else f"{v:+.0f}"


def draw_injury_overlay(
    frame: np.ndarray,
    points: list[tuple[int, int, float]],
    angles: dict[str, float | None],
    flags: dict[str, bool],
    min_visibility: float,
    label: str = "",
) -> None:
    """Draw the joint-angle panel and per-joint risk markers in place."""
    s = _ui_scale(frame)
    risky = (0, 0, 255)
    ok = (180, 255, 180)

    for kp_idx, fields in ((13, ("knee_flex_l", "knee_valgus_l")),
                           (14, ("knee_flex_r", "knee_valgus_r"))):
        if kp_idx < len(points):
            x, y, score = points[kp_idx]
            if score >= min_visibility:
                hot = any(flags.get(f) for f in fields)
                cv2.circle(frame, (x, y), round(8 * s), risky if hot else ok,
                           _th(2, s), cv2.LINE_AA)

    pad = round(6 * s)
    x0, y0 = round(12 * s), round(60 * s)
    row_h = round(18 * s)
    cv2.rectangle(frame, (x0 - pad, y0 - round(22 * s)),
                  (x0 + round(256 * s), y0 + round(96 * s)), (0, 0, 0), -1)
    cv2.putText(frame, f"INJURY ANALYSIS {label}".strip(), (x0, y0 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), _th(1, s), cv2.LINE_AA)
    n_flags = sum(1 for v in flags.values() if v)
    for i, (name, fl, fr) in enumerate(_INJURY_ROWS):
        y = y0 + round(16 * s) + i * row_h
        hot = flags.get(fl, False) or (fr is not None and flags.get(fr, False))
        right = "" if fr is None else f"  R {_fmt(angles.get(fr))}"
        text = f"{name:<11} L {_fmt(angles.get(fl))}{right}"
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s,
                    risky if hot else (220, 220, 220), _th(1, s), cv2.LINE_AA)
    cv2.putText(frame, f"RISK FLAGS: {n_flags}",
                (x0, y0 + round(16 * s) + len(_INJURY_ROWS) * row_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s,
                risky if n_flags else ok, _th(2, s), cv2.LINE_AA)


_RISK_COLORS = {
    "HIGH": (0, 0, 255),
    "MODERATE": (0, 165, 255),
    "LOW": (0, 200, 0),
}


def draw_risk_card(
    frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]],
    rating: str, title: str = "INJURY RISK",
) -> None:
    """Draw a compact, colour-coded risk panel in the top-right corner."""
    s = _ui_scale(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = round(8 * s)
    line_h = round(20 * s)
    col = _RISK_COLORS.get(rating, (200, 200, 200))
    panel_w = round(250 * s)
    panel_h = pad * 2 + round(28 * s) + len(lines) * line_h
    x1 = frame.shape[1] - panel_w - round(12 * s)
    y1 = round(12 * s)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + panel_w, y1 + panel_h), col, _th(2, s))
    cv2.putText(frame, f"{title}: {rating}", (x1 + pad, y1 + pad + round(18 * s)),
                font, 0.6 * s, col, _th(2, s), cv2.LINE_AA)
    y = y1 + pad + round(28 * s)
    for text, tcol in lines:
        y += line_h
        cv2.putText(frame, text, (x1 + pad, y), font, 0.45 * s, tcol,
                    _th(1, s), cv2.LINE_AA)


def draw_fall_banner(
    frame: np.ndarray, index: int, confidence: float, sudden: bool
) -> None:
    """Centre banner + red frame border flagging a detected fall / on-ground event."""
    h, w = frame.shape[:2]
    s = _ui_scale(frame)
    red = (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), red, _th(6, s), cv2.LINE_AA)
    kind = "FALL (sudden)" if sudden else "FALL / ON GROUND"
    text = f"{kind} #{index}   conf {int(confidence * 100)}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, _th(2, s))
    x = (w - tw) // 2
    pad = round(12 * s)
    y0 = round(78 * s)
    cv2.rectangle(frame, (x - pad, y0), (x + tw + pad, y0 + round(40 * s)), red, -1)
    cv2.putText(frame, text, (x, y0 + th + round(8 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), _th(2, s), cv2.LINE_AA)


def draw_jump_banner(
    frame: np.ndarray, index: int, score: int, fails: list[str]
) -> None:
    """Top-centre flag marking a detected jump and its landing LESS-subset score."""
    w = frame.shape[1]
    s = _ui_scale(frame)
    hot = score >= 4
    col = (0, 0, 255) if hot else (0, 200, 255)
    text = f"JUMP #{index}   LESS {score}/9"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, _th(2, s))
    x = (w - tw) // 2
    pad = round(12 * s)
    cv2.rectangle(frame, (x - pad, round(8 * s)),
                  (x + tw + pad, round(70 * s)), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, round(8 * s) + th + round(6 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, col, _th(2, s), cv2.LINE_AA)
    if fails:
        sub = "fail: " + ", ".join(fails[:4]) + (" ..." if len(fails) > 4 else "")
        cv2.putText(frame, sub, (x, round(63 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * s, (180, 180, 255), _th(1, s), cv2.LINE_AA)
