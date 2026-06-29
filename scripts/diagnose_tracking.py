"""Diagnostic: investigate ID swap / loss in basket_S1T1_pre.mp4.

Runs track-all mode with full per-frame logging of:
  - ByteTrack raw ids
  - Colour similarity (1 - colour_cost) between each track <-> detection pair
  - Kalman predicted vs actual position
  - Final Hungarian assignment decision
  - Whether hard-gate blocked a pair

Outputs:
  - output/diag_tracking.csv   (full per-frame log)
  - output/diag_frames/        (annotated debug frames)
  - Console summary of swap / loss events
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sport_monitoring import config
from sport_monitoring.video import probe, read_frames
from sport_monitoring.perception.detector import PersonDetector
from sport_monitoring.perception.rtm_pose import RTMPoseEstimator
from sport_monitoring.tracking.color_id import keypoint_color
from sport_monitoring.tracking.tracker import (
    FusedTracker, _centroid, _color_cost, _position_cost,
    _bytetrack_cost, _appearance_from_box,
)

# ── Config ─────────────────────────────────────────────────────────────────
VIDEO      = ROOT / "data" / "basket_S1T1_pre.mp4"
OUT_CSV    = ROOT / "output" / "diag_tracking.csv"
OUT_FRAMES = ROOT / "output" / "diag_frames"
FPS        = 19.98
DIAG_IDS   = {2, 3, 6}          # stable ids to watch
SWAP_SEC   = 3.0                 # reported swap time
WATCH_SECS = (2.0, 5.0)         # window to save annotated frames
# ───────────────────────────────────────────────────────────────────────────

WATCH_FRAMES = range(
    int(WATCH_SECS[0] * FPS),
    int(WATCH_SECS[1] * FPS) + 1,
)

OUT_FRAMES.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

COLORS = {
    2: (0, 255, 0),    # green
    3: (0, 165, 255),  # orange
    6: (255, 0, 0),    # blue
}
GENERIC = (200, 200, 200)


def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    iw = max(0, min(ax2,bx2)-max(ax1,bx1))
    ih = max(0, min(ay2,by2)-max(ay1,by1))
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1)
    ub = (bx2-bx1)*(by2-by1)
    return inter / (ua+ub-inter) if (ua+ub-inter)>0 else 0.0


def draw_debug_frame(
    frame: np.ndarray,
    frame_idx: int,
    tracks,
    detections,
    det_colors,
    predicted,
    assignment: dict[int,int],  # stable_id -> det index
    blocked: set[tuple[int,int]],  # (track_idx, det_idx) pairs blocked by hard gate
    active_tracks,
    diag: float,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    sec = frame_idx / FPS

    # Draw all detections (grey)
    for di, det in enumerate(detections):
        x1,y1,x2,y2 = det.box
        cv2.rectangle(canvas,(x1,y1),(x2,y2),GENERIC,1)
        cv2.putText(canvas,f"raw{det.track_id}",(x1,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,GENERIC,1)

    # Draw Kalman predictions for watched ids
    for t in active_tracks:
        if t.stable_id not in DIAG_IDS:
            continue
        pred = predicted.get(t.stable_id)
        if pred:
            px,py = int(pred[0]), int(pred[1])
            col = COLORS.get(t.stable_id, GENERIC)
            cv2.drawMarker(canvas,(px,py),col,cv2.MARKER_CROSS,15,2)
            cv2.putText(canvas,f"pred{t.stable_id}",(px+8,py),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)

    # Draw assignments for watched ids
    for ti, t in enumerate(active_tracks):
        if t.stable_id not in DIAG_IDS:
            continue
        di = assignment.get(ti)
        if di is None:
            # Lost
            cx,cy = int(t.kalman.position[0]), int(t.kalman.position[1])
            cv2.putText(canvas,f"ID{t.stable_id}:LOST",(cx,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            continue
        det = detections[di]
        x1,y1,x2,y2 = det.box
        col = COLORS.get(t.stable_id, GENERIC)
        cv2.rectangle(canvas,(x1,y1),(x2,y2),col,3)
        color_sim = 1.0 - _color_cost(t.color_hist, det_colors.get(det.track_id))
        c_byte = _bytetrack_cost(t, det)
        pred = predicted.get(t.stable_id,(t.kalman.position))
        cx2,cy2 = _centroid(det.box)
        c_traj = _position_cost(pred,cx2,cy2,diag)
        label = (f"ID{t.stable_id} raw{det.track_id} "
                 f"sim={color_sim:.2f} byte={'same' if c_byte==0 else 'NEW'} "
                 f"traj={c_traj:.2f}")
        cv2.putText(canvas,label,(x1,y2+14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1,cv2.LINE_AA)

    # Blocked pairs (hard gate fired)
    for (ti,di) in blocked:
        if active_tracks[ti].stable_id in DIAG_IDS:
            det = detections[di]
            x1,y1,x2,y2 = det.box
            cv2.rectangle(canvas,(x1-2,y1-2),(x2+2,y2+2),(0,0,200),1)
            cv2.putText(canvas,"GATE",(x1,y1-14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,200),1)

    cv2.putText(canvas,f"Frame {frame_idx}  t={sec:.2f}s",
                (8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return canvas


def main():
    info = probe(VIDEO)
    diag_px = float(np.hypot(info.width, info.height))

    detector = PersonDetector(confidence=config.DEFAULT_PERSON_CONFIDENCE)
    detector.reset()
    pose = RTMPoseEstimator()
    tracker = FusedTracker()

    csv_rows = []
    events = []

    print(f"Processing {VIDEO.name}  ({info.frame_count} frames @ {FPS:.2f}fps)")
    print(f"Watching IDs {DIAG_IDS}  |  saving annotated frames {WATCH_SECS[0]:.1f}s-{WATCH_SECS[1]:.1f}s")

    # Track previous assignment to detect swaps/losses
    prev_stable_to_raw: dict[int, int | None] = {}
    prev_present: set[int] = set()

    for frame_idx, frame in read_frames(VIDEO, start_frame=0):
        sec = frame_idx / FPS
        detections = detector.detect(frame)
        det_colors: dict[int, np.ndarray | None] = {}
        if detections:
            kp, sc = pose.estimate(frame, [d.box for d in detections])
            for i, det in enumerate(detections):
                pts = [(int(x),int(y),float(s)) for (x,y),s in zip(kp[i],sc[i])]
                kp_hist = keypoint_color(frame, pts)
                det_colors[det.track_id] = (
                    kp_hist if kp_hist is not None
                    else _appearance_from_box(frame, det.box)
                )

        # ── Replicate the cost matrix step for logging ──────────────────
        active_tracks = [
            t for t in tracker._tracks.values()
            if t.lost_frames <= config.FUSED_MAX_LOST_FRAMES
        ]
        predicted: dict[int, tuple[float,float]] = {}
        for t in active_tracks:
            # Don't actually advance the filter here; use current state
            predicted[t.stable_id] = t.kalman.position

        cost_log: list[dict] = []
        blocked_pairs: set[tuple[int,int]] = set()
        assignment_map: dict[int,int] = {}  # ti -> di (for debug draw)

        if active_tracks and detections:
            n_t, n_d = len(active_tracks), len(detections)
            cost_mat = np.full((n_t, n_d), 1e6, dtype=np.float64)
            for ti, track in enumerate(active_tracks):
                pred = predicted[track.stable_id]
                for di, det in enumerate(detections):
                    cx, cy = _centroid(det.box)
                    c_byte  = _bytetrack_cost(track, det)
                    c_color = _color_cost(track.color_hist, det_colors.get(det.track_id))
                    c_traj  = _position_cost(pred, cx, cy, diag_px)
                    gate_fired = c_byte > 0 and c_color > config.FUSED_COLOR_HARD_GATE
                    combined = (config.FUSED_W_BYTE*c_byte
                                + config.FUSED_W_COLOR*c_color
                                + config.FUSED_W_TRAJ*c_traj)
                    if gate_fired:
                        cost_mat[ti,di] = 1e6
                        blocked_pairs.add((ti,di))
                    else:
                        cost_mat[ti,di] = combined

                    if track.stable_id in DIAG_IDS:
                        cost_log.append({
                            "frame": frame_idx, "time_s": round(sec,3),
                            "stable_id": track.stable_id,
                            "raw_track_id": det.track_id,
                            "c_byte": round(c_byte,3),
                            "colour_sim": round(1-c_color,3),
                            "c_traj": round(c_traj,3),
                            "combined_cost": round(combined,3),
                            "gate_blocked": gate_fired,
                        })

        # ── Now actually run the tracker update ──────────────────────────
        assigned = tracker.update(detections, frame, det_colors)
        stable_to_det = {sid: det for sid,det in assigned}
        present_ids = {sid for sid,_ in assigned}

        # Build assignment_map for draw (ti -> di)
        if active_tracks and detections:
            for ti, t in enumerate(active_tracks):
                if t.stable_id in stable_to_det:
                    det = stable_to_det[t.stable_id]
                    for di, d in enumerate(detections):
                        if d.track_id == det.track_id:
                            assignment_map[ti] = di
                            break

        # ── Detect swap / loss events ────────────────────────────────────
        for sid in DIAG_IDS:
            cur_raw = stable_to_det[sid].track_id if sid in stable_to_det else None
            prev_raw = prev_stable_to_raw.get(sid)
            was_present = sid in prev_present

            if cur_raw is not None and prev_raw is not None and cur_raw != prev_raw:
                # Raw id changed — could be a swap or ByteTrack reassignment
                events.append(f"⚠  FRAME {frame_idx} t={sec:.2f}s  "
                              f"ID {sid}: raw tracklet changed {prev_raw}→{cur_raw}")

            if was_present and cur_raw is None:
                events.append(f"❌ FRAME {frame_idx} t={sec:.2f}s  "
                              f"ID {sid}: LOST  (was raw {prev_raw})")

            if not was_present and cur_raw is not None:
                events.append(f"✅ FRAME {frame_idx} t={sec:.2f}s  "
                              f"ID {sid}: RECOVERED raw {cur_raw}")

        prev_stable_to_raw = {sid: (stable_to_det[sid].track_id
                                     if sid in stable_to_det else None)
                               for sid in DIAG_IDS}
        prev_present = present_ids & DIAG_IDS

        csv_rows.extend(cost_log)

        # ── Save annotated debug frame ───────────────────────────────────
        if frame_idx in WATCH_FRAMES:
            debug = draw_debug_frame(
                frame, frame_idx, tracker._tracks.values(),
                detections, det_colors, predicted,
                assignment_map, blocked_pairs,
                active_tracks, diag_px,
            )
            out_path = OUT_FRAMES / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ── Write CSV ────────────────────────────────────────────────────────
    if csv_rows:
        fields = list(csv_rows[0].keys())
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nCost log saved → {OUT_CSV}")

    print(f"\n{'='*60}")
    print(f"EVENTS for IDs {DIAG_IDS}:")
    if events:
        for e in events:
            print(" ", e)
    else:
        print("  (no swap or loss events detected)")

    print(f"\nAnnotated frames → {OUT_FRAMES}/")
    print(f"  ({len(list(OUT_FRAMES.glob('*.jpg')))} frames saved)")


if __name__ == "__main__":
    main()
