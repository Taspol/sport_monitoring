"""End-to-end pose-tracking pipeline for a single video.

YOLO11 + ByteTrack locate and track players; RTMPose estimates each player's 17
COCO keypoints from their box; results are drawn and written out.

Three modes:
* analyze -- one player is picked (clicked or auto-selected) and their landing
  biomechanics (LESS-inspired joint angles, jumps, falls) are scored.
* interactive -- the user picks players on a preview frame, and only those
  (matched by shirt/pant colour) are pose-tracked; everyone else is drawn faint.
* track-all -- every detected player is tracked, with FusedTracker keeping ids
  stable through occlusion (see tracker.py).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

from . import config, video
from .analysis.biomechanics import JointAngles, joint_angles, risk_flags
from .tracking.color_id import (
    color_name,
    dominant_hsv,
    hsv_to_bgr,
    keypoint_color,
    similarity,
)
from .perception.backbones import build_detector, build_pose
from .perception.detector import Detection, PersonDetector
from .analysis.fall import Fall, detect_falls, fall_metrics
from .analysis.landing import LESS_ITEMS, Landing, detect_landings, sample_from_points
from .analysis.risk import RiskSummary, assess_player_risk, format_risk_report
from .perception.rtm_pose import RTMPoseEstimator
from .tracking.selector import auto_select_player, register_all_players, select_one_player
from .tracking.tracker import FusedSelectedTracker, FusedTracker
from .render.visualize import (
    COCO_KEYPOINT_NAMES,
    PersonPose,
    draw_clean_poses,
    draw_fall_banner,
    draw_injury_overlay,
    draw_jump_banner,
    draw_pose_trajectory,
    draw_risk_card,
    draw_tracked_poses,
    draw_trajectories,
    draw_trajectory_map,
)

Box = tuple[int, int, int, int]


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


FrameHandler = Callable[
    ["np.ndarray"], tuple[list[PersonPose], list[PersonPose], "np.ndarray"]
]


def _save_identity_registry(
    registry: dict[int, np.ndarray], output_dir: Path, stem: str
) -> Path:
    """Print and write the saved ID -> jersey-colour identities to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}_identities.json"
    data: dict[str, dict] = {}
    print("Saved identity registry (ID -> jersey colour):")
    for sid in sorted(registry):
        hsv = dominant_hsv(registry[sid])
        name = color_name(hsv)
        print(f"  ID {sid}: {name:<6} HSV{tuple(int(c) for c in hsv)}")
        data[str(sid)] = {"name": name, "hsv": [int(c) for c in hsv]}
    path.write_text(json.dumps(data, indent=2))
    print(f"  -> saved {path}")
    return path


@dataclass
class ProcessResult:
    """Summary of a processed video."""

    source: Path
    output: Path | None
    frames: int
    max_people: float
    avg_people: float


def _poses_for(
    pose: RTMPoseEstimator,
    frame,
    items: list[tuple[int, Detection]],
    tracker: FusedTracker | None = None,
) -> list[PersonPose]:
    """Run RTMPose on the given (stable_id, detection) pairs."""
    boxes = [det.box for _, det in items]
    keypoints, scores = pose.estimate(frame, boxes)
    people: list[PersonPose] = []
    for idx, (stable_id, det) in enumerate(items):
        points = [
            (int(x), int(y), float(s))
            for (x, y), s in zip(keypoints[idx], scores[idx])
        ]
        track = tracker._tracks.get(stable_id) if tracker else None
        sf = track.color_suspect_frames if track else 0
        people.append(PersonPose(
            track_id=stable_id,
            box=det.box,
            points=points,
            raw_id=det.track_id,
            suspect_frames=sf,
            confidence=det.confidence,
        ))
    return people


def _run(
    source: Path,
    output_dir: Path,
    write_video: bool,
    handle_frame: FrameHandler,
    start_frame: int = 0,
    debug: bool = False,
    three_d: bool = False,
    store=None,
) -> ProcessResult:
    """Drive the frame loop with a per-frame handler, writing an annotated mp4.

    ``store`` is the live TrajectoryStore; in ``--3d`` it feeds each person's
    real-time move arrow inside ``draw_tracked_poses``.
    """
    info = video.probe(source)

    out_path: Path | None = None
    writer: video.VideoWriter | None = None
    if write_video:
        out_path = output_dir / f"{source.stem}_pose.mp4"
        writer = video.VideoWriter(out_path, info.fps, info.width, info.height)

    total = max(0, (info.frame_count or 0) - start_frame) or None
    people_per_frame: list[int] = []
    try:
        for _, frame in tqdm(
            video.read_frames(source, start_frame=start_frame),
            total=total,
            desc=source.name,
            unit="f",
        ):
            people, others, canvas = handle_frame(frame)
            people_per_frame.append(len(people))
            if writer is not None:
                writer.write(
                    draw_tracked_poses(
                        canvas,
                        people,
                        config.MIN_LANDMARK_VISIBILITY,
                        others=others,
                        debug=debug,
                        three_d=three_d,
                        store=store,
                    )
                )
    finally:
        if writer is not None:
            writer.close()

    frames = len(people_per_frame)
    return ProcessResult(
        source=source,
        output=out_path,
        frames=frames,
        max_people=max(people_per_frame, default=0),
        avg_people=(sum(people_per_frame) / frames) if frames else 0.0,
    )


def process_video(
    source: Path,
    person_confidence: float = config.DEFAULT_PERSON_CONFIDENCE,
    output_dir: Path = config.OUTPUT_DIR,
    write_video: bool = True,
    debug: bool = False,
    three_d: bool = False,
) -> ProcessResult:
    """Track-all mode: detect and pose-estimate every player (FusedTracker)."""
    detector = PersonDetector(confidence=person_confidence)
    detector.reset()
    pose = RTMPoseEstimator()
    tracker = FusedTracker()

    def handle(frame) -> tuple[list[PersonPose], list[PersonPose], np.ndarray]:
        detections = detector.detect(frame)
        boxes = [det.box for det in detections]
        kp_colors: dict[int, np.ndarray | None] = {}
        if detections:
            kp, sc = pose.estimate(frame, boxes)
            for i, det in enumerate(detections):
                pts = [(int(x), int(y), float(s))
                       for (x, y), s in zip(kp[i], sc[i])]
                kp_colors[det.track_id] = keypoint_color(frame, pts)
        assigned = tracker.update(detections, frame, kp_colors)
        canvas = frame.copy()
        draw_trajectories(canvas, tracker.trajectories, three_d=three_d)
        people = _poses_for(pose, frame, assigned, tracker)
        return people, [], canvas

    return _run(source, output_dir, write_video, handle, debug=debug,
                three_d=three_d, store=tracker.trajectories)


def process_video_interactive(
    source: Path,
    person_confidence: float = config.DEFAULT_PERSON_CONFIDENCE,
    output_dir: Path = config.OUTPUT_DIR,
    write_video: bool = True,
    select_frame: int = 0,
    debug: bool = False,
    three_d: bool = False,
) -> ProcessResult | None:
    """Registry mode: register every person in the reference frame, then track.

    Every person detected in the reference frame is assigned a stable id;
    FusedSelectedTracker follows each one using weighted ByteTrack + colour +
    Kalman-trajectory cues.  Trajectories are drawn as fading polyline trails.

    Returns ``None`` if no person is detected in the reference frame.
    """
    targets, sel_frame = register_all_players(
        source, person_confidence=person_confidence, start_frame=select_frame
    )
    if not targets:
        print(f"No person detected in frame {sel_frame} of {source.name} -- skipping.")
        return None

    print(
        f"Registered {len(targets)} player(s) "
        f"{[t.stable_id for t in targets]} in reference frame {sel_frame}"
    )

    detector = PersonDetector(confidence=person_confidence)
    detector.reset()
    pose = RTMPoseEstimator()
    tracker = FusedSelectedTracker(targets)

    _save_identity_registry(tracker.registry(), output_dir, source.stem)

    def handle(frame) -> tuple[list[PersonPose], list[PersonPose], np.ndarray]:
        detections = detector.detect(frame)

        boxes = [det.box for det in detections]
        keypoints, scores = pose.estimate(frame, boxes)
        points_by_id: dict[int, list[tuple[int, int, float]]] = {}
        colors: dict[int, np.ndarray | None] = {}
        for i, det in enumerate(detections):
            pts = [(int(x), int(y), float(s))
                   for (x, y), s in zip(keypoints[i], scores[i])]
            points_by_id[det.track_id] = pts
            colors[det.track_id] = keypoint_color(frame, pts)

        result = tracker.update(detections, frame, colors)

        people = []
        for sid, det in result.selected:
            ref_hist = tracker.reference_color(sid)
            cur_hist = colors.get(det.track_id)
            track = tracker._fused._tracks.get(sid)
            sf = track.color_suspect_frames if track else 0
            people.append(PersonPose(
                track_id=sid,
                box=det.box,
                points=points_by_id.get(det.track_id),
                ref_color=(
                    hsv_to_bgr(dominant_hsv(ref_hist)) if ref_hist is not None else None
                ),
                cur_color=(
                    hsv_to_bgr(dominant_hsv(cur_hist)) if cur_hist is not None else None
                ),
                similarity=(
                    similarity(ref_hist, cur_hist)
                    if ref_hist is not None and cur_hist is not None else None
                ),
                raw_id=det.track_id,
                suspect_frames=sf,
                confidence=det.confidence,
            ))

        others = []
        for sid, det in result.others:
            cur_hist = colors.get(det.track_id)
            others.append(PersonPose(
                track_id=sid,
                box=det.box,
                points=points_by_id.get(det.track_id),
                cur_color=(
                    hsv_to_bgr(dominant_hsv(cur_hist)) if cur_hist is not None else None
                ),
                raw_id=det.track_id,
                confidence=det.confidence,
            ))

        canvas = frame.copy()
        draw_trajectories(canvas, tracker.trajectories, three_d=three_d)
        return people, others, canvas

    return _run(source, output_dir, write_video, handle, start_frame=sel_frame,
                debug=debug, three_d=three_d, store=tracker.trajectories)


def _write_joint_csv(
    path: Path, rows: list[tuple[int, JointAngles, dict[str, bool]]]
) -> None:
    fields = list(JointAngles().as_dict().keys())
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", *fields, "n_risk_flags"])
        for frame_idx, angles, flags in rows:
            vals = angles.as_dict()
            w.writerow(
                [frame_idx]
                + ["" if vals[f] is None else round(vals[f], 1) for f in fields]
                + [sum(1 for v in flags.values() if v)]
            )


def _print_injury_summary(
    rows: list[tuple[int, JointAngles, dict[str, bool]]], total_frames: int
) -> None:
    if not rows:
        print("  (player never tracked -- no joint data)")
        return
    fields = list(JointAngles().as_dict().keys())
    print(f"Injury-risk summary ({len(rows)}/{total_frames} frames with the player):")
    for f in fields:
        vals = [a.as_dict()[f] for _, a, _ in rows if a.as_dict()[f] is not None]
        if not vals:
            print(f"  {f:<14}: no data")
            continue
        print(f"  {f:<14}: min {min(vals):+.0f}  max {max(vals):+.0f}  "
              f"mean {sum(vals)/len(vals):+.0f}  (deg)")
    risky_frames = sum(1 for _, _, fl in rows if any(fl.values()))
    print(f"  frames with >=1 risk flag: {risky_frames}/{len(rows)} "
          f"({100*risky_frames//max(1,len(rows))}%)")


def process_video_analyze(
    source: Path,
    person_confidence: float = config.DEFAULT_PERSON_CONFIDENCE,
    output_dir: Path = config.OUTPUT_DIR,
    write_video: bool = True,
    select_frame: int = 0,
    debug: bool = False,
    three_d: bool = False,
    detector_name: str = "yolo",
    pose_name: str = "rtm",
    auto_select: bool = False,
    preselected_box: Box | None = None,
    show_trajectory: bool = True,
) -> ProcessResult | None:
    """Injury-risk analysis: track everyone, analyse one selected player.

    Pops up a selection window to pick the player to ANALYSE, but registers and
    tracks **every** person (stable id + colour identity). For the selected id,
    the LESS-inspired joint angles are computed each frame -- overlaid, written to
    ``output/<clip>_joint_angles.csv``, and summarised. With ``auto_select`` the
    largest detected person is picked headlessly instead of via the click window.

    ``preselected_box`` lets a caller (e.g. the Gradio web demo) supply the chosen
    player's box from ``select_frame`` directly, skipping both the click window and
    auto-select so analysis runs headless. ``show_trajectory`` (default on) draws
    the movement trail + arrows; the web demo turns it off for a cleaner overlay.
    The CLI never passes either, so its default behaviour is unchanged.
    """
    if preselected_box is not None:
        target_box, ref_frame = preselected_box, select_frame
        print(f"Using caller-selected player at frame {ref_frame}.")
    elif auto_select:
        target_box, ref_frame = auto_select_player(
            source, person_confidence, select_frame
        )
        if target_box is not None:
            print(f"Auto-selected the largest player at frame {ref_frame}.")
    else:
        target_box, ref_frame = select_one_player(
            source, person_confidence, select_frame
        )
    if target_box is None:
        print(f"Analysis cancelled for {source.name}.")
        return None

    targets, _ = register_all_players(source, person_confidence, ref_frame)
    if not targets:
        print(f"No person detected in frame {ref_frame} of {source.name} -- skipping.")
        return None

    analyze_id = max(targets, key=lambda t: _iou(t.box, target_box)).stable_id
    print(
        f"Analysing ID {analyze_id} while tracking {len(targets)} player(s) "
        f"from reference frame {ref_frame}."
    )

    detector = build_detector(detector_name, person_confidence)
    pose = build_pose(pose_name)
    tracker = FusedSelectedTracker(targets)
    _save_identity_registry(tracker.registry(), output_dir, source.stem)

    info = video.probe(source)
    total = max(0, (info.frame_count or 0) - ref_frame) or None

    payloads: list[tuple[int, list[PersonPose], list[PersonPose],
                         list[tuple[int, int, float]] | None, JointAngles | None]] = []
    rows: list[tuple[int, JointAngles, dict[str, bool]]] = []
    samples = []
    fall_samples = []
    traj_full: dict[int, list[tuple[float, float]]] = {}
    pose_track: list[tuple[int, list[tuple[int, int, float]]]] = []
    bg_frame: np.ndarray | None = None
    for idx, frame in tqdm(
        video.read_frames(source, start_frame=ref_frame),
        total=total, desc=f"{source.name} (analyse)", unit="f",
    ):
        if bg_frame is None:
            bg_frame = frame.copy()
        detections = detector.detect(frame)
        points_by_id: dict[int, list[tuple[int, int, float]]] = {}
        colors: dict[int, np.ndarray | None] = {}
        if detections:
            kp, sc = pose.estimate(frame, [d.box for d in detections])
            for i, d in enumerate(detections):
                pts = [(int(x), int(y), float(s)) for (x, y), s in zip(kp[i], sc[i])]
                points_by_id[d.track_id] = pts
                colors[d.track_id] = keypoint_color(frame, pts)

        result = tracker.update(detections, frame, colors)
        people: list[PersonPose] = []
        angles = None
        tpts = None
        for sid, det in result.selected:
            pts = points_by_id.get(det.track_id, [])
            ref_hist = tracker.reference_color(sid)
            cur_hist = colors.get(det.track_id)
            track = tracker._fused._tracks.get(sid)
            sf = track.color_suspect_frames if track else 0
            people.append(PersonPose(
                track_id=sid, box=det.box, points=pts,
                ref_color=hsv_to_bgr(dominant_hsv(ref_hist)) if ref_hist is not None else None,
                cur_color=hsv_to_bgr(dominant_hsv(cur_hist)) if cur_hist is not None else None,
                similarity=(similarity(ref_hist, cur_hist)
                            if ref_hist is not None and cur_hist is not None else None),
                raw_id=det.track_id,
                suspect_frames=sf,
                confidence=det.confidence,
            ))
            if sid == analyze_id:
                tpts = pts
                angles = joint_angles(pts)
                rows.append((idx, angles, risk_flags(angles)))
                if pts:
                    pose_track.append((idx, pts))

        others = [
            PersonPose(
                track_id=sid, box=d.box,
                points=points_by_id.get(d.track_id, []),
                raw_id=d.track_id,
                confidence=d.confidence,
            )
            for sid, d in result.others
        ]
        payloads.append((idx, people, others, tpts, angles))
        for pp in (*people, *others):
            bx1, by1, bx2, by2 = pp.box
            traj_full.setdefault(pp.track_id, []).append(
                ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))
        samples.append(sample_from_points(
            idx, tpts, angles if angles is not None else joint_angles([]),
            config.JOINT_MIN_SCORE))
        fall_samples.append(fall_metrics(idx, tpts, config.JOINT_MIN_SCORE))

    landings = detect_landings(samples)
    falls = detect_falls(fall_samples)
    risk = assess_player_risk(rows, landings)
    risk.n_falls = len(falls)
    banner_frames = max(1, int(round(info.fps)))
    frame_banner: dict[int, Landing] = {}
    for L in landings:
        for f in range(L.apex_frame, L.ic_frame + banner_frames):
            frame_banner[f] = L
    frame_fall: dict[int, Fall] = {}
    for fl in falls:
        for f in range(fl.start_frame, fl.end_frame + 1):
            frame_fall[f] = fl

    out_path: Path | None = None
    clean_path: Path | None = None
    if write_video:
        out_path = output_dir / f"{source.stem}_injury.mp4"
        clean_path = output_dir / f"{source.stem}_clean.mp4"
        writer = video.VideoWriter(out_path, info.fps, info.width, info.height)
        clean_writer = video.VideoWriter(clean_path, info.fps, info.width, info.height)
        for (idx, frame), (_, people, others, tpts, angles) in zip(
            video.read_frames(source, start_frame=ref_frame), payloads
        ):
            clean = draw_clean_poses(
                frame, people, others, config.MIN_LANDMARK_VISIBILITY,
                highlight_id=analyze_id,
            )
            clean_writer.write(clean)

            canvas = draw_tracked_poses(
                frame, people, config.MIN_LANDMARK_VISIBILITY, others=others,
                debug=debug, three_d=three_d,
                store=tracker.trajectories if show_trajectory else None,
            )
            if show_trajectory:
                draw_trajectories(canvas, tracker.trajectories, three_d=three_d)
            if angles is not None and tpts is not None:
                draw_injury_overlay(
                    canvas, tpts, angles.as_dict(), risk_flags(angles),
                    config.MIN_LANDMARK_VISIBILITY, label=f"ID {analyze_id}",
                )
            banner = frame_banner.get(idx)
            if banner is not None:
                draw_jump_banner(
                    canvas, banner.index, banner.score,
                    [k for k, v in banner.items.items() if v],
                )
            fall = frame_fall.get(idx)
            if fall is not None:
                draw_fall_banner(
                    canvas, fall.index, fall.max_confidence, fall.sudden_drop
                )
            writer.write(canvas)
        writer.close()
        clean_writer.close()
        print(f"  -> clean video: {clean_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    joint_csv = output_dir / f"{source.stem}_joint_angles.csv"
    _write_joint_csv(joint_csv, rows)
    _print_injury_summary(rows, info.frame_count or 0)
    print(f"  -> joint angles: {joint_csv}")

    jumps_csv = output_dir / f"{source.stem}_jumps.csv"
    _write_jumps_csv(jumps_csv, landings)
    _print_jump_report(landings)
    print(f"  -> jumps: {jumps_csv}")

    falls_csv = output_dir / f"{source.stem}_falls.csv"
    _write_falls_csv(falls_csv, falls)
    _print_fall_report(falls)
    print(f"  -> falls: {falls_csv}")

    risk_lines = format_risk_report(risk, f"ID {analyze_id}")
    print("\n" + "\n".join(risk_lines))
    risk_txt = output_dir / f"{source.stem}_risk_summary.txt"
    risk_txt.write_text("\n".join(risk_lines) + "\n")
    print(f"  -> risk summary: {risk_txt}")

    pose_csv = output_dir / f"{source.stem}_pose_track.csv"
    _write_pose_track_csv(pose_csv, pose_track)
    print(f"Pose track: {len(pose_track)} frames with ID {analyze_id}'s skeleton")
    print(f"  -> pose track: {pose_csv}")
    if bg_frame is not None and pose_track:
        risk_frames = {idx for idx, _, fl in rows if any(fl.values())}
        strobe = draw_pose_trajectory(
            bg_frame, pose_track, config.MIN_LANDMARK_VISIBILITY,
            max_samples=config.POSE_TRAIL_SAMPLES, label=f"ID {analyze_id}",
            risk_frames=risk_frames,
        )
        draw_risk_card(strobe, _risk_card_lines(risk), risk.rating)
        pose_png = output_dir / f"{source.stem}_pose_trajectory.png"
        cv2.imwrite(str(pose_png), strobe)
        print(f"  -> pose trajectory image: {pose_png}")

        strobe_risk = draw_pose_trajectory(
            bg_frame, pose_track, config.MIN_LANDMARK_VISIBILITY,
            max_samples=config.POSE_TRAIL_SAMPLES, label=f"ID {analyze_id}",
            risk_frames=risk_frames, risk_only=True,
        )
        draw_risk_card(strobe_risk, _risk_card_lines(risk), risk.rating)
        pose_risk_png = output_dir / f"{source.stem}_pose_trajectory_risk.png"
        cv2.imwrite(str(pose_risk_png), strobe_risk)
        print(f"  -> risk-only pose trajectory image: {pose_risk_png}")

    if bg_frame is not None and traj_full.get(analyze_id):
        traj_map = draw_trajectory_map(
            bg_frame, {analyze_id: traj_full[analyze_id]},
            label=f"(analysed ID {analyze_id})",
        )
        traj_png = output_dir / f"{source.stem}_trajectory_map.png"
        cv2.imwrite(str(traj_png), traj_map)
        print(f"  -> trajectory map image: {traj_png}")

    return ProcessResult(
        source=source, output=out_path, frames=len(rows),
        max_people=1, avg_people=len(rows) / (total or 1),
    )


def _risk_card_lines(risk: RiskSummary) -> list[tuple[str, tuple[int, int, int]]]:
    """Compact, colour-coded metric lines for the strobe-image risk card."""
    green, yellow, red, grey = (
        (120, 230, 120), (60, 200, 255), (80, 80, 255), (190, 190, 190))

    def sev(rate: float, mod: float, high: float) -> tuple[int, int, int]:
        return red if rate >= high else yellow if rate >= mod else green

    ml = "--" if risk.mean_less is None else f"{risk.mean_less:.1f}"
    lines = [
        (f"Composite exposure: {risk.composite}/100", grey),
        (f"Knee valgus: {risk.valgus_frame_rate*100:.0f}% frames",
         sev(risk.valgus_frame_rate, config.RISK_VALGUS_FRAME_MOD,
             config.RISK_VALGUS_FRAME_HIGH)),
        (f"Trunk lean: {risk.trunk_frame_rate*100:.0f}% frames",
         sev(risk.trunk_frame_rate, config.RISK_TRUNK_FRAME_MOD, 0.30)),
        (f"Jumps: {risk.n_landings}  landing LESS {ml}/9", grey),
        (f"Stiff landings: {risk.stiff_landings}/{risk.n_landings}",
         red if risk.stiff_landings else green),
        (f"Falls (on-ground): {risk.n_falls}",
         red if risk.n_falls else green),
    ]
    return lines


def _write_pose_track_csv(
    path: Path, pose_track: list[tuple[int, list[tuple[int, int, float]]]]
) -> None:
    """Write the analysed player's 17 COCO keypoints (x, y, score) per frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame"]
    for name in COCO_KEYPOINT_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_score"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for frame_idx, pts in pose_track:
            row: list[object] = [frame_idx]
            for i in range(len(COCO_KEYPOINT_NAMES)):
                if i < len(pts):
                    x, y, score = pts[i]
                    row += [x, y, round(float(score), 4)]
                else:
                    row += ["", "", ""]
            w.writerow(row)


def _write_jumps_csv(path: Path, landings: list[Landing]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["jump", "apex_frame", "ic_frame", "maxflex_frame",
                    "less_score", *LESS_ITEMS])
        for L in landings:
            w.writerow([L.index, L.apex_frame, L.ic_frame, L.maxflex_frame, L.score,
                        *[int(L.items.get(k, False)) for k in LESS_ITEMS]])


def _write_falls_csv(path: Path, falls: list[Fall]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fall", "start_frame", "end_frame", "lowest_frame",
                    "duration_frames", "max_confidence", "sudden_drop"])
        for fl in falls:
            w.writerow([fl.index, fl.start_frame, fl.end_frame, fl.lowest_frame,
                        fl.duration_frames, round(fl.max_confidence, 3),
                        int(fl.sudden_drop)])


def _print_fall_report(falls: list[Fall]) -> None:
    if not falls:
        print("Falls: none detected (player stayed upright in this clip).")
        return
    print(f"Falls: {len(falls)} on-ground event(s) detected (keypoint alignment):")
    for fl in falls:
        kind = "sudden fall" if fl.sudden_drop else "on-ground"
        print(f"  #{fl.index} frames {fl.start_frame}-{fl.end_frame} "
              f"({fl.duration_frames}f, {kind}, conf {int(fl.max_confidence*100)}%)")


def _print_jump_report(landings: list[Landing]) -> None:
    if not landings:
        print("Jumps: none detected (no clear jump-landings in this clip).")
        return
    scores = [L.score for L in landings]
    print(f"Jumps: {len(landings)} detected | landing "
          f"LESS-subset mean {sum(scores)/len(scores):.1f}/9 max {max(scores)}/9")
    for L in landings:
        fails = ", ".join(k for k, v in L.items.items() if v) or "none"
        print(f"  #{L.index} jump@frame {L.apex_frame}, land@frame {L.ic_frame}: "
              f"LESS {L.score}/9  (fail: {fails})")
