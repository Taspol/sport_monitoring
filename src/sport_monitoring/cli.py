"""Command-line entry point for the sport-monitoring pose tracker."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config, video
from .perception.backbones import DETECTORS, POSES, variant_name
from .pipeline import (
    process_video,
    process_video_analyze,
    process_video_interactive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sport-monitoring",
        description=(
            "Track the pose of multiple players in basketball videos using "
            "YOLO11 + ByteTrack detection and RTMPose, rendering annotated clips."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Video file(s) to process. Defaults to every video in ./data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Folder scanned for videos when no inputs are given.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Where annotated videos are written.",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=config.DEFAULT_PERSON_CONFIDENCE,
        help="Minimum YOLO person-detection confidence (0-1).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Injury-risk mode: pick ONE player and compute LESS-inspired joint "
        "angles (knee flexion/valgus, hip flexion, trunk lean) -- overlaid on the "
        "video, exported to a CSV, and summarised.",
    )
    parser.add_argument(
        "--track-all",
        action="store_true",
        help="Use the IDStabilizer method instead (re-id by colour histogram + "
        "position) rather than the reference-frame colour registry.",
    )
    parser.add_argument(
        "--select-frame",
        type=int,
        default=0,
        help="Reference frame index to register all players from.",
    )
    parser.add_argument(
        "--3d",
        dest="three_d",
        action="store_true",
        help="Render each person as an orientation-aware 3D bounding cuboid "
        "instead of a flat box. Body yaw is read from shoulder/hip foreshortening "
        "in the 2D keypoints (no extra model) so the box turns with the player.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rendering; only print per-video statistics.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Draw a compact debug panel for every person showing stable id, "
        "raw ByteTrack id, colour-sim %%, watchdog count, and YOLO confidence.",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="(analyse mode) skip the click window and auto-pick the player to "
        "analyse as the one with the largest vertical foot-line excursion (the "
        "jumper). Lets --analyze run headless / in batch (e.g. for the backbone "
        "comparison).",
    )
    parser.add_argument(
        "--detector",
        choices=DETECTORS,
        default="yolo",
        help="(analyse mode) person detector backbone. 'yolo' (default) or 'rfdetr' "
        "(RF-DETR + supervision ByteTrack). Non-default variants write to "
        "output/<detector>_<pose>/ so backbones can be compared side by side.",
    )
    parser.add_argument(
        "--pose",
        choices=POSES,
        default="rtm",
        help="(analyse mode) pose backbone. 'rtm' (RTMPose, default) or 'mediapipe' "
        "(BlazePose remapped to COCO-17).",
    )
    return parser


def _write_timing(
    out_dir: Path, stem: str, frames: int, seconds: float, variant: str
) -> None:
    """Record wall-clock throughput for the FPS column of the comparison table."""
    fps = frames / seconds if seconds > 0 else 0.0
    (out_dir / f"{stem}_timing.json").write_text(
        json.dumps(
            {"variant": variant, "frames": frames, "seconds": round(seconds, 3),
             "fps": round(fps, 2)},
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.inputs:
        sources = args.inputs
    else:
        sources = video.find_videos(args.data_dir)

    mode = (
        "injury analysis" if args.analyze
        else "IDStabilizer" if args.track_all
        else "reference-frame registry"
    )
    variant = variant_name(args.detector, args.pose)
    analyze_out = (
        args.output_dir if variant == "yolo_rtm" else args.output_dir / variant
    )
    label = "YOLO + RTMPose" if variant == "yolo_rtm" else f"{args.detector} + {args.pose}"
    print(f"Processing {len(sources)} video(s) with {label} ({mode}).\n")
    for source in sources:
        if args.analyze:
            analyze_out.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            result = process_video_analyze(
                source,
                person_confidence=args.person_conf,
                output_dir=analyze_out,
                write_video=not args.no_video,
                select_frame=args.select_frame,
                debug=args.debug,
                three_d=args.three_d,
                detector_name=args.detector,
                pose_name=args.pose,
                auto_select=args.auto_select,
            )
            if result is not None:
                _write_timing(analyze_out, source.stem, result.frames,
                              time.perf_counter() - t0, variant)
        elif args.track_all:
            result = process_video(
                source,
                person_confidence=args.person_conf,
                output_dir=args.output_dir,
                write_video=not args.no_video,
                debug=args.debug,
                three_d=args.three_d,
            )
        else:
            result = process_video_interactive(
                source,
                person_confidence=args.person_conf,
                output_dir=args.output_dir,
                write_video=not args.no_video,
                select_frame=args.select_frame,
                debug=args.debug,
                three_d=args.three_d,
            )
        if result is None:
            continue
        print(
            f"\n{result.source.name}: {result.frames} frames | "
            f"max {result.max_people} people | "
            f"avg {result.avg_people:.1f} people"
        )
        if result.output is not None:
            print(f"  -> saved {result.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
