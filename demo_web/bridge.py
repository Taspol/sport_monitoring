"""Glue between the Gradio UI and the existing sport_monitoring pipeline.

This module contains **no** analysis logic of its own -- it only:

1. extracts a reference frame and runs person detection on it (for the picker),
2. maps a click coordinate to the chosen detection box,
3. runs the unchanged ``process_video_analyze`` with that box, and
4. re-encodes the output to browser-playable H.264 and gathers the artifacts.

Everything heavy lives in ``sport_monitoring``; the CLI is untouched.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sport_monitoring import config, video
from sport_monitoring.perception.detector import PersonDetector
from sport_monitoring.pipeline import process_video_analyze

Box = tuple[int, int, int, int]

# Cache one detector per confidence so re-loading the same frame is cheap and we
# don't re-init YOLO on every slider nudge.
_detector_cache: dict[float, PersonDetector] = {}


def _get_detector(confidence: float) -> PersonDetector:
    det = _detector_cache.get(confidence)
    if det is None:
        det = PersonDetector(confidence=confidence)
        _detector_cache[confidence] = det
    return det


def list_sample_videos() -> list[tuple[str, str]]:
    """``(display_name, path)`` for every clip in ``data/`` (for the picker)."""
    try:
        return [(p.name, str(p)) for p in video.find_videos(config.DATA_DIR)]
    except FileNotFoundError:
        return []


# Coloured badge per overall-risk rating from risk.py (LOW / MODERATE / HIGH).
_RATING_BADGE = {"LOW": "🟢 **LOW**", "MODERATE": "🟡 **MODERATE**",
                 "HIGH": "🔴 **HIGH**"}


def decorate_summary(text: str) -> str:
    """Turn the plain ``_risk_summary.txt`` into nicely formatted Markdown.

    Parses the known report structure (title / overall risk / metric lines /
    "Why:" reasons / NOTE) into a heading, a colour-coded risk badge, a metrics
    table and a bullet list. Falls back to a code block if the shape is unexpected.
    """
    title = "Injury-Risk Summary"
    overall = ""
    metrics: list[tuple[str, str]] = []
    reasons: list[str] = []
    note = ""
    section = "head"

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("===="):
            who = line.strip("= ").replace("INJURY-RISK SUMMARY", "").lstrip("- ").strip()
            title = f"Injury-Risk Summary — {who}" if who else title
        elif line.startswith("Overall risk:"):
            overall = line
        elif line == "Why:":
            section = "why"
        elif line.startswith("NOTE:"):
            note = line[len("NOTE:"):].strip()
        elif section == "why" and line.startswith("- "):
            reasons.append(line[2:].strip())
        elif ":" in line:
            label, _, value = line.partition(":")
            metrics.append((label.strip(), value.strip()))

    if not overall and not metrics:  # unexpected format -> show raw
        return f"```\n{text}\n```"

    md = [f"## 🩺 {title}"]

    rating = next((r for r in _RATING_BADGE if r in overall), None)
    if rating:
        line = f"### Overall risk: {_RATING_BADGE[rating]}"
        comp = re.search(r"composite exposure\s*(\d+/100)", overall)
        if comp:
            line += f" &nbsp; · &nbsp; composite exposure `{comp.group(1)}`"
        md.append(line)
    elif overall:
        md.append(f"**{overall}**")

    if metrics:
        md += ["", "| Metric | Value |", "|---|---|"]
        md += [f"| {label} | {value} |" for label, value in metrics]

    if reasons:
        md += ["", "**🔎 Why it was flagged**"]
        md += [f"- {r}" for r in reasons]

    if note:
        md += ["", f"> ⚠️ _{note}_"]

    return "\n".join(md)


def frame_count(video_path: str | Path) -> int:
    """Total number of frames in the uploaded clip (for the slider range)."""
    return max(1, video.probe(Path(video_path)).frame_count)


def read_frame_rgb(video_path: str | Path, frame_idx: int) -> np.ndarray | None:
    """Read a single frame as an RGB array (Gradio displays RGB)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_on_frame(
    video_path: str | Path, frame_idx: int, confidence: float
) -> tuple[np.ndarray | None, list[Box]]:
    """Detect people on one frame; return ``(annotated_rgb, boxes)``.

    The returned image has every candidate drawn with a faint box + index so the
    user can see who is selectable before clicking.
    """
    rgb = read_frame_rgb(video_path, frame_idx)
    if rgb is None:
        return None, []
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    boxes = _get_detector(confidence).detect_boxes(bgr)
    return render_boxes(rgb, boxes, selected=None), boxes


def render_boxes(
    rgb: np.ndarray, boxes: list[Box], selected: int | None
) -> np.ndarray:
    """Draw numbered candidate boxes; highlight the selected one in green."""
    canvas = rgb.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        picked = i == selected
        col = (0, 200, 0) if picked else (170, 170, 170)  # RGB
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3 if picked else 1)
        cv2.putText(canvas, str(i), (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return canvas


def pick_box(boxes: list[Box], x: int, y: int) -> int | None:
    """Return the index of the box the click landed in.

    Prefers a box that *contains* the click (smallest such box wins when nested);
    otherwise falls back to the box whose centre is nearest.
    """
    if not boxes:
        return None
    containing = [
        i for i, (x1, y1, x2, y2) in enumerate(boxes)
        if x1 <= x <= x2 and y1 <= y <= y2
    ]
    if containing:
        return min(
            containing,
            key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
        )
    return min(
        range(len(boxes)),
        key=lambda i: (
            (x - (boxes[i][0] + boxes[i][2]) / 2) ** 2
            + (y - (boxes[i][1] + boxes[i][3]) / 2) ** 2
        ),
    )


def _reencode_h264(src: Path) -> Path:
    """Re-encode an mp4v clip to H.264/yuv420p so browsers can play it.

    The pipeline writes mp4v (not web-playable). If ffmpeg is missing we return
    the original path and let the UI warn the user.
    """
    if shutil.which("ffmpeg") is None:
        return src
    dst = src.with_name(f"{src.stem}_web.mp4")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)],
        capture_output=True, text=True,
    )
    return dst if proc.returncode == 0 and dst.exists() else src


@dataclass
class AnalysisResult:
    """Artifact paths produced by one analyze run (any may be ``None``)."""

    video: Path | None
    summary: str
    strobes: list[Path]
    trajectory_map: Path | None
    jumps_csv: Path | None
    falls_csv: Path | None
    joint_angles_csv: Path | None
    frames: int


def run_analysis(
    video_path: str | Path,
    target_box: Box,
    frame_idx: int,
    confidence: float,
    three_d: bool,
    detector_name: str,
    pose_name: str,
    work_dir: Path,
) -> AnalysisResult:
    """Run the unchanged analyze pipeline for the clicked player.

    Returns gathered artifact paths; the output video is re-encoded to H.264.
    Raises ``RuntimeError`` if the pipeline produced no result (e.g. nobody
    detected at the reference frame).
    """
    source = Path(video_path)
    work_dir.mkdir(parents=True, exist_ok=True)

    result = process_video_analyze(
        source,
        person_confidence=confidence,
        output_dir=work_dir,
        write_video=True,
        select_frame=frame_idx,
        three_d=three_d,
        detector_name=detector_name,
        pose_name=pose_name,
        preselected_box=tuple(int(v) for v in target_box),  # type: ignore[arg-type]
        show_trajectory=False,  # web demo: cleaner overlay, no trail/arrow
    )
    if result is None:
        raise RuntimeError(
            "Analysis produced no result -- no player was detected at the "
            "selected frame. Try a different frame or lower the confidence."
        )

    stem = source.stem

    def _opt(name: str) -> Path | None:
        p = work_dir / f"{stem}{name}"
        return p if p.exists() else None

    injury = _opt("_injury.mp4")
    web_video = _reencode_h264(injury) if injury else None
    summary_path = _opt("_risk_summary.txt")
    raw_summary = (
        summary_path.read_text() if summary_path else "(no risk summary written)"
    )
    summary = decorate_summary(raw_summary)
    strobes = [p for p in (_opt("_pose_trajectory.png"),
                           _opt("_pose_trajectory_risk.png")) if p]

    return AnalysisResult(
        video=web_video,
        summary=summary,
        strobes=strobes,
        trajectory_map=_opt("_trajectory_map.png"),
        jumps_csv=_opt("_jumps.csv"),
        falls_csv=_opt("_falls.csv"),
        joint_angles_csv=_opt("_joint_angles.csv"),
        frames=result.frames,
    )


# Re-exported defaults so app.py doesn't reach into config directly.
DEFAULT_CONF = config.DEFAULT_PERSON_CONFIDENCE
