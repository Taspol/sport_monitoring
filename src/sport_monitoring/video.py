"""Video discovery, reading and writing helpers built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class VideoInfo:
    """Basic properties of an opened video."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


def find_videos(data_dir: Path = config.DATA_DIR) -> list[Path]:
    """Return all supported video files in ``data_dir``, sorted by name."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    videos = [
        p
        for p in sorted(data_dir.iterdir())
        if p.suffix.lower() in config.VIDEO_EXTENSIONS
    ]
    if not videos:
        raise FileNotFoundError(f"No video files found in {data_dir}")
    return videos


def probe(path: Path) -> VideoInfo:
    """Read metadata for a single video without decoding all frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frames(
    path: Path, start_frame: int = 0
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` pairs from ``start_frame`` onward.

    ``frame_index`` is the absolute index in the video (so it still lines up with
    the frame the user selected on).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        index = start_frame
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


class VideoWriter:
    """Thin wrapper around ``cv2.VideoWriter`` with sensible mp4 defaults."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise OSError(f"Could not open writer for: {path}")
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
