"""Pluggable detector / pose backbones for the backbone-comparison benchmark.

The injury pipeline factors into a DETECTOR (person boxes + ByteTrack ids) and a
top-down POSE estimator (17 COCO keypoints per box). Both sit behind a tiny
interface so they can be swapped for an A/B comparison:

    detector:  yolo   -> YOLO + ByteTrack            (default)
               rfdetr -> RF-DETR + supervision ByteTrack
    pose:      rtm       -> RTMPose                   (default)
               mediapipe -> BlazePose, remapped to COCO-17

``build_detector`` / ``build_pose`` construct one by name. The alternative
adapters import their heavy packages (``rfdetr``, ``supervision``, ``mediapipe``)
*lazily*, so the default ``yolo`` + ``rtm`` path keeps working without the
benchmark extras installed. The tracker (ByteTrack) is held constant across
detectors so the comparison isolates the detector, not the association method.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .. import config

DETECTORS: tuple[str, ...] = ("yolo", "rfdetr")
POSES: tuple[str, ...] = ("rtm", "mediapipe")
DEFAULT_VARIANT = "yolo_rtm"


@runtime_checkable
class Detector(Protocol):
    """Per-frame person detection with stable ByteTrack ids."""

    def detect(self, frame_bgr: np.ndarray) -> list: ...
    def reset(self) -> None: ...


@runtime_checkable
class PoseBackend(Protocol):
    """Top-down pose: boxes -> ``(keypoints (N,17,2), scores (N,17))`` in COCO order."""

    def estimate(
        self, frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]: ...


def build_detector(name: str, confidence: float) -> Detector:
    name = name.lower()
    if name == "yolo":
        from .detector import PersonDetector

        det = PersonDetector(confidence=confidence)
        det.reset()
        return det
    if name == "rfdetr":
        from .detector_rfdetr import RFDetrDetector

        return RFDetrDetector(confidence=confidence)
    raise ValueError(f"unknown detector {name!r}; choose from {DETECTORS}")


def build_pose(name: str) -> PoseBackend:
    name = name.lower()
    if name == "rtm":
        from .rtm_pose import RTMPoseEstimator

        return RTMPoseEstimator()
    if name == "mediapipe":
        from .pose_mediapipe import MediaPipePoseEstimator

        return MediaPipePoseEstimator(
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=config.MEDIAPIPE_MIN_DET_CONF,
        )
    raise ValueError(f"unknown pose backend {name!r}; choose from {POSES}")


def variant_name(detector: str, pose: str) -> str:
    """Canonical ``<detector>_<pose>`` id used for output subdirs and report rows."""
    return f"{detector.lower()}_{pose.lower()}"
