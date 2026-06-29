"""MediaPipe BlazePose adapter exposing the RTMPose ``.estimate`` interface.

BlazePose returns 33 landmarks; the injury pipeline (and every downstream LESS /
angle calculation) expects the 17 COCO keypoints in COCO order. ``_BLAZE_TO_COCO``
does that remap so the rest of the system is *identical* regardless of pose
backend -- otherwise the benchmark would compare adapters, not models.

This uses MediaPipe's **Tasks** API (``PoseLandmarker``); recent mediapipe wheels
ship only that, not the legacy ``mp.solutions.pose``. The ``.task`` model asset is
downloaded once on first use and cached under ``~/.cache/sport_monitoring``.
MediaPipe Pose is single-person, so we run it once per detected box crop and place
the normalized landmarks back into full-frame pixel coordinates; ``visibility`` is
the per-keypoint score (the role RTMPose's confidence plays).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import requests

_BLAZE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

_VARIANTS = {0: "pose_landmarker_lite", 1: "pose_landmarker_full", 2: "pose_landmarker_heavy"}
_BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
_CACHE_DIR = Path.home() / ".cache" / "sport_monitoring"


def _ensure_model(variant: str) -> Path:
    """Download (once) and cache the ``<variant>.task`` BlazePose bundle."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / f"{variant}.task"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{_BASE_URL}/{variant}/float16/latest/{variant}.task"
    print(f"Downloading MediaPipe model {variant} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".task.part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        tmp.replace(dest)
    return dest


class MediaPipePoseEstimator:
    """BlazePose (Tasks API) per-crop estimator that emits 17 COCO keypoints/box."""

    def __init__(
        self, model_complexity: int = 1, min_detection_confidence: float = 0.5
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = _ensure_model(_VARIANTS.get(model_complexity, _VARIANTS[1]))
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def estimate(
        self, frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match RTMPose: return ``(keypoints (N,17,2), scores (N,17))``.

        Order matches ``boxes``; a box where BlazePose finds nothing yields zeros
        with score 0 (treated as not-visible downstream).
        """
        if not boxes:
            return np.empty((0, 17, 2)), np.empty((0, 17))
        h, w = frame_bgr.shape[:2]
        kpts = np.zeros((len(boxes), 17, 2), dtype=float)
        scores = np.zeros((len(boxes), 17), dtype=float)
        for i, (bx1, by1, bx2, by2) in enumerate(boxes):
            x1, y1 = max(0, int(bx1)), max(0, int(by1))
            x2, y2 = min(w, int(bx2)), min(h, int(by2))
            if x2 <= x1 or y2 <= y1:
                continue
            rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            res = self._landmarker.detect(mp_image)
            if not res.pose_landmarks:
                continue
            lm = res.pose_landmarks[0]
            cw, ch = x2 - x1, y2 - y1
            for c, b in enumerate(_BLAZE_TO_COCO):
                p = lm[b]
                kpts[i, c, 0] = x1 + p.x * cw
                kpts[i, c, 1] = y1 + p.y * ch
                scores[i, c] = float(p.visibility)
        return kpts, scores
