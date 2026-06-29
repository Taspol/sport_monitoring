"""Person detection + short-term tracking with YOLO11 + ByteTrack.

``model.track(persist=True)`` runs YOLO detection and ByteTrack association, so
each person carries a stable *tracklet* id across frames. The colour-identity
layer downstream maps each tracklet to a selected player id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .. import config

_PERSON_CLASS_ID = 0

Box = tuple[int, int, int, int]


def _overlap(a: Box, b: Box) -> tuple[float, float]:
    """Return ``(iou, intersection_over_smaller)`` for two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0, 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    iou = inter / union if union > 0 else 0.0
    ios = inter / smaller if smaller > 0 else 0.0
    return iou, ios


def _is_duplicate(a: Box, b: Box) -> bool:
    """True if box ``b`` is a near-duplicate of (or nested inside) box ``a``."""
    iou, ios = _overlap(a, b)
    return iou >= config.DETECT_DEDUP_IOU or ios >= config.DETECT_DEDUP_CONTAIN


def _dedup_boxes(boxes: list[Box]) -> list[Box]:
    """Greedy duplicate suppression for plain boxes (keep the larger one)."""
    order = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
        reverse=True,
    )
    kept: list[Box] = []
    for i in order:
        if not any(_is_duplicate(k, boxes[i]) for k in kept):
            kept.append(boxes[i])
    return kept


@dataclass(frozen=True)
class Detection:
    """A detected person with a ByteTrack tracklet id."""

    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def _dedup_detections(dets: list[Detection]) -> list[Detection]:
    """Drop near-duplicate detections, keeping the higher-confidence box."""
    order = sorted(range(len(dets)), key=lambda i: dets[i].confidence, reverse=True)
    kept: list[Detection] = []
    for i in order:
        if not any(_is_duplicate(k.box, dets[i].box) for k in kept):
            kept.append(dets[i])
    return kept


class PersonDetector:
    """Detect and track people across frames with YOLO11 + ByteTrack."""

    def __init__(
        self,
        model_path: Path = config.YOLO_MODEL_PATH,
        confidence: float = config.DEFAULT_PERSON_CONFIDENCE,
    ) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def reset(self) -> None:
        """Drop tracker state so tracklet ids restart for a new video."""
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers = None

    def detect_boxes(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect person boxes on a single frame without tracking (for selection)."""
        results = self.model.predict(
            frame_bgr,
            classes=[_PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        h, w = frame_bgr.shape[:2]
        boxes = [
            (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
            for x1, y1, x2, y2 in results[0].boxes.xyxy.cpu().numpy()
        ]
        return _dedup_boxes(boxes)

    def _track(self, frame_bgr: np.ndarray):
        return self.model.track(
            frame_bgr,
            classes=[_PERSON_CLASS_ID],
            conf=self.confidence,
            persist=True,
            tracker=str(config.BYTETRACK_CONFIG),
            verbose=False,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Return tracked person detections (each with a ByteTrack tracklet id)."""
        results = self._track(frame_bgr)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        h, w = frame_bgr.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        detections: list[Detection] = []
        for (x1, y1, x2, y2), track_id, conf in zip(xyxy, ids, confs):
            detections.append(
                Detection(
                    track_id=int(track_id),
                    x1=max(0, int(x1)),
                    y1=max(0, int(y1)),
                    x2=min(w, int(x2)),
                    y2=min(h, int(y2)),
                    confidence=float(conf),
                )
            )
        return _dedup_detections(detections)
