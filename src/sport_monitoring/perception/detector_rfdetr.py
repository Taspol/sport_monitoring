"""RF-DETR person detector with ByteTrack ids, mirroring ``PersonDetector``.

RF-DETR has no built-in tracker, so we pair it with ``supervision``'s ByteTrack to
mint stable tracklet ids -- the exact role ultralytics' ByteTrack plays for YOLO,
so the *association* stays constant and the benchmark isolates the detector. Both
``rfdetr`` and ``supervision`` import lazily, so this module costs nothing unless
``--detector rfdetr`` is selected.

NOTE: ``config.RFDETR_PERSON_CLASS_ID`` is the person class in RF-DETR's label
space (COCO-91 -> 1). If RF-DETR drops persons, check this against the build you
installed (some COCO-80 variants use 0).
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .detector import Detection, _dedup_detections


class RFDetrDetector:
    """RF-DETR detection + supervision ByteTrack, returning ``Detection`` objects."""

    def __init__(self, confidence: float = config.DEFAULT_PERSON_CONFIDENCE) -> None:
        from rfdetr import RFDETRBase
        import supervision as sv

        self.model = RFDETRBase()
        self.confidence = confidence
        self._sv = sv
        self._tracker = sv.ByteTrack()

    def reset(self) -> None:
        """Drop tracker state so tracklet ids restart for a new video."""
        self._tracker = self._sv.ByteTrack()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.model.predict(rgb, threshold=self.confidence)
        det = det[det.class_id == config.RFDETR_PERSON_CLASS_ID]
        det = self._tracker.update_with_detections(det)
        h, w = frame_bgr.shape[:2]
        out: list[Detection] = []
        ids = det.tracker_id if det.tracker_id is not None else [None] * len(det.xyxy)
        for (x1, y1, x2, y2), conf, tid in zip(det.xyxy, det.confidence, ids):
            if tid is None:
                continue
            out.append(
                Detection(
                    track_id=int(tid),
                    x1=max(0, int(x1)),
                    y1=max(0, int(y1)),
                    x2=min(w, int(x2)),
                    y2=min(h, int(y2)),
                    confidence=float(conf),
                )
            )
        return _dedup_detections(out)
