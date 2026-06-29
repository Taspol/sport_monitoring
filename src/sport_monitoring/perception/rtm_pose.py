"""Multi-person pose estimation with RTMPose (via rtmlib + ONNXRuntime).

RTMPose is a top-down estimator: given the frame and a set of person bounding
boxes, it returns 17 COCO keypoints per box in full-frame image coordinates.
We feed it the boxes produced by the YOLO + ByteTrack detector, so each pose is
already associated with a stable track id.
"""

from __future__ import annotations

import numpy as np
from rtmlib import RTMPose

from .. import config


class RTMPoseEstimator:
    """Thin wrapper around rtmlib's ``RTMPose`` for batched box → pose inference."""

    def __init__(
        self,
        onnx_model: str = config.RTMPOSE_ONNX_URL,
        input_size: tuple[int, int] = config.RTMPOSE_INPUT_SIZE,
        backend: str = config.RTMPOSE_BACKEND,
        device: str = config.RTMPOSE_DEVICE,
    ) -> None:
        self.model = RTMPose(
            onnx_model=onnx_model,
            model_input_size=input_size,
            backend=backend,
            device=device,
        )

    def estimate(
        self, frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate poses for ``boxes`` in one frame.

        Returns ``(keypoints, scores)`` with shapes ``(N, 17, 2)`` and
        ``(N, 17)``. ``N`` matches ``len(boxes)`` and preserves their order, so
        results line up with the detections they came from.
        """
        if not boxes:
            return np.empty((0, 17, 2)), np.empty((0, 17))
        keypoints, scores = self.model(frame_bgr, bboxes=[list(b) for b in boxes])
        return np.asarray(keypoints), np.asarray(scores)
