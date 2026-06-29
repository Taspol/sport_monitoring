"""Person detection and pose-estimation backbones.

YOLO + ByteTrack (default) or RF-DETR for detection; RTMPose (default) or
MediaPipe BlazePose for pose. The RF-DETR and MediaPipe backends are imported
lazily by :mod:`.backbones`, so the default ``yolo + rtm`` path needs neither
of their extra dependencies installed.
"""

from .backbones import DETECTORS, POSES, build_detector, build_pose, variant_name
from .detector import Detection, PersonDetector
from .rtm_pose import RTMPoseEstimator

__all__ = [
    "DETECTORS",
    "POSES",
    "build_detector",
    "build_pose",
    "variant_name",
    "Detection",
    "PersonDetector",
    "RTMPoseEstimator",
]
