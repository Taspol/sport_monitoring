"""Player identity, multi-frame tracking and reference-frame selection.

``FusedTracker`` / ``FusedSelectedTracker`` fuse ByteTrack motion with a
pose-guided jersey-colour cue to keep stable ids through occlusion. ``selector``
registers players from a reference frame (or auto-picks the jumper).
"""

from .color_id import (
    color_name,
    dominant_hsv,
    hsv_to_bgr,
    keypoint_color,
    similarity,
)
from .selector import (
    SelectedTarget,
    auto_select_player,
    register_all_players,
    select_one_player,
)
from .tracker import FusedSelectedTracker, FusedTracker, TrajectoryStore

__all__ = [
    "color_name",
    "dominant_hsv",
    "hsv_to_bgr",
    "keypoint_color",
    "similarity",
    "SelectedTarget",
    "auto_select_player",
    "register_all_players",
    "select_one_player",
    "FusedSelectedTracker",
    "FusedTracker",
    "TrajectoryStore",
]
