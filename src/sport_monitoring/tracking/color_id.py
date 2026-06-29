"""Pose-guided colour identity (torso-polygon HS histogram).

A player's jersey colour is captured as a **Hue-Saturation histogram** over the
torso region defined by their RTMPose keypoints (shoulders + hips). The polygon
hugs the body, so the histogram is dominated by jersey pixels rather than
background. Histograms capture the colour *distribution* (patterns, numbers,
trim), which separates players better than a single averaged colour, and they
degrade gracefully under partial occlusion. Two identities are compared by
histogram correlation.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import config

Point = tuple[int, int, float]


def keypoint_color(frame: np.ndarray, points: list[Point]) -> np.ndarray | None:
    """HS histogram over the torso polygon, or ``None`` if it can't be sampled.

    ``None`` means too few torso keypoints are confidently visible to form a
    region (need at least 3 of shoulders/hips above ``COLOR_ID_MIN_SCORE``).
    """
    h, w = frame.shape[:2]
    pts: list[tuple[int, int]] = []
    for idx in config.COLOR_ID_KEYPOINTS:
        if idx < len(points):
            x, y, score = points[idx]
            if score >= config.COLOR_ID_MIN_SCORE:
                pts.append((int(x), int(y)))
    if len(pts) < 3:
        return None

    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if int(cv2.countNonZero(mask)) < 12:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bins = list(config.COLOR_ID_HIST_BINS)
    hist = cv2.calcHist([hsv], [0, 1], mask, bins, [0, 180, 0, 256])
    hist = cv2.GaussianBlur(hist, (3, 3), 0)
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram correlation in [0, 1] (1 = identical distribution)."""
    corr = cv2.compareHist(
        np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32),
        cv2.HISTCMP_CORREL,
    )
    return max(0.0, float(corr))


def ema(ref: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average of two histograms."""
    return (1.0 - alpha) * ref + alpha * new


def dominant_hsv(hist: np.ndarray) -> np.ndarray:
    """Representative HSV colour (peak histogram bin) for display swatches."""
    bins_h, bins_s = config.COLOR_ID_HIST_BINS
    grid = np.asarray(hist, dtype=np.float32).reshape(bins_h, bins_s)
    hi, si = np.unravel_index(int(grid.argmax()), grid.shape)
    hue = (hi + 0.5) * 180.0 / bins_h
    sat = (si + 0.5) * 256.0 / bins_s
    return np.array([hue, sat, 200.0], dtype=np.float32)


def hsv_to_bgr(hsv: np.ndarray) -> tuple[int, int, int]:
    """Convert a single HSV colour to a BGR tuple for drawing swatches."""
    px = np.uint8([[hsv]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def color_name(hsv: np.ndarray) -> str:
    """Rough human-readable jersey-colour name from an (OpenCV) HSV colour."""
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 180 else "grey"
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"
