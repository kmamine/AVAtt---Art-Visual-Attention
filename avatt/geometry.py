"""Viewing geometry of the AVAtt recordings.

All values were measured from the recordings: the display scale comes from regressing Tobii's pixel gaze
coordinates on its millimetre coordinates, and the viewing distance is the mean eye-screen distance over fixations.
"""
import math

CANVAS_W, CANVAS_H = 1600, 900        # stimulus size in pixels (painting centred on grey RGB 170)
SCREEN_W, SCREEN_H = 1920, 1080       # display resolution
PX_PER_MM = 3.6161                    # display pixels per millimetre (531 mm wide display)
MEAN_DISTANCE_MM = 568.4              # mean eye-screen distance over all fixations
GREY = (170, 170, 170)                # background colour around the painting


def px_per_degree(distance_mm: float = MEAN_DISTANCE_MM) -> float:
    """Pixels spanned by one degree of visual angle at a given viewing distance."""
    return 2 * distance_mm * math.tan(math.radians(0.5)) * PX_PER_MM


PX_PER_DEGREE = px_per_degree()       # 35.87 px


def pixels_to_degrees(px: float, distance_mm: float = MEAN_DISTANCE_MM) -> float:
    """Convert an on-screen distance in pixels to degrees of visual angle."""
    return math.degrees(math.atan(px / PX_PER_MM / distance_mm))
