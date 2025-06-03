from typing import Iterable, Tuple, Optional
import numpy as np

def get_midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Calculates the midpoint between two 2D points."""
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

def get_bbox_center(points: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Returns the center of a bounding box defined by two points.
    Assumes points is a 2x2 np.array like [[x1, y1], [x2, y2]].
    """
    if not isinstance(points, np.ndarray) or points.shape != (2, 2):
        # Consider raising ValueError for invalid input
        return None
    return get_midpoint(points[0], points[1])

def round_tuple_coords(coords: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
    """Rounds all elements in a tuple of coordinates to integers."""
    if coords is None:
        return None
    return tuple(np.round(coords).astype(int))

def round_iterable(iterable: Iterable) -> list:
    """Rounds all numeric items in an iterable to integers."""
    return [int(round(item)) for item in iterable if isinstance(item, (int, float))]