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

# Helper function to determine orientation of ordered triplet (p, q, r).
# Returns:
# 0 --> p, q, r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def get_orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

# Helper function to check if point q lies on line segment pr
def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def line_segments_intersect(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> bool:
    """
    Checks if line segment 'p1q1' intersects line segment 'p2q2'.
    Points are expected as numpy arrays e.g., np.array([x, y]).
    """
    # Find the four orientations needed for general and special cases
    o1 = get_orientation(p1, q1, p2)
    o2 = get_orientation(p1, q1, q2)
    o3 = get_orientation(p2, q2, p1)
    o4 = get_orientation(p2, q2, q1)

    # General case: segments intersect if orientations are different
    if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        if o1 != o2 and o3 != o4:
            return True

    # Special Cases for collinear points:
    # Check if the segments are collinear and overlap.

    # p1, q1, p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1): return True
    # p1, q1, q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1): return True
    # p2, q2, p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2): return True
    # p2, q2, q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False # Doesn't fall in any of the above cases