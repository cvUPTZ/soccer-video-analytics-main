from typing import Optional, Tuple # Add List if you use List[foo] as well
import norfair
import numpy as np

from soccer.draw import Draw
from soccer.utils import get_bbox_center, round_tuple_coords
# Forward declaration for type hinting
if False: # This is fine, it's just for static type checkers
    from soccer.match import Match


class Ball:
    def __init__(self, detection: Optional[norfair.Detection]):
        """
        Initialize Ball

        Parameters
        ----------
        detection : Optional[norfair.Detection]
            norfair.Detection containing the ball, or None.
        """
        self.detection = detection
        self.color: Optional[Tuple[int, int, int]] = None # Tuple is used here

    def set_color(self, match: "Match"):
        """
        Sets the color of the ball to the team color with the ball possession in the match.

        Parameters
        ----------
        match : Match
            Match object
        """
        if match.team_possession is None:
            self.color = None # Default color or clear
            if self.detection and self.detection.data and "color" in self.detection.data: # Check .data
                 del self.detection.data["color"] # Or set to a default
            return

        self.color = match.team_possession.color

        if self.detection:
            if self.detection.data is None: # Ensure data dict exists
                self.detection.data = {}
            self.detection.data["color"] = match.team_possession.color

    @property
    def center(self) -> Optional[Tuple[int, int]]: # Tuple used here
        """
        Returns the center of the ball, rounded to integers.

        Returns
        -------
        Optional[Tuple[int, int]]
            Center of the ball (x, y), or None if no detection.
        """
        if self.detection is None or self.detection.points is None:
            return None
        
        center_float = get_bbox_center(self.detection.points)
        return round_tuple_coords(center_float)

    @property
    def center_abs(self) -> Optional[Tuple[int, int]]: # Tuple used here
        """
        Returns the center of the ball in absolute coordinates, rounded to integers.

        Returns
        -------
        Optional[Tuple[int, int]]
            Center of the ball (x, y) in absolute coordinates, or None.
        """
        if self.detection is None or self.detection.absolute_points is None:
            return None
            
        center_float = get_bbox_center(self.detection.absolute_points)
        return round_tuple_coords(center_float)

    def draw(self, frame: np.ndarray) -> np.ndarray: # Assuming frame is np.ndarray from context, PIL was in run.py
        """
        Draw the ball on the frame.

        Parameters
        ----------
        frame : np.ndarray # Changed from PIL.Image.Image as Draw.draw_detection likely expects np.ndarray
            Frame to draw on.

        Returns
        -------
        np.ndarray
            Frame with ball drawn.
        """
        if self.detection is None:
            return frame
        # Ensure Draw.draw_detection is compatible with the frame type (np.ndarray or PIL.Image)
        # The error was in run.py, so this `draw` method signature needs careful check
        # If run.py converts to PIL then passes it, this should be PIL.Image.Image
        # If Draw.draw_detection works on np.ndarray, this is fine.
        # Given the original code, Draw.draw_detection was called with PIL image.
        # So, let's assume frame here is PIL.Image.Image for consistency with how it was used.
        # If frame here is meant to be np.ndarray, then Draw.draw_detection must support it.
        # For now, keeping as np.ndarray as it was previously, but this might need adjustment
        # based on Draw.draw_detection's actual implementation.

        # Reverting to PIL.Image.Image based on previous run.py logic
        # if self.detection is None:
        #     return frame # frame is PIL.Image.Image
        # return Draw.draw_detection(self.detection, frame) # frame is PIL.Image.Image
        
        # Let's stick to what the original `Ball.draw` was called with in `run.py` (which was a PIL Image)
        # However, the method `Draw.draw_detection` in `draw.py` was also updated to take `PIL.Image.Image`.
        # So, this should be:
        # def draw(self, frame: PIL.Image.Image) -> PIL.Image.Image:

        # To match the previous usage in run.py where frame was converted to PIL.Image
        # and then Ball.draw() was called:
        # The type hint should be PIL.Image.Image
        # def draw(self, frame: PIL.Image.Image) -> PIL.Image.Image:
        if self.detection is None:
            return frame # frame is PIL.Image.Image
        return Draw.draw_detection(self.detection, frame) # frame is PIL.Image.Image

    def __str__(self):
        return f"Ball: {self.center}"