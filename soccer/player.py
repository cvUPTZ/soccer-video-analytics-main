
from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
from norfair import Detection

from soccer.ball import Ball
from soccer.draw import Draw # Assuming Draw class handles PIL.Image
from soccer.team import Team


class Player:
    def __init__(self, detection: Optional[Detection]):
        self.detection: Optional[Detection] = detection
        self.team: Optional[Team] = None

        if detection and detection.data and "team" in detection.data:
            # Ensure 'team' is actually a Team object, not just a name string
            if isinstance(detection.data["team"], Team):
                self.team = detection.data["team"]
            # else: print warning or handle if it's a name to be resolved

    def _get_foot_position(self, points: np.ndarray, side: str) -> Optional[np.ndarray]:
        """Helper to get foot position, assumes points is [[x1,y1],[x2,y2]]."""
        if points is None or points.shape != (2,2): return None
        x1, y1 = points[0]
        x2, y2 = points[1]
        if side == "left":
            return np.array([x1, y2]) # Bottom-left
        elif side == "right":
            return np.array([x2, y2]) # Bottom-right
        return None

    @property
    def left_foot(self) -> Optional[np.ndarray]: # Relative coordinates
        return self._get_foot_position(self.detection.points, "left") if self.detection else None

    @property
    def right_foot(self) -> Optional[np.ndarray]: # Relative coordinates
        return self._get_foot_position(self.detection.points, "right") if self.detection else None

    @property
    def left_foot_abs(self) -> Optional[np.ndarray]: # Absolute coordinates
        return self._get_foot_position(self.detection.absolute_points, "left") if self.detection else None

    @property
    def right_foot_abs(self) -> Optional[np.ndarray]: # Absolute coordinates
        return self._get_foot_position(self.detection.absolute_points, "right") if self.detection else None

    @property
    def feet(self) -> Optional[np.ndarray]: # Relative coordinates
        lf, rf = self.left_foot, self.right_foot
        if lf is not None and rf is not None:
            return np.array([lf, rf])
        return None

    def distance_to_ball(self, ball: Ball) -> Optional[float]:
        if self.detection is None or ball.center is None:
            return None

        left_f = self.left_foot
        right_f = self.right_foot
        
        distances = []
        if left_f is not None:
            distances.append(np.linalg.norm(ball.center - left_f))
        if right_f is not None:
            distances.append(np.linalg.norm(ball.center - right_f))
        
        return min(distances) if distances else None


    def closest_foot_to_ball(self, ball: Ball) -> Optional[np.ndarray]: # Relative
        if self.detection is None or ball.center is None:
            return None

        left_f = self.left_foot
        right_f = self.right_foot
        ball_c = ball.center

        dist_lf = np.linalg.norm(ball_c - left_f) if left_f is not None else float('inf')
        dist_rf = np.linalg.norm(ball_c - right_f) if right_f is not None else float('inf')

        if dist_lf == float('inf') and dist_rf == float('inf'): return None
        return left_f if dist_lf <= dist_rf else right_f

    def closest_foot_to_ball_abs(self, ball: Ball) -> Optional[np.ndarray]: # Absolute
        if self.detection is None or ball.center_abs is None:
            return None

        left_f_abs = self.left_foot_abs
        right_f_abs = self.right_foot_abs
        ball_c_abs = ball.center_abs
        
        dist_lf = np.linalg.norm(ball_c_abs - left_f_abs) if left_f_abs is not None else float('inf')
        dist_rf = np.linalg.norm(ball_c_abs - right_f_abs) if right_f_abs is not None else float('inf')
        
        if dist_lf == float('inf') and dist_rf == float('inf'): return None
        return left_f_abs if dist_lf <= dist_rf else right_f_abs

    def draw(
        self, frame: PIL.Image.Image, confidence: bool = False, id_label: bool = False # <--- Ensure this is 'id_label'
    ) -> PIL.Image.Image:
        if self.detection is None:
            return frame

        if self.detection.data is None: self.detection.data = {}
        
        if self.team is not None:
            self.detection.data["color"] = self.team.color
        elif "color" in self.detection.data:
            del self.detection.data["color"]

        # Pass the id_label argument correctly
        return Draw.draw_detection(self.detection, frame, confidence=confidence, id_label=id_label)
    
    
    # def draw(
    #     self, frame: PIL.Image.Image, confidence: bool = False, id_label: bool = False
    # ) -> PIL.Image.Image:
    #     if self.detection is None:
    #         return frame

    #     # Ensure detection.data exists
    #     if self.detection.data is None: self.detection.data = {}
        
    #     if self.team is not None:
    #         self.detection.data["color"] = self.team.color # For drawing
    #     elif "color" in self.detection.data: # Remove color if no team, or set default
    #         del self.detection.data["color"]

    #     return Draw.draw_detection(self.detection, frame, confidence=confidence, id_label=id_label)

    def draw_pointer(self, frame: PIL.Image.Image) -> PIL.Image.Image:
        if self.detection is None:
            return frame
        
        pointer_color = self.team.color if self.team else (128, 128, 128) # Grey if no team
        return Draw.draw_pointer(detection=self.detection, img=frame, color=pointer_color)

    def __str__(self):
        team_name = self.team.name if self.team else "N/A"
        return f"Player ID: {self.detection.data.get('id', 'N/A') if self.detection else 'N/A'}, Team: {team_name}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Player):
            return NotImplemented
        if self.detection is None or other.detection is None:
            return False # Or handle as per desired logic for None detections
        if self.detection.data is None or other.detection.data is None:
            return False

        self_id = self.detection.data.get("id")
        other_id = other.detection.data.get("id")

        return self_id is not None and self_id == other_id

    @staticmethod
    def have_same_id(player1: Optional["Player"], player2: Optional["Player"]) -> bool:
        if not player1 or not player2: return False
        if player1.detection is None or player2.detection is None: return False
        if player1.detection.data is None or player2.detection.data is None: return False
        
        p1_id = player1.detection.data.get("id")
        p2_id = player2.detection.data.get("id")
        
        return p1_id is not None and p1_id == p2_id

    @staticmethod
    def draw_players(
        players: List["Player"],
        frame: PIL.Image.Image,
        confidence: bool = False,
        id_label: bool = False,
    ) -> PIL.Image.Image:
        for player in players:
            frame = player.draw(frame, confidence=confidence, id_label=id_label)
        return frame

    @staticmethod
    def from_detections(
        detections: List[Optional[Detection]], teams: List[Team]
    ) -> List["Player"]:
        players = []
        for detection in detections:
            if detection is None:
                continue
            
            # Ensure data dictionary exists
            if detection.data is None: detection.data = {}

            if "classification" in detection.data:
                team_name = detection.data["classification"]
                # Team.from_name should handle if team_name is not found
                team_obj = Team.from_name(teams=teams, name=str(team_name))
                if team_obj: # Only assign if team is found
                    detection.data["team"] = team_obj 
                # else: 'team' field remains unset or handled by Player.__init__
            
            players.append(Player(detection=detection))
        return players