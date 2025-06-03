
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw

from soccer.ball import Ball
from soccer.draw import Draw
from soccer.pass_event import Pass, PassEvent
from soccer.player import Player
from soccer.team import Team

# Default paths for board images, consider making these configurable
DEFAULT_POSSESSION_BOARD_IMG_PATH = Path("./images/possession_board.png")
DEFAULT_PASSES_BOARD_IMG_PATH = Path("./images/passes_board.png")

class Match:
    # Configuration constants
    DEFAULT_POSSESSION_COUNTER_THRESHOLD = 20
    DEFAULT_BALL_DISTANCE_THRESHOLD = 45 # pixels
    BAR_HEIGHT = 29
    BAR_WIDTH = 310
    BAR_RATIO_MIN_PROTECTION = 0.07
    BAR_RATIO_MAX_PROTECTION = 0.93
    BAR_TEXT_MIN_RATIO_DISPLAY = 0.15 # Min ratio to display text for home team
    BAR_TEXT_MAX_RATIO_DISPLAY = 0.85 # Max ratio to display text for away team (1.0 - 0.15)
    COUNTER_RECT_HEIGHT = 31
    COUNTER_RECT_WIDTH = 150
    COUNTER_RECT_SPACING = 10
    COUNTER_BACKGROUND_OFFSET_X = 540
    COUNTER_BACKGROUND_OFFSET_Y = 40
    COUNTER_ELEMENTS_ORIGIN_X_OFFSET = 35
    COUNTER_ELEMENTS_ORIGIN_Y_OFFSET = 130
    COUNTER_BAR_ORIGIN_Y_OFFSET = 195


    def __init__(
        self, 
        home: Team, 
        away: Team, 
        fps: int = 30,
        possession_counter_threshold: int = DEFAULT_POSSESSION_COUNTER_THRESHOLD,
        ball_distance_threshold: int = DEFAULT_BALL_DISTANCE_THRESHOLD,
        possession_board_img_path: str = str(DEFAULT_POSSESSION_BOARD_IMG_PATH),
        passes_board_img_path: str = str(DEFAULT_PASSES_BOARD_IMG_PATH),
    ):
        self.duration: int = 0
        self.home: Team = home
        self.away: Team = away
        self.team_possession: Optional[Team] = home # Start with home possession or None
        self.current_team: Optional[Team] = home    # Team currently closest to ball (before possession change)
        self.possession_counter: int = 0
        self.closest_player: Optional[Player] = None
        self.ball: Optional[Ball] = None
        
        self.possession_counter_threshold: int = possession_counter_threshold
        self.ball_distance_threshold: int = ball_distance_threshold
        self.fps: int = fps
        
        self.pass_event: PassEvent = PassEvent()

        # Pre-load background images
        self.possession_background_img: Optional[PIL.Image.Image] = self._load_background_image(possession_board_img_path)
        self.passes_background_img: Optional[PIL.Image.Image] = self._load_background_image(passes_board_img_path)

    def _load_background_image(self, img_path_str: str) -> Optional[PIL.Image.Image]:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"Warning: Background image not found at {img_path}")
            return None
        try:
            counter = PIL.Image.open(img_path).convert("RGBA")
            counter = Draw.add_alpha(counter, 210) # Add alpha channel
            # BGR to RGB conversion for PIL (if original was cv2 format, PIL opens as RGB)
            # counter_np = np.array(counter)
            # if counter_np.shape[2] == 4: # RGBA
            #     red, green, blue, alpha = counter_np.T
            #     # Assuming PIL ImageDraw works with RGB, this might not be needed if PIL handles it.
            #     # For safety, ensure it's in a standard format if issues arise.
            #     # counter_np_swapped = np.array([blue, green, red, alpha]).T 
            #     # counter = PIL.Image.fromarray(counter_np_swapped)
            # else: # RGB
            #     red, green, blue = counter_np.T
            #     counter_np_swapped = np.array([blue, green, red]).T
            #     counter = PIL.Image.fromarray(counter_np_swapped)
            
            counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
            return counter
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}")
            return None
        except Exception as e:
            print(f"Error loading background image {img_path}: {e}")
            return None

    def update(self, players: List[Player], ball: Optional[Ball]):
        self.update_possession_duration()

        if ball is None or ball.detection is None:
            self.closest_player = None
            # Pass event might still need to process if ball becomes None after possession
            self.pass_event.update(closest_player=None, ball=None) 
            self.pass_event.process_pass()
            return

        self.ball = ball

        if not players: # No players detected
            self.closest_player = None
            self.pass_event.update(closest_player=None, ball=ball)
            self.pass_event.process_pass()
            return

        # Find closest player
        try:
            # Filter out players with no valid distance to ball
            valid_players = [p for p in players if p.distance_to_ball(ball) is not None]
            if not valid_players:
                self.closest_player = None
                self.pass_event.update(closest_player=None, ball=ball)
                self.pass_event.process_pass()
                return

            closest_player = min(valid_players, key=lambda player: player.distance_to_ball(ball))
        except ValueError: # Should be caught by empty valid_players check
            self.closest_player = None
            self.pass_event.update(closest_player=None, ball=ball)
            self.pass_event.process_pass()
            return
            
        self.closest_player = closest_player
        ball_distance = closest_player.distance_to_ball(ball)

        if ball_distance is None or ball_distance > self.ball_distance_threshold:
            self.closest_player = None # Player is too far, effectively no one has the ball
            # Update pass event: player might have lost ball
            self.pass_event.update(closest_player=None, ball=ball) 
            self.pass_event.process_pass()
            return

        # Possession logic
        if closest_player.team != self.current_team :
            self.possession_counter = 0
            self.current_team = closest_player.team
        
        self.possession_counter += 1

        if (
            self.possession_counter >= self.possession_counter_threshold
            and closest_player.team is not None
        ):
            self.change_team_possession(closest_player.team)

        # Pass detection
        self.pass_event.update(closest_player=closest_player, ball=ball)
        self.pass_event.process_pass()

    def change_team_possession(self, team: Optional[Team]):
        self.team_possession = team

    def update_possession_duration(self):
        if self.team_possession is not None:
            self.team_possession.possession += 1
        self.duration += 1

    @property
    def home_possession_str(self) -> str:
        return f"{self.home.abbreviation}: {self.home.get_time_possession(self.fps)}"

    @property
    def away_possession_str(self) -> str:
        return f"{self.away.abbreviation}: {self.away.get_time_possession(self.fps)}"

    def __str__(self) -> str:
        return f"{self.home_possession_str} | {self.away_possession_str}"

    @property
    def time_possessions(self) -> str:
        return f"{self.home.name}: {self.home.get_time_possession(self.fps)} | {self.away.name}: {self.away.get_time_possession(self.fps)}"

    @property
    def passes(self) -> List["Pass"]:
        return self.home.passes + self.away.passes

    def _draw_stats_bar(
        self,
        frame: PIL.Image.Image,
        origin: Tuple[int, int],
        home_value: float,
        away_value: float,
        home_team: Team,
        away_team: Team,
    ) -> PIL.Image.Image:
        bar_x, bar_y = origin
        
        total_value = home_value + away_value
        ratio = home_value / total_value if total_value > 0 else 0.5 # Default to 50/50 if no values

        # Protect against too small rectangles for visual appeal
        if ratio < self.BAR_RATIO_MIN_PROTECTION:
            ratio = self.BAR_RATIO_MIN_PROTECTION
        if ratio > self.BAR_RATIO_MAX_PROTECTION:
            ratio = self.BAR_RATIO_MAX_PROTECTION

        # Define rectangles (these are tuples, so they won't be modified by draw_counter_rectangle)
        left_rect_points = (
            origin,
            (int(bar_x + ratio * self.BAR_WIDTH), int(bar_y + self.BAR_HEIGHT)),
        )
        right_rect_points = (
            (int(bar_x + ratio * self.BAR_WIDTH), bar_y),
            (int(bar_x + self.BAR_WIDTH), int(bar_y + self.BAR_HEIGHT)),
        )

        frame = self._draw_split_rounded_rectangle(
            frame=frame,
            ratio=ratio,
            left_rectangle_points=left_rect_points,
            left_color=home_team.board_color,
            right_rectangle_points=right_rect_points,
            right_color=away_team.board_color,
        )
        
        # Draw home text
        if ratio > self.BAR_TEXT_MIN_RATIO_DISPLAY: # Only draw if there's enough space
            home_text = f"{int(home_value / total_value * 100) if total_value > 0 else 0}%"
            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rect_points[0],
                width=left_rect_points[1][0] - left_rect_points[0][0],
                height=left_rect_points[1][1] - left_rect_points[0][1],
                text=home_text,
                color=home_team.text_color,
            )

        # Draw away text
        if (1 - ratio) > self.BAR_TEXT_MIN_RATIO_DISPLAY: # Symmetric condition for away
            away_text = f"{int(away_value / total_value * 100) if total_value > 0 else 0}%"
            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rect_points[0],
                width=right_rect_points[1][0] - right_rect_points[0][0],
                height=right_rect_points[1][1] - right_rect_points[0][1],
                text=away_text,
                color=away_team.text_color,
            )
        return frame

    def possession_bar(self, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
        return self._draw_stats_bar(
            frame,
            origin,
            self.home.possession,
            self.away.possession,
            self.home,
            self.away,
        )

    def passes_bar(self, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
        return self._draw_stats_bar(
            frame,
            origin,
            float(len(self.home.passes)),
            float(len(self.away.passes)),
            self.home,
            self.away,
        )

    def _draw_split_rounded_rectangle(
        self,
        frame: PIL.Image.Image,
        ratio: float,
        left_rectangle_points: tuple, # ((x1,y1), (x2,y2))
        left_color: tuple,
        right_rectangle_points: tuple, # ((x1,y1), (x2,y2))
        right_color: tuple,
        radius: int = 15,
        overlap_compensation: int = 20 # To make the larger rectangle slightly bigger for better rounded corners
    ) -> PIL.Image.Image:
        
        # Create copies of points to modify for drawing overlap
        mod_left_rect_points = (left_rectangle_points[0], list(left_rectangle_points[1]))
        mod_right_rect_points = (list(right_rectangle_points[0]), right_rectangle_points[1])

        # Draw first one rectangle or another in order to make the
        # rectangle bigger for better rounded corners. The larger part is drawn first.
        if ratio < 0.5: # Right side is larger or equal
            # Make right rectangle a bit wider to cover the seam
            mod_right_rect_points[0][0] -= overlap_compensation 
            frame = Draw.half_rounded_rectangle(
                frame, rectangle=tuple(map(tuple,mod_right_rect_points)), color=right_color, left=True, radius=radius
            )
            # Then draw left
            frame = Draw.half_rounded_rectangle(
                frame, rectangle=left_rectangle_points, color=left_color, radius=radius
            )
        else: # Left side is larger
            # Make left rectangle a bit wider
            mod_left_rect_points[1][0] += overlap_compensation
            frame = Draw.half_rounded_rectangle(
                frame, rectangle=tuple(map(tuple,mod_left_rect_points)), color=left_color, radius=radius
            )
            # Then draw right
            frame = Draw.half_rounded_rectangle(
                frame, rectangle=right_rectangle_points, color=right_color, left=True, radius=radius
            )
        return frame

    def draw_counter_background(
        self,
        frame: PIL.Image.Image,
        origin: tuple,
        counter_background: Optional[PIL.Image.Image],
    ) -> PIL.Image.Image:
        if counter_background:
            frame.paste(counter_background, origin, counter_background)
        return frame

    def draw_team_stat_counter(
        self,
        frame: PIL.Image.Image,
        text: str,
        counter_text: str,
        origin: tuple,
        color: tuple,
        text_color: tuple,
        height: int = COUNTER_RECT_HEIGHT,
        width: int = COUNTER_RECT_WIDTH,
        team_width_ratio: float = 0.417, # Proportion of width for team abbreviation
        radius: int = 20
    ) -> PIL.Image.Image:
        team_begin = origin
        team_w = int(width * team_width_ratio)

        team_rectangle_points = (
            team_begin,
            (team_begin[0] + team_w, team_begin[1] + height),
        )

        time_begin = (origin[0] + team_w, origin[1])
        time_w = width - team_w # Remaining width

        time_rectangle_points = (
            time_begin,
            (time_begin[0] + time_w, time_begin[1] + height),
        )

        frame = Draw.half_rounded_rectangle(
            img=frame, rectangle=team_rectangle_points, color=color, radius=radius, left=False # Rounded on left
        )
        frame = Draw.half_rounded_rectangle(
            img=frame, rectangle=time_rectangle_points, color=(239, 234, 229), radius=radius, left=True # Rounded on right
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame, origin=team_rectangle_points[0], height=height, width=team_w, text=text, color=text_color
        )
        frame = Draw.text_in_middle_rectangle(
            img=frame, origin=time_rectangle_points[0], height=height, width=time_w, text=counter_text, color="black"
        )
        return frame

    def draw_debug_info(self, frame: PIL.Image.Image) -> PIL.Image.Image:
        if self.closest_player and self.ball and self.closest_player.detection and self.ball.center:
            closest_foot = self.closest_player.closest_foot_to_ball(self.ball)
            if closest_foot is None or self.ball.center is None:
                return frame

            line_color = (0, 0, 0) # Default: black
            distance = self.closest_player.distance_to_ball(self.ball)
            if distance is not None and distance > self.ball_distance_threshold:
                line_color = (255, 0, 0) # Red if over threshold (original was white, red is more visible)

            draw = PIL.ImageDraw.Draw(frame)
            draw.line(
                [ tuple(closest_foot), tuple(self.ball.center) ],
                fill=line_color, width=2,
            )
        return frame

    def _draw_generic_counter_display(
        self,
        frame: PIL.Image.Image,
        counter_background: Optional[PIL.Image.Image],
        home_stat_str: str,
        away_stat_str: str,
        bar_draw_func, # e.g., self.possession_bar or self.passes_bar
        debug: bool = False,
    ) -> PIL.Image.Image:
        frame_width = frame.size[0]
        counter_origin = (frame_width - self.COUNTER_BACKGROUND_OFFSET_X, self.COUNTER_BACKGROUND_OFFSET_Y)

        frame = self.draw_counter_background(frame, origin=counter_origin, counter_background=counter_background)

        home_counter_origin_x = counter_origin[0] + self.COUNTER_ELEMENTS_ORIGIN_X_OFFSET
        home_counter_origin_y = counter_origin[1] + self.COUNTER_ELEMENTS_ORIGIN_Y_OFFSET
        
        frame = self.draw_team_stat_counter(
            frame,
            origin=(home_counter_origin_x, home_counter_origin_y),
            text=self.home.abbreviation,
            counter_text=home_stat_str,
            color=self.home.board_color,
            text_color=self.home.text_color,
        )
        
        away_counter_origin_x = home_counter_origin_x + self.COUNTER_RECT_WIDTH + self.COUNTER_RECT_SPACING
        
        frame = self.draw_team_stat_counter(
            frame,
            origin=(away_counter_origin_x, home_counter_origin_y),
            text=self.away.abbreviation,
            counter_text=away_stat_str,
            color=self.away.board_color,
            text_color=self.away.text_color,
        )
        
        bar_origin_y = counter_origin[1] + self.COUNTER_BAR_ORIGIN_Y_OFFSET
        frame = bar_draw_func(frame, origin=(home_counter_origin_x, bar_origin_y))

        if self.closest_player:
            frame = self.closest_player.draw_pointer(frame)

        if debug:
            frame = self.draw_debug_info(frame=frame)
        return frame

    def draw_possession_counter(
        self, frame: PIL.Image.Image, debug: bool = False
    ) -> PIL.Image.Image:
        return self._draw_generic_counter_display(
            frame,
            self.possession_background_img,
            self.home.get_time_possession(self.fps),
            self.away.get_time_possession(self.fps),
            self.possession_bar,
            debug,
        )

    def draw_passes_counter(
        self, frame: PIL.Image.Image, debug: bool = False
    ) -> PIL.Image.Image:
        return self._draw_generic_counter_display(
            frame,
            self.passes_background_img,
            str(len(self.home.passes)),
            str(len(self.away.passes)),
            self.passes_bar,
            debug,
        )