
from typing import List, Optional, Tuple, Dict, Any # Added Dict, Any
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
from config_loader import config
from soccer.utils import get_bbox_center, round_tuple_coords, line_segments_intersect # Added line_segments_intersect

# Default paths for board images, consider making these configurable
DEFAULT_POSSESSION_BOARD_IMG_PATH = Path(config['paths']['possession_board_img'])
DEFAULT_PASSES_BOARD_IMG_PATH = Path(config['paths']['passes_board_img'])

class Match:
    # Configuration constants from config.yaml
    ui_cfg = config['ui_constants']
    match_defaults_cfg = ui_cfg['match_defaults']
    score_bar_cfg = ui_cfg['score_bar']
    counter_rect_cfg = ui_cfg['counter_rect']
    counter_layout_cfg = ui_cfg['counter_layout']

    DEFAULT_POSSESSION_COUNTER_THRESHOLD = match_defaults_cfg['possession_counter_threshold']
    DEFAULT_BALL_DISTANCE_THRESHOLD = match_defaults_cfg['ball_distance_threshold'] # pixels

    BAR_HEIGHT = score_bar_cfg['height']
    BAR_WIDTH = score_bar_cfg['width']
    BAR_RATIO_MIN_PROTECTION = score_bar_cfg['ratio_min_protection']
    BAR_RATIO_MAX_PROTECTION = score_bar_cfg['ratio_max_protection']
    BAR_TEXT_MIN_RATIO_DISPLAY = score_bar_cfg['text_min_ratio_display_home'] # Min ratio to display text for home team
    BAR_TEXT_MAX_RATIO_DISPLAY = score_bar_cfg['text_max_ratio_display_away'] # Max ratio to display text for away team

    COUNTER_RECT_HEIGHT = counter_rect_cfg['height']
    COUNTER_RECT_WIDTH = counter_rect_cfg['width']
    COUNTER_RECT_SPACING = counter_rect_cfg['spacing']

    COUNTER_BACKGROUND_OFFSET_X = counter_layout_cfg['background_offset_x']
    COUNTER_BACKGROUND_OFFSET_Y = counter_layout_cfg['background_offset_y']
    COUNTER_ELEMENTS_ORIGIN_X_OFFSET = counter_layout_cfg['elements_origin_x_offset']
    COUNTER_ELEMENTS_ORIGIN_Y_OFFSET = counter_layout_cfg['elements_origin_y_offset']
    COUNTER_BAR_ORIGIN_Y_OFFSET = counter_layout_cfg['bar_origin_y_offset']


    def __init__(
        self, 
        home: Team, 
        away: Team, 
        fps: int = 30,
        possession_counter_threshold: int = DEFAULT_POSSESSION_COUNTER_THRESHOLD, # Uses class/module level var
        ball_distance_threshold: int = DEFAULT_BALL_DISTANCE_THRESHOLD, # Uses class/module level var
        possession_board_img_path: str = str(DEFAULT_POSSESSION_BOARD_IMG_PATH), # Uses module level var
        passes_board_img_path: str = str(DEFAULT_PASSES_BOARD_IMG_PATH), # Uses module level var
    ):
        self.duration: int = 0
        self.home: Team = home
        self.away: Team = away
        self.team_possession: Optional[Team] = home # Start with home possession or None
        # self.current_team: Optional[Team] = home    # Removed, replaced by current_controlling_team
        self.possession_counter: int = 0 # Counts towards confirmed possession change
        self.closest_player: Optional[Player] = None
        self.ball: Optional[Ball] = None
        
        # New attributes for refined possession logic
        self.current_controlling_team: Optional[Team] = None # Stores the team currently 'in control'
        self.control_continuity_counter: int = 0 # Frames this team has been 'in control'

        # Load thresholds from config (via class attributes that already load from config)
        self.possession_counter_threshold: int = possession_counter_threshold # Existing
        self.ball_distance_threshold: int = ball_distance_threshold # Existing
        self.control_continuity_threshold: int = self.match_defaults_cfg['control_continuity_threshold'] # New

        self.fps: int = fps
        
        # Attributes for ball speed calculation
        self.previous_ball_center_abs: Optional[np.ndarray] = None
        self.current_ball_speed_abs: float = 0.0 # pixels per second

        # Load min_pass_speed_pps from config
        pass_event_config = config.get('ui_constants', {}).get('pass_event', {})
        self.min_pass_speed_pps: float = pass_event_config.get('min_pass_speed_pps', 150.0)

        self.pass_event: PassEvent = PassEvent(min_pass_speed_pps=self.min_pass_speed_pps)

        # Pre-load background images
        self.possession_background_img: Optional[PIL.Image.Image] = self._load_background_image(possession_board_img_path)
        self.passes_background_img: Optional[PIL.Image.Image] = self._load_background_image(passes_board_img_path)

        self.player_positions_history: List[Dict[str, Any]] = []

        # Load goal definitions
        self.goal_definitions = config.get('pitch_layout', {}).get('goals', [])
        self.goals_by_team: Dict[str, Dict[str, List[int]]] = {} # Stores {team_name: {'post1': [x,y], 'post2': [x,y]}} for the goal they ATTACK

        if len(self.goal_definitions) == 2 and len(config.get('teams', [])) == 2:
            # This assumes self.home and self.away are set based on the first two teams in config.teams
            # A more robust mapping would use team names directly if available from earlier team loading.
            # For now, using the assumption that self.home.name and self.away.name match config['teams'][0]['name'] and config['teams'][1]['name']

            # Determine team names from the already instantiated self.home and self.away objects
            # This is safer than re-accessing config['teams'] and assuming order.
            # The Match instance receives Team objects, so we should use their names.

            # We need to map which goal (by its defended_by_team_index) corresponds to self.home and self.away.
            # Let's find which team in config.teams matches self.home and self.away

            config_teams_data = config.get('teams', [])
            home_team_name_from_config = None
            away_team_name_from_config = None

            if self.home and self.away: # Ensure home and away teams are set
                # This logic assumes that the order of teams passed to Match constructor (home, away)
                # corresponds to the order in config['teams'] if we want to use indices 0 and 1 from config.
                # A better way is to match by name if the 'teams' in config has names 'Chelsea', 'Man City'
                # and self.home/self.away also have these names.

                # Simplified assumption: The first team in config.teams is one of (home/away), second is the other.
                # And defended_by_team_index refers to this order.
                team_0_cfg_name = config_teams_data[0]['name']
                team_1_cfg_name = config_teams_data[1]['name']

                for goal_def in self.goal_definitions:
                    def_team_idx = goal_def.get('defended_by_team_index')
                    goal_coords = {'post1': goal_def['post1'], 'post2': goal_def['post2']}

                    if def_team_idx == 0: # Goal defended by team at index 0 in config.teams
                        # The other team (index 1) attacks this goal.
                        self.goals_by_team[team_1_cfg_name] = goal_coords
                    elif def_team_idx == 1: # Goal defended by team at index 1 in config.teams
                        # The other team (index 0) attacks this goal.
                        self.goals_by_team[team_0_cfg_name] = goal_coords
            else: # Should not happen if Match is initialized correctly
                 print("Warning: Home or Away team not set in Match object during goal assignment.")

        else:
            print("Warning: Could not map goals to teams. Expected 2 goals and 2 teams in config for current auto-assignment logic.")
            # self.goals_by_team will be empty, shot detection might not work.

        # Load shot detection parameters
        shot_config = config.get('shot_detection', {})
        self.shot_speed_threshold_pps: float = shot_config.get('shot_speed_threshold_pps', 500.0)

        # Initialize storage for shot attempts
        self.shot_attempts_history: List[Dict[str, Any]] = [] # To store detected shot attempts

        self.coord_transformations: Optional[Any] = None


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

    def update(self, players: List[Player], ball: Optional[Ball], coord_transformations: Optional[Any]): # Using Any for now
        self.coord_transformations = coord_transformations # Store it
        self.update_possession_duration() # This still increments self.duration and active team's possession time

        if ball is None or ball.detection is None:
            self.closest_player = None
            self.current_controlling_team = None
            self.control_continuity_counter = 0
            self.possession_counter = 0
            self.previous_ball_center_abs = None # Add this reset
            self.current_ball_speed_abs = 0.0    # Add this reset
            self.pass_event.update(closest_player=None, ball=None, ball_speed=0.0)
            self.pass_event.process_pass()
            return

        self.ball = ball

        # New Ball Speed Calculation Logic:
        if self.ball and self.ball.detection and self.ball.center_abs is not None:
            current_center_abs = self.ball.center_abs
            if self.previous_ball_center_abs is not None:
                displacement_vector = np.array(current_center_abs) - np.array(self.previous_ball_center_abs)
                displacement_pixels = np.linalg.norm(displacement_vector)
                if self.fps > 0:
                    self.current_ball_speed_abs = displacement_pixels * self.fps
                else:
                    self.current_ball_speed_abs = 0.0
            else:
                self.current_ball_speed_abs = 0.0
            self.previous_ball_center_abs = np.array(current_center_abs).copy()
        else:
            self.previous_ball_center_abs = None
            self.current_ball_speed_abs = 0.0

        if not players:  # No players detected
            self.closest_player = None
            self.current_controlling_team = None
            self.control_continuity_counter = 0
            self.possession_counter = 0
            self.pass_event.update(closest_player=None, ball=ball, ball_speed=self.current_ball_speed_abs)
            self.pass_event.process_pass()
            return

        # Find closest player (existing logic for this can be kept)
        valid_players = [p for p in players if p.distance_to_ball(ball) is not None]
        if not valid_players:
            self.closest_player = None
            self.current_controlling_team = None
            self.control_continuity_counter = 0
            self.possession_counter = 0
            self.pass_event.update(closest_player=None, ball=ball, ball_speed=self.current_ball_speed_abs)
            self.pass_event.process_pass()
            return

        self.closest_player = min(valid_players, key=lambda player: player.distance_to_ball(ball))
        ball_distance = self.closest_player.distance_to_ball(ball)

        if ball_distance is None or ball_distance > self.ball_distance_threshold:
            self.closest_player = None # Player is too far
            self.current_controlling_team = None
            self.control_continuity_counter = 0
            self.possession_counter = 0
            self.pass_event.update(closest_player=None, ball=ball, ball_speed=self.current_ball_speed_abs)
            self.pass_event.process_pass()
            return

        # New Possession Logic Starts Here
        # ==================================

        # Control Establishment Phase
        if self.closest_player.team is not None: # Ensure player has a team
            if self.closest_player.team == self.current_controlling_team:
                self.control_continuity_counter += 1
            else:
                # New team is now closest, or was previously None
                self.current_controlling_team = self.closest_player.team
                self.control_continuity_counter = 1
                self.possession_counter = 0 # Reset confirmed possession counter for the new controlling team
        else:
            # Closest player has no team, treat as loss of control
            self.current_controlling_team = None
            self.control_continuity_counter = 0
            self.possession_counter = 0


        # Possession Confirmation Phase
        if self.current_controlling_team is not None and \
           self.control_continuity_counter >= self.control_continuity_threshold:

            # Current controlling team has established stable control.
            # Now, check if they meet the criteria for confirmed possession.
            self.possession_counter += 1

            if self.possession_counter >= self.possession_counter_threshold:
                if self.team_possession != self.current_controlling_team:
                    self.change_team_possession(self.current_controlling_team)
        else:
            # Control not yet stable enough for the current_controlling_team,
            # or no team is currently controlling.
            # If a team *had* confirmed possession, they keep it for now unless challenged
            # and overcome by another team. If current_controlling_team is None,
            # and possession_counter was building for a previous team, it's already reset.
            # If self.current_controlling_team became None in this frame, possession_counter was reset too.
            pass # No change to self.possession_counter unless it was reset above


        # Pass detection (original logic)

        # --- Start of new block for recording player positions ---
        if players: # Ensure there are players
            for player_obj in players: # Changed variable name from 'player' to 'player_obj' to avoid conflict if 'player' is used elsewhere
                if player_obj.detection and \
                   player_obj.detection.absolute_points is not None and \
                   player_obj.detection.data and \
                   player_obj.detection.data.get('id') is not None:

                    # Get absolute center of the player's bounding box
                    # player_obj.detection.absolute_points is [[x1,y1],[x2,y2]]
                    abs_center_float = get_bbox_center(player_obj.detection.absolute_points)
                    if abs_center_float: # Ensure center calculation was successful
                        abs_center_coords = round_tuple_coords(abs_center_float)

                        history_entry = {
                            'frame': self.duration, # self.duration is incremented in update_possession_duration
                            'player_id': player_obj.detection.data.get('id'),
                            'team_name': player_obj.team.name if player_obj.team else "Unknown",
                            'position_abs_center': abs_center_coords
                        }
                        self.player_positions_history.append(history_entry)
        # --- End of new block ---

        # --- Shot Attempt Detection Logic ---
        if self.current_ball_speed_abs > self.shot_speed_threshold_pps and \
           self.ball and self.ball.center_abs is not None and \
           self.previous_ball_center_abs is not None:

            shooter_team_name = None
            # Prioritize team in official possession, then controlling team
            if self.team_possession:
                shooter_team_name = self.team_possession.name
            elif self.current_controlling_team: # Team that is currently controlling but not yet confirmed possession
                shooter_team_name = self.current_controlling_team.name

            if shooter_team_name:
                target_goal_data = self.goals_by_team.get(shooter_team_name) # Goal this team attacks

                if target_goal_data:
                    curr_ball_pos_np = np.array(self.ball.center_abs)
                    prev_ball_pos_np = np.array(self.previous_ball_center_abs) # Already stored as np.array

                    goal_post1_np = np.array(target_goal_data['post1'])
                    goal_post2_np = np.array(target_goal_data['post2'])

                    is_shot_towards_goal = False
                    if line_segments_intersect(prev_ball_pos_np, curr_ball_pos_np, goal_post1_np, goal_post2_np):
                        is_shot_towards_goal = True

                    if is_shot_towards_goal:
                        shooter_player_id = None
                        if self.closest_player and self.closest_player.team and self.closest_player.team.name == shooter_team_name:
                            if self.closest_player.detection and self.closest_player.detection.data:
                                 shooter_player_id = self.closest_player.detection.data.get('id')

                        target_goal_id_str = "Unknown"
                        for g_def in self.goal_definitions:
                            if g_def['post1'] == target_goal_data['post1'] and g_def['post2'] == target_goal_data['post2']:
                                target_goal_id_str = g_def.get('id', "Unknown")
                                break

                        shot_attempt = {
                            'frame': self.duration,
                            'shooter_team_name': shooter_team_name,
                            'shooter_player_id': shooter_player_id,
                            'ball_start_pos': prev_ball_pos_np.tolist(),
                            'ball_current_pos': curr_ball_pos_np.tolist(),
                            'ball_speed_pps': self.current_ball_speed_abs,
                            'target_goal_id': target_goal_id_str,
                            'target_goal_posts': {'post1': target_goal_data['post1'], 'post2': target_goal_data['post2']}
                        }
                        self.shot_attempts_history.append(shot_attempt)
                        # print(f"Shot attempt recorded at frame {self.duration} by team {shooter_team_name}") # Optional debug
        # --- End of Shot Attempt Detection Logic ---

        self.pass_event.update(closest_player=self.closest_player, ball=ball, ball_speed=self.current_ball_speed_abs)
        self.pass_event.process_pass()

    def draw_shot_attempts(self, frame: PIL.Image.Image) -> PIL.Image.Image:
        if self.coord_transformations is None:
            return frame

        draw = PIL.ImageDraw.Draw(frame)

        for shot_attempt in self.shot_attempts_history:
            # Only draw shots detected in the current frame for non-persistent visualization
            if shot_attempt['frame'] == self.duration:
                abs_start_pos = np.array(shot_attempt['ball_start_pos'])
                abs_current_pos = np.array(shot_attempt['ball_current_pos'])

                rel_start_pos_list = self.coord_transformations.abs_to_rel(np.array([abs_start_pos]))
                rel_current_pos_list = self.coord_transformations.abs_to_rel(np.array([abs_current_pos]))

                if rel_start_pos_list is not None and rel_current_pos_list is not None:
                    rel_start_pos = tuple(map(int, rel_start_pos_list[0]))
                    rel_current_pos = tuple(map(int, rel_current_pos_list[0]))

                    draw.line([rel_start_pos, rel_current_pos], fill=(255, 255, 0, 200), width=3) # Yellow, slightly transparent
        return frame

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