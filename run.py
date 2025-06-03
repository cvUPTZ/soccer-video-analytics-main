import argparse

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from config_loader import config # Add this import
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass
from visualizations import generate_player_heatmap # Added import

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default=config['paths']['default_video'], # Use config value
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default=config['paths']['default_ball_model'], type=str, help="Path to the model" # Use config value
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
player_detector = YoloV5()
ball_detector = YoloV5(model_path=args.model)

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
teams_data = config.get('teams', [])
teams = []
for team_data in teams_data:
    teams.append(
        Team(
            name=team_data['name'],
            abbreviation=team_data['abbreviation'],
            color=tuple(team_data['color']), # Ensure tuple
            board_color=tuple(team_data['board_color']) if team_data.get('board_color') else None, # Ensure tuple
            text_color=tuple(team_data['text_color']) if team_data.get('text_color') else None # Ensure tuple
        )
    )

# For simplicity, assume first team in config is home, second is away, or add specific keys in config.
# For now, let's assume:
home_team_data_cfg = next((t for t in teams_data if t['name'] == "Chelsea"), None)
away_team_data_cfg = next((t for t in teams_data if t['name'] == "Man City"), None)

if not home_team_data_cfg or not away_team_data_cfg:
    raise ValueError("Home or away team not found in config. Please define 'Chelsea' and 'Man City' in config.yaml for current setup.")

chelsea = next((t for t in teams if t.name == home_team_data_cfg['name']), None)
man_city = next((t for t in teams if t.name == away_team_data_cfg['name']), None)

if not chelsea or not man_city:
    raise ValueError("Could not instantiate Team objects for Chelsea or Man City from config.")

match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city # Or determine from config, for now keep Man City

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=20,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

for i, frame in enumerate(video):

    # Get Detections
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, ball, coord_transformations) # Add coord_transformations here

    # Draw
    frame_pil = PIL.Image.fromarray(frame) # Changed variable name for clarity
    frame_pil = frame_pil.convert("RGBA") # Ensure RGBA for drawing with alpha

    if args.possession:
        frame_pil = Player.draw_players(
            players=players, frame=frame_pil, confidence=False, id=True
        )

        frame_pil = path.draw(
            img=frame_pil,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame_pil = match.draw_possession_counter(
            frame_pil, counter_background=possession_background, debug=False
        )

        if ball:
            frame_pil = ball.draw(frame_pil)

    if args.passes:
        pass_list = match.passes

        frame_pil = Pass.draw_pass_list(
            img=frame_pil, passes=pass_list, coord_transformations=coord_transformations
        )

        frame_pil = match.draw_passes_counter(
            frame_pil, counter_background=passes_background, debug=False
        )

    # Draw shot attempts
    frame_pil = match.draw_shot_attempts(frame_pil)

    frame = np.array(frame_pil) # Convert back to numpy array for OpenCV

    # Write video
    video.write(frame)
# This section was already part of the replacement block above.
# The diff tool might sometimes show it this way if the search block is small
# and the replace block is large and contains the search block's lines.
# No further changes needed for this part based on the provided diff.

print("Video processing complete. Generating heatmaps...")

if match.player_positions_history:
    video_width = int(video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_dimensions = (video_width, video_height)

    # Generate overall heatmap
    generate_player_heatmap(
        positions_history=match.player_positions_history,
        output_path="heatmap_all_players.png",
        img_size=img_dimensions
    )

    # Generate heatmap for home team
    if match.home: # Ensure home team exists
        generate_player_heatmap(
            positions_history=match.player_positions_history,
            output_path=f"heatmap_{match.home.name.lower().replace(' ', '_')}.png",
            filter_team_name=match.home.name,
            img_size=img_dimensions
        )

    # Generate heatmap for away team
    if match.away: # Ensure away team exists
        generate_player_heatmap(
            positions_history=match.player_positions_history,
            output_path=f"heatmap_{match.away.name.lower().replace(' ', '_')}.png",
            filter_team_name=match.away.name,
            img_size=img_dimensions
        )

    print("Heatmap generation complete.")
else:
    print("No player position history recorded. Skipping heatmap generation.")
