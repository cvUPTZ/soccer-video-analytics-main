from typing import List

import norfair
import numpy as np
from norfair import Detection
from norfair.camera_motion import MotionEstimator

from config_loader import config
from inference import Converter, YoloV5
from soccer import Ball, Match


def get_ball_detections(
    ball_detector: YoloV5, frame: np.ndarray
) -> List[norfair.Detection]:
    """
    Uses custom Yolov5 detector in order
    to get the predictions of the ball and converts it to
    Norfair.Detection list.

    Parameters
    ----------
    ball_detector : YoloV5
        YoloV5 detector for balls
    frame : np.ndarray
        Frame to get the ball detections from

    Returns
    -------
    List[norfair.Detection]
        List of ball detections
    """
    ball_df = ball_detector.predict(frame)
    ball_df = ball_df[ball_df["confidence"] > config['detection']['ball_confidence_threshold']]
    return Converter.DataFrame_to_Detections(ball_df)


def get_player_detections(
    person_detector: YoloV5, frame: np.ndarray
) -> List[norfair.Detection]:
    """
    Uses YoloV5 Detector in order to detect the players
    in a match and filter out the detections that are not players
    and have confidence lower than 0.35.

    Parameters
    ----------
    person_detector : YoloV5
        YoloV5 detector
    frame : np.ndarray
        _description_

    Returns
    -------
    List[norfair.Detection]
        List of player detections
    """

    person_df = person_detector.predict(frame)
    person_df = person_df[person_df["name"] == "person"]
    person_df = person_df[person_df["confidence"] > config['detection']['player_confidence_threshold']]
    person_detections = Converter.DataFrame_to_Detections(person_df)
    return person_detections


def create_mask(frame: np.ndarray, detections: List[norfair.Detection]) -> np.ndarray:
    """

    Creates mask in order to hide detections and goal counter for motion estimation

    Parameters
    ----------
    frame : np.ndarray
        Frame to create mask for.
    detections : List[norfair.Detection]
        Detections to hide.

    Returns
    -------
    np.ndarray
        Mask.
    """

    if not detections:
        mask = np.ones(frame.shape[:2], dtype=frame.dtype)
    else:
        detections_df = Converter.Detections_to_DataFrame(detections)
        mask = YoloV5.generate_predictions_mask(detections_df, frame, margin=40)

    # remove goal counter
    mask_config = config['detection']['goal_counter_mask']
    mask[
        mask_config['y_start']:mask_config['y_end'],
        mask_config['x_start']:mask_config['x_end']
    ] = 0

    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an img

    Parameters
    ----------
    img : np.ndarray
        Image to apply the mask to
    mask : np.ndarray
        Mask to apply

    Returns
    -------
    np.ndarray
        img with mask applied
    """
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img


def update_motion_estimator(
    motion_estimator: MotionEstimator,
    detections: List[Detection],
    frame: np.ndarray,
) -> "CoordinatesTransformation":
    """

    Update coordinate transformations every frame

    Parameters
    ----------
    motion_estimator : MotionEstimator
        Norfair motion estimator class
    detections : List[Detection]
        List of detections to hide in the mask
    frame : np.ndarray
        Current frame

    Returns
    -------
    CoordinatesTransformation
        Coordinate transformation for the current frame
    """

    mask = create_mask(frame=frame, detections=detections)
    coord_transformations = motion_estimator.update(frame, mask=mask)
    return coord_transformations


def get_main_ball(detections: List[Detection], match: Match = None) -> Ball:
    """
    Gets the main ball from a list of balls detection.
    If multiple balls are detected, the one with the highest confidence score is chosen.

    The match is used in order to set the color of the ball to
    the color of the team in possession of the ball.

    Parameters
    ----------
    detections : List[Detection]
        List of ball detections from the detector. Each detection is expected
        to have a confidence score in `detection.data["p"]`.
    match : Match, optional
        Match object, by default None

    Returns
    -------
    Ball
        Main ball object (can be a Ball with no detection if no balls are found).
    """
    ball = Ball(detection=None)

    if match:
        ball.set_color(match)

    if not detections:
        return ball

    # Sort detections by confidence score in descending order
    # Confidence is expected to be in detection.data["p"]
    # Add a check for "p" in data to avoid KeyError if a detection somehow misses it.
    # And handle cases where data might be None (though Norfair typically initializes it)
    valid_detections_with_confidence = []
    for det in detections:
        if det.data is not None and "p" in det.data:
            valid_detections_with_confidence.append(det)
        # else:
            # Optionally log a warning for detections without confidence or data
            # print(f"Warning: Ball detection {det} missing confidence score or data.")

    if not valid_detections_with_confidence:
        # No valid detections with confidence found, return ball with no detection
        return ball

    # Sort by confidence: detection.data["p"]
    # The lambda function will access detection.data["p"]
    sorted_detections = sorted(
        valid_detections_with_confidence,
        key=lambda det: det.data["p"],
        reverse=True
    )

    # The ball with the highest confidence is the first one after sorting
    ball.detection = sorted_detections[0]

    return ball
