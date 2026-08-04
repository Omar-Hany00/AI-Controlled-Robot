"""
Shared robot command definitions.

Single source of truth for robot commands used across the project.

Import this module from both the Computer Vision and Text Inference
pipelines so every subsystem uses identical command names.
"""

from enum import Enum


class RobotCommand(str, Enum):
    STOP = "STOP"

    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"

    LEFT = "LEFT"
    RIGHT = "RIGHT"

    YAW_LEFT = "YAW_LEFT"
    YAW_RIGHT = "YAW_RIGHT"

    SHOULDER_FORWARD = "SHOULDER_FORWARD"
    SHOULDER_BACKWARD = "SHOULDER_BACKWARD"

    ELBOW_UP = "ELBOW_UP"
    ELBOW_DOWN = "ELBOW_DOWN"

    GRIPPER_OPEN = "GRIPPER_OPEN"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"


# HaGRID gesture label -> RobotCommand
# Replace the labels below with your model's actual output labels if needed.
GESTURE_TO_COMMAND = {
    "stop": RobotCommand.STOP,

    "fist": RobotCommand.FORWARD,
    "one": RobotCommand.BACKWARD,

    "point_left": RobotCommand.LEFT,
    "point_right": RobotCommand.RIGHT,

    "middle": RobotCommand.YAW_LEFT,
    "ring": RobotCommand.YAW_RIGHT,

    "thumbs_up": RobotCommand.SHOULDER_FORWARD,
    "thumbs_down": RobotCommand.SHOULDER_BACKWARD,

    "shaka": RobotCommand.ELBOW_UP,
    "rock": RobotCommand.ELBOW_DOWN,

    "peace": RobotCommand.GRIPPER_OPEN,
    "four": RobotCommand.GRIPPER_CLOSE,
}


def gesture_to_command(gesture_label: str) -> RobotCommand | None:
    """
    Convert a raw gesture label into a RobotCommand.

    Accepts:
        point_left
        Point Left
        POINT LEFT
        point-left
        point left

    Returns:
        RobotCommand or None
    """
    if not gesture_label:
        return None

    key = (
        gesture_label.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    return GESTURE_TO_COMMAND.get(key)


ALL_COMMANDS = [command.value for command in RobotCommand]