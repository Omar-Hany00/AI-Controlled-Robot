"""
===================================================================
   COMPUTER VISION MODULE FOR ROBOT TELEOPERATION (ALL-IN-ONE)
===================================================================
Features:
 - Config & Tracking (MediaPipe)
 - Smart Gesture Detection & Temporal Smoothing
 - Gesture & Text Inference Engines
 - Command Queue & Cooldown Management
 - Complete UI Dashboard & FPS Performance Monitor
===================================================================

NOTE: detect_raw_gesture() below is a geometric/rule-based placeholder
(finger-open/closed states) standing in for the EfficientNetV2-S HaGRID
CNN. It outputs the same 13 RobotCommand values from commands.py so the
rest of the pipeline (CommandManager -> FastAPI -> robot_sim) doesn't
need to change when the CNN replaces this detector later.
"""

import sys
import os
import cv2
import time
import logging
import requests
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from collections import deque, Counter
from queue import Queue
import mediapipe as mp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from commands import RobotCommand, ALL_COMMANDS, gesture_to_command  # noqa: E402

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s")

# FastAPI broker endpoint (see main_1_.py)
API_BASE_URL = "http://127.0.0.1:8000"


# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass(frozen=True)
class Config:
    # Camera Settings
    CAM_INDEX: int = 0
    WIDTH: int = 1280
    HEIGHT: int = 720

    # Vision & Detection Thresholds
    MAX_HANDS: int = 1
    MIN_DETECTION_CONF: float = 0.7
    MIN_TRACKING_CONF: float = 0.7
    SMOOTHING_WINDOW: int = 5

    # Command Engine
    COOLDOWN_SEC: float = 0.8
    QUEUE_SIZE: int = 10

    # Networking
    SEND_TO_API: bool = True
    API_TIMEOUT_SEC: float = 0.5

    # UI Styling (BGR Format)
    TEXT_COLOR: Tuple[int, int, int] = (0, 255, 0)       # Green
    HIGHLIGHT_COLOR: Tuple[int, int, int] = (0, 165, 255)  # Orange
    FONT_SCALE: float = 0.7
    THICKNESS: int = 2


# ==========================================
# 2. HAND TRACKER
# ==========================================
class HandTracker:
    """Handles camera input and landmark extraction using MediaPipe."""
    def __init__(self, config: Config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.MAX_HANDS,
            min_detection_confidence=self.config.MIN_DETECTION_CONF,
            min_tracking_confidence=self.config.MIN_TRACKING_CONF
        )

    def process_frame(self, frame):
        """Extracts landmark objects, landmark list, and identifies Left/Right hand."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks_obj = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks_obj.landmark
            handedness = results.multi_handedness[0].classification[0].label
            return hand_landmarks_obj, landmarks, handedness
        return None, None, None

    def draw_landmarks(self, frame, hand_landmarks_obj) -> None:
        """Draws hand connections and joint points."""
        if hand_landmarks_obj:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks_obj,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )


# ==========================================
# 3. GESTURE DETECTOR
# ==========================================
class GestureDetector:
    """Calculates finger extension states and applies temporal smoothing.

    Maps 13 finger-open/closed signatures directly onto the 13
    RobotCommand values from commands.py. This is a placeholder for the
    EfficientNetV2-S HaGRID CNN — swap detect_raw_gesture()'s internals
    for a model.predict() call later without touching anything downstream.
    """
    def __init__(self, config: Config):
        self.config = config
        self.history: deque = deque(maxlen=self.config.SMOOTHING_WINDOW)

    def _get_finger_states(self, lm, handedness: str) -> List[bool]:
        """Returns boolean array [Thumb, Index, Middle, Ring, Pinky]."""
        fingers = []

        # 1. Thumb State (Horizontal comparison based on hand side)
        if handedness == "Right":
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)

        # 2. Other 4 Fingers (Vertical comparison: Tip higher than PIP)
        tip_ids = [8, 12, 16, 20]
        for tip in tip_ids:
            fingers.append(lm[tip].y < lm[tip - 2].y)

        return fingers

    def detect_raw_gesture(self, lm, handedness: str) -> Tuple[str, int]:
        """Translates finger states into one of the 13 RobotCommand values.

        Signature legend (Thumb, Index, Middle, Ring, Pinky):
          STOP               11111  open palm
          FORWARD            00000  fist
          BACKWARD           01000  index only
          LEFT               01100  index + middle
          RIGHT              11000  thumb + index
          YAW_LEFT           00100  middle only
          YAW_RIGHT          00010  ring only
          SHOULDER_FORWARD   10000  thumb only, pointing UP
          SHOULDER_BACKWARD  10000  thumb only, pointing DOWN
          ELBOW_UP           10001  thumb + pinky
          ELBOW_DOWN         01001  index + pinky
          GRIPPER_OPEN       11100  thumb + index + middle
          GRIPPER_CLOSE      01111  index + middle + ring + pinky (no thumb)
        """
        if not lm:
            return "NO_HAND", 0

        states = self._get_finger_states(lm, handedness)
        finger_count = sum(states)

        if states == [True, True, True, True, True]:
            gesture = RobotCommand.STOP.value
        elif states == [False, False, False, False, False]:
            gesture = RobotCommand.FORWARD.value
        elif states == [False, True, False, False, False]:
            gesture = RobotCommand.BACKWARD.value
        elif states == [False, True, True, False, False]:
            gesture = RobotCommand.LEFT.value
        elif states == [True, True, False, False, False]:
            gesture = RobotCommand.RIGHT.value
        elif states == [False, False, True, False, False]:
            gesture = RobotCommand.YAW_LEFT.value
        elif states == [False, False, False, True, False]:
            gesture = RobotCommand.YAW_RIGHT.value
        elif states == [True, False, False, False, False]:
            # Thumb-only: direction (up vs down) decides shoulder move.
            # Landmark 4 = thumb tip, landmark 2 = thumb MCP joint.
            gesture = (
                RobotCommand.SHOULDER_FORWARD.value
                if lm[4].y < lm[2].y
                else RobotCommand.SHOULDER_BACKWARD.value
            )
        elif states == [True, False, False, False, True]:
            gesture = RobotCommand.ELBOW_UP.value
        elif states == [False, True, False, False, True]:
            gesture = RobotCommand.ELBOW_DOWN.value
        elif states == [True, True, True, False, False]:
            gesture = RobotCommand.GRIPPER_OPEN.value
        elif states == [False, True, True, True, True]:
            gesture = RobotCommand.GRIPPER_CLOSE.value
        else:
            gesture = "UNRECOGNIZED"

        return gesture, finger_count

    def get_smoothed_gesture(self, raw_gesture: str) -> str:
        """Applies majority voting over recent frames to stabilize output."""
        self.history.append(raw_gesture)
        most_common = Counter(self.history).most_common(1)
        return most_common[0][0] if most_common else "UNKNOWN"


# ==========================================
# 4. INFERENCE ENGINES (GESTURE & TEXT)
# ==========================================
class GestureInference:
    """High-level gesture inference module. detect_raw_gesture already
    returns RobotCommand values directly, so this just smooths + passes
    them through (finger_count kept for the UI dashboard)."""
    def __init__(self, config: Config):
        self.detector = GestureDetector(config)

    def infer(self, landmarks, handedness: str) -> Dict[str, Any]:
        if not landmarks:
            return {"gesture": "NO_HAND", "action": None, "finger_count": 0}

        raw_gesture, finger_count = self.detector.detect_raw_gesture(landmarks, handedness)
        smoothed = self.detector.get_smoothed_gesture(raw_gesture)

        action = smoothed if smoothed in ALL_COMMANDS else None
        return {
            "gesture": smoothed,
            "finger_count": finger_count,
            "action": action,
        }


class CNNGestureInference:
    """STUB — not yet functional. Drop-in replacement for GestureInference
    once the EfficientNetV2-S HaGRID model exists.

    Unlike GestureInference (which invents its own gesture names directly
    as RobotCommand values), this class is meant to output the model's
    real HaGRID label — e.g. "two_up", "peace_inverted", "like" — in the
    "gesture" field, exactly like the rule-based version currently shows
    raw gesture names in the UI dashboard (`main()`'s "Gesture: {...}"
    line). The mapping from that HaGRID label to a RobotCommand goes
    through gesture_to_command() from commands.py, which already has the
    13-command mapping defined and ready.

    TODO once the model is trained:
      1. Load the fine-tuned EfficientNetV2-S checkpoint in __init__.
      2. In infer(), crop/preprocess the hand region from the frame,
         run the model, and get back a HaGRID label string + confidence.
      3. Everything else below already works unchanged.
    """
    def __init__(self, config: Config, model_path: str = "gesture_cnn.pt"):
        self.config = config
        self.model = None  # TODO: load EfficientNetV2-S checkpoint here
        self.history: deque = deque(maxlen=self.config.SMOOTHING_WINDOW)
        logging.info(f"[CNNGestureInference] STUB — no model loaded from '{model_path}' yet.")

    def infer(self, frame, landmarks, handedness: str) -> Dict[str, Any]:
        if not landmarks:
            return {"gesture": "NO_HAND", "action": None, "finger_count": 0, "confidence": 0.0}

        # TODO: replace with real inference, e.g.:
        #   crop = crop_hand_region(frame, landmarks)
        #   raw_label, confidence = self.model.predict(crop)
        raw_label, confidence = "stop", 0.0  # placeholder until the model exists

        self.history.append(raw_label)
        smoothed = Counter(self.history).most_common(1)[0][0]

        command = gesture_to_command(smoothed)  # HaGRID label -> RobotCommand | None
        return {
            "gesture": smoothed,                        # e.g. "two_up" — shown in UI as-is
            "action": command.value if command else None,
            "finger_count": 0,                           # not meaningful for CNN output
            "confidence": confidence,
        }


# ==========================================
# 5. COMMAND MANAGER
# ==========================================
class CommandManager:
    """Manages command dispatch, duplicate prevention, cooldowns, and
    forwarding to the FastAPI broker (main_1_.py)."""
    def __init__(self, config: Config):
        self.config = config
        self.queue: Queue = Queue(maxsize=self.config.QUEUE_SIZE)
        self.last_command: Optional[str] = None
        self.last_time: float = 0.0

    def process_command(self, action: Optional[str]) -> Optional[str]:
        if action is None or action not in ALL_COMMANDS:
            return None

        now = time.time()
        # Cooldown & Duplicate Check
        if (now - self.last_time) >= self.config.COOLDOWN_SEC:
            if action != self.last_command:
                self.last_command = action
                self.last_time = now
                if not self.queue.full():
                    self.queue.put(action)
                logging.info(f"[DISPATCH] -> Action: {action}")
                if self.config.SEND_TO_API:
                    self._send_to_api(action)
                return action
        return None

    def _send_to_api(self, action: str) -> None:
        """POSTs the command to the FastAPI broker so robot_sim.py (which
        polls /state) picks it up. Failures are logged, not raised, so a
        dropped connection never crashes the vision loop."""
        try:
            requests.post(
                f"{API_BASE_URL}/command",
                json={"command": action, "source": "gesture"},
                timeout=self.config.API_TIMEOUT_SEC,
            )
        except requests.exceptions.RequestException as e:
            logging.warning(f"[API] Failed to send '{action}': {e}")

    def pop_command_for_pybullet(self) -> Optional[str]:
        """Local queue accessor, kept for any consumer polling this
        process directly instead of going through the FastAPI broker."""
        if not self.queue.empty():
            return self.queue.get()
        return None


# ==========================================
# 6. MAIN PIPELINE & UI RUNNER
# ==========================================
def main():
    cfg = Config()

    cap = cv2.VideoCapture(cfg.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.HEIGHT)

    tracker = HandTracker(cfg)
    gesture_engine = GestureInference(cfg)
    cmd_manager = CommandManager(cfg)

    # ==========================================
    # Create a resizable OpenCV window
    # ==========================================
    WINDOW_NAME = "Robot Vision Control Center"

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    )

    # Initial window size (you can change these)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    prev_time = 0.0
    print("[INFO] Computer Vision Pipeline Initialized. Press 'q' to Quit.")
    print(f"[INFO] Forwarding commands to {API_BASE_URL} (set Config.SEND_TO_API=False to disable)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        # Flip for intuitive selfie view
        frame = cv2.flip(frame, 1)

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # 1. Vision Processing Pipeline
        hand_landmarks_obj, landmarks, handedness = tracker.process_frame(frame)

        inference_res = {"gesture": "NO_HAND", "action": None, "finger_count": 0}

        if landmarks:
            tracker.draw_landmarks(frame, hand_landmarks_obj)
            inference_res = gesture_engine.infer(landmarks, handedness)
            cmd_manager.process_command(inference_res["action"])

        # # 2. Render Telemetry UI Dashboard
        # cv2.rectangle(frame, (10, 10), (330, 220), (0, 0, 0), -1)
        # cv2.rectangle(frame, (10, 10), (330, 220), cfg.HIGHLIGHT_COLOR, 2)

        # cv2.putText(frame, f"FPS: {int(fps)}", (25, 45),
        #             cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.THICKNESS)
        # cv2.putText(frame, f"Hand: {handedness if handedness else 'N/A'}", (25, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.THICKNESS)
        # cv2.putText(frame, f"Fingers Count: {inference_res['finger_count']}", (25, 115),
        #             cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.THICKNESS)
        # cv2.putText(frame, f"Gesture: {inference_res['gesture']}", (25, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.HIGHLIGHT_COLOR, cfg.THICKNESS)
        # cv2.putText(frame, f"Robot Action: {inference_res['action']}", (25, 185),
        #             cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, (0, 255, 255), cfg.THICKNESS)
        
        # Define cropping coordinates
        start_x, start_y = 0, 0
        end_x, end_y = 800, 700

        # Crop the image using NumPy slicing
        cropped = frame[start_y:end_y, start_x:end_x]

        # Show Window
        cv2.imshow(WINDOW_NAME, cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()