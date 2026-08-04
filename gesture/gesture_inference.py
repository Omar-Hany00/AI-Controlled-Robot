"""
===============================================================
Gesture Inference (CNN Version)
===============================================================

AI-based hand gesture recognition using:

- MediaPipe Hands
- EfficientNetV2-S
- PyTorch
- FastAPI

Pipeline

Camera
   ↓
MediaPipe
   ↓
Crop Hand
   ↓
EfficientNetV2-S
   ↓
Predicted Letter
   ↓
LETTER_TO_COMMAND
   ↓
Robot Command
"""

import os
import cv2
import time
import logging
import requests

import torch
import torch.nn as nn

from PIL import Image

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from collections import deque, Counter

import mediapipe as mp

from torchvision import transforms
from torchvision.models import efficientnet_v2_s

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

API_BASE_URL = "http://127.0.0.1:8000"

@dataclass(frozen=True)
class Config:

    CAMERA_INDEX = 0

    WIDTH = 1280
    HEIGHT = 720

    MAX_HANDS = 1

    DETECTION_CONFIDENCE = 0.7
    TRACKING_CONFIDENCE = 0.7

    SMOOTHING_WINDOW = 5

    CONFIDENCE_THRESHOLD = 0.75

    COOLDOWN = 0.8

    MODEL_PATH = "models/best_gesture_model.pth"

class CNNGestureInference:

    def __init__(self, config: Config):

        self.config = config

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint = torch.load(
            config.MODEL_PATH,
            map_location=self.device,
        )

        self.classes = checkpoint["classes"]

        self.command_map = checkpoint["letter_to_command"]

        self.image_size = checkpoint["image_size"]

        self.transform = transforms.Compose([
            transforms.Resize(
                (
                    self.image_size,
                    self.image_size,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                checkpoint["mean"],
                checkpoint["std"],
            ),
        ])

        self.model = efficientnet_v2_s(
            weights=None
        )

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            len(self.classes),
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

        self.history = deque(
            maxlen=config.SMOOTHING_WINDOW
        )

        logging.info(
            "Gesture model loaded successfully."
        )

    def preprocess(self, image):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(image)

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, image):

        tensor = self.preprocess(image)

        with torch.inference_mode():

            output = self.model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1,
            )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        confidence = confidence.item()

        letter = self.classes[
            prediction.item()
        ]

        return letter, confidence
    
    def infer(self, image):

        letter, confidence = self.predict(image)

        if confidence < self.config.CONFIDENCE_THRESHOLD:

            return {
                "gesture": "UNKNOWN",
                "action": None,
                "confidence": confidence,
            }

        self.history.append(letter)

        smoothed = Counter(
            self.history
        ).most_common(1)[0][0]

        action = self.command_map.get(smoothed)

        return {

            "gesture": smoothed,

            "action": action,

            "confidence": confidence,

        }

class HandTracker:

    def __init__(self, config: Config):

        self.config = config

        self.mp_hands = mp.solutions.hands

        self.drawer = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=config.MAX_HANDS,

            min_detection_confidence=config.DETECTION_CONFIDENCE,

            min_tracking_confidence=config.TRACKING_CONFIDENCE,

        )

    def process(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:

            return None, None

        landmarks = results.multi_hand_landmarks[0]

        return landmarks, results
    
    def draw(self, frame, landmarks):

        self.drawer.draw_landmarks(

            frame,

            landmarks,

            self.mp_hands.HAND_CONNECTIONS,

        )

    def crop_hand(
        self,
        frame,
        landmarks,
        padding=40,
    ):

        h, w = frame.shape[:2]

        xs = []

        ys = []

        for lm in landmarks.landmark:

            xs.append(int(lm.x * w))

            ys.append(int(lm.y * h))

        x1 = max(min(xs) - padding, 0)

        y1 = max(min(ys) - padding, 0)

        x2 = min(max(xs) + padding, w)

        y2 = min(max(ys) + padding, h)

        hand = frame[
            y1:y2,
            x1:x2,
        ]

        return hand, (
            x1,
            y1,
            x2,
            y2,
        )

class Camera:

    def __init__(self, config: Config):

        self.cap = cv2.VideoCapture(
            config.CAMERA_INDEX
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.WIDTH,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.HEIGHT,
        )

    def read(self):

        success, frame = self.cap.read()

        if not success:

            return None

        frame = cv2.flip(
            frame,
            1,
        )

        return frame
    
    def release(self):

        self.cap.release()

class VisionPipeline:

    def __init__(self, config):

        self.config = config

        self.camera = Camera(config)

        self.tracker = HandTracker(config)

        self.cnn = CNNGestureInference(config)

    def process(self):

        frame = self.camera.read()

        if frame is None:

            return None

        landmarks, results = self.tracker.process(
            frame
        )

        if landmarks is None:

            return frame, None

        self.tracker.draw(
            frame,
            landmarks,
        )

        hand, bbox = self.tracker.crop_hand(

            frame,

            landmarks,

        )

        if hand.size == 0:
            return frame, None

        prediction = self.cnn.infer(hand)

        x1, y1, x2, y2 = bbox

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2,

        )

        return frame, prediction

class CommandManager:

    def __init__(self, config: Config):

        self.config = config

        self.last_command = None

        self.last_time = 0.0

    def send(self, command):

        if command is None:
            return

        now = time.time()

        if (
            command == self.last_command
            and
            now - self.last_time < self.config.COOLDOWN
        ):
            return

        self.last_command = command
        self.last_time = now

        try:

            requests.post(

                f"{API_BASE_URL}/command",

                json={
                    "command": command,
                    "source": "gesture",
                },

                timeout=0.5,

            )

        except Exception as e:

            logging.warning(e)

def main():

    cfg = Config()

    pipeline = VisionPipeline(cfg)

    manager = CommandManager(cfg)

    window = "AI Gesture Robot"

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    previous = time.time()

    while True:

        result = pipeline.process()

        if result is None:
            break

        frame, prediction = result

        now = time.time()

        fps = 1.0 / max(
            now - previous,
            1e-6,
        )

        previous = now

        if prediction is not None:

            manager.send(
                prediction["action"]
            )

        cv2.imshow(

        window,

        frame,

    )
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    pipeline.camera.release()

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()