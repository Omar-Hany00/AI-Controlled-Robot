"""
main.py

Gesture Robot - System Launcher

Starts:
1. FastAPI Backend
2. Gesture Inference
3. Text Inference
4. Robot Simulation

Everything communicates through FastAPI.
"""

import logging
import signal
import subprocess
import sys
import threading
import time
import requests

# Flat project layout — every file lives in this same folder.
from gesture.gesture_inference_rule_based import main as run_gesture_inference
from simulation.robot_simulation import RobotSimulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class GestureRobotSystem:
    """Main application controller."""

    def __init__(self):
        self.fastapi_process = None
        self.text_process = None
        self.gesture_thread = None

    # --------------------------------------------------
    # FastAPI
    # --------------------------------------------------

    def start_fastapi(self):
        """Launch FastAPI and wait until it is ready."""

        logger.info("Starting FastAPI server...")

        self.fastapi_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main_api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]
        )

        logger.info("Waiting for FastAPI...")

        for _ in range(30):
            try:
                response = requests.get(
                    "http://127.0.0.1:8000/health",
                    timeout=0.5,
                )

                if response.status_code == 200:
                    logger.info("FastAPI is ready.")
                    return

            except requests.RequestException:
                pass

            time.sleep(0.5)

        raise RuntimeError("FastAPI failed to start.")

    # --------------------------------------------------
    # Gesture Thread
    # --------------------------------------------------

    def start_gesture_inference(self):
        logger.info("Starting Gesture Inference...")

        self.gesture_thread = threading.Thread(
            target=run_gesture_inference,
            daemon=True,
            name="GestureInference",
        )

        self.gesture_thread.start()

        logger.info("Gesture Inference running.")

    # --------------------------------------------------
    # Text Thread
    # --------------------------------------------------

    def start_text_inference(self):
        """Launch the Text Inference GUI."""

        logger.info("Starting Text Inference...")

        self.text_process = subprocess.Popen(
            [
                sys.executable,
                "text/text_inference.py",
                "gui",
            ]
        )

        logger.info("Text Inference running.")

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

    def start_simulation(self):
        """Blocks until the simulation window is closed — RobotSimulation()
        owns its own GLFW loop and cleans up its own resources internally
        (see robot_simulation.py), so there's nothing to store or close here."""
        logger.info("Starting Robot Simulation...")
        exit_code = RobotSimulation()
        if exit_code != 0:
            logger.warning(f"Robot Simulation exited with code {exit_code}.")

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    def shutdown(self):
        logger.info("Shutting down...")

        if self.text_process is not None:
            logger.info("Stopping Text Inference...")

            self.text_process.terminate()

            try:
                self.text_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.text_process.kill()

        # Robot Simulation cleans up its own GLFW/MuJoCo resources
        # internally (see robot_simulation.py's try/finally) before
        # start_simulation() returns, so there's nothing to close here.

        if self.fastapi_process is not None:
            logger.info("Stopping FastAPI...")

            self.fastapi_process.terminate()

            try:
                self.fastapi_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.fastapi_process.kill()

        logger.info("Shutdown complete.")

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(self):
        try:
            self.start_fastapi()
            self.start_gesture_inference()
            self.start_text_inference()
            self.start_simulation()

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")

        except Exception:
            logger.exception("Unexpected error.")

        finally:
            self.shutdown()


system = GestureRobotSystem()


def signal_handler(sig, frame):
    system.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Gesture Robot System")
    logger.info("=" * 60)

    system.run()