"""
FastAPI backend — receives commands from the gesture pipeline (Member 2)
and text pipeline (Member 3's text_inference.py), and exposes the current
robot command state for the simulation (Member 4) to poll.

Run: uvicorn main_api:app --reload --port 8000

Member 3 (Backend & NLP) owns this file.
"""

import os
import sys
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from commands import ALL_COMMANDS  # noqa: E402
from text.text_inference import TextIntentClassifier  # noqa: E402

app = FastAPI(title="Gesture/Text Robot Control API")
text_classifier = TextIntentClassifier()

# In-memory state — simple and sufficient for a single-robot demo.
# Swap for Redis/a message queue if this needs to scale beyond one process.
state = {
    "command": "STOP",
    "source": None,       # "gesture" | "text"
    "updated_at": time.time(),
}


class CommandRequest(BaseModel):
    command: str
    source: str = "unknown"


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/commands")
def list_commands():
    return {"commands": ALL_COMMANDS}


@app.get("/state")
def get_state():
    """Polled by simulation/robot_sim.py to drive the robot."""
    return state


@app.post("/command")
def post_command(req: CommandRequest):
    """Direct command submission — used by gesture_inference.py."""
    if req.command not in ALL_COMMANDS:
        raise HTTPException(status_code=400, detail=f"Unknown command '{req.command}'. Valid: {ALL_COMMANDS}")
    state["command"] = req.command
    state["source"] = req.source
    state["updated_at"] = time.time()
    return state


@app.post("/text-command")
def post_text_command(req: TextRequest):
    """Natural-language command submission — runs the DistilBERT/fallback classifier."""
    result = text_classifier.predict(req.text)
    if result["command"] is None:
        raise HTTPException(status_code=422, detail=f"Could not map '{req.text}' to a known command.")
    state["command"] = result["command"]
    state["source"] = "text"
    state["updated_at"] = time.time()
    return {**state, "confidence": result["confidence"], "method": result["method"]}
