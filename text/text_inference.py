"""
Text -> robot command intent classifier, fine-tuned DistilBERT.

Two modes:
  1. Terminal inference: `python text_inference.py predict "move forward"`
  2. GUI: `python text_inference.py gui`
     Opens a small local web page in your default browser: type a command,
     see the predicted intent, and it forwards the result to the FastAPI
     broker (main_api.py) at /command. Uses only Python's built-in
     http.server — no extra packages, no Tkinter, nothing to install.

Training lives in text_model_train.py — this file only does inference.

Assumes a flat project layout: this file, commands.py, main_api.py, etc.
all live in the same project root folder, with a models/ subfolder next
to them holding the trained checkpoint.
"""

import argparse
import json
import os
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

MODEL_DIR = os.path.join(THIS_DIR, "models", "text_intent_model")
API_BASE_URL = "http://127.0.0.1:8000"

try:
    from commands import ALL_COMMANDS
except ImportError:
    ALL_COMMANDS = [
        "STOP", "FORWARD", "BACKWARD", "LEFT", "RIGHT",
        "YAW_LEFT", "YAW_RIGHT", "SHOULDER_FORWARD", "SHOULDER_BACKWARD",
        "ELBOW_UP", "ELBOW_DOWN", "GRIPPER_OPEN", "GRIPPER_CLOSE",
    ]
    print(f"[warning] Could not import ALL_COMMANDS from commands.py — using default list: {ALL_COMMANDS}")

# Label mapping is only needed if a fine-tuned model actually exists.
# Missing it must NOT crash the whole module — fall back to a mapping
# built straight from ALL_COMMANDS so `predict` mode / the GUI still work.
_label_mapping_path = os.path.join(MODEL_DIR, "label_mapping.json")
if os.path.isfile(_label_mapping_path):
    with open(_label_mapping_path, "r") as f:
        ID2LABEL = {int(k): v for k, v in json.load(f).items()}
else:
    ID2LABEL = dict(enumerate(ALL_COMMANDS))

LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class TextIntentClassifier:
    """Loads the fine-tuned DistilBERT model if available, otherwise falls
    back to simple keyword matching so the GUI always works end-to-end —
    including before the model has ever been trained."""

    # IMPORTANT — dict order = match priority (first match wins, longest-
    # phrase-wins within a pass). STOP is checked first as the
    # safety-critical command.
    _FALLBACK_KEYWORDS = {
        "STOP": [
            "stop", "halt", "freeze", "pause", "brake", "shutdown",
            "shut down", "power off", "turn off", "hold", "stay still",
            "don't move",
        ],
        "FORWARD": ["forward", "move forward", "go forward", "ahead", "advance", "straight"],
        "BACKWARD": ["backward", "move backward", "go backward", "back", "reverse", "retreat"],
        "LEFT": ["left", "move left", "go left"],
        "RIGHT": ["right", "move right", "go right"],
        "YAW_LEFT": ["yaw left", "rotate left", "turn arm left", "rotate base left"],
        "YAW_RIGHT": ["yaw right", "rotate right", "turn arm right", "rotate base right"],
        "SHOULDER_FORWARD": ["shoulder forward", "arm forward", "move shoulder forward", "extend arm"],
        "SHOULDER_BACKWARD": ["shoulder backward", "arm backward", "move shoulder backward", "retract arm"],
        "ELBOW_UP": ["elbow up", "raise elbow", "lift elbow", "bend elbow up"],
        "ELBOW_DOWN": ["elbow down", "lower elbow", "bend elbow down"],
        "GRIPPER_OPEN": ["open gripper", "open claw", "release", "open hand"],
        "GRIPPER_CLOSE": ["close gripper", "close claw", "grip", "grab", "grasp", "clamp", "squeeze"],
    }

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cpu"):
        self.device = device
        self.model = None
        self.tokenizer = None

        if os.path.isdir(model_dir):
            try:
                import torch  # noqa: F401
                from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
                self.model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
                self.model.eval()
                print(f"Loaded fine-tuned text intent model from {model_dir}/")
            except Exception as e:
                print(f"[warning] Failed to load fine-tuned model ({e}). Using keyword fallback.")
                self.model = None
        else:
            print(f"No fine-tuned model found at {model_dir}/ — using keyword fallback. "
                  f"Run text_model_train.py to train one.")

    def predict(self, text: str) -> dict:
        if self.model is not None:
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0]
                idx = int(probs.argmax())
                return {"command": ID2LABEL[idx], "confidence": float(probs[idx]), "method": "distilbert"}

        lowered = text.lower().strip()

        # Search for the longest matching keyword first — prevents
        # "yaw right" from being caught by the generic "right" keyword.
        matches = []
        for command, keywords in self._FALLBACK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered:
                    matches.append((len(keyword), command, keyword))

        if matches:
            matches.sort(reverse=True)
            _, command, matched_keyword = matches[0]
            confidence = min(0.95, 0.55 + len(matched_keyword) * 0.03)
            return {"command": command, "confidence": round(confidence, 2), "method": "keyword_fallback"}

        return {"command": None, "confidence": 0.0, "method": "keyword_fallback"}


# ---------------------------------------------------------------------------
# Browser-based GUI (stdlib only — no Tkinter, no extra pip installs)
# ---------------------------------------------------------------------------

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Robot Command Intent Classifier</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:32px; }}
  .card {{ max-width:640px; margin:0 auto; background:#1e293b; border-radius:12px; padding:24px 28px; box-shadow:0 4px 24px rgba(0,0,0,0.4); }}
  h1 {{ font-size:20px; margin-top:0; }}
  input[type=text] {{ width:100%; box-sizing:border-box; padding:12px; font-size:16px; border-radius:8px; border:1px solid #334155; background:#0f172a; color:#e2e8f0; }}
  input[type=text]:focus {{ outline:2px solid #3b82f6; }}
  button {{ margin-top:10px; padding:10px 18px; font-size:14px; border:none; border-radius:8px; background:#3b82f6; color:white; cursor:pointer; }}
  button:hover {{ background:#2563eb; }}
  button.secondary {{ background:#475569; margin-left:8px; }}
  button.secondary:hover {{ background:#334155; }}
  #result {{ margin-top:16px; font-size:18px; font-weight:600; color:#60a5fa; min-height:26px; }}
  #commands {{ font-size:12px; color:#94a3b8; margin-top:4px; }}
  #history {{ margin-top:18px; background:#0f172a; border-radius:8px; padding:10px 14px; height:260px; overflow-y:auto; font-family:Consolas, monospace; font-size:13px; }}
  .row {{ padding:4px 0; border-bottom:1px solid #1e293b; }}
  .cmd {{ color:#4ade80; font-weight:bold; }}
  .conf {{ color:#94a3b8; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Robot Command Intent Classifier</h1>
    <input id="text" type="text" placeholder="Type a command, e.g. 'move forward'" autofocus />
    <div>
      <button onclick="predict()">Predict (Enter)</button>
      <button class="secondary" onclick="clearHistory()">Clear history</button>
    </div>
    <div id="result">Prediction will appear here.</div>
    <div id="commands">Known commands ({n}): {commands}</div>
    <div id="history"></div>
  </div>

<script>
const input = document.getElementById('text');
const resultEl = document.getElementById('result');
const historyEl = document.getElementById('history');

input.addEventListener('keydown', e => {{ if (e.key === 'Enter') predict(); }});

async function predict() {{
  const text = input.value.trim();
  if (!text) return;
  resultEl.textContent = 'Thinking...';
  try {{
    const res = await fetch('/predict', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{text}})
    }});
    const data = await res.json();
    const cmd = data.command || 'UNRECOGNIZED';
    const conf = (data.confidence * 100).toFixed(0);
    resultEl.innerHTML = `→ <span class="cmd">${{cmd}}</span> <span class="conf">(${{conf}}% conf, ${{data.method}})</span>`;
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `"${{text}}" → <span class="cmd">${{cmd}}</span> <span class="conf">(${{conf}}%, ${{data.method}})</span>`;
    historyEl.prepend(row);
    input.value = '';
    input.focus();
  }} catch (err) {{
    resultEl.textContent = 'Error: ' + err;
  }}
}}

function clearHistory() {{ historyEl.innerHTML = ''; }}
</script>
</body>
</html>
"""


def launch_gui(port: int = 8765):
    print("Loading classifier...")
    clf = TextIntentClassifier()

    page = PAGE_HTML.format(n=len(ALL_COMMANDS), commands=", ".join(ALL_COMMANDS))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # silence default request logging

        def do_GET(self):
            if self.path == "/":
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/predict":
                # Classification and "forward to FastAPI" are two
                # independent steps — a failure in the second must never
                # discard a result the first step already produced.
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    text = payload.get("text", "")
                    result = clf.predict(text)
                except Exception:
                    traceback.print_exc()
                    result = {"command": None, "confidence": 0.0, "method": "error"}

                if result.get("command"):
                    try:
                        requests.post(
                            f"{API_BASE_URL}/command",
                            json={"command": result["command"], "source": "text"},
                            timeout=0.5,
                        )
                    except requests.exceptions.RequestException as e:
                        print(f"[warning] Could not forward '{result['command']}' to FastAPI: {e}")

                body = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving GUI at {url}")
    print(f"Forwarding recognized commands to {API_BASE_URL}/command")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    predict_p = sub.add_parser("predict")
    predict_p.add_argument("text")

    gui_p = sub.add_parser("gui")
    gui_p.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    try:
        if args.mode == "predict":
            clf = TextIntentClassifier()
            print(clf.predict(args.text))
        elif args.mode == "gui":
            launch_gui(port=args.port)
    except Exception:
        traceback.print_exc()
        input("\nAn error occurred (see above). Press Enter to close...")
