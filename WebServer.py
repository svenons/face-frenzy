import json
import threading
import time

from flask import Flask, Response, jsonify, request


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face Frenzy</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #101418;
      color: #f4f7f8;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 16px;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 28px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #d14f4f;
      transition: background 0.2s;
    }
    .dot.live { background: #64d17a; }
    .video {
      width: 100%;
      background: #050607;
      border: 1px solid #2b343b;
      border-radius: 8px;
      display: block;
    }
    .panel {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .stat {
      border: 1px solid #2b343b;
      border-radius: 8px;
      padding: 10px;
      background: #182127;
    }
    .label {
      color: #a9b8c0;
      font-size: 13px;
    }
    .value {
      font-size: 24px;
      margin-top: 4px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      border: 1px solid #5d6b73;
      border-radius: 8px;
      background: #f4f7f8;
      color: #101418;
      font-size: 18px;
      padding: 12px 16px;
      min-width: 120px;
      cursor: pointer;
    }
    button.primary {
      background: #64d17a;
      border-color: #64d17a;
    }
    .hint {
      color: #a9b8c0;
      line-height: 1.4;
    }
  </style>
</head>
<body>
  <main>
    <h1>Face Frenzy <span id="conn" class="dot" title="disconnected"></span></h1>
    <img class="video" src="/video_feed" alt="Live camera feed">
    <div class="panel">
      <div class="stat"><div class="label">State</div><div class="value" id="state">-</div></div>
      <div class="stat"><div class="label">Players</div><div class="value" id="players">-</div></div>
      <div class="stat"><div class="label">Live Faces</div><div class="value" id="live">-</div></div>
      <div class="stat"><div class="label">Target</div><div class="value" id="target">-</div></div>
      <div class="stat"><div class="label">Score</div><div class="value" id="score">-</div></div>
      <div class="stat"><div class="label">Strikes</div><div class="value" id="strikes">-</div></div>
    </div>
    <p class="hint" id="error"></p>
    <div class="controls">
      <button onclick="control('decrement_players')">Less</button>
      <button onclick="control('increment_players')">More</button>
      <button class="primary" onclick="control('start')">Start</button>
      <button onclick="control('reset')">Reset</button>
    </div>
    <p class="hint">BTN0/BTN1 also change player count. BTN3 starts the game.
    LEDs flash all four, then the selected player count, while waiting.</p>
  </main>
  <script>
    function render(s) {
      document.getElementById('state').textContent = s.current_state;
      document.getElementById('players').textContent = s.selected_players;
      document.getElementById('live').textContent = s.live_face_count;
      document.getElementById('target').textContent = s.target_faces;
      document.getElementById('score').textContent = s.score;
      document.getElementById('strikes').textContent = s.strikes;
      document.getElementById('error').textContent = s.last_error ? `FPGA: ${s.last_error}` : '';
    }
    async function control(action) {
      const r = await fetch('/api/control', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action})
      });
      const data = await r.json();
      if (data && data.state) render(data.state);
    }
    function connect() {
      const src = new EventSource('/api/state/stream');
      const dot = document.getElementById('conn');
      src.onopen = () => { dot.classList.add('live'); dot.title = 'live'; };
      src.onmessage = (ev) => { try { render(JSON.parse(ev.data)); } catch (e) {} };
      src.onerror = () => {
        dot.classList.remove('live');
        dot.title = 'reconnecting';
        src.close();
        setTimeout(connect, 1000);
      };
    }
    connect();
  </script>
</body>
</html>
"""


class WebServer(threading.Thread):
    def __init__(self, exporter, controller, port=5000, stream_interval_s=0.1):
        super().__init__(daemon=True)
        self.exporter = exporter
        self.controller = controller
        self.port = port
        self.stream_interval_s = stream_interval_s
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/", methods=["GET"])
        def index():
            return INDEX_HTML

        @self.app.route("/api/state", methods=["GET"])
        def get_state():
            return jsonify(self.exporter.export())

        @self.app.route("/api/state/stream", methods=["GET"])
        def state_stream():
            return Response(
                self._state_events(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        @self.app.route("/api/control", methods=["POST"])
        def control():
            data = request.get_json(silent=True) or {}
            action = data.get("action")
            if action == "increment_players":
                self.controller.increment_players()
            elif action == "decrement_players":
                self.controller.decrement_players()
            elif action == "start":
                self.controller.request_start()
            elif action == "reset":
                self.controller.reset_game()
            else:
                return jsonify({"ok": False, "error": "unknown action"}), 400
            return jsonify({"ok": True, "state": self.exporter.export()})

        @self.app.route("/video_feed", methods=["GET"])
        def video_feed():
            return Response(
                self._mjpeg_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def _state_events(self):
        last_seq = -1
        last_payload = None
        last_emit = time.time()
        while True:
            last_seq = self.controller.wait_for_state(last_seq, timeout=15.0)
            state = self.exporter.export()
            payload = json.dumps(state, separators=(",", ":"))
            now = time.time()
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
                last_emit = now
            elif now - last_emit >= 15.0:
                yield ": keepalive\n\n"
                last_emit = now

    def _mjpeg_frames(self):
        last_seq = -1
        while True:
            last_seq, jpeg = self.controller.wait_for_jpeg(last_seq, timeout=5.0)
            if jpeg is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

    def run(self):
        self.app.run(host="0.0.0.0", port=self.port, use_reloader=False, threaded=True)
