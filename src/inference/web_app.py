from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from src.inference.realtime import (
    EMOTION_COLORS,
    EMOTION_LABELS,
    EmotionPrediction,
    EmotionRecognizer,
    EmotionTimeline,
    FPSCounter,
    FaceBox,
    MediaPipeFaceDetector,
    crop_face,
    draw_video_overlay,
)


def _bgr_to_hex(color: tuple[int, int, int]) -> str:
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"


WEB_COLORS = {label: _bgr_to_hex(color) for label, color in EMOTION_COLORS.items()}


@dataclass
class WebConfig:
    camera: int
    checkpoint: Path
    model: str
    device: str
    host: str
    port: int
    min_detection_confidence: float
    frame_width: int
    frame_height: int
    jpeg_quality: int
    history_length: int


class SharedWebState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg_frame: bytes | None = None
        self._payload: dict[str, Any] = {
            "status": "starting",
            "error": None,
            "labels": EMOTION_LABELS,
            "colors": WEB_COLORS,
            "fps": 0.0,
            "duration": 0.0,
            "samples": 0,
            "faceDetected": False,
            "emotion": None,
            "confidence": 0.0,
            "probabilities": _empty_probabilities(),
            "topProbabilities": [],
            "timeline": [],
        }

    def set_frame(self, jpeg_frame: bytes) -> None:
        with self._lock:
            self._jpeg_frame = jpeg_frame

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._jpeg_frame

    def set_payload(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._payload = payload

    def get_payload(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._payload))


class RealtimeWebWorker:
    def __init__(self, config: WebConfig, state: SharedWebState) -> None:
        self.config = config
        self.state = state
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started_at = time.perf_counter()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="realtime-web-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        detector: MediaPipeFaceDetector | None = None
        camera: cv2.VideoCapture | None = None

        try:
            detector = MediaPipeFaceDetector(
                min_detection_confidence=self.config.min_detection_confidence
            )
            recognizer = EmotionRecognizer(
                checkpoint_path=self.config.checkpoint,
                model_name=self.config.model,
                device_name=self.config.device,
            )
            timeline = EmotionTimeline(max_length=self.config.history_length)
            fps_counter = FPSCounter()

            camera = cv2.VideoCapture(self.config.camera)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            if not camera.isOpened():
                raise RuntimeError(f"Unable to open camera: {self.config.camera}")

            while not self._stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    self._set_error("Failed to read a camera frame.")
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                boxes = detector.detect(frame)
                fps = fps_counter.update()
                prediction: EmotionPrediction | None = None
                active_box: FaceBox | None = None

                if boxes:
                    face_crop, expanded_box = crop_face(frame, boxes[0])
                    if face_crop.size > 0:
                        prediction = recognizer.predict(face_crop)
                        active_box = expanded_box
                        timeline.append(prediction)

                display = draw_video_overlay(frame, active_box, prediction)
                self._publish_frame(display)
                self.state.set_payload(
                    _build_payload(
                        status="running",
                        error=None,
                        model=self.config.model,
                        device=str(recognizer.device),
                        detector_backend=detector.backend,
                        fps=fps,
                        duration=time.perf_counter() - self._started_at,
                        prediction=prediction,
                        face_box=active_box,
                        timeline=timeline,
                    )
                )
        except Exception as error:
            self._set_error(str(error))
        finally:
            if camera is not None:
                camera.release()
            if detector is not None:
                detector.close()

    def _publish_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )
        if ok:
            self.state.set_frame(encoded.tobytes())

    def _set_error(self, message: str) -> None:
        payload = self.state.get_payload()
        payload["status"] = "error"
        payload["error"] = message
        self.state.set_payload(payload)


def _empty_probabilities() -> list[dict[str, Any]]:
    return [
        {"label": label, "value": 0.0, "percent": 0.0, "color": WEB_COLORS[label]}
        for label in EMOTION_LABELS
    ]


def _build_payload(
    status: str,
    error: str | None,
    model: str,
    device: str,
    detector_backend: str,
    fps: float,
    duration: float,
    prediction: EmotionPrediction | None,
    face_box: FaceBox | None,
    timeline: EmotionTimeline,
) -> dict[str, Any]:
    probabilities = _format_probabilities(prediction)
    return {
        "status": status,
        "error": error,
        "model": model,
        "device": device,
        "detectorBackend": detector_backend,
        "labels": EMOTION_LABELS,
        "colors": WEB_COLORS,
        "fps": round(float(fps), 2),
        "duration": round(float(duration), 2),
        "samples": len(timeline.entries),
        "faceDetected": face_box is not None,
        "faceBox": _format_face_box(face_box),
        "emotion": prediction.label if prediction else None,
        "confidence": round(float(prediction.confidence), 4) if prediction else 0.0,
        "probabilities": probabilities,
        "topProbabilities": sorted(probabilities, key=lambda item: item["value"], reverse=True)[:3],
        "timeline": _format_timeline(timeline),
    }


def _format_probabilities(prediction: EmotionPrediction | None) -> list[dict[str, Any]]:
    if prediction is None:
        return _empty_probabilities()

    values: list[dict[str, Any]] = []
    for index, label in enumerate(EMOTION_LABELS):
        value = float(prediction.probabilities[index])
        values.append(
            {
                "label": label,
                "value": round(value, 4),
                "percent": round(value * 100.0, 1),
                "color": WEB_COLORS[label],
            }
        )
    return values


def _format_face_box(face_box: FaceBox | None) -> dict[str, Any] | None:
    if face_box is None:
        return None
    return {
        "x1": face_box.x1,
        "y1": face_box.y1,
        "x2": face_box.x2,
        "y2": face_box.y2,
        "score": round(face_box.score, 3),
    }


def _format_timeline(timeline: EmotionTimeline) -> list[dict[str, Any]]:
    entries = list(timeline.entries)
    if not entries:
        return []

    start = entries[0].timestamp
    formatted = []
    for entry in entries:
        formatted.append(
            {
                "time": round(entry.timestamp - start, 3),
                "label": entry.label,
                "confidence": round(float(entry.confidence), 4),
                "probabilities": {
                    label: round(float(entry.probabilities[index]), 4)
                    for index, label in enumerate(EMOTION_LABELS)
                },
            }
        )
    return formatted


class WebRequestHandler(BaseHTTPRequestHandler):
    state: SharedWebState

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html()
        elif self.path == "/api/state":
            self._send_json(self.state.get_payload())
        elif self.path == "/video_feed":
            self._stream_video()
        elif self.path == "/health":
            self._send_json({"ok": True})
        elif self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_html(self) -> None:
        content = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any]) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _stream_video(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        while True:
            frame = self.state.get_frame()
            if frame is None:
                frame = _placeholder_frame()
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                break


def _placeholder_frame() -> bytes:
    frame = np.full((540, 960, 3), (11, 15, 22), dtype=np.uint8)
    cv2.putText(
        frame,
        "Waiting for camera stream...",
        (260, 275),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (180, 190, 205),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return encoded.tobytes() if ok else b""


def serve_web_app(config: WebConfig) -> None:
    state = SharedWebState()
    worker = RealtimeWebWorker(config=config, state=state)
    handler_class = type("BoundWebRequestHandler", (WebRequestHandler,), {"state": state})
    server = ThreadingHTTPServer((config.host, config.port), handler_class)

    worker.start()
    try:
        print(f"Web demo running at http://{config.host}:{config.port}")
        print("Press Ctrl+C to stop.")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        worker.stop()


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Realtime Emotion Dashboard</title>
  <style>
    :root {
      --bg: #080d15;
      --panel: rgba(19, 27, 40, 0.94);
      --panel-2: rgba(15, 22, 34, 0.96);
      --line: rgba(148, 163, 184, 0.14);
      --text: #e7eef8;
      --muted: #8290a4;
      --soft: #c9d2df;
      --teal: #32d7b5;
      --blue: #68a7ff;
      --amber: #ffbd55;
      --danger: #ff6b7b;
      --shadow: 0 22px 70px rgba(0, 0, 0, 0.45);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 18% 8%, rgba(45, 212, 191, 0.12), transparent 28%),
        radial-gradient(circle at 85% 12%, rgba(96, 165, 250, 0.11), transparent 34%),
        linear-gradient(180deg, #060a11 0%, #0b111d 50%, #080c14 100%);
      color: var(--text);
      font-family: "Avenir Next", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
      letter-spacing: 0.01em;
    }

    .page {
      width: min(1500px, calc(100vw - 36px));
      margin: 0 auto;
      padding: 26px 0 34px;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 20px;
    }

    .eyebrow {
      color: var(--teal);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }

    h1 {
      margin: 6px 0 0;
      font-size: clamp(28px, 4vw, 54px);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }

    .status-pill {
      min-width: 190px;
      padding: 14px 18px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.7);
      color: var(--soft);
      text-align: center;
      box-shadow: var(--shadow);
    }

    .status-dot {
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 9px;
      border-radius: 999px;
      background: var(--teal);
      box-shadow: 0 0 22px var(--teal);
    }

    .dashboard {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .side-stack,
    .main-stack {
      display: grid;
      gap: 18px;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(22, 31, 46, 0.96), rgba(12, 18, 29, 0.96));
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .hero-card {
      position: relative;
      padding: 30px 26px;
      min-height: 210px;
      background:
        radial-gradient(circle at 55% 95%, rgba(45, 212, 191, 0.22), transparent 38%),
        linear-gradient(160deg, rgba(23, 43, 60, 0.98), rgba(11, 17, 28, 0.98));
    }

    .hero-card::after {
      content: "";
      position: absolute;
      inset: auto 24px 20px 24px;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(94, 234, 212, 0.5), transparent);
    }

    .emotion-name {
      margin-top: 28px;
      font-size: 38px;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .confidence-label {
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .confidence-value {
      margin-top: 6px;
      font-size: 56px;
      font-weight: 300;
      letter-spacing: -0.06em;
    }

    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 20px 6px;
      font-size: 16px;
      font-weight: 800;
    }

    .dots {
      color: var(--muted);
      letter-spacing: 0.2em;
    }

    .prob-list,
    .summary-list {
      padding: 8px 20px 20px;
      display: grid;
      gap: 16px;
    }

    .prob-row {
      display: grid;
      grid-template-columns: 88px 1fr 48px;
      align-items: center;
      gap: 12px;
      color: var(--soft);
      font-size: 15px;
      font-weight: 700;
    }

    .bar-track {
      height: 8px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.16);
      overflow: hidden;
    }

    .bar-fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: var(--teal);
      transition: width 180ms ease;
    }

    .summary-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      color: var(--muted);
      font-weight: 700;
    }

    .summary-row strong {
      color: var(--text);
      font-size: 20px;
      font-weight: 600;
    }

    .video-card {
      padding: 12px;
    }

    .video-frame {
      position: relative;
      overflow: hidden;
      border-radius: 17px;
      background: #05080d;
      aspect-ratio: 16 / 8.2;
    }

    .video-frame img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .video-glow {
      pointer-events: none;
      position: absolute;
      inset: 0;
      box-shadow: inset 0 -120px 130px rgba(0, 0, 0, 0.45);
    }

    .video-caption {
      position: absolute;
      left: 18px;
      bottom: 16px;
      padding: 10px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(5, 9, 16, 0.72);
      backdrop-filter: blur(10px);
      color: var(--soft);
      font-weight: 800;
    }

    .micro-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
    }

    .mini-card {
      padding: 16px 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel-2);
      min-height: 96px;
    }

    .mini-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .mini-value {
      margin-top: 14px;
      font-size: 28px;
      font-weight: 700;
    }

    .chart-card {
      padding: 18px 20px 22px;
    }

    .chart-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }

    .chart-header h2 {
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.02em;
    }

    .legend {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--soft);
      font-weight: 800;
      font-size: 13px;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }

    .legend-color {
      width: 11px;
      height: 11px;
      border-radius: 999px;
      background: var(--teal);
    }

    .chart-wrap {
      height: 270px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(rgba(148, 163, 184, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px),
        rgba(9, 14, 23, 0.74);
      background-size: 100% 25%, 10% 100%, 100% 100%;
      overflow: hidden;
    }

    svg {
      width: 100%;
      height: 100%;
      display: block;
    }

    .axis-label {
      fill: #8390a4;
      font-size: 12px;
      font-weight: 700;
    }

    @media (max-width: 980px) {
      .dashboard {
        grid-template-columns: 1fr;
      }

      .micro-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="header">
      <div>
        <div class="eyebrow">Realtime FER System</div>
        <h1>Emotion<br />Intelligence</h1>
      </div>
      <div class="status-pill"><span class="status-dot"></span><span id="statusText">Starting</span></div>
    </header>

    <main class="dashboard">
      <aside class="side-stack">
        <section class="card hero-card">
          <div class="eyebrow">Detected Emotion</div>
          <div id="emotionName" class="emotion-name">NO FACE</div>
          <div class="confidence-label">Confidence</div>
          <div id="confidenceValue" class="confidence-value">0.0%</div>
        </section>

        <section class="card">
          <div class="section-title"><span>Top Probabilities</span><span class="dots">••</span></div>
          <div id="probabilityList" class="prob-list"></div>
        </section>

        <section class="card">
          <div class="section-title"><span>Session Summary</span><span class="dots">••</span></div>
          <div class="summary-list">
            <div class="summary-row"><span>Duration</span><strong id="durationValue">00:00</strong></div>
            <div class="summary-row"><span>Total Samples</span><strong id="samplesValue">0</strong></div>
            <div class="summary-row"><span>FPS</span><strong id="fpsValue">0.0</strong></div>
          </div>
        </section>
      </aside>

      <section class="main-stack">
        <section class="card video-card">
          <div class="video-frame">
            <img src="/video_feed" alt="Realtime camera feed" />
            <div class="video-glow"></div>
            <div class="video-caption" id="videoCaption">Waiting for camera stream</div>
          </div>
        </section>

        <section class="micro-grid">
          <div class="mini-card"><div class="mini-label">Model</div><div class="mini-value" id="modelValue">-</div></div>
          <div class="mini-card"><div class="mini-label">Device</div><div class="mini-value" id="deviceValue">-</div></div>
          <div class="mini-card"><div class="mini-label">Detector</div><div class="mini-value" id="detectorValue">-</div></div>
        </section>

        <section class="card chart-card">
          <div class="chart-header">
            <h2>Emotion Trends</h2>
            <div id="legend" class="legend"></div>
          </div>
          <div class="chart-wrap">
            <svg id="emotionChart" viewBox="0 0 900 270" preserveAspectRatio="none"></svg>
          </div>
        </section>
      </section>
    </main>
  </div>

  <script>
    const labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];
    let colors = {};

    function formatDuration(seconds) {
      const total = Math.max(0, Math.floor(seconds || 0));
      const minutes = String(Math.floor(total / 60)).padStart(2, "0");
      const secs = String(total % 60).padStart(2, "0");
      return `${minutes}:${secs}`;
    }

    function updateProbabilityList(items) {
      const root = document.getElementById("probabilityList");
      const sorted = [...items].sort((a, b) => b.value - a.value).slice(0, 5);
      root.innerHTML = sorted.map(item => `
        <div class="prob-row">
          <span>${item.label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${item.percent}%; background:${item.color}"></div></div>
          <strong>${item.percent.toFixed(0)}%</strong>
        </div>
      `).join("");
    }

    function updateLegend(series) {
      const legend = document.getElementById("legend");
      legend.innerHTML = series.map(name => `
        <span class="legend-item"><span class="legend-color" style="background:${colors[name]}"></span>${name}</span>
      `).join("");
    }

    function buildPath(values, width, height, pad) {
      if (values.length < 2) return "";
      return values.map((value, index) => {
        const x = pad.left + (width - pad.left - pad.right) * index / (values.length - 1);
        const y = pad.top + (1 - value) * (height - pad.top - pad.bottom);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
    }

    function drawChart(state) {
      const svg = document.getElementById("emotionChart");
      const width = 900;
      const height = 270;
      const pad = { left: 46, right: 22, top: 22, bottom: 36 };
      const timeline = state.timeline || [];
      const topSeries = [...(state.topProbabilities || [])].map(item => item.label);
      const series = [...topSeries, "intensity"];
      updateLegend(series);

      let grid = "";
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (height - pad.top - pad.bottom) * i / 4;
        const value = (1 - i * 0.25).toFixed(2);
        grid += `<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="rgba(148,163,184,.15)" />`;
        grid += `<text class="axis-label" x="8" y="${y + 4}">${value}</text>`;
      }

      if (timeline.length < 2) {
        svg.innerHTML = `${grid}<text x="360" y="138" fill="#8390a4" font-size="18" font-weight="700">Waiting for emotion samples...</text>`;
        return;
      }

      let lines = "";
      for (const name of series) {
        const values = timeline.map(point => name === "intensity" ? point.confidence : (point.probabilities[name] || 0));
        const path = buildPath(values, width, height, pad);
        lines += `<polyline points="${path}" fill="none" stroke="${colors[name]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />`;
        const latest = values[values.length - 1] || 0;
        const y = pad.top + (1 - latest) * (height - pad.top - pad.bottom);
        lines += `<text x="${width - pad.right - 56}" y="${y - 8}" fill="${colors[name]}" font-size="18" font-weight="800">${Math.round(latest * 100)}%</text>`;
      }

      const firstTime = timeline[0].time || 0;
      const lastTime = timeline[timeline.length - 1].time || 0;
      const footer = `
        <text class="axis-label" x="${pad.left}" y="${height - 10}">${firstTime.toFixed(1)}s</text>
        <text class="axis-label" x="${width - 78}" y="${height - 10}">${lastTime.toFixed(1)}s</text>
      `;
      svg.innerHTML = grid + lines + footer;
    }

    function applyState(state) {
      colors = state.colors || colors;
      const emotion = state.emotion || "NO FACE";
      const confidence = Math.round((state.confidence || 0) * 1000) / 10;
      const faceStatus = state.faceDetected ? "Face detected" : "No face detected";
      const status = state.error ? "Error" : state.status;

      document.getElementById("statusText").textContent = status;
      document.getElementById("emotionName").textContent = emotion.toUpperCase();
      document.getElementById("emotionName").style.color = colors[emotion] || "#e7eef8";
      document.getElementById("confidenceValue").textContent = `${confidence.toFixed(1)}%`;
      document.getElementById("durationValue").textContent = formatDuration(state.duration);
      document.getElementById("samplesValue").textContent = state.samples || 0;
      document.getElementById("fpsValue").textContent = (state.fps || 0).toFixed(1);
      document.getElementById("modelValue").textContent = state.model || "-";
      document.getElementById("deviceValue").textContent = state.device || "-";
      document.getElementById("detectorValue").textContent = state.detectorBackend || "-";
      document.getElementById("videoCaption").textContent = `${faceStatus} · ${emotion} · ${(state.fps || 0).toFixed(1)} FPS`;

      updateProbabilityList(state.probabilities || []);
      drawChart(state);
    }

    async function refreshState() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        const state = await response.json();
        applyState(state);
      } catch (error) {
        document.getElementById("statusText").textContent = "Disconnected";
      }
    }

    refreshState();
    setInterval(refreshState, 350);
  </script>
</body>
</html>
"""
