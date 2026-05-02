from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np
import torch

from src.models.factory import build_model

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_COLORS = {
    "angry": (70, 70, 255),
    "disgust": (80, 200, 120),
    "fear": (255, 120, 60),
    "happy": (0, 215, 255),
    "neutral": (190, 190, 190),
    "sad": (255, 140, 0),
    "surprise": (255, 80, 180),
    "intensity": (0, 255, 255),
}


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current environment.")
        return torch.device("cuda")

    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in the current environment.")
        return torch.device("mps")

    return torch.device("cpu")


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def area(self) -> int:
        return max(self.x2 - self.x1, 0) * max(self.y2 - self.y1, 0)


@dataclass
class EmotionPrediction:
    label: str
    confidence: float
    probabilities: np.ndarray


@dataclass
class TimelineEntry:
    timestamp: float
    probabilities: np.ndarray
    confidence: float
    label: str


class MediaPipeFaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self.backend = "opencv_haar"
        self._face_detection = None
        self._cascade = None
        self._min_detection_confidence = min_detection_confidence

        try:
            import mediapipe as mp

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_detection_confidence,
                )
                self.backend = "mediapipe_solutions"
                return
        except ModuleNotFoundError:
            pass

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade face detector.")
        self._cascade = cascade

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        boxes: list[FaceBox] = []

        if self._face_detection is not None:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._face_detection.process(image_rgb)
            detections = result.detections or []
            frame_height, frame_width = frame_bgr.shape[:2]

            for detection in detections:
                relative = detection.location_data.relative_bounding_box
                x1 = max(int(relative.xmin * frame_width), 0)
                y1 = max(int(relative.ymin * frame_height), 0)
                x2 = min(int((relative.xmin + relative.width) * frame_width), frame_width - 1)
                y2 = min(int((relative.ymin + relative.height) * frame_height), frame_height - 1)
                score = float(detection.score[0]) if detection.score else 0.0
                boxes.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score))
        elif self._cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detections = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            for (x, y, w, h) in detections:
                boxes.append(FaceBox(x1=int(x), y1=int(y), x2=int(x + w), y2=int(y + h), score=1.0))

        boxes.sort(key=lambda item: item.area, reverse=True)
        return boxes

    def close(self) -> None:
        if self._face_detection is not None:
            self._face_detection.close()


class EmotionRecognizer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "resnet_cbam",
        device_name: str = "auto",
        ema_alpha: float = 0.6,
    ) -> None:
        self.device = resolve_device(device_name)
        self.model = build_model(model_name, num_classes=len(EMOTION_LABELS)).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.ema_alpha = ema_alpha
        self.smoothed_probabilities: np.ndarray | None = None

    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        tensor = self._preprocess(face_bgr).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        if self.smoothed_probabilities is None:
            self.smoothed_probabilities = probabilities
        else:
            self.smoothed_probabilities = (
                self.ema_alpha * probabilities + (1.0 - self.ema_alpha) * self.smoothed_probabilities
            )

        label_index = int(np.argmax(self.smoothed_probabilities))
        confidence = float(self.smoothed_probabilities[label_index])
        return EmotionPrediction(
            label=EMOTION_LABELS[label_index],
            confidence=confidence,
            probabilities=self.smoothed_probabilities.copy(),
        )

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.timestamps: list[float] = []

    def update(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / max(elapsed, 1e-6)


class EmotionTimeline:
    def __init__(self, max_length: int = 180) -> None:
        self.entries: deque[TimelineEntry] = deque(maxlen=max_length)

    def append(self, prediction: EmotionPrediction) -> None:
        self.entries.append(
            TimelineEntry(
                timestamp=time.perf_counter(),
                probabilities=prediction.probabilities.copy(),
                confidence=prediction.confidence,
                label=prediction.label,
            )
        )

    def is_empty(self) -> bool:
        return not self.entries

    def latest(self) -> TimelineEntry | None:
        return self.entries[-1] if self.entries else None


def crop_face(frame_bgr: np.ndarray, face_box: FaceBox, padding: float = 0.18) -> tuple[np.ndarray, FaceBox]:
    frame_height, frame_width = frame_bgr.shape[:2]
    width = face_box.x2 - face_box.x1
    height = face_box.y2 - face_box.y1
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    x1 = max(face_box.x1 - pad_x, 0)
    y1 = max(face_box.y1 - pad_y, 0)
    x2 = min(face_box.x2 + pad_x, frame_width - 1)
    y2 = min(face_box.y2 + pad_y, frame_height - 1)
    return frame_bgr[y1:y2, x1:x2].copy(), FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=face_box.score)


def draw_video_overlay(
    frame_bgr: np.ndarray,
    face_box: FaceBox | None,
    prediction: EmotionPrediction | None,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if face_box is not None:
        color = EMOTION_COLORS.get(prediction.label if prediction else "neutral", (0, 220, 120))
        cv2.rectangle(canvas, (face_box.x1, face_box.y1), (face_box.x2, face_box.y2), color, 3)
    return canvas


def build_dashboard(
    frame_bgr: np.ndarray,
    prediction: EmotionPrediction | None,
    face_box: FaceBox | None,
    fps: float,
    detector_backend: str,
    timeline: EmotionTimeline,
) -> np.ndarray:
    sidebar_width = 340
    chart_height = 260
    frame_height, frame_width = frame_bgr.shape[:2]

    video_panel = draw_video_overlay(frame_bgr, face_box, prediction)
    sidebar = np.full((frame_height, sidebar_width, 3), (22, 24, 28), dtype=np.uint8)
    chart = np.full((chart_height, sidebar_width + frame_width, 3), (18, 20, 24), dtype=np.uint8)

    _draw_sidebar(sidebar, prediction, fps, detector_backend, timeline)
    _draw_chart(chart, timeline)

    top = np.hstack([sidebar, video_panel])
    dashboard = np.vstack([top, chart])
    return dashboard


def _draw_sidebar(
    sidebar: np.ndarray,
    prediction: EmotionPrediction | None,
    fps: float,
    detector_backend: str,
    timeline: EmotionTimeline,
) -> None:
    cv2.putText(
        sidebar,
        "Emotion Monitor",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sidebar,
        f"FPS {fps:.1f}",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sidebar,
        f"Detector {detector_backend}",
        (24, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (170, 170, 170),
        2,
        cv2.LINE_AA,
    )

    if prediction is None:
        cv2.putText(
            sidebar,
            "NO FACE",
            (24, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 180, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            sidebar,
            "Waiting for a face...",
            (24, 238),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        return

    dominant_color = EMOTION_COLORS.get(prediction.label, (255, 255, 255))
    cv2.putText(
        sidebar,
        prediction.label.upper(),
        (24, 198),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        dominant_color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        sidebar,
        f"Intensity {prediction.confidence * 100:.1f}%",
        (24, 242),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        sidebar,
        "Current Scores",
        (24, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (225, 225, 225),
        2,
        cv2.LINE_AA,
    )

    top_indices = np.argsort(prediction.probabilities)[::-1][:5]
    bar_start_y = 324
    bar_x = 24
    bar_width = sidebar.shape[1] - 48
    bar_height = 18
    for offset, index in enumerate(top_indices):
        label = EMOTION_LABELS[int(index)]
        value = float(prediction.probabilities[int(index)])
        y = bar_start_y + offset * 44
        color = EMOTION_COLORS.get(label, (255, 255, 255))
        cv2.putText(
            sidebar,
            f"{label:>8s}  {value:.3f}",
            (bar_x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(sidebar, (bar_x, y), (bar_x + bar_width, y + bar_height), (55, 58, 66), -1)
        cv2.rectangle(sidebar, (bar_x, y), (bar_x + int(bar_width * value), y + bar_height), color, -1)

    latest = timeline.latest()
    if latest is not None:
        cv2.putText(
            sidebar,
            f"Samples {len(timeline.entries)}",
            (24, sidebar.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )


def _draw_chart(chart: np.ndarray, timeline: EmotionTimeline) -> None:
    cv2.putText(
        chart,
        "Realtime Emotion Curve",
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    left = 64
    right = chart.shape[1] - 26
    top = 66
    bottom = chart.shape[0] - 34
    cv2.rectangle(chart, (left, top), (right, bottom), (50, 54, 62), 1)

    for row in range(5):
        y = int(top + (bottom - top) * row / 4)
        cv2.line(chart, (left, y), (right, y), (38, 42, 48), 1)
        value = 1.0 - row * 0.25
        cv2.putText(
            chart,
            f"{value:.2f}",
            (14, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    if timeline.is_empty():
        cv2.putText(
            chart,
            "Waiting for predictions...",
            (left + 18, top + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (190, 190, 190),
            2,
            cv2.LINE_AA,
        )
        return

    latest = timeline.latest()
    assert latest is not None
    top_indices = np.argsort(latest.probabilities)[::-1][:3]
    legend_items = [EMOTION_LABELS[int(index)] for index in top_indices] + ["intensity"]

    legend_x = left + 6
    for item in legend_items:
        color = EMOTION_COLORS[item]
        cv2.rectangle(chart, (legend_x, 40), (legend_x + 16, 56), color, -1)
        cv2.putText(
            chart,
            item,
            (legend_x + 24, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 225, 225),
            2,
            cv2.LINE_AA,
        )
        legend_x += 150

    entries = list(timeline.entries)
    if len(entries) < 2:
        return

    series_map: dict[str, list[float]] = {name: [] for name in legend_items}
    for entry in entries:
        for name in legend_items:
            if name == "intensity":
                series_map[name].append(entry.confidence)
            else:
                series_map[name].append(float(entry.probabilities[EMOTION_LABELS.index(name)]))

    for name in legend_items:
        values = series_map[name]
        points: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            x = int(left + (right - left) * index / (len(values) - 1))
            y = int(bottom - value * (bottom - top))
            points.append((x, y))
        polyline = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(chart, [polyline], False, EMOTION_COLORS[name], 2, cv2.LINE_AA)

    cv2.putText(
        chart,
        f"Recent samples: {len(entries)}",
        (right - 180, chart.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
