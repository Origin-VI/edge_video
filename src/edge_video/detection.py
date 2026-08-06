"""Detection backends used on edge device B."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from edge_video.tracking import RegionCounter


@dataclass(frozen=True, slots=True)
class DetectionResult:
    annotated_frame: np.ndarray
    objects: list[dict[str, object]]
    inference_ms: float
    tracking: dict[str, object] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> DetectionResult: ...


class FaceIdentityProvider(Protocol):
    def identify_people(self, frame: np.ndarray, people: list[dict[str, object]]) -> None: ...


class MockDetector:
    """A dependency-light backend for validating transport before AI setup."""

    name = "mock"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        output = frame.copy()
        label = "Transport test - AI disabled"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        x = max(16, output.shape[1] - text_width - 24)
        y = 38
        cv2.rectangle(
            output,
            (x - 10, y - text_height - 9),
            (x + text_width + 10, y + 9),
            (18, 24, 22),
            -1,
        )
        cv2.putText(
            output,
            label,
            (x, y),
            font,
            font_scale,
            (40, 220, 255),
            thickness,
            cv2.LINE_AA,
        )
        return DetectionResult(
            annotated_frame=output,
            objects=[],
            inference_ms=(time.perf_counter() - started) * 1000,
        )


class YoloDetector:
    name = "yolo"

    def __init__(
        self,
        model_path: str,
        confidence: float,
        classes: list[int] | None,
        device: str | None,
        tracking: bool,
        roi: tuple[float, float, float, float],
        missing_grace_frames: int,
        face_registry: FaceIdentityProvider | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is unavailable; install the project with the 'edge' extra"
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = classes
        self.device = device
        self.tracking = tracking
        self.region_counter = RegionCounter(roi, missing_grace_frames)
        self.face_registry = face_registry

    def detect(self, frame: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        inference_options = {
            "source": frame,
            "imgsz": 640,
            "conf": self.confidence,
            "classes": self.classes,
            "device": self.device,
            "verbose": False,
        }
        if self.tracking:
            predictions = self.model.track(
                **inference_options,
                persist=True,
                tracker="bytetrack.yaml",
            )
        else:
            predictions = self.model.predict(**inference_options)
        result = predictions[0]
        objects: list[dict[str, object]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                coordinates = [round(value, 1) for value in box.xyxy[0].tolist()]
                detected: dict[str, object] = {
                    "class_id": class_id,
                    "label": str(result.names[class_id]),
                    "confidence": round(float(box.conf[0].item()), 3),
                    "xyxy": coordinates,
                }
                if box.id is not None:
                    detected["track_id"] = int(box.id[0].item())
                objects.append(detected)

        people = [detected for detected in objects if detected["label"] == "person"]
        if self.face_registry is not None:
            self.face_registry.identify_people(frame, people)

        annotated = result.plot()
        if self.face_registry is not None:
            _draw_identity_labels(annotated, people)
        tracking_summary: dict[str, object] = {"tracking_enabled": False}
        events: list[dict[str, object]] = []
        if self.tracking:
            tracking_summary, events = self.region_counter.update(
                people,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
            _draw_region_overlay(annotated, self.region_counter, tracking_summary)

        return DetectionResult(
            annotated_frame=annotated,
            objects=objects,
            inference_ms=(time.perf_counter() - started) * 1000,
            tracking=tracking_summary,
            events=events,
        )


def build_detector(
    kind: str,
    model_path: str,
    confidence: float,
    classes: list[int] | None,
    device: str | None,
    tracking: bool = True,
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    missing_grace_frames: int = 10,
    face_registry: FaceIdentityProvider | None = None,
) -> Detector:
    if kind == "mock":
        return MockDetector()
    if kind == "yolo":
        return YoloDetector(
            model_path,
            confidence,
            classes,
            device,
            tracking,
            roi,
            missing_grace_frames,
            face_registry,
        )
    raise ValueError(f"Unknown detector: {kind}")


def _draw_identity_labels(frame: np.ndarray, people: list[dict[str, object]]) -> None:
    labels: list[tuple[int, int, str, tuple[int, int, int]]] = []
    frame_height, frame_width = frame.shape[:2]
    for person in people:
        coordinates = person.get("xyxy")
        identity = person.get("identity")
        if not isinstance(coordinates, list) or len(coordinates) != 4 or not identity:
            continue
        x1, y1, _, _ = (round(float(value)) for value in coordinates)
        label = str(identity)
        color = (130, 220, 70) if identity != "stranger" else (245, 170, 70)
        labels.append((x1, y1, label, color))

    if not labels:
        return

    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _identity_font()
    for x1, y1, label, color in labels:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width = right - left
        text_height = bottom - top
        label_x = max(0, min(x1, frame_width - text_width - 12))
        label_y = max(0, min(y1 - text_height - 14, frame_height - text_height - 12))
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 10, label_y + text_height + 9),
            fill=(22, 24, 18),
        )
        draw.text((label_x + 5, label_y + 2 - top), label, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=1)
def _identity_font():
    from PIL import ImageFont

    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), 22)
    return ImageFont.load_default()


def _draw_region_overlay(
    frame: np.ndarray,
    counter: RegionCounter,
    summary: dict[str, object],
) -> None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = counter.pixel_roi(frame_width, frame_height)
    color = (70, 220, 130)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = (
        f"ROI  Inside {summary['current_inside']}  "
        f"In {summary['total_entered']}  Out {summary['total_exited']}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = max(8, min(x1 + 8, frame_width - text_width - 16))
    label_y = max(text_height + 12, min(y2 - 10, frame_height - 12))
    cv2.rectangle(
        frame,
        (label_x - 7, label_y - text_height - 7),
        (label_x + text_width + 7, label_y + 7),
        (18, 24, 22),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
