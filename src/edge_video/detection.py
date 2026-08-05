"""Detection backends used on edge device B."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DetectionResult:
    annotated_frame: np.ndarray
    objects: list[dict[str, object]]
    inference_ms: float


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> DetectionResult: ...


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

    def detect(self, frame: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        predictions = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        result = predictions[0]
        objects: list[dict[str, object]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                coordinates = [round(value, 1) for value in box.xyxy[0].tolist()]
                objects.append(
                    {
                        "class_id": class_id,
                        "label": str(result.names[class_id]),
                        "confidence": round(float(box.conf[0].item()), 3),
                        "xyxy": coordinates,
                    }
                )

        return DetectionResult(
            annotated_frame=result.plot(),
            objects=objects,
            inference_ms=(time.perf_counter() - started) * 1000,
        )


def build_detector(
    kind: str,
    model_path: str,
    confidence: float,
    classes: list[int] | None,
    device: str | None,
) -> Detector:
    if kind == "mock":
        return MockDetector()
    if kind == "yolo":
        return YoloDetector(model_path, confidence, classes, device)
    raise ValueError(f"Unknown detector: {kind}")
