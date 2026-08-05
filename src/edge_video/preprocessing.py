"""Endpoint-side image preprocessing and compression."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    max_width: int = 960
    jpeg_quality: int = 75
    enhance_contrast: bool = True


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    jpeg: bytes
    width: int
    height: int
    processing_ms: float


def preprocess_frame(frame: np.ndarray, config: PreprocessConfig) -> ProcessedFrame:
    started = time.perf_counter()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")

    output = frame
    height, width = output.shape[:2]
    if config.max_width > 0 and width > config.max_width:
        scale = config.max_width / width
        output = cv2.resize(
            output,
            (config.max_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    if config.enhance_contrast:
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        output = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    quality = min(100, max(1, config.jpeg_quality))
    ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("OpenCV failed to encode a JPEG frame")

    output_height, output_width = output.shape[:2]
    return ProcessedFrame(
        jpeg=encoded.tobytes(),
        width=output_width,
        height=output_height,
        processing_ms=(time.perf_counter() - started) * 1000,
    )

