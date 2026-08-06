"""Video sources for a camera, file, CSI camera, or local simulation."""

from __future__ import annotations

import time
from typing import Protocol

import cv2
import numpy as np


class VideoSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class OpenCVSource:
    def __init__(self, source: str, width: int, height: int) -> None:
        parsed_source: int | str = int(source) if source.isdecimal() else source
        self.capture = cv2.VideoCapture(parsed_source)
        if isinstance(parsed_source, int):
            # UVC cameras can sustain higher resolutions with hardware MJPEG than raw YUYV.
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class Picamera2Source:
    def __init__(self, width: int, height: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is unavailable; install python3-picamera2 on the Raspberry Pi"
            ) from exc

        self.camera = Picamera2()
        size = (width if width > 0 else 1280, height if height > 0 else 720)
        configuration = self.camera.create_video_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self.camera.configure(configuration)
        self.camera.start()

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self.camera.capture_array("main")
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self.camera.stop()


class SyntheticSource:
    def __init__(self, width: int, height: int) -> None:
        self.width = width if width > 0 else 1280
        self.height = height if height > 0 else 720
        self.frame_id = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (32, 38, 42)
        x = (self.frame_id * 13) % max(1, self.width - 180)
        cv2.rectangle(frame, (x, 120), (x + 180, 420), (30, 180, 235), -1)
        cv2.putText(
            frame,
            f"Synthetic frame {self.frame_id}",
            (40, self.height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (120, 220, 160),
            2,
            cv2.LINE_AA,
        )
        self.frame_id += 1
        return True, frame

    def release(self) -> None:
        return None


def open_video_source(source: str, width: int, height: int) -> VideoSource:
    if source == "synthetic":
        return SyntheticSource(width, height)
    if source == "picamera2":
        return Picamera2Source(width, height)
    return OpenCVSource(source, width, height)
