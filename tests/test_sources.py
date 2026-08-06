import cv2

from edge_video.sources import OpenCVSource


def test_camera_source_requests_mjpeg_and_single_frame_buffer(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.settings: list[tuple[int, float]] = []

        def set(self, prop: int, value: float) -> bool:
            self.settings.append((prop, value))
            return True

        def isOpened(self) -> bool:
            return True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    OpenCVSource("0", 1280, 720)

    assert (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) in capture.settings
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in capture.settings
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in capture.settings
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in capture.settings
