import cv2
import numpy as np

from edge_video.preprocessing import PreprocessConfig, preprocess_frame


def test_preprocess_resizes_and_encodes_jpeg() -> None:
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)

    result = preprocess_frame(
        frame,
        PreprocessConfig(max_width=640, jpeg_quality=70, enhance_contrast=False),
    )

    decoded = cv2.imdecode(np.frombuffer(result.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert (result.width, result.height) == (640, 360)
    assert decoded.shape[:2] == (360, 640)
    assert result.processing_ms >= 0


def test_preprocess_rejects_grayscale_input() -> None:
    frame = np.zeros((100, 100), dtype=np.uint8)

    try:
        preprocess_frame(frame, PreprocessConfig())
    except ValueError as exc:
        assert "three channels" in str(exc)
    else:
        raise AssertionError("Expected grayscale input to be rejected")
