from pathlib import Path

import cv2
import numpy as np
import pytest

from edge_video.recognition import EnrollmentError, FaceRegistry


class FakeFaceBackend:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = [np.asarray(value, dtype=np.float32) for value in embeddings]

    def features(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if not self.embeddings:
            return []
        embedding = self.embeddings.pop(0)
        face = np.asarray([2, 2, 8, 8] + [0] * 11, dtype=np.float32)
        return [(face, embedding)]


def photo_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_registry_enrolls_persists_and_deletes(tmp_path: Path) -> None:
    registry = FaceRegistry(tmp_path, FakeFaceBackend([[1.0, 0.0]]))

    enrolled = registry.enroll("Alice", photo_bytes())
    assert enrolled["name"] == "Alice"
    assert registry.photo_path(enrolled["id"]) is not None

    reloaded = FaceRegistry(tmp_path, FakeFaceBackend([]))
    assert reloaded.list_identities()[0]["name"] == "Alice"
    assert reloaded.delete(enrolled["id"]) is True
    assert reloaded.list_identities() == []


def test_registry_rejects_duplicate_name(tmp_path: Path) -> None:
    registry = FaceRegistry(tmp_path, FakeFaceBackend([[1.0, 0.0], [1.0, 0.0]]))
    registry.enroll("Alice", photo_bytes())

    with pytest.raises(EnrollmentError, match="已经登记"):
        registry.enroll("alice", photo_bytes())


def test_registry_labels_known_and_unknown_people(tmp_path: Path) -> None:
    backend = FakeFaceBackend([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    registry = FaceRegistry(tmp_path, backend, similarity_threshold=0.5)
    registry.enroll("Alice", photo_bytes())

    known = {"track_id": 1, "xyxy": [0.0, 0.0, 20.0, 20.0]}
    registry.identify_people(np.zeros((20, 20, 3), dtype=np.uint8), [known])
    assert known["identity"] == "Alice"
    assert known["face_similarity"] == pytest.approx(0.9)

    unknown = {"track_id": 2, "xyxy": [0.0, 0.0, 20.0, 20.0]}
    registry.identify_people(np.zeros((20, 20, 3), dtype=np.uint8), [unknown])
    assert unknown["identity"] == "stranger"
