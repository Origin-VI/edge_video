"""Persistent face enrollment and OpenCV-based identity matching."""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

LOG = logging.getLogger("edge_video.recognition")

FACE_MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": [
        (
            "https://hf-mirror.com/opencv/face_detection_yunet/resolve/main/"
            "face_detection_yunet_2023mar.onnx"
        ),
        (
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
    ],
    "face_recognition_sface_2021dec.onnx": [
        (
            "https://hf-mirror.com/opencv/face_recognition_sface/resolve/main/"
            "face_recognition_sface_2021dec.onnx"
        ),
        (
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
    ],
}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class EnrollmentError(ValueError):
    """Raised when an uploaded reference photo cannot be enrolled."""


class FaceBackend(Protocol):
    def features(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]: ...


@dataclass(frozen=True, slots=True)
class Identity:
    identity_id: str
    name: str
    photo_filename: str
    embedding: np.ndarray


class OpenCvFaceBackend:
    """YuNet face detection plus SFace embedding extraction."""

    def __init__(self, detector_model: Path, recognizer_model: Path) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model.resolve()), "", (320, 320), 0.8, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model.resolve()), "")
        self._lock = threading.Lock()

    def features(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if image.ndim != 3 or image.shape[2] != 3:
            return []
        height, width = image.shape[:2]
        if width < 2 or height < 2:
            return []

        with self._lock:
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(image)
            if faces is None:
                return []
            extracted: list[tuple[np.ndarray, np.ndarray]] = []
            for face in faces:
                aligned = self.recognizer.alignCrop(image, face)
                feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
                norm = float(np.linalg.norm(feature))
                if norm > 0:
                    extracted.append((face.copy(), feature / norm))
            return extracted


class FaceRegistry:
    def __init__(
        self,
        storage_dir: Path,
        backend: FaceBackend,
        similarity_threshold: float = 0.363,
        track_cache_frames: int = 50,
    ) -> None:
        self.storage_dir = storage_dir
        self.backend = backend
        self.similarity_threshold = similarity_threshold
        self.track_cache_frames = max(1, track_cache_frames)
        self.index_path = storage_dir / "index.json"
        self.photo_dir = storage_dir / "photos"
        self._lock = threading.RLock()
        self._identities: list[Identity] = []
        self._frame_index = 0
        self._track_cache: dict[int, tuple[str, str | None, float, int]] = {}
        self._load()

    def list_identities(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                {
                    "id": identity.identity_id,
                    "name": identity.name,
                    "photo_url": f"/api/faces/{identity.identity_id}/photo",
                }
                for identity in self._identities
            ]

    def photo_path(self, identity_id: str) -> Path | None:
        with self._lock:
            identity = self._find(identity_id)
            if identity is None:
                return None
            path = self.photo_dir / identity.photo_filename
            return path if path.is_file() else None

    def enroll(self, name: str, photo: bytes) -> dict[str, str]:
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 64 or any(ord(char) < 32 for char in clean_name):
            raise EnrollmentError("姓名必须为 1 到 64 个有效字符")
        if not photo or len(photo) > MAX_UPLOAD_BYTES:
            raise EnrollmentError("照片不能为空且不能超过 8 MB")

        encoded = np.frombuffer(photo, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise EnrollmentError("无法读取照片，请上传 JPEG、PNG 或 WebP 图片")
        faces = self.backend.features(image)
        if len(faces) != 1:
            raise EnrollmentError("证件照中必须清晰可见且只能有一张人脸")

        with self._lock:
            if any(identity.name.casefold() == clean_name.casefold() for identity in self._identities):
                raise EnrollmentError("该姓名已经登记，请先删除原记录")

            identity_id = uuid.uuid4().hex
            photo_filename = f"{identity_id}.jpg"
            self.photo_dir.mkdir(parents=True, exist_ok=True)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise EnrollmentError("照片转换失败")
            (self.photo_dir / photo_filename).write_bytes(jpeg.tobytes())
            identity = Identity(identity_id, clean_name, photo_filename, faces[0][1])
            self._identities.append(identity)
            self._save()
            return self.list_identities()[-1]

    def delete(self, identity_id: str) -> bool:
        with self._lock:
            identity = self._find(identity_id)
            if identity is None:
                return False
            self._identities.remove(identity)
            (self.photo_dir / identity.photo_filename).unlink(missing_ok=True)
            self._track_cache = {
                track_id: cached
                for track_id, cached in self._track_cache.items()
                if cached[1] != identity_id
            }
            self._save()
            return True

    def identify_people(self, frame: np.ndarray, people: list[dict[str, object]]) -> None:
        self._frame_index += 1
        faces = self.backend.features(frame) if people and self._identities else []
        unused_faces = list(faces)

        for person in people:
            track_id = person.get("track_id")
            box = person.get("xyxy")
            best_face = _pop_largest_face_inside(unused_faces, box)
            match = self._match(best_face[1]) if best_face is not None else None

            if match is not None:
                name, identity_id, similarity = match
                if isinstance(track_id, int):
                    self._track_cache[track_id] = (
                        name,
                        identity_id,
                        similarity,
                        self._frame_index,
                    )
            elif isinstance(track_id, int):
                cached = self._track_cache.get(track_id)
                if cached is not None:
                    name, identity_id, similarity, _ = cached
                    self._track_cache[track_id] = (
                        name,
                        identity_id,
                        similarity,
                        self._frame_index,
                    )
                else:
                    name, identity_id, similarity = "stranger", None, 0.0
            else:
                name, identity_id, similarity = "stranger", None, 0.0

            person["identity"] = name
            person["identity_id"] = identity_id
            person["face_similarity"] = round(similarity, 3)

        cutoff = self._frame_index - self.track_cache_frames
        self._track_cache = {
            track_id: cached
            for track_id, cached in self._track_cache.items()
            if cached[3] >= cutoff
        }

    def _match(self, embedding: np.ndarray) -> tuple[str, str, float] | None:
        with self._lock:
            if not self._identities:
                return None
            scores = [float(np.dot(embedding, identity.embedding)) for identity in self._identities]
            best_index = int(np.argmax(scores))
            score = scores[best_index]
            if score < self.similarity_threshold:
                return None
            identity = self._identities[best_index]
            return identity.name, identity.identity_id, score

    def _find(self, identity_id: str) -> Identity | None:
        return next(
            (identity for identity in self._identities if identity.identity_id == identity_id),
            None,
        )

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            identities = []
            for item in payload.get("identities", []):
                identities.append(
                    Identity(
                        identity_id=str(item["id"]),
                        name=str(item["name"]),
                        photo_filename=str(item["photo"]),
                        embedding=np.asarray(item["embedding"], dtype=np.float32),
                    )
                )
            self._identities = identities
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            LOG.exception("Could not load face registry from %s", self.index_path)

    def _save(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "identities": [
                {
                    "id": identity.identity_id,
                    "name": identity.name,
                    "photo": identity.photo_filename,
                    "embedding": identity.embedding.tolist(),
                }
                for identity in self._identities
            ],
        }
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.index_path)


def ensure_face_models(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, urls in FACE_MODEL_URLS.items():
        path = model_dir / filename
        if not path.is_file() or path.stat().st_size < 10_000:
            LOG.info("Downloading face model %s", filename)
            temporary = path.with_suffix(".download")
            errors = []
            for url in urls:
                try:
                    with (
                        urllib.request.urlopen(url, timeout=60) as response,
                        temporary.open("wb") as model_file,
                    ):
                        while chunk := response.read(1024 * 1024):
                            model_file.write(chunk)
                    if temporary.stat().st_size < 10_000:
                        raise RuntimeError(f"Downloaded model is unexpectedly small: {filename}")
                    temporary.replace(path)
                    break
                except (OSError, RuntimeError) as exc:
                    errors.append(f"{url}: {exc}")
                    temporary.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    f"Could not download {filename}. Attempts: {'; '.join(errors)}"
                )
        paths.append(path)
    return paths[0], paths[1]


def _pop_largest_face_inside(
    faces: list[tuple[np.ndarray, np.ndarray]],
    box: object,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    candidates = []
    for index, (face, _) in enumerate(faces):
        face_x, face_y, face_width, face_height = face[:4]
        center_x = float(face_x + face_width / 2)
        center_y = float(face_y + face_height / 2)
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            candidates.append((float(face_width * face_height), index))
    if not candidates:
        return None
    _, index = max(candidates)
    return faces.pop(index)
