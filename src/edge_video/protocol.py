"""Binary frame protocol shared by the device and edge server."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

HEADER = struct.Struct("!I")
MAX_METADATA_BYTES = 16 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class ProtocolError(ValueError):
    """Raised when a received frame packet is malformed."""


@dataclass(frozen=True, slots=True)
class FramePacket:
    metadata: dict[str, Any]
    jpeg: bytes


def pack_frame(metadata: dict[str, Any], jpeg: bytes) -> bytes:
    if not jpeg:
        raise ProtocolError("JPEG payload is empty")
    if len(jpeg) > MAX_IMAGE_BYTES:
        raise ProtocolError("JPEG payload exceeds the size limit")

    encoded_metadata = json.dumps(
        metadata, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded_metadata) > MAX_METADATA_BYTES:
        raise ProtocolError("Metadata exceeds the size limit")
    return HEADER.pack(len(encoded_metadata)) + encoded_metadata + jpeg


def unpack_frame(payload: bytes) -> FramePacket:
    if len(payload) < HEADER.size:
        raise ProtocolError("Frame packet is shorter than its header")

    (metadata_size,) = HEADER.unpack_from(payload)
    if metadata_size > MAX_METADATA_BYTES:
        raise ProtocolError("Metadata exceeds the size limit")

    image_offset = HEADER.size + metadata_size
    if image_offset > len(payload):
        raise ProtocolError("Frame packet contains truncated metadata")

    jpeg = payload[image_offset:]
    if not jpeg:
        raise ProtocolError("JPEG payload is empty")
    if len(jpeg) > MAX_IMAGE_BYTES:
        raise ProtocolError("JPEG payload exceeds the size limit")

    try:
        metadata = json.loads(payload[HEADER.size:image_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Metadata is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict):
        raise ProtocolError("Metadata must be a JSON object")

    return FramePacket(metadata=metadata, jpeg=jpeg)

