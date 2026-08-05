import pytest

from edge_video.protocol import HEADER, ProtocolError, pack_frame, unpack_frame


def test_frame_packet_round_trip() -> None:
    metadata = {"frame_id": 42, "sent_at_ns": 123, "width": 640}
    packet = unpack_frame(pack_frame(metadata, b"jpeg-data"))

    assert packet.metadata == metadata
    assert packet.jpeg == b"jpeg-data"


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        HEADER.pack(5) + b"{}",
        HEADER.pack(2) + b"[]" + b"image",
        HEADER.pack(1) + b"{" + b"image",
    ],
)
def test_invalid_packets_are_rejected(payload: bytes) -> None:
    with pytest.raises(ProtocolError):
        unpack_frame(payload)


def test_empty_jpeg_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        pack_frame({}, b"")
