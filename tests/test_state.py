import asyncio

from edge_video.protocol import FramePacket
from edge_video.state import RuntimeState


def test_queue_discards_old_frame_when_full() -> None:
    state = RuntimeState()
    first = FramePacket(metadata={"frame_id": 1}, jpeg=b"first")
    second = FramePacket(metadata={"frame_id": 2}, jpeg=b"second")

    state.submit(first, 100)
    state.submit(second, 200)

    queued = state.queue.get_nowait()
    assert queued.packet == second
    assert state.received_frames == 2
    assert state.dropped_frames == 1


def test_published_frame_updates_snapshot() -> None:
    async def scenario() -> None:
        state = RuntimeState()
        await state.publish(b"jpeg", {"frame_id": 3, "inference_ms": 12.5})

        sequence, jpeg = await state.wait_for_frame(0)
        snapshot = state.snapshot()

        assert sequence == 1
        assert jpeg == b"jpeg"
        assert snapshot["frame_id"] == 3
        assert snapshot["processed_frames"] == 1

    asyncio.run(scenario())

