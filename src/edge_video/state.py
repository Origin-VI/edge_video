"""Latest-frame queue and observable runtime state."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from edge_video.protocol import FramePacket


@dataclass(frozen=True, slots=True)
class QueuedFrame:
    packet: FramePacket
    received_at_ns: int


class RuntimeState:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[QueuedFrame] = asyncio.Queue(maxsize=1)
        self.condition = asyncio.Condition()
        self.latest_jpeg: bytes | None = None
        self.latest_stats: dict[str, Any] = {}
        self.sequence = 0
        self.connected_devices: set[str] = set()
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.started_at = time.time()
        self._ingest_times: deque[float] = deque(maxlen=60)
        self._process_times: deque[float] = deque(maxlen=60)

    def connect(self, device_id: str) -> None:
        self.connected_devices.add(device_id)

    def disconnect(self, device_id: str) -> None:
        self.connected_devices.discard(device_id)

    def submit(self, packet: FramePacket, received_at_ns: int) -> None:
        self.received_frames += 1
        self._ingest_times.append(time.monotonic())
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.dropped_frames += 1
        self.queue.put_nowait(QueuedFrame(packet, received_at_ns))

    async def publish(self, jpeg: bytes, stats: dict[str, Any]) -> None:
        self.processed_frames += 1
        self._process_times.append(time.monotonic())
        async with self.condition:
            self.latest_jpeg = jpeg
            self.latest_stats = stats
            self.sequence += 1
            self.condition.notify_all()

    async def wait_for_frame(self, after_sequence: int) -> tuple[int, bytes]:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.sequence > after_sequence and self.latest_jpeg is not None
            )
            assert self.latest_jpeg is not None
            return self.sequence, self.latest_jpeg

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": "streaming" if self.connected_devices else "waiting",
            "connected_devices": sorted(self.connected_devices),
            "uptime_s": round(time.time() - self.started_at, 1),
            "received_frames": self.received_frames,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
            "ingest_fps": _window_rate(self._ingest_times),
            "processing_fps": _window_rate(self._process_times),
            **self.latest_stats,
        }


def _window_rate(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    return round((len(timestamps) - 1) / duration, 1) if duration > 0 else 0.0

