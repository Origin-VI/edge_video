"""Stateful region counting built on detector-provided track IDs."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrackState:
    inside: bool
    last_seen_frame: int


class RegionCounter:
    def __init__(
        self,
        roi: tuple[float, float, float, float],
        missing_grace_frames: int = 10,
        event_history_size: int = 50,
    ) -> None:
        self.roi = roi
        self.missing_grace_frames = max(0, missing_grace_frames)
        self.frame_index = 0
        self.total_entered = 0
        self.total_exited = 0
        self._tracks: dict[int, TrackState] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=event_history_size)

    def update(
        self,
        objects: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        timestamp: float | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.frame_index += 1
        event_time = time.time() if timestamp is None else timestamp
        seen_ids: set[int] = set()
        current_inside = 0
        emitted: list[dict[str, Any]] = []

        for detected in objects:
            track_id = detected.get("track_id")
            coordinates = detected.get("xyxy")
            if not isinstance(track_id, int) or not _valid_box(coordinates):
                continue

            inside = self.contains_box_center(coordinates, frame_width, frame_height)
            detected["inside_roi"] = inside
            seen_ids.add(track_id)
            if inside:
                current_inside += 1

            previous = self._tracks.get(track_id)
            if previous is None:
                if inside:
                    emitted.append(self._record_event("entered", track_id, event_time))
            elif inside != previous.inside:
                event_type = "entered" if inside else "exited"
                emitted.append(self._record_event(event_type, track_id, event_time))

            self._tracks[track_id] = TrackState(
                inside=inside,
                last_seen_frame=self.frame_index,
            )

        for track_id, state in list(self._tracks.items()):
            if track_id in seen_ids:
                continue
            missing_frames = self.frame_index - state.last_seen_frame
            if missing_frames <= self.missing_grace_frames:
                continue
            if state.inside:
                emitted.append(self._record_event("exited", track_id, event_time))
            del self._tracks[track_id]

        summary = {
            "tracking_enabled": True,
            "current_inside": current_inside,
            "active_tracks": len(seen_ids),
            "total_entered": self.total_entered,
            "total_exited": self.total_exited,
            "roi": list(self.roi),
            "recent_events": list(self._events),
        }
        return summary, emitted

    def contains_box_center(
        self,
        coordinates: list[float],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        x1, y1, x2, y2 = coordinates
        center_x = ((x1 + x2) / 2) / max(1, frame_width)
        center_y = ((y1 + y2) / 2) / max(1, frame_height)
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        return roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2

    def pixel_roi(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.roi
        return (
            round(x1 * (frame_width - 1)),
            round(y1 * (frame_height - 1)),
            round(x2 * (frame_width - 1)),
            round(y2 * (frame_height - 1)),
        )

    def _record_event(self, event_type: str, track_id: int, timestamp: float) -> dict[str, Any]:
        if event_type == "entered":
            self.total_entered += 1
        else:
            self.total_exited += 1
        event = {
            "type": event_type,
            "track_id": track_id,
            "timestamp": timestamp,
        }
        self._events.appendleft(event)
        return event


def _valid_box(value: object) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(
        isinstance(coordinate, (int, float)) for coordinate in value
    )

