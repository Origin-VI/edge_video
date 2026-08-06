from edge_video.tracking import RegionCounter


def tracked_person(track_id: int, xyxy: list[float]) -> dict[str, object]:
    return {"track_id": track_id, "xyxy": xyxy, "label": "person"}


def test_counter_records_region_transitions() -> None:
    counter = RegionCounter((0.25, 0.25, 0.75, 0.75), missing_grace_frames=2)

    outside = tracked_person(7, [0, 0, 10, 10])
    summary, events = counter.update([outside], 100, 100, timestamp=1.0)
    assert summary["current_inside"] == 0
    assert events == []

    inside = tracked_person(7, [40, 40, 60, 60])
    summary, events = counter.update([inside], 100, 100, timestamp=2.0)
    assert summary["current_inside"] == 1
    assert summary["total_entered"] == 1
    assert events == [{"type": "entered", "track_id": 7, "timestamp": 2.0}]
    assert inside["inside_roi"] is True

    summary, events = counter.update([outside], 100, 100, timestamp=3.0)
    assert summary["total_exited"] == 1
    assert events == [{"type": "exited", "track_id": 7, "timestamp": 3.0}]


def test_missing_inside_track_exits_after_grace_period() -> None:
    counter = RegionCounter((0, 0, 1, 1), missing_grace_frames=1)
    person = tracked_person(2, [10, 10, 20, 20])

    counter.update([person], 100, 100, timestamp=1.0)
    summary, events = counter.update([], 100, 100, timestamp=2.0)
    assert summary["total_exited"] == 0
    assert events == []

    summary, events = counter.update([], 100, 100, timestamp=3.0)
    assert summary["total_exited"] == 1
    assert events == [{"type": "exited", "track_id": 2, "timestamp": 3.0}]


def test_pixel_roi_converts_normalized_coordinates() -> None:
    counter = RegionCounter((0.1, 0.2, 0.9, 0.8))
    assert counter.pixel_roi(101, 51) == (10, 10, 90, 40)
