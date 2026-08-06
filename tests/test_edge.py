import argparse

import pytest

from edge_video.edge import parse_roi


def test_parse_roi_accepts_normalized_rectangle() -> None:
    assert parse_roi("0.1,0.2,0.9,0.8") == (0.1, 0.2, 0.9, 0.8)


@pytest.mark.parametrize("value", ["0,0,1", "0.8,0,0.2,1", "-0.1,0,1,1"])
def test_parse_roi_rejects_invalid_rectangle(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_roi(value)
