from urllib.parse import parse_qs, urlsplit

from edge_video.device import add_token, estimate_clock_offset


def test_token_is_added_to_existing_query_string() -> None:
    result = add_token("ws://localhost/ws?mode=test", "a token/with symbols")
    query = parse_qs(urlsplit(result).query)

    assert query == {"mode": ["test"], "token": ["a token/with symbols"]}


def test_empty_token_does_not_change_url() -> None:
    url = "ws://localhost/ws"
    assert add_token(url, "") == url


def test_clock_offset_uses_ntp_estimate() -> None:
    offset_ns, rtt_ms = estimate_clock_offset(
        client_send_ns=1_000_000_000,
        server_receive_ns=1_600_000_000,
        server_send_ns=1_620_000_000,
        client_receive_ns=1_220_000_000,
    )

    assert offset_ns == 500_000_000
    assert rtt_ms == 200.0
