from urllib.parse import parse_qs, urlsplit

from edge_video.device import add_token


def test_token_is_added_to_existing_query_string() -> None:
    result = add_token("ws://localhost/ws?mode=test", "a token/with symbols")
    query = parse_qs(urlsplit(result).query)

    assert query == {"mode": ["test"], "token": ["a token/with symbols"]}


def test_empty_token_does_not_change_url() -> None:
    url = "ws://localhost/ws"
    assert add_token(url, "") == url
