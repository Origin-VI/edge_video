"""Device A: capture, preprocess, and transmit frames to the edge server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from edge_video.preprocessing import PreprocessConfig, preprocess_frame
from edge_video.protocol import pack_frame
from edge_video.sources import VideoSource, open_video_source

LOG = logging.getLogger("edge_video.device")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and stream video from device A")
    parser.add_argument("--server", required=True, help="WebSocket ingest URL on device B")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, 'picamera2', or 'synthetic'",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height")
    parser.add_argument("--max-width", type=int, default=960, help="Transmitted frame width limit")
    parser.add_argument("--fps", type=float, default=10.0, help="Maximum transmitted FPS")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument(
        "--enhance-contrast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CLAHE contrast enhancement on device A",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("EDGE_STREAM_TOKEN", ""),
        help="Shared token (defaults to EDGE_STREAM_TOKEN)",
    )
    return parser


def add_token(url: str, token: str) -> str:
    if not token:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["token"] = token
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


async def stream_once(
    source: VideoSource,
    server_url: str,
    config: PreprocessConfig,
    target_fps: float,
) -> None:
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0
    frame_id = 0
    async with connect(
        server_url,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
    ) as websocket:
        LOG.info("Connected to edge server: %s", urlsplit(server_url)._replace(query="").geturl())
        while True:
            loop_started = time.perf_counter()
            ok, frame = await asyncio.to_thread(source.read)
            if not ok or frame is None:
                raise RuntimeError("Video source ended or failed to return a frame")

            processed = await asyncio.to_thread(preprocess_frame, frame, config)
            metadata = {
                "frame_id": frame_id,
                "sent_at_ns": time.time_ns(),
                "width": processed.width,
                "height": processed.height,
                "endpoint_processing_ms": round(processed.processing_ms, 3),
                "jpeg_bytes": len(processed.jpeg),
            }
            await websocket.send(pack_frame(metadata, processed.jpeg))
            frame_id += 1

            elapsed = time.perf_counter() - loop_started
            if frame_interval > elapsed:
                await asyncio.sleep(frame_interval - elapsed)


async def run(args: argparse.Namespace) -> None:
    source = open_video_source(args.source, args.width, args.height)
    preprocess_config = PreprocessConfig(
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
        enhance_contrast=args.enhance_contrast,
    )
    server_url = add_token(args.server, args.token)
    reconnect_delay = 1.0
    try:
        while True:
            try:
                await stream_once(source, server_url, preprocess_config, args.fps)
            except (ConnectionClosed, OSError) as exc:
                LOG.warning("Connection lost (%s); retrying in %.0f s", exc, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 15.0)
            else:
                reconnect_delay = 1.0
    finally:
        source.release()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    LOG.info("Starting device %s with source %s", socket.gethostname(), args.source)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        LOG.info("Stopped by user")


if __name__ == "__main__":
    main()

