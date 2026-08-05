"""Device B: receive frames, run AI inference, and serve a live dashboard."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from edge_video.detection import Detector, build_detector
from edge_video.protocol import ProtocolError, unpack_frame
from edge_video.state import RuntimeState

LOG = logging.getLogger("edge_video.edge")
WEB_DIR = Path(__file__).with_name("web")


async def inference_worker(state: RuntimeState, detector: Detector) -> None:
    while True:
        queued = await state.queue.get()
        try:
            encoded = np.frombuffer(queued.packet.jpeg, dtype=np.uint8)
            frame = await asyncio.to_thread(cv2.imdecode, encoded, cv2.IMREAD_COLOR)
            if frame is None:
                LOG.warning("Discarded a JPEG frame that OpenCV could not decode")
                continue

            result = await asyncio.to_thread(detector.detect, frame)
            ok, output_jpeg = await asyncio.to_thread(
                cv2.imencode,
                ".jpg",
                result.annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 82],
            )
            if not ok:
                LOG.warning("Failed to encode an annotated frame")
                continue

            now_ns = time.time_ns()
            sent_at_ns = queued.packet.metadata.get("sent_at_ns")
            network_ms = None
            end_to_end_ms = None
            if isinstance(sent_at_ns, int) and sent_at_ns <= now_ns:
                network_ms = round((queued.received_at_ns - sent_at_ns) / 1_000_000, 1)
                end_to_end_ms = round((now_ns - sent_at_ns) / 1_000_000, 1)

            stats = {
                "detector": detector.name,
                "frame_id": queued.packet.metadata.get("frame_id"),
                "frame_width": queued.packet.metadata.get("width"),
                "frame_height": queued.packet.metadata.get("height"),
                "jpeg_kb": round(len(queued.packet.jpeg) / 1024, 1),
                "endpoint_processing_ms": queued.packet.metadata.get("endpoint_processing_ms"),
                "network_ms": network_ms,
                "inference_ms": round(result.inference_ms, 1),
                "end_to_end_ms": end_to_end_ms,
                "object_count": len(result.objects),
                "objects": result.objects,
            }
            await state.publish(output_jpeg.tobytes(), stats)
        except Exception:
            LOG.exception("Inference worker failed to process a frame")
        finally:
            state.queue.task_done()


def create_app(detector: Detector, token: str = "") -> FastAPI:
    state = RuntimeState()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker = asyncio.create_task(inference_worker(state, detector))
        try:
            yield
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    app = FastAPI(title="Edge Video System", lifespan=lifespan)
    app.state.runtime = state

    @app.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True, "detector": detector.name})

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.get("/stream.mjpg")
    async def stream() -> StreamingResponse:
        async def frames():
            sequence = 0
            while True:
                sequence, jpeg = await state.wait_for_frame(sequence)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/ws/ingest/{device_id}")
    async def ingest(websocket: WebSocket, device_id: str) -> None:
        supplied_token = websocket.query_params.get("token", "")
        if token and not hmac.compare_digest(supplied_token, token):
            await websocket.close(code=1008, reason="Invalid stream token")
            return

        await websocket.accept()
        state.connect(device_id)
        LOG.info("Device connected: %s", device_id)
        try:
            while True:
                payload = await websocket.receive_bytes()
                try:
                    packet = unpack_frame(payload)
                except ProtocolError as exc:
                    LOG.warning("Rejected malformed frame from %s: %s", device_id, exc)
                    continue
                state.submit(packet, time.time_ns())
        except WebSocketDisconnect:
            LOG.info("Device disconnected: %s", device_id)
        finally:
            state.disconnect(device_id)

    return app


def parse_classes(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run edge inference on device B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--detector", choices=("mock", "yolo"), default="yolo")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--classes",
        default="0",
        help="Comma-separated COCO class IDs; default 0 detects people, empty detects all",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device such as cpu, 0, or cuda:0")
    parser.add_argument(
        "--token",
        default=os.getenv("EDGE_STREAM_TOKEN", ""),
        help="Shared token (defaults to EDGE_STREAM_TOKEN)",
    )
    return parser


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    detector = build_detector(
        args.detector,
        args.model,
        args.confidence,
        parse_classes(args.classes),
        args.device,
    )
    uvicorn.run(create_app(detector, args.token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

