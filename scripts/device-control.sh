#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${EDGE_CONFIG_FILE:-$ROOT_DIR/.env}"
PID_FILE="$ROOT_DIR/artifacts/device.pid"
LOG_FILE="$ROOT_DIR/artifacts/device.log"
PYTHON="$ROOT_DIR/.venv/bin/python"

load_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Missing configuration: $CONFIG_FILE" >&2
        echo "Create it from .env.example and set EDGE_SERVER_URL and EDGE_STREAM_TOKEN." >&2
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    set +a

    : "${EDGE_SERVER_URL:?EDGE_SERVER_URL is required}"
    : "${EDGE_DEVICE_ID:=rpi5}"
    : "${EDGE_SOURCE:=0}"
    : "${EDGE_CAMERA_CODEC:=YUYV}"
    : "${EDGE_CAPTURE_WIDTH:=960}"
    : "${EDGE_CAPTURE_HEIGHT:=544}"
    : "${EDGE_MAX_WIDTH:=960}"
    : "${EDGE_FPS:=5}"
    : "${EDGE_JPEG_QUALITY:=75}"
}

running_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    printf '%s\n' "$pid"
}

start_device() {
    load_config
    if pid="$(running_pid)"; then
        echo "Device sender is already running (PID $pid)."
        return
    fi
    rm -f "$PID_FILE"
    mkdir -p "$ROOT_DIR/artifacts"

    nohup env EDGE_STREAM_TOKEN="${EDGE_STREAM_TOKEN:-}" \
        "$PYTHON" -m edge_video.device \
        --server "${EDGE_SERVER_URL%/}/ws/ingest/$EDGE_DEVICE_ID" \
        --source "$EDGE_SOURCE" \
        --camera-codec "$EDGE_CAMERA_CODEC" \
        --width "$EDGE_CAPTURE_WIDTH" \
        --height "$EDGE_CAPTURE_HEIGHT" \
        --max-width "$EDGE_MAX_WIDTH" \
        --fps "$EDGE_FPS" \
        --jpeg-quality "$EDGE_JPEG_QUALITY" \
        >"$LOG_FILE" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Device sender failed to start. Last log lines:" >&2
        tail -n 20 "$LOG_FILE" >&2 || true
        exit 1
    fi
    echo "Device sender started (PID $pid)."
}

stop_device() {
    if ! pid="$(running_pid)"; then
        rm -f "$PID_FILE"
        echo "Device sender is not running."
        return
    fi

    kill "$pid"
    for _ in {1..50}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo "Device sender stopped."
            return
        fi
        sleep 0.1
    done
    echo "Device sender did not stop within 5 seconds." >&2
    exit 1
}

status_device() {
    if pid="$(running_pid)"; then
        echo "Device sender is running (PID $pid)."
        tail -n 5 "$LOG_FILE" 2>/dev/null || true
    else
        echo "Device sender is not running."
        return 1
    fi
}

case "${1:-status}" in
    start)
        start_device
        ;;
    stop)
        stop_device
        ;;
    restart)
        stop_device
        start_device
        ;;
    status)
        status_device
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}" >&2
        exit 2
        ;;
esac
