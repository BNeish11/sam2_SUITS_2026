"""Terminal-friendly remote AI bridge.

This script shows a clear place to set the server address via `tools/remote_config.py`
or environment variables. It can send one image frame to the external server,
print the AI response in the terminal, and optionally forward the JSON result back
to a server path intended for Unreal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from tools.remote_adapter import is_enabled, send_image_bytes_rest
from tools.remote_config import print_remote_config, load_remote_config


def _read_image_bytes(image_path: str) -> bytes:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)
    return path.read_bytes()


def _encode_first_frame_from_video(video_path: str) -> bytes:
    cap = cv2.VideoCapture(video_path)
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read first frame from: {video_path}")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Could not encode frame to JPEG")
        return bytes(buf)
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a frame to a remote AI server and print the result.")
    parser.add_argument("--image", help="Path to an image file to send")
    parser.add_argument("--video", help="Path to a video file; first frame will be sent")
    parser.add_argument("--forward-result", action="store_true", help="POST the result JSON back to the server for Unreal forwarding")
    parser.add_argument("--name", default="frame.jpg", help="Filename reported to the server")
    args = parser.parse_args()

    cfg = load_remote_config()
    print_remote_config()

    if not is_enabled():
        print("ENABLE_REMOTE is false; remote bridge is idle.")
        return 0

    if args.image:
        frame_bytes = _read_image_bytes(args.image)
        source_desc = args.image
    elif args.video:
        frame_bytes = _encode_first_frame_from_video(args.video)
        source_desc = f"first frame of {args.video}"
    else:
        raise SystemExit("Provide --image or --video")

    print(f"AI running on {source_desc} ...")
    print(f"Sending frame to {cfg.server_url.rstrip('/')}{cfg.remote_path} ...")
    result = send_image_bytes_rest(frame_bytes, filename=args.name)

    print("AI result received from server:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.forward_result:
        # This path is intentionally configurable and left to the server to relay to Unreal.
        print(f"Result ready for Unreal relay at {cfg.server_url.rstrip('/')}{cfg.remote_result_path} ...")
        try:
            import requests

            response = requests.post(
                cfg.server_url.rstrip("/") + cfg.remote_result_path,
                json={"result": result},
                timeout=cfg.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            print("Result forwarded successfully.")
        except Exception as exc:
            print(f"Forwarding result failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
