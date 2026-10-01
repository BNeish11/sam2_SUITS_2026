"""Single, obvious place for remote server settings.

Edit the values below by setting environment variables or by changing the
defaults for local testing. This is the clear location to put the server address
for the remote AI bridge.

Recommended env vars:
  SERVER_URL=http://YOUR_SERVER_IP:3000
  WS_URL=ws://YOUR_SERVER_IP:3001
  ENABLE_REMOTE=true
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteConfig:
    server_url: str
    ws_url: str
    enable_remote: bool
    remote_path: str
    remote_result_path: str
    timeout_seconds: float


def load_remote_config() -> RemoteConfig:
    return RemoteConfig(
        server_url=os.getenv("SERVER_URL", "http://172.29.84.254:5000"),
        ws_url=os.getenv("WS_URL", "ws://172.29.84.254:5001"),
        enable_remote=os.getenv("ENABLE_REMOTE", "false").lower() in ("1", "true", "yes"),
        remote_path=os.getenv("REMOTE_PATH", "/segment"),
        remote_result_path=os.getenv("REMOTE_RESULT_PATH", "/unreal"),
        timeout_seconds=float(os.getenv("REMOTE_TIMEOUT", "10")),
    )


def print_remote_config() -> None:
    cfg = load_remote_config()
    print("=== Remote AI Bridge Configuration ===")
    print(f"SERVER_URL         : {cfg.server_url}")
    print(f"WS_URL             : {cfg.ws_url}")
    print(f"ENABLE_REMOTE      : {cfg.enable_remote}")
    print(f"REMOTE_PATH        : {cfg.remote_path}")
    print(f"REMOTE_RESULT_PATH : {cfg.remote_result_path}")
    print(f"REMOTE_TIMEOUT     : {cfg.timeout_seconds}s")
    print("======================================")
