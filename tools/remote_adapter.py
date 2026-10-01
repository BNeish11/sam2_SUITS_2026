"""Minimal remote adapter for sending/receiving image frames to external server.

Non-invasive: feature-flagged by ENABLE_REMOTE env var.

Usage:
    from tools.remote_adapter import is_enabled, send_image_bytes_rest
    if is_enabled():
            resp = send_image_bytes_rest(image_bytes)
            # resp is dict parsed from JSON returned by remote

This module prefers `requests` for REST multipart upload. For WebSocket there is
an optional placeholder `send_image_ws` which will attempt to import `websocket`.
"""
from __future__ import annotations

import os
import logging
import base64
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from tools.remote_config import load_remote_config

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:
    requests = None  # type: ignore

_LOG = logging.getLogger(__name__)


def _cfg() -> Dict[str, Any]:
    cfg = load_remote_config()
    return {
        "SERVER_URL": cfg.server_url,
        "WS_URL": cfg.ws_url,
        "ENABLE_REMOTE": str(cfg.enable_remote).lower(),
        "REMOTE_PATH": cfg.remote_path,
        "REMOTE_RESULT_PATH": cfg.remote_result_path,
        "REMOTE_TIMEOUT": cfg.timeout_seconds,
    }


def is_enabled() -> bool:
    return _cfg().get("ENABLE_REMOTE", "false").lower() in ("1", "true", "yes")


def _validate_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https", "ws", "wss") and bool(p.netloc)
    except Exception:
        return False


def _session_with_retries(total: int = 3, backoff: float = 0.5, status_forcelist=(500, 502, 504)):
    if requests is None:
        raise RuntimeError("requests library is required for remote adapter")
    s = requests.Session()
    retries = Retry(total=total, backoff_factor=backoff, status_forcelist=status_forcelist)
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def send_image_bytes_rest(
    image_bytes: bytes,
    filename: str = "frame.jpg",
    server_url: Optional[str] = None,
    path: Optional[str] = None,
    timeout: Optional[float] = None,
    use_base64_fallback: bool = True,
) -> Dict[str, Any]:
    """Send image bytes to remote server using multipart/form-data POST.

    Returns parsed JSON response from server.
    """
    cfg = _cfg()
    server_url = server_url or cfg.get("SERVER_URL")
    path = path or cfg.get("REMOTE_PATH", "/segment")
    timeout = float(timeout or cfg.get("REMOTE_TIMEOUT", 10))

    if not server_url:
        raise ValueError("SERVER_URL not configured")
    if not _validate_url(server_url):
        raise ValueError(f"Invalid SERVER_URL: {server_url}")

    url = server_url.rstrip("/") + (path if path.startswith("/") else f"/{path}")

    sess = _session_with_retries()

    # try multipart/form-data
    try:
        files = {"image": (filename, image_bytes, "image/jpeg")}
        headers = {"Accept": "application/json"}
        r = sess.post(url, files=files, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _LOG.debug("multipart upload failed: %s", e)
        if not use_base64_fallback:
            raise

    # fallback: base64 JSON
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {"image_base64": b64}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        r = sess.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _LOG.exception("base64 JSON fallback failed: %s", e)
        raise


def send_image_ws(image_bytes: bytes, server_ws: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Attempt WebSocket send. Requires `websocket-client` package.

    Sends binary frame if server supports; otherwise can send base64 JSON.
    Returns parsed JSON response or raises informative error.
    """
    cfg = _cfg()
    server_ws = server_ws or cfg.get("WS_URL")
    timeout = float(timeout or cfg.get("REMOTE_TIMEOUT", 10))

    if not server_ws:
        raise ValueError("WS_URL not configured")
    if not _validate_url(server_ws):
        raise ValueError(f"Invalid WS_URL: {server_ws}")

    try:
        import websocket
    except Exception:
        raise RuntimeError("websocket-client package required for WS support")

    # simple blocking connect/send/recv
    ws = websocket.create_connection(server_ws, timeout=timeout)
    try:
        ws.send_binary(image_bytes)
        resp = ws.recv()
        # try parse JSON
        try:
            import json

            return json.loads(resp)
        except Exception:
            return {"response": resp}
    finally:
        ws.close()
