#!/usr/bin/env python3
"""Unreal Engine Bridge: Accept frames via HTTP, run SAM2 segmentation, return results.

Usage:
    python run_unreal_bridge.py

Flow:
    1. GET http://172.29.84.254:5000/data → fetch frame
    2. Process with SAM2
    3. POST http://172.29.84.254:5000/receive → submit results
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import cv2

try:
    import requests
except ImportError:
    print("[ERROR] requests library required. Install with: pip install requests")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOG = logging.getLogger(__name__)

# Load environment
def load_env_file(path: Path) -> Dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env

repo_root = Path(__file__).resolve().parent
env_file = repo_root / ".env"
ENV = load_env_file(env_file)

# Config
SERVER_URL = ENV.get("SERVER_URL", "").strip()
POLL_INTERVAL = float(ENV.get("POLL_INTERVAL", 1.0))
SAM2_TIMEOUT = float(ENV.get("SAM2_TIMEOUT", 30))
REMOTE_TIMEOUT = float(ENV.get("REMOTE_TIMEOUT", 10))

if not SERVER_URL:
    print("[ERROR] SERVER_URL not set in .env")
    sys.exit(1)

FETCH_ENDPOINT = SERVER_URL.rstrip("/") + "/data"
SUBMIT_ENDPOINT = SERVER_URL.rstrip("/") + "/receive"

# Lazy-load SAM2 predictor (initialize once on first frame)
_sam2_predictor = None


def get_sam2_predictor():
    global _sam2_predictor
    if _sam2_predictor is None:
        try:
            from sam2.build_sam import build_sam2_image_predictor
            LOG.info("[SAM2] Loading model...")
            _sam2_predictor = build_sam2_image_predictor(
                model_type="hiera_base_plus",
                checkpoint=(repo_root / "checkpoints" / "sam2.1_hiera_base_plus.pt").as_posix()
            )
            LOG.info("[SAM2] Model loaded.")
        except Exception as e:
            LOG.error(f"[SAM2] Failed to load: {e}")
            raise
    return _sam2_predictor
def fetch_frame() -> Optional[Dict[str, Any]]:
    """Poll external server for next frame."""
    try:
        resp = requests.get(FETCH_ENDPOINT, timeout=REMOTE_TIMEOUT)
        if resp.status_code == 204:  # No content / no frame available
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        LOG.warning(f"[FETCH] Failed to get frame: {e}")
        return None




def process_frame_sam2(image_bytes: bytes) -> Dict[str, Any]:
    """Run SAM2 automatic mask generation on image.
    
    Returns: {"masks": [...], "boxes": [...], "labels": [...], "time_ms": float}
    """
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB for SAM2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictor = get_sam2_predictor()
        start = time.time()
        
        # Use automatic mask generator
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            mask_generator = SAM2AutomaticMaskGenerator(predictor)
            masks = mask_generator.generate(image_rgb)
            elapsed = time.time() - start
            
            # Format results
            results = {
                "masks": len(masks),
                "boxes": [m.get("bbox", []) for m in masks],
                "labels": [m.get("predicted_iou", 0.0) for m in masks],
                "time_ms": elapsed * 1000,
            }
        except Exception:
            # Fallback
            elapsed = time.time() - start
            results = {
                "boxes": [],
                "labels": [],
                "time_ms": elapsed * 1000,
            }
        
        return results
    except Exception as e:
        LOG.error(f"[SAM2] Processing failed: {e}")
        raise


def submit_results(frame_id: str, results: Dict[str, Any]) -> bool:
    """Submit segmentation results to external server."""
    try:
        payload = {
            "frame_id": frame_id,
            "segmentation": results,
        }
        resp = requests.post(
            SUBMIT_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REMOTE_TIMEOUT
        )
        resp.raise_for_status()
        LOG.info(f"[SUBMIT] Results sent for frame {frame_id}, status={resp.status_code}")
        return True
    except Exception as e:
        LOG.warning(f"[SUBMIT] Failed to submit results: {e}")
        return False


def main():
    LOG.info("=" * 80)
    LOG.info("Unreal Engine Bridge (Client Mode)")
    LOG.info("=" * 80)
    LOG.info(f"SERVER_URL={SERVER_URL}")
    LOG.info(f"FETCH={FETCH_ENDPOINT}")
    LOG.info(f"SUBMIT={SUBMIT_ENDPOINT}")
    LOG.info(f"POLL_INTERVAL={POLL_INTERVAL}s")
    LOG.info(f"SAM2_TIMEOUT={SAM2_TIMEOUT}s")
    LOG.info(f"REMOTE_TIMEOUT={REMOTE_TIMEOUT}s")
    LOG.info("=" * 80)
    
    LOG.info("[STARTUP] Polling for frames from external server")
    LOG.info("[STARTUP] Press Ctrl+C to stop")
    LOG.info("=" * 80)
    
    try:
        while True:
            # Poll for next frame
            frame_data = fetch_frame()
            
            if not frame_data:
                time.sleep(POLL_INTERVAL)
                continue
            
            frame_id = frame_data.get("frame_id", "unknown")
            image_base64 = frame_data.get("image", "")
            
            if not image_base64:
                LOG.warning(f"[FRAME {frame_id}] No image data in response")
                time.sleep(POLL_INTERVAL)
                continue
            
            # Decode base64
            try:
                import base64
                image_bytes = base64.b64decode(image_base64)
            except Exception as e:
                LOG.error(f"[FRAME {frame_id}] Failed to decode base64: {e}")
                time.sleep(POLL_INTERVAL)
                continue
            
            LOG.info(f"[FRAME RECEIVED] id={frame_id}, size={len(image_bytes)} bytes")
            
            # Process
            try:
                LOG.info(f"[PROCESSING] SAM2 segmentation started for frame {frame_id}")
                results = process_frame_sam2(image_bytes)
                LOG.info(f"[RESULTS] masks={results['masks']}, boxes={len(results['boxes'])}, time={results['time_ms']:.1f}ms")
                
                # Submit
                submit_results(frame_id, results)
            except Exception as e:
                LOG.error(f"[ERROR] Failed to process frame {frame_id}: {e}")
            
            time.sleep(POLL_INTERVAL)
    
    except KeyboardInterrupt:
        LOG.info("[SHUTDOWN] Bridge stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
