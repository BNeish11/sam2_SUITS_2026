### Unreal Engine Bridge (Server Mode)

This project now operates as a **client** that polls an external Unreal server, processes frames with SAM2, and submits results back.

#### Quick Start

1. **Configure `.env`** (in repo root):
   ```dotenv
   SERVER_URL=http://172.29.84.254:5000
   POLL_INTERVAL=1.0
   REMOTE_TIMEOUT=10
   SAM2_TIMEOUT=30
   ```

2. **Start the bridge** (from repo root, with venv active):
   ```bash
   python run_unreal_bridge.py
   ```

   Expected output:
   ```
   ================================================================================
   Unreal Engine Bridge (Client Mode)
   ================================================================================
   SERVER_URL=http://172.29.84.254:5000
   FETCH=http://172.29.84.254:5000/data
   SUBMIT=http://172.29.84.254:5000/receive
   POLL_INTERVAL=1.0s
   ...
   [STARTUP] Polling for frames from external server
   ```

#### How It Works

1. **Poll for frames:** Sends `GET http://172.29.84.254:5000/data`
   - Response (if frame available):
     ```json
     {
       "frame_id": "frame_123",
       "image": "<base64-encoded-jpeg>"
     }
     ```
   - Response (if no frame): HTTP 204 (No Content)

2. **Process with SAM2:** Decodes image, runs segmentation

3. **Submit results:** Sends `POST http://172.29.84.254:5000/receive`
   - Payload:
     ```json
     {
       "frame_id": "frame_123",
       "segmentation": {
         "masks": 3,
         "boxes": [[x1, y1, x2, y2], ...],
         "labels": [0.95, 0.87, ...],
         "time_ms": 1234.5
       }
     }
     ```

#### Terminal Output

Each frame logs clearly:
```
[FETCH] Polling external server
[FRAME RECEIVED] id=frame_123, size=152403 bytes
[PROCESSING] SAM2 segmentation started for frame frame_123
[RESULTS] masks=3, boxes=3, time=1234.5ms
[SUBMIT] Results sent for frame frame_123, status=200
```

#### Configuration Details

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVER_URL` | — | Base URL of external server |
| `POLL_INTERVAL` | 1.0 | Seconds to wait between polls |
| `SAM2_TIMEOUT` | 30 | Max seconds for SAM2 processing |
| `REMOTE_TIMEOUT` | 10 | Max seconds for HTTP requests |

#### Testing

Run unit tests:
```bash
python -m unittest tests.test_unreal_bridge
```

Or manually test by checking logs (make sure external server is running and sending frames):
```bash
python run_unreal_bridge.py
```

#### Troubleshooting

- **"Cannot import sam2":** SAM2 models not installed. Ensure checkpoints are in `checkpoints/`.
- **"Failed to get frame":** External server not responding. Check `SERVER_URL` is correct and server is running.
- **Frames not processing:** Check `POLL_INTERVAL` isn't too long, and SAM2 model loads successfully.
