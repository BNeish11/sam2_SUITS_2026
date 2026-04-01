from __future__ import annotations

import argparse
import io
import socket
import threading
import time
from pathlib import Path
from queue import Queue

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO rover detection on a HoloLens2 camera stream.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained YOLO weights (best.pt).")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Camera source: rtsp/http URL, video file path, or webcam index like 0.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--rover-class-id",
        type=int,
        default=None,
        help="Class id for rover. If omitted, class name 'rover' will be used.",
    )
    parser.add_argument("--show", action="store_true", help="Display annotated live window.")
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optional output MP4 path for annotated stream.",
    )
    parser.add_argument(
        "--stream-http",
        type=str,
        default=None,
        help="Enable HTTP MJPEG streaming on host:port (e.g., 0.0.0.0:8080). Open http://<server_ip>:8080/stream in browser.",
    )
    return parser.parse_args()


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def get_rover_class_id(model: YOLO, explicit_id: int | None) -> int:
    if explicit_id is not None:
        return explicit_id

    for class_id, class_name in model.names.items():
        if str(class_name).lower() == "rover":
            return int(class_id)

    raise ValueError(
        "Could not find class name 'rover' in model names. Pass --rover-class-id explicitly."
    )


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def http_mjpeg_server(
    frame_queue: Queue,
    host: str,
    port: int,
    stop_event: threading.Event,
) -> None:
    """Simple HTTP MJPEG streamer."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
        sock.settimeout(1.0)
        print(f"[HTTP] MJPEG stream server listening on http://{host}:{port}/stream")

        while not stop_event.is_set():
            try:
                client, addr = sock.accept()
                print(f"[HTTP] Client connected from {addr}")
                threading.Thread(
                    target=handle_mjpeg_client,
                    args=(client, frame_queue, stop_event),
                    daemon=True,
                ).start()
            except socket.timeout:
                pass
    except Exception as e:
        print(f"[HTTP] Server error: {e}")
    finally:
        sock.close()


def handle_mjpeg_client(
    client: socket.socket,
    frame_queue: Queue,
    stop_event: threading.Event,
) -> None:
    """Handle a single MJPEG client."""
    try:
        boundary = "frame"
        request = client.recv(4096).decode()
        if "GET" not in request:
            client.close()
            return

        http_header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n"
            "Connection: keep-alive\r\n"
            "Cache-Control: no-cache\r\n"
            "\r\n"
        )
        client.send(http_header.encode())

        last_frame = None
        while not stop_event.is_set():
            try:
                with frame_queue.mutex:
                    if not frame_queue.empty():
                        last_frame = frame_queue.get_nowait()

                if last_frame is not None:
                    _, buffer = cv2.imencode(".jpg", last_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    frame_data = (
                        f"--{boundary}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame_bytes)}\r\n\r\n"
                    ).encode() + frame_bytes + b"\r\n"
                    client.send(frame_data)
            except Exception:
                break

            if stop_event.is_set():
                break
            time.sleep(0.033)  # ~30 fps
    except Exception as e:
        print(f"[HTTP] Client error: {e}")
    finally:
        try:
            client.close()
        except Exception:
            pass



def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(str(args.model))
    rover_class_id = get_rover_class_id(model, args.rover_class_id)
    print(f"Loaded model: {args.model}")
    print(f"Rover class id: {rover_class_id}")
    print(f"Class names: {model.names}")

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.save_video is not None:
        writer = make_writer(args.save_video, fps, width, height)
        print(f"Saving annotated stream to: {args.save_video}")

    frame_queue = Queue(maxsize=2)
    stop_event = threading.Event()
    http_thread = None

    if args.stream_http:
        host, port = args.stream_http.split(":")
        port = int(port)
        http_thread = threading.Thread(
            target=http_mjpeg_server,
            args=(frame_queue, host, port, stop_event),
            daemon=True,
        )
        http_thread.start()

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            rover_count = 0
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    if class_id == rover_class_id:
                        rover_count += 1

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = f"{model.names[class_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), max(int(y1) - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            rover_detected = rover_count > 0
            status_text = f"ROVER DETECTED: {rover_detected} (count={rover_count})"
            cv2.putText(
                frame,
                status_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if rover_detected else (0, 0, 255),
                2,
            )

            if frame_idx % 30 == 0:
                print(f"frame={frame_idx} | {status_text}")

            if writer is not None:
                writer.write(frame)

            if args.stream_http:
                try:
                    frame_queue.put_nowait(frame.copy())
                except Exception:
                    pass

            if args.show:
                cv2.imshow("YOLO HoloLens2 Rover Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        stop_event.set()
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        if http_thread is not None:
            http_thread.join(timeout=2.0)



if __name__ == "__main__":
    main()
