from __future__ import annotations

import argparse
from pathlib import Path

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

            if args.show:
                cv2.imshow("YOLO HoloLens2 Rover Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
