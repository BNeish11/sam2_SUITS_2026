from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO rover dataset from segmented images/masks and train a rover detector."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Repo root path.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("runs/rover_retrain_dataset"),
        help="Output YOLO dataset folder.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("yolo26n.pt"),
        help="Base YOLO weights for fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/yolo_runs"),
        help="Ultralytics project output directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rover_seg_retrain",
        help="Ultralytics run name.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "train_images": base / "images" / "train",
        "val_images": base / "images" / "val",
        "train_labels": base / "labels" / "train",
        "val_labels": base / "labels" / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def mask_to_bbox(mask: np.ndarray, min_area: float = 50.0) -> tuple[int, int, int, int] | None:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    points = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            points.append(contour)
    if not points:
        return None

    all_points = np.vstack(points)
    x, y, w, h = cv2.boundingRect(all_points)
    if w <= 1 or h <= 1:
        return None
    return x, y, w, h


def yolo_line_from_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def overlay_to_mask_rgb(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask = (r > 120) & (r > g + 20) & (r > b + 20)
    return (mask.astype(np.uint8) * 255)


def find_image_by_stem(folder: Path, stem: str) -> Path | None:
    desired = stem.lower()
    for candidate in folder.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            if candidate.stem.lower() == desired:
                return candidate
    return None


def gather_samples(workspace: Path) -> list[tuple[Path, np.ndarray, str]]:
    samples: list[tuple[Path, np.ndarray, str]] = []
    sam2_count = 0
    overlay_count = 0

    rover_pictures = workspace / "rover pictures"
    sam2_mask_dirs = [
        workspace / "runs" / "rover_batch_customyolo_conf0001" / "sam2_masks",
        workspace / "runs" / "rover_batch_customyolo_conf001" / "sam2_masks",
        workspace / "runs" / "rover_batch_conf005" / "sam2_masks",
        workspace / "runs" / "rover_batch" / "sam2_masks",
    ]
    if rover_pictures.exists():
        for sam2_masks_dir in sam2_mask_dirs:
            if not sam2_masks_dir.exists():
                continue
            for mask_path in sorted(sam2_masks_dir.glob("*_mask.png")):
                stem = mask_path.stem.replace("_mask", "")
                image_path = find_image_by_stem(rover_pictures, stem)
                if image_path is None:
                    continue
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                samples.append((image_path, mask, f"sam2mask:{stem}"))
                sam2_count += 1
            if sam2_count > 0:
                break

    rover_training = workspace / "training" / "rover images"
    if rover_training.exists():
        for overlay_path in sorted(rover_training.glob("*_sam2_overlay.jpg")):
            overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
            if overlay is None:
                continue

            base_stem = overlay_path.stem.replace("_sam2_overlay", "")
            mask = overlay_to_mask_rgb(overlay)
            samples.append((overlay_path, mask, f"overlay:{base_stem}"))
            overlay_count += 1

    print(f"Gathered segmented samples: sam2_masks={sam2_count}, overlays={overlay_count}, total={len(samples)}")

    return samples


def build_dataset(workspace: Path, out_dir: Path, val_ratio: float, seed: int) -> tuple[Path, int, int]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    dirs = ensure_dirs(out_dir)

    samples = gather_samples(workspace)
    if not samples:
        raise RuntimeError("No segmented samples found to build training dataset.")

    valid_items: list[tuple[Path, np.ndarray, str]] = []
    for image_path, mask, source in samples:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        valid_items.append((image_path, mask, source))

    if len(valid_items) < 2:
        raise RuntimeError("Not enough valid segmented samples to train.")

    random.Random(seed).shuffle(valid_items)
    val_count = max(1, int(len(valid_items) * val_ratio)) if val_ratio > 0 else 0
    val_set = set(id(item) for item in valid_items[:val_count])

    train_written = 0
    val_written = 0

    for index, item in enumerate(valid_items):
        image_path, mask, source = item
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        x, y, bw, bh = bbox

        split = "val" if id(item) in val_set else "train"
        name = f"{image_path.stem}_{index:04d}"
        out_image = dirs[f"{split}_images"] / f"{name}.jpg"
        out_label = dirs[f"{split}_labels"] / f"{name}.txt"

        shutil.copy2(image_path, out_image)
        out_label.write_text(yolo_line_from_bbox(x, y, bw, bh, w, h) + "\n", encoding="utf-8")

        if split == "train":
            train_written += 1
        else:
            val_written += 1

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val" if val_written > 0 else "val: images/train",
                "names: ['rover']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return yaml_path, train_written, val_written


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dataset = (workspace / args.output_dataset).resolve()

    yaml_path, train_count, val_count = build_dataset(
        workspace=workspace,
        out_dir=output_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Dataset built: {output_dataset}")
    print(f"Train samples: {train_count} | Val samples: {val_count}")
    print(f"YAML: {yaml_path}")

    base_model = (workspace / args.base_model).resolve()
    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")

    model = YOLO(str(base_model))
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str((workspace / args.project).resolve()),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()