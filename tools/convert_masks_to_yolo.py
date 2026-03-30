from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert segmentation masks to YOLO detection or segmentation labels."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=["detection", "segmentation"],
        default="segmentation",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["indexed", "binary"],
        default="indexed",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["rover", "wall", "floor"],
    )
    parser.add_argument(
        "--mask-stem-prefix",
        type=str,
        default="",
        help="Optional prefix added to image stem when resolving mask filenames (e.g., frame_).",
    )
    parser.add_argument(
        "--indexed-class-values",
        type=int,
        nargs="+",
        default=None,
        help="Pixel values in indexed masks corresponding to classes order. Default: 1..N",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--approx-epsilon-ratio",
        type=float,
        default=0.002,
        help="Polygon simplification factor for segmentation mode.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of data for validation split. Use 0 for train-only dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_image_files(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.stem)


def ensure_splits(output_dir: Path, has_val: bool) -> dict[str, dict[str, Path]]:
    splits = {"train": {"images": output_dir / "images" / "train", "labels": output_dir / "labels" / "train"}}
    if has_val:
        splits["val"] = {"images": output_dir / "images" / "val", "labels": output_dir / "labels" / "val"}
    for split_dirs in splits.values():
        split_dirs["images"].mkdir(parents=True, exist_ok=True)
        split_dirs["labels"].mkdir(parents=True, exist_ok=True)
    return splits


def split_items(items: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if val_ratio <= 0:
        return items, []
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_items = sorted(shuffled[:val_count], key=lambda p: p.stem)
    train_items = sorted(shuffled[val_count:], key=lambda p: p.stem)
    return train_items, val_items


def load_indexed_mask(mask_path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def load_binary_mask(
    masks_dir: Path,
    stem: str,
    prefixed_stem: str,
    class_name: str,
) -> np.ndarray | None:
    candidates = [
        masks_dir / class_name / f"{stem}.png",
        masks_dir / class_name / f"{prefixed_stem}.png",
        masks_dir / f"{stem}.png",
        masks_dir / f"{prefixed_stem}.png",
        masks_dir / f"{stem}_{class_name}.png",
        masks_dir / f"{prefixed_stem}_{class_name}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
    return None


def contours_from_mask(binary_mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            kept.append(contour)
    return kept


def contour_to_detection_line(contour: np.ndarray, class_id: int, width: int, height: int) -> str:
    x, y, w, h = cv2.boundingRect(contour)
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    nw = w / width
    nh = h / height
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def contour_to_segmentation_line(
    contour: np.ndarray,
    class_id: int,
    width: int,
    height: int,
    approx_epsilon_ratio: float,
) -> str | None:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = approx_epsilon_ratio * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    points = simplified.reshape(-1, 2)
    if len(points) < 3:
        return None

    coords: list[str] = []
    for x, y in points:
        nx = min(max(float(x) / width, 0.0), 1.0)
        ny = min(max(float(y) / height, 0.0), 1.0)
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")

    return f"{class_id} " + " ".join(coords)


def build_labels_for_image(
    image_path: Path,
    args: argparse.Namespace,
    class_values: list[int],
) -> list[str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    stem = image_path.stem
    prefixed_stem = f"{args.mask_stem_prefix}{stem}"

    lines: list[str] = []

    if args.mask_mode == "indexed":
        indexed_candidates = [
            args.masks_dir / f"{stem}.png",
            args.masks_dir / f"{prefixed_stem}.png",
        ]
        mask = None
        for mask_path in indexed_candidates:
            mask = load_indexed_mask(mask_path)
            if mask is not None:
                break
        if mask is None:
            return lines

        for class_id, class_value in enumerate(class_values):
            class_mask = np.where(mask == class_value, 255, 0).astype(np.uint8)
            contours = contours_from_mask(class_mask, args.min_area)
            for contour in contours:
                if args.task == "detection":
                    lines.append(contour_to_detection_line(contour, class_id, width, height))
                else:
                    line = contour_to_segmentation_line(
                        contour, class_id, width, height, args.approx_epsilon_ratio
                    )
                    if line is not None:
                        lines.append(line)

        return lines

    for class_id, class_name in enumerate(args.classes):
        binary_mask = load_binary_mask(args.masks_dir, stem, prefixed_stem, class_name)
        if binary_mask is None:
            continue
        class_mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
        contours = contours_from_mask(class_mask, args.min_area)
        for contour in contours:
            if args.task == "detection":
                lines.append(contour_to_detection_line(contour, class_id, width, height))
            else:
                line = contour_to_segmentation_line(
                    contour, class_id, width, height, args.approx_epsilon_ratio
                )
                if line is not None:
                    lines.append(line)

    return lines


def write_yaml(output_dir: Path, classes: list[str], has_val: bool) -> None:
    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val" if has_val else "val: images/train",
        f"names: {classes}",
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def process_split(
    split_name: str,
    image_paths: list[Path],
    split_dirs: dict[str, Path],
    args: argparse.Namespace,
    class_values: list[int],
) -> tuple[int, int]:
    written_images = 0
    written_labels = 0

    for image_path in image_paths:
        target_image = split_dirs["images"] / image_path.name
        shutil.copy2(image_path, target_image)
        written_images += 1

        label_lines = build_labels_for_image(image_path, args, class_values)
        target_label = split_dirs["labels"] / f"{image_path.stem}.txt"
        target_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        written_labels += 1

    print(f"[{split_name}] images={written_images}, labels={written_labels}")
    return written_images, written_labels


def main() -> None:
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {args.images_dir}")
    if not args.masks_dir.exists():
        raise FileNotFoundError(f"masks-dir not found: {args.masks_dir}")

    image_files = get_image_files(args.images_dir)
    if not image_files:
        raise RuntimeError(f"No image files found in: {args.images_dir}")

    if args.indexed_class_values is None:
        class_values = list(range(1, len(args.classes) + 1))
    else:
        class_values = args.indexed_class_values

    if len(class_values) != len(args.classes):
        raise ValueError("indexed-class-values length must match number of classes")

    train_items, val_items = split_items(image_files, args.val_ratio, args.seed)
    has_val = len(val_items) > 0
    splits = ensure_splits(args.output_dir, has_val)

    print(f"Classes: {args.classes}")
    print(f"Mask mode: {args.mask_mode}")
    print(f"Task: {args.task}")
    print(f"Min area: {args.min_area}")
    print(f"Total images: {len(image_files)}")

    process_split("train", train_items, splits["train"], args, class_values)
    if has_val:
        process_split("val", val_items, splits["val"], args, class_values)

    write_yaml(args.output_dir, args.classes, has_val)
    print(f"Done. YOLO dataset written to: {args.output_dir}")
    print(f"YAML: {args.output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
