from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 first, then YOLO identification from SAM2-derived ROI/masked crop."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--yolo-model", type=Path, required=True)
    parser.add_argument(
        "--sam2-run-dir",
        type=Path,
        default=None,
        help="Optional precomputed SAM2 run dir from tools/sam2_first_segment.py",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/sam2_then_yolo"))
    parser.add_argument("--run-name", type=str, default="default_run")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--yolo-input", choices=["bbox", "masked"], default="masked")
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def ensure_dirs(base: Path) -> dict[str, Path]:
    mapping = {
        "yolo_detections": base / "yolo_detections",
        "sam2_masks": base / "sam2_masks",
        "sam2_overlays": base / "sam2_overlays",
        "meta": base / "meta",
    }
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def select_mask(mask_items: list[dict], image_shape: tuple[int, int], min_ratio: float, max_ratio: float):
    h, w = image_shape
    img_area = float(max(1, h * w))
    best_score = -1e9
    best_mask = None
    best_item = None
    for item in mask_items:
        seg = item.get("segmentation")
        if seg is None:
            continue
        mask = seg.astype(bool)
        area = float(mask.sum())
        ratio = area / img_area
        if not (min_ratio <= ratio <= max_ratio):
            continue
        score = 0.6 * float(item.get("predicted_iou", 0.0)) + 0.4 * float(item.get("stability_score", 0.0))
        if score > best_score:
            best_score = score
            best_mask = mask
            best_item = item
    return best_mask, best_item


def draw_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy().astype(np.float32)
    out[mask] = out[mask] * 0.45 + np.array([255, 0, 0], dtype=np.float32) * 0.55
    return out.astype(np.uint8)


def roi_from_mask(image_rgb: np.ndarray, mask: np.ndarray, mode: str):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None
    x1, y1, x2, y2 = bbox
    if mode == "bbox":
        roi = image_rgb[y1 : y2 + 1, x1 : x2 + 1]
    else:
        masked = image_rgb.copy()
        masked[~mask] = 0
        roi = masked[y1 : y2 + 1, x1 : x2 + 1]
    return roi, bbox


def main() -> None:
    args = parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {args.images_dir}")
    if not args.sam2_checkpoint.exists():
        raise FileNotFoundError(f"sam2-checkpoint not found: {args.sam2_checkpoint}")
    if not args.yolo_model.exists():
        raise FileNotFoundError(f"yolo-model not found: {args.yolo_model}")
    if args.sam2_run_dir is not None and not args.sam2_run_dir.exists():
        raise FileNotFoundError(f"sam2-run-dir not found: {args.sam2_run_dir}")

    images = get_images(args.images_dir)
    if not images:
        raise RuntimeError(f"No images found in: {args.images_dir}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.output_root / args.run_name
    out = ensure_dirs(run_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    precomputed_masks_dir = None
    precomputed_meta_dir = None
    amg = None
    if args.sam2_run_dir is not None:
        precomputed_masks_dir = args.sam2_run_dir / "masks"
        precomputed_meta_dir = args.sam2_run_dir / "meta"
        if not precomputed_masks_dir.exists():
            raise FileNotFoundError(f"Precomputed SAM2 masks not found: {precomputed_masks_dir}")
        print(f"Using precomputed SAM2 outputs from: {args.sam2_run_dir}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Building SAM2 on {device}...")
        sam2_model = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=device)
        amg = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )

    yolo = YOLO(str(args.yolo_model))
    print(f"Loaded YOLO model: {args.yolo_model}")
    print(f"YOLO classes: {yolo.names}")

    rows: list[dict] = []
    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] SAM2->YOLO {image_path.name}")
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        h, w = image_rgb.shape[:2]

        best_mask = None
        best_item = None
        if precomputed_masks_dir is not None:
            mask_path = precomputed_masks_dir / f"{image_path.stem}_mask.png"
            if mask_path.exists():
                best_mask = np.array(Image.open(mask_path).convert("L")) > 0
            if precomputed_meta_dir is not None:
                meta_path = precomputed_meta_dir / f"{image_path.stem}.json"
                if meta_path.exists():
                    try:
                        best_item = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        best_item = None
        else:
            mask_items = amg.generate(image_rgb)
            best_mask, best_item = select_mask(
                mask_items,
                image_shape=(h, w),
                min_ratio=args.min_area_ratio,
                max_ratio=args.max_area_ratio,
            )

        status = "no_mask"
        failure_reason = "no_mask_selected"
        pred_class_name = None
        pred_conf = None
        yolo_box_full = None
        mask_pixels = 0
        bbox = None

        det_annot = image_rgb.copy()

        if best_mask is not None:
            mask_pixels = int(best_mask.sum())
            Image.fromarray((best_mask.astype(np.uint8) * 255)).save(
                out["sam2_masks"] / f"{image_path.stem}_mask.png"
            )

            overlay = draw_overlay(image_rgb, best_mask)
            Image.fromarray(overlay).save(out["sam2_overlays"] / f"{image_path.stem}_overlay.jpg")

            roi, bbox = roi_from_mask(image_rgb, best_mask, args.yolo_input)
            if roi is not None and roi.size > 0:
                yolo_result = yolo.predict(
                    source=roi,
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]

                if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
                    best_box = max(yolo_result.boxes, key=lambda b: float(b.conf.item()))
                    local_x1, local_y1, local_x2, local_y2 = [float(v) for v in best_box.xyxy[0].tolist()]
                    class_id = int(best_box.cls.item())
                    pred_class_name = str(yolo.names[class_id])
                    pred_conf = float(best_box.conf.item())

                    if bbox is not None:
                        bx1, by1, _, _ = bbox
                        full_x1 = int(bx1 + local_x1)
                        full_y1 = int(by1 + local_y1)
                        full_x2 = int(bx1 + local_x2)
                        full_y2 = int(by1 + local_y2)
                        yolo_box_full = [full_x1, full_y1, full_x2, full_y2]
                        label = f"{pred_class_name} {pred_conf:.3f}"
                        import cv2

                        cv2.rectangle(det_annot, (full_x1, full_y1), (full_x2, full_y2), (0, 255, 0), 2)
                        cv2.putText(
                            det_annot,
                            label,
                            (full_x1, max(0, full_y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        status = "ok"
                        failure_reason = None
                else:
                    status = "mask_only"
                    failure_reason = "no_yolo_boxes"
            else:
                status = "mask_only"
                failure_reason = "empty_roi"

        Image.fromarray(det_annot.astype(np.uint8)).save(
            out["yolo_detections"] / f"{image_path.stem}_yolo.jpg"
        )

        meta = {
            "image": image_path.name,
            "status": status,
            "mask_pixels": mask_pixels,
            "predicted_class": pred_class_name,
            "confidence": pred_conf,
            "bbox": yolo_box_full,
            "failure_reason": failure_reason,
            "sam2_bbox_xyxy": None if bbox is None else list(map(int, bbox)),
            "sam2_predicted_iou": None
            if best_item is None
            else float(best_item.get("predicted_iou", best_item.get("sam2_predicted_iou", 0.0))),
            "sam2_stability_score": None
            if best_item is None
            else float(best_item.get("stability_score", best_item.get("sam2_stability_score", 0.0))),
            "yolo_pred_class": pred_class_name,
            "yolo_pred_conf": pred_conf,
            "yolo_box_xyxy": yolo_box_full,
        }
        (out["meta"] / f"{image_path.stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(meta)

    csv_path = run_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "status",
                "predicted_class",
                "confidence",
                "bbox",
                "mask_pixels",
                "failure_reason",
                "sam2_bbox_xyxy",
                "sam2_predicted_iou",
                "sam2_stability_score",
                "yolo_pred_class",
                "yolo_pred_conf",
                "yolo_box_xyxy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    detections = sum(1 for row in rows if row["status"] == "ok")
    masks_found = sum(1 for row in rows if row["status"] in {"ok", "mask_only"})
    failures = sum(1 for row in rows if row["status"] != "ok")
    print(f"Done. images={len(rows)}, masks={masks_found}, yolo_detections={detections}, failures={failures}")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
