from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


PromptPoints = dict[str, list[tuple[float, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO rover detection and SAM2 segmentation on a folder of images."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("rover pictures"),
        help="Input image folder.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("yolo26n.pt"),
        help="Path to YOLO model weights.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path("checkpoints/sam2.1_hiera_large.pt"),
        help="Path to SAM2 checkpoint.",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/rover_batch"),
        help="Output directory.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size.")
    parser.add_argument(
        "--rover-class-id",
        type=int,
        default=None,
        help="Optional explicit rover class id for YOLO.",
    )
    parser.add_argument(
        "--points-file",
        type=Path,
        default=None,
        help="Optional per-image manual SAM2 point prompts file.",
    )
    parser.add_argument(
        "--manual-min-area-ratio",
        type=float,
        default=0.0015,
        help="Minimum allowed mask area ratio for manual point prompting.",
    )
    parser.add_argument(
        "--manual-max-area-ratio",
        type=float,
        default=0.45,
        help="Maximum allowed mask area ratio for manual point prompting.",
    )
    parser.add_argument(
        "--manual-box-scale",
        type=float,
        default=0.22,
        help="Relative image size for local box prompt around manual points.",
    )
    parser.add_argument(
        "--manual-require-pos-ratio",
        type=float,
        default=0.8,
        help="Minimum fraction of positive points that must lie inside the selected mask.",
    )
    parser.add_argument(
        "--manual-max-neg-ratio",
        type=float,
        default=0.2,
        help="Maximum fraction of negative points allowed inside the selected mask.",
    )
    parser.add_argument(
        "--manual-target-area-ratio",
        type=float,
        default=0.08,
        help="Preferred mask area ratio for manual prompt scoring.",
    )
    parser.add_argument(
        "--allow-yolo-fallback-with-points",
        action="store_true",
        help="If set, allows YOLO fallback when manual points are present but no valid manual mask is selected.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for mask generation.",
    )
    parser.add_argument(
        "--mask-close-kernel",
        type=int,
        default=7,
        help="Morphological close kernel size for mask cleanup (odd number, <=1 disables).",
    )
    parser.add_argument(
        "--mask-open-kernel",
        type=int,
        default=3,
        help="Morphological open kernel size for mask cleanup (odd number, <=1 disables).",
    )
    parser.add_argument(
        "--mask-min-component-area-ratio",
        type=float,
        default=0.0002,
        help="Minimum connected-component area ratio retained during mask cleanup.",
    )
    return parser.parse_args()


def parse_points_file(points_file: Path) -> dict[str, PromptPoints]:
    points_map: dict[str, PromptPoints] = {"*": {"pos": [], "neg": []}}
    current_key = "*"
    current_mode = "pos"
    point_re = re.compile(r"\(([-\d.]+)\s*,\s*([-\d.]+)\)")

    for raw_line in points_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            header = line.lstrip("#").strip()
            if not header:
                continue
            low = header.lower()
            if low.startswith("image"):
                parts = header.split(maxsplit=1)
                if len(parts) == 2:
                    current_key = Path(parts[1].strip()).stem.lower()
                    points_map.setdefault(current_key, {"pos": [], "neg": []})
                    current_mode = "pos"
            elif low in {"pos", "positive", "rover", "+"}:
                current_mode = "pos"
            elif low in {"neg", "negative", "background", "obstacle", "floor", "-"}:
                current_mode = "neg"
            continue

        match = point_re.search(line)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            points_map.setdefault(current_key, {"pos": [], "neg": []})
            points_map[current_key][current_mode].append((x, y))

    return points_map


def points_for_image(image_path: Path, points_map: dict[str, PromptPoints]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    stem = image_path.stem.lower()
    if stem in points_map and (points_map[stem]["pos"] or points_map[stem]["neg"]):
        bucket = points_map[stem]
    else:
        bucket = points_map.get("*", {"pos": [], "neg": []})
    return bucket.get("pos", []), bucket.get("neg", [])


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def point_hit_ratio(mask: np.ndarray, points: list[tuple[float, float]]) -> float:
    if not points:
        return 1.0
    h, w = mask.shape[:2]
    hits = 0
    for px, py in points:
        ix = int(np.clip(px, 0, w - 1))
        iy = int(np.clip(py, 0, h - 1))
        if bool(mask[iy, ix]):
            hits += 1
    return hits / float(max(1, len(points)))


def cleanup_mask(
    mask: np.ndarray,
    pos_points: list[tuple[float, float]] | None = None,
    close_kernel: int = 7,
    open_kernel: int = 3,
    min_component_area_ratio: float = 0.0002,
) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return mask_bool

    h, w = mask_bool.shape[:2]
    img_area = float(max(1, h * w))
    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels > 1:
        min_area_pixels = int(max(1.0, min_component_area_ratio * img_area))
        keep = np.zeros_like(mask_bool)

        hit_labels: set[int] = set()
        if pos_points:
            for px, py in pos_points:
                ix = int(np.clip(px, 0, w - 1))
                iy = int(np.clip(py, 0, h - 1))
                label_id = int(labels[iy, ix])
                if label_id > 0:
                    hit_labels.add(label_id)

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if hit_labels:
                if label_id in hit_labels:
                    keep[labels == label_id] = True
                elif area >= min_area_pixels:
                    keep[labels == label_id] = True
            elif area >= min_area_pixels:
                keep[labels == label_id] = True

        if np.any(keep):
            mask_bool = keep
            mask_u8 = (mask_bool.astype(np.uint8) * 255)

    if close_kernel > 1:
        k = close_kernel if close_kernel % 2 == 1 else close_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    if open_kernel > 1:
        k = open_kernel if open_kernel % 2 == 1 else open_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    return mask_u8 > 0


def mask_from_points(
    predictor: SAM2ImagePredictor,
    image_rgb: np.ndarray,
    pos_points: list[tuple[float, float]],
    neg_points: list[tuple[float, float]] | None = None,
    min_area_ratio: float = 0.0015,
    max_area_ratio: float = 0.45,
    local_box_scale: float = 0.22,
    require_pos_ratio: float = 0.8,
    max_neg_ratio: float = 0.2,
    target_area_ratio: float = 0.08,
    debug: bool = False,
) -> tuple[np.ndarray | None, float | None, tuple[int, int, int, int] | None]:
    if not pos_points:
        return None, None, None

    neg_points = neg_points or []

    h, w = image_rgb.shape[:2]
    pos_coords = np.array(pos_points, dtype=np.float32)
    pos_x1 = float(np.min(pos_coords[:, 0]))
    pos_y1 = float(np.min(pos_coords[:, 1]))
    pos_x2 = float(np.max(pos_coords[:, 0]))
    pos_y2 = float(np.max(pos_coords[:, 1]))
    pos_box_area = float(max(1.0, (pos_x2 - pos_x1) * (pos_y2 - pos_y1)))

    predictor.set_image(image_rgb)
    img_area = float(max(1, h * w))
    point_variants: list[tuple[str, np.ndarray, np.ndarray]] = []

    pos_only_coords = np.array(pos_points, dtype=np.float32)
    pos_only_coords[:, 0] = np.clip(pos_only_coords[:, 0], 0, w - 1)
    pos_only_coords[:, 1] = np.clip(pos_only_coords[:, 1], 0, h - 1)
    point_variants.append(("pos_only", pos_only_coords, np.ones(len(pos_points), dtype=np.int32)))

    if neg_points:
        all_points = pos_points + neg_points
        labels_list = ([1] * len(pos_points)) + ([0] * len(neg_points))
        coords = np.array(all_points, dtype=np.float32)
        coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)
        labels = np.array(labels_list, dtype=np.int32)
        point_variants.append(("pos_neg", coords, labels))

    scale_multipliers = [1.0, 1.6, 2.4, 3.4]

    def build_local_box(scale_multiplier: float) -> np.ndarray:
        pad_x = max(64.0, local_box_scale * w * 0.5 * scale_multiplier)
        pad_y = max(64.0, local_box_scale * h * 0.5 * scale_multiplier)
        box_x1 = max(0.0, pos_x1 - pad_x)
        box_y1 = max(0.0, pos_y1 - pad_y)
        box_x2 = min(float(w - 1), pos_x2 + pad_x)
        box_y2 = min(float(h - 1), pos_y2 + pad_y)
        return np.array([box_x1, box_y1, box_x2, box_y2], dtype=np.float32)

    best_strict_mask = None
    best_strict_score = -1e9
    best_strict_bbox = None
    best_relaxed_mask = None
    best_relaxed_score = -1e9
    best_relaxed_bbox = None
    rejected_reasons = []
    candidate_count = 0

    def evaluate_candidate(mask: np.ndarray, sam_score: float, tag: str) -> None:
        nonlocal best_strict_mask, best_strict_score, best_strict_bbox
        nonlocal best_relaxed_mask, best_relaxed_score, best_relaxed_bbox, candidate_count

        candidate_count += 1
        area_ratio = float(mask.sum()) / img_area
        pos_ratio = point_hit_ratio(mask, pos_points)
        neg_ratio = point_hit_ratio(mask, neg_points)

        if debug:
            print(
                f"    Candidate {candidate_count - 1} [{tag}]: sam_score={sam_score:.4f}, area_ratio={area_ratio:.6f}, pos_ratio={pos_ratio:.4f}, neg_ratio={neg_ratio:.4f}"
            )

        bbox = mask_bbox(mask)
        if bbox is None:
            if debug:
                print(f"      REJECTED: no bbox")
            rejected_reasons.append((candidate_count - 1, f"{tag}: no bbox"))
            return

        area_penalty = abs(area_ratio - target_area_ratio)
        bx1, by1, bx2, by2 = bbox
        bbox_area = float(max(1.0, (bx2 - bx1) * (by2 - by1)))
        bbox_expand_ratio = bbox_area / pos_box_area
        expansion_penalty = max(0.0, bbox_expand_ratio - 8.0) * 0.02
        strict_ok = (
            min_area_ratio <= area_ratio <= max_area_ratio
            and pos_ratio >= require_pos_ratio
            and neg_ratio <= max_neg_ratio
            and bbox_expand_ratio <= 22.0
        )

        strict_score = (
            0.65 * sam_score
            + 0.2 * pos_ratio
            + 0.15 * (1.0 - neg_ratio)
            - 0.35 * area_penalty
            - 0.12 * expansion_penalty
        )
        relaxed_score = (
            0.62 * pos_ratio
            + 0.30 * (1.0 - neg_ratio)
            + 0.08 * sam_score
            - 0.18 * area_penalty
            - 0.55 * expansion_penalty
        )

        if strict_ok:
            if debug:
                print(f"      ACCEPTED(strict): score={strict_score:.4f}")
            if strict_score > best_strict_score:
                best_strict_score = strict_score
                best_strict_mask = mask
                best_strict_bbox = bbox
            return

        if debug:
            reason_parts = []
            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                reason_parts.append(f"area {area_ratio:.6f} outside [{min_area_ratio:.6f}, {max_area_ratio:.6f}]")
            if pos_ratio < require_pos_ratio:
                reason_parts.append(f"pos_ratio {pos_ratio:.4f} < {require_pos_ratio:.4f}")
            if neg_ratio > max_neg_ratio:
                reason_parts.append(f"neg_ratio {neg_ratio:.4f} > {max_neg_ratio:.4f}")
            if bbox_expand_ratio > 22.0:
                reason_parts.append(f"expand_ratio {bbox_expand_ratio:.2f} > 22.0")
            print(f"      FALLBACK: {'; '.join(reason_parts) if reason_parts else 'no strict match'}")

        if bbox_expand_ratio > 40.0 and pos_ratio < 0.9:
            if debug:
                print(
                    f"      REJECTED(relaxed): over-expanded bbox ratio {bbox_expand_ratio:.2f} with insufficient pos coverage {pos_ratio:.4f}"
                )
            return

        if relaxed_score > best_relaxed_score:
            best_relaxed_score = relaxed_score
            best_relaxed_mask = mask
            best_relaxed_bbox = bbox

    for variant_name, coords, labels in point_variants:
        if debug:
            print(f"  DEBUG: Trying point variant '{variant_name}' with {len(coords)} points")

        with torch.inference_mode():
            point_masks, point_scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=True,
            )

        if debug:
            print(f"  DEBUG: Point scores ({variant_name}): {[f'{s:.4f}' for s in point_scores]}")

        for idx, (mask, sam_score) in enumerate(zip(point_masks, point_scores)):
            evaluate_candidate(mask.astype(bool), float(sam_score), f"{variant_name}/points/{idx}")

        for scale_multiplier in scale_multipliers:
            local_box = build_local_box(scale_multiplier)
            if debug:
                print(
                    f"  DEBUG: Local box ({variant_name}, x{scale_multiplier:.1f}): [{local_box[0]:.1f}, {local_box[1]:.1f}, {local_box[2]:.1f}, {local_box[3]:.1f}]"
                )
            with torch.inference_mode():
                box_masks, box_scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=local_box,
                    multimask_output=True,
                )
            if debug:
                print(f"  DEBUG: Box scores ({variant_name}, x{scale_multiplier:.1f}): {[f'{s:.4f}' for s in box_scores]}")
            for idx, (mask, sam_score) in enumerate(zip(box_masks, box_scores)):
                evaluate_candidate(mask.astype(bool), float(sam_score), f"{variant_name}/box/{scale_multiplier:.1f}/{idx}")

    best_mask = best_strict_mask if best_strict_mask is not None else best_relaxed_mask
    best_bbox = best_strict_bbox if best_strict_mask is not None else best_relaxed_bbox
    best_score = best_strict_score if best_strict_mask is not None else best_relaxed_score

    if debug and best_strict_mask is None and best_relaxed_mask is None:
        print(f"  DEBUG: No valid mask found. Rejection summary:")
        for idx, reason in rejected_reasons:
            print(f"    Candidate {idx}: {reason}")
    elif debug and best_strict_mask is None and best_relaxed_mask is not None:
        print(f"  DEBUG: Using relaxed fallback mask (no strict mask matched).")

    if best_mask is None:
        return None, None, None

    return best_mask, best_score, best_bbox


def pick_rover_class_id(model: YOLO, explicit: int | None) -> int | None:
    if explicit is not None:
        return int(explicit)

    for class_id, class_name in model.names.items():
        if str(class_name).lower() == "rover":
            return int(class_id)

    if len(model.names) == 1:
        return int(next(iter(model.names.keys())))

    return None


def choose_detection(result, rover_class_id: int | None):
    if result.boxes is None or len(result.boxes) == 0:
        return None, False, 0

    boxes = result.boxes

    rover_count = 0
    if rover_class_id is not None:
        rover_count = int((boxes.cls.cpu().numpy().astype(int) == rover_class_id).sum())

    if rover_class_id is not None:
        rover_boxes = [b for b in boxes if int(b.cls.item()) == rover_class_id]
        if rover_boxes:
            return max(rover_boxes, key=lambda b: float(b.conf.item())), False, rover_count
        return max(boxes, key=lambda b: float(b.conf.item())), True, rover_count

    return max(boxes, key=lambda b: float(b.conf.item())), True, rover_count


def is_unreliable_rover_box(box, rover_count: int, image_shape: tuple[int, int]) -> bool:
    if box is None:
        return True
    h, w = image_shape
    img_area = float(max(1, h * w))
    conf = float(box.conf.item())
    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area_ratio = (bw * bh) / img_area

    if conf < 0.02:
        return True
    if area_ratio > 0.55 or area_ratio < 0.0004:
        return True
    if rover_count > 120 and conf < 0.08:
        return True
    return False


def main() -> None:
    args = parse_args()

    image_dir = args.images
    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not args.yolo_model.exists():
        raise FileNotFoundError(f"YOLO model not found: {args.yolo_model}")
    if not args.sam2_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
    if args.points_file is not None and not args.points_file.exists():
        raise FileNotFoundError(f"Points file not found: {args.points_file}")

    image_paths = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        ]
    )
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    det_dir = args.out / "yolo_detections"
    seg_dir = args.out / "sam2_overlays"
    mask_dir = args.out / "sam2_masks"
    det_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(args.yolo_model))
    rover_class_id = pick_rover_class_id(yolo, args.rover_class_id)
    print(f"Loaded YOLO model: {args.yolo_model}")
    print(f"YOLO class names: {yolo.names}")
    print(f"Using rover class id: {rover_class_id}")

    points_map = parse_points_file(args.points_file) if args.points_file is not None else {"*": {"pos": [], "neg": []}}
    if args.points_file is not None:
        print(f"Using manual per-image prompts from: {args.points_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM2 on {device}...")
    sam_model = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=device)
    predictor = SAM2ImagePredictor(sam_model)

    rows: list[dict] = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path.name}")
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)

        det_annot = img_np.copy()
        seg_overlay = img_np.copy().astype(np.float32)
        status = "no_detection"
        det_conf = None
        det_class_id = None
        det_class_name = None
        bbox = None
        mask_pixels = 0

        manual_pos_points, manual_neg_points = points_for_image(image_path, points_map)
        used_manual_prompt = False
        if manual_pos_points:
            best_mask, point_score, point_bbox = mask_from_points(
                predictor=predictor,
                image_rgb=img_np,
                pos_points=manual_pos_points,
                neg_points=manual_neg_points,
                min_area_ratio=args.manual_min_area_ratio,
                max_area_ratio=args.manual_max_area_ratio,
                local_box_scale=args.manual_box_scale,
                require_pos_ratio=args.manual_require_pos_ratio,
                max_neg_ratio=args.manual_max_neg_ratio,
                target_area_ratio=args.manual_target_area_ratio,
                debug=args.debug,
            )
            if best_mask is not None and point_bbox is not None:
                raw_best_mask = best_mask
                raw_pos_ratio = point_hit_ratio(raw_best_mask, manual_pos_points)
                best_mask = cleanup_mask(
                    best_mask,
                    pos_points=manual_pos_points,
                    close_kernel=args.mask_close_kernel,
                    open_kernel=args.mask_open_kernel,
                    min_component_area_ratio=args.mask_min_component_area_ratio,
                )
                cleaned_pos_ratio = point_hit_ratio(best_mask, manual_pos_points)
                min_allowed_ratio = min(0.95, max(0.55, raw_pos_ratio * 0.78))
                if cleaned_pos_ratio < min_allowed_ratio:
                    best_mask = raw_best_mask
                point_bbox = mask_bbox(best_mask)
            if best_mask is not None and point_bbox is not None:
                used_manual_prompt = True
                x1, y1, x2, y2 = [float(v) for v in point_bbox]
                bbox = [x1, y1, x2, y2]
                det_conf = point_score
                det_class_id = rover_class_id
                det_class_name = "rover"
                label = f"{det_class_name} manual {0.0 if point_score is None else point_score:.3f}"

                cv2.rectangle(det_annot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.putText(
                    det_annot,
                    label,
                    (int(x1), max(0, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                mask_pixels = int(best_mask.sum())
                Image.fromarray((best_mask.astype(np.uint8) * 255)).save(
                    mask_dir / f"{image_path.stem}_mask.png"
                )

                seg_overlay[best_mask] = (
                    seg_overlay[best_mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
                )
                cv2.rectangle(seg_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    seg_overlay,
                    label,
                    (int(x1), max(0, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                status = "ok_manual"

            elif not args.allow_yolo_fallback_with_points:
                status = "manual_failed"

        rover_count = 0
        chosen = None
        use_yolo = not used_manual_prompt
        if manual_pos_points and not used_manual_prompt and not args.allow_yolo_fallback_with_points:
            use_yolo = False

        if use_yolo:
            result = yolo.predict(
                source=img_np,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            chosen, _, rover_count = choose_detection(result, rover_class_id)

            if is_unreliable_rover_box(chosen, rover_count, img_np.shape[:2]):
                chosen = None

        if chosen is not None and not used_manual_prompt:
            x1, y1, x2, y2 = [float(v) for v in chosen.xyxy[0].tolist()]
            bbox = [x1, y1, x2, y2]
            det_conf = float(chosen.conf.item())
            det_class_id = int(chosen.cls.item())
            det_class_name = "rover"
            label = f"{det_class_name} {det_conf:.3f}"

            cv2.rectangle(det_annot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(
                det_annot,
                label,
                (int(x1), max(0, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            predictor.set_image(img_np)
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    multimask_output=True,
                )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(bool)
            best_mask = cleanup_mask(
                best_mask,
                pos_points=None,
                close_kernel=args.mask_close_kernel,
                open_kernel=args.mask_open_kernel,
                min_component_area_ratio=args.mask_min_component_area_ratio,
            )
            cleaned_bbox = mask_bbox(best_mask)
            if cleaned_bbox is not None:
                x1, y1, x2, y2 = [float(v) for v in cleaned_bbox]
                bbox = [x1, y1, x2, y2]

            mask_pixels = int(best_mask.sum())

            Image.fromarray((best_mask.astype(np.uint8) * 255)).save(
                mask_dir / f"{image_path.stem}_mask.png"
            )

            seg_overlay[best_mask] = (
                seg_overlay[best_mask] * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
            )
            cv2.rectangle(seg_overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                seg_overlay,
                label,
                (int(x1), max(0, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            status = "ok"

        Image.fromarray(det_annot.astype(np.uint8)).save(det_dir / f"{image_path.stem}_yolo.jpg")
        Image.fromarray(seg_overlay.astype(np.uint8)).save(seg_dir / f"{image_path.stem}_sam2_overlay.jpg")

        rows.append(
            {
                "image": image_path.name,
                "status": status,
                "rover_class_id": rover_class_id,
                "rover_count": rover_count,
                "det_class_id": det_class_id,
                "det_class_name": det_class_name,
                "det_conf": det_conf,
                "bbox_x1": None if bbox is None else bbox[0],
                "bbox_y1": None if bbox is None else bbox[1],
                "bbox_x2": None if bbox is None else bbox[2],
                "bbox_y2": None if bbox is None else bbox[3],
                "mask_pixels": mask_pixels,
            }
        )

    csv_path = args.out / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if str(row["status"]).startswith("ok"))
    print(f"Done: {ok_count}/{len(rows)} images segmented.")
    print(f"YOLO outputs: {det_dir}")
    print(f"SAM2 overlays: {seg_dir}")
    print(f"SAM2 masks: {mask_dir}")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()