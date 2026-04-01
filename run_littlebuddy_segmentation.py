from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
frames_dir = Path("training/HighQualityHololensFootage_frames")
source_video = Path("training/HighQualityHololensFootage.mp4")
output_path = Path("video_output/Hololens_segmented.mp4")
picked_points_file = Path("picked_points.txt")


def load_picked_points() -> list[tuple[int, int]]:
    """Load manually-picked points from picked_points.txt if it exists."""
    if not picked_points_file.exists():
        return []
    
    points = []
    with open(picked_points_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            # Parse (x, y) format
            line = line.strip("()")
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    x = int(parts[0].strip())
                    y = int(parts[1].strip())
                    points.append((x, y))
                except ValueError:
                    continue
    return points


def main() -> None:
    frame_names = sorted(
        [p.name for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}],
        key=lambda name: int(Path(name).stem),
    )
    if not frame_names:
        raise RuntimeError(f"No JPEG frames found in: {frames_dir}")

    first_bgr = cv2.imread(str(frames_dir / frame_names[0]))
    if first_bgr is None:
        raise RuntimeError("Could not read first frame")
    height, width = first_bgr.shape[:2]

    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Load manually-picked rover points if available
    picked_rover_points = load_picked_points()
    
    class_points = {
        "rover": picked_rover_points if picked_rover_points else [(width // 2, int(height * 0.68))],
        "obstacle": [
            (width // 2, height // 6),
            (width // 4, height // 4),
            ((3 * width) // 4, height // 4),
        ],
        "floor": [
            (width // 2, (5 * height) // 6),
            (width // 4, (4 * height) // 5),
            ((3 * width) // 4, (4 * height) // 5),
        ],
    }
    
    if picked_rover_points:
        print(f"Loaded {len(picked_rover_points)} manually-picked rover points from {picked_points_file}")

    class_obj_ids = {"rover": 1, "obstacle": 2, "floor": 3}
    class_colors = {
        1: np.array([255, 0, 0]),
        2: np.array([0, 0, 255]),
        3: np.array([0, 255, 0]),
    }
    render_priority = [3, 2, 1]

    print(f"Frames: {len(frame_names)}")
    print(f"Resolution: {width}x{height}, FPS={fps}")
    print("Prompts:")
    for class_name in ("rover", "obstacle", "floor"):
        print(f"  {class_name}: {class_points[class_name]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM2 predictor on {device}...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    with torch.inference_mode():
        print("Initializing video state from frame directory...")
        state = predictor.init_state(video_path=str(frames_dir))

        for class_name, obj_id in class_obj_ids.items():
            positive_points = class_points[class_name]
            negative_points = [
                point_xy
                for other_class_name, point_list in class_points.items()
                if other_class_name != class_name
                for point_xy in point_list
            ]
            all_points = positive_points + negative_points
            all_labels = ([1] * len(positive_points)) + ([0] * len(negative_points))

            points = np.array(all_points, dtype=np.float32)
            labels = np.array(all_labels, dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        print("Propagating masks through video...")
        video_segments = {}
        masks_dir = Path(__file__).resolve().parent / "video_output" / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving class masks to: {masks_dir}")

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            video_segments[out_frame_idx] = {
                int(out_obj_id): (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            if out_frame_idx % 50 == 0:
                print(f"[DEBUG] propagated frame {out_frame_idx}; out_obj_ids={list(map(int, out_obj_ids))}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("Rendering output video...")
    written_masks = 0
    for frame_idx, frame_name in enumerate(frame_names):
        frame_bgr = cv2.imread(str(frames_dir / frame_name))
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        overlay = frame_rgb.copy().astype(float)

        if frame_idx in video_segments:
            class_map = np.zeros((height, width), dtype=np.uint8)

            for obj_id in render_priority:
                mask = video_segments[frame_idx].get(obj_id)
                if mask is None:
                    continue
                mask = mask.squeeze()
                if mask.ndim > 2:
                    mask = mask[0]
                mask_bool = mask.astype(bool)
                class_map[mask_bool] = obj_id

            alpha = 0.5
            for obj_id, color in class_colors.items():
                mask_bool = class_map == obj_id
                if np.any(mask_bool):
                    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color * alpha

            rover_mask = (class_map == 1)
            obstacle_mask = (class_map == 2)
            floor_mask = (class_map == 3)

            masks_to_save = {
                1: rover_mask,
                2: obstacle_mask,
                3: floor_mask,
            }

            for obj_id, mask_bool in masks_to_save.items():
                mask_u8 = np.ascontiguousarray(mask_bool.astype(np.uint8) * 255)
                mask_path = masks_dir / f"{frame_idx:05d}_{obj_id}.png"
                ok = cv2.imwrite(str(mask_path), mask_u8)
                if not ok:
                    print(f"[WARN] failed to write mask: {mask_path}")
                else:
                    written_masks += 1
                    if not np.any(mask_u8):
                        print(f"[WARN] empty mask: {mask_path}")
        else:
            print(f"[WARN] missing propagated masks for frame {frame_idx}")

        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out_video.write(overlay_bgr)

        if (frame_idx + 1) % 25 == 0:
            print(f"Rendered {frame_idx + 1}/{len(frame_names)} frames")

    out_video.release()
    print(f"Saved segmented video: {output_path}")
    print(f"Saved mask files: {written_masks} to {masks_dir}")


if __name__ == "__main__":
    main()
