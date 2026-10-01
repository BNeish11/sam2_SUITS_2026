from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
frames_dir = Path("training/HighQualityHololensFootage_frames")
source_video = Path("training/HighQualityHololensFootage.mp4")
output_path = Path("video_output/Hololens_segmented_multiframe.mp4")
picked_points_file_multiframe = Path("picked_hololens_points_multiframe.txt")
picked_points_file_legacy = Path("picked_hololens_points.txt")


def load_multiframe_picked_points() -> dict[int, dict[str, list[tuple[int, int]]]]:
    """Load manually-picked points organized by frame then class."""
    frame_points = {}
    
    points_file = picked_points_file_multiframe if picked_points_file_multiframe.exists() else picked_points_file_legacy
    
    if not points_file.exists():
        return frame_points
    
    current_frame = None
    current_class = None
    
    with open(points_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse frame headers like "# FRAME 0"
            if line.startswith("# FRAME"):
                frame_str = line.replace("# FRAME", "").strip()
                try:
                    current_frame = int(frame_str)
                    if current_frame not in frame_points:
                        frame_points[current_frame] = {"rover": [], "obstacle": [], "floor": []}
                except ValueError:
                    continue
                current_class = None
                continue
            
            # Parse class headers like "# ROVER"
            if line.startswith("# "):
                class_name = line[2:].lower()
                if class_name in ["rover", "obstacle", "floor"]:
                    current_class = class_name
                continue
            
            # Parse point coordinates
            if line.startswith("(") and current_frame is not None and current_class:
                try:
                    line = line.strip("()")
                    parts = line.split(",")
                    if len(parts) == 2:
                        x = int(parts[0].strip())
                        y = int(parts[1].strip())
                        frame_points[current_frame][current_class].append((x, y))
                except ValueError:
                    continue
    
    return frame_points


def load_picked_points() -> dict[str, list[tuple[int, int]]]:
    """Load manually-picked points organized by class (legacy single-frame format)."""
    class_points = {"rover": [], "obstacle": [], "floor": []}
    
    if not picked_points_file_legacy.exists():
        return class_points
    
    current_class = None
    with open(picked_points_file_legacy, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check for class headers like "# ROVER"
            if line.startswith("# "):
                class_name = line[2:].lower()
                if class_name in class_points:
                    current_class = class_name
                continue
            
            # Parse point coordinates
            if line.startswith("(") and current_class:
                try:
                    line = line.strip("()")
                    parts = line.split(",")
                    if len(parts) == 2:
                        x = int(parts[0].strip())
                        y = int(parts[1].strip())
                        class_points[current_class].append((x, y))
                except ValueError:
                    continue
    
    return class_points


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

    # Try loading multiframe points first, fall back to legacy single-frame points
    multiframe_points = load_multiframe_picked_points()
    legacy_points = load_picked_points()
    
    # If we have multiframe points, use them; otherwise fall back to legacy
    use_multiframe = len(multiframe_points) > 0 and any(
        any(pts for pts in frame_data.values()) for frame_data in multiframe_points.values()
    )
    
    if use_multiframe:
        print(f"Using multiframe points from {picked_points_file_multiframe}")
        print(f"  Loaded {len(multiframe_points)} keyframes")
        for frame_idx in sorted(multiframe_points.keys()):
            frame_data = multiframe_points[frame_idx]
            for class_name in ["rover", "obstacle", "floor"]:
                pts = frame_data.get(class_name, [])
                if pts:
                    print(f"  Frame {frame_idx} - {class_name}: {len(pts)} points")
    else:
        print(f"Using legacy single-frame points from {picked_points_file_legacy}")
        total_picked = sum(len(pts) for pts in legacy_points.values())
        if total_picked > 0:
            for cls_name, pts in legacy_points.items():
                if pts:
                    print(f"  {cls_name}: {len(pts)} points")
        # Convert legacy to frame 0
        multiframe_points = {0: legacy_points}

    class_obj_ids = {"rover": 1, "obstacle": 2, "floor": 3}
    class_colors = {
        1: np.array([255, 0, 0]),
        2: np.array([0, 0, 255]),
        3: np.array([0, 255, 0]),
    }
    render_priority = [3, 2, 1]

    print(f"Frames: {len(frame_names)}")
    print(f"Resolution: {width}x{height}, FPS={fps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM2 predictor on {device}...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    with torch.inference_mode():
        print("Initializing video state from frame directory...")
        state = predictor.init_state(video_path=str(frames_dir))

        # Get the first frame with points (prioritize frame 0)
        init_frames = sorted(multiframe_points.keys())
        first_init_frame = init_frames[0] if init_frames else 0
        
        # Find first frame that has at least one class with points
        first_frame_with_points = None
        for frame_idx in init_frames:
            if any(multiframe_points[frame_idx].get(cls, []) for cls in class_obj_ids.keys()):
                first_frame_with_points = frame_idx
                break
        
        if first_frame_with_points is None and init_frames:
            first_frame_with_points = init_frames[0]
        elif first_frame_with_points is None:
            first_frame_with_points = 0
        
        print(f"Initializing objects at frame {first_frame_with_points}")
        
        # Initialize all classes at the first frame with any points
        for class_name, obj_id in class_obj_ids.items():
            class_points_at_frame = multiframe_points.get(first_frame_with_points, {}).get(class_name, [])
            
            # Get points from other classes as negative examples
            negative_points = [
                point_xy
                for other_class_name in class_obj_ids.keys()
                if other_class_name != class_name
                for point_xy in multiframe_points.get(first_frame_with_points, {}).get(other_class_name, [])
            ]
            
            all_points = class_points_at_frame + negative_points
            all_labels = ([1] * len(class_points_at_frame)) + ([0] * len(negative_points))

            if all_points:
                points = np.array(all_points, dtype=np.float32)
                labels = np.array(all_labels, dtype=np.int32)
                
                print(f"  {class_name}: {len(class_points_at_frame)} positive + {len(negative_points)} negative points")
                
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=first_frame_with_points,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
            else:
                print(f"  {class_name}: no points at frame {first_frame_with_points} (will initialize without)")


        print("Propagating masks through video...")
        video_segments = {}
        masks_dir = Path(__file__).resolve().parent / "video_output" / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving class masks to: {masks_dir}")

        # Track which keyframes we've processed
        processed_keyframes = set()
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            # Check if this frame has new points to add
            if out_frame_idx in multiframe_points and out_frame_idx not in processed_keyframes:
                processed_keyframes.add(out_frame_idx)
                frame_data = multiframe_points[out_frame_idx]
                
                # Add new points for this frame
                for class_name, obj_id in class_obj_ids.items():
                    class_points_at_frame = frame_data.get(class_name, [])
                    negative_points = [
                        point_xy
                        for other_class_name in class_obj_ids.keys()
                        if other_class_name != class_name
                        for point_xy in frame_data.get(other_class_name, [])
                    ]
                    
                    all_points = class_points_at_frame + negative_points
                    all_labels = ([1] * len(class_points_at_frame)) + ([0] * len(negative_points))
                    
                    if all_points:
                        points = np.array(all_points, dtype=np.float32)
                        labels = np.array(all_labels, dtype=np.int32)
                        
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=out_frame_idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )
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

            #Create a single RGB mask image 
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)

            #apply colors
            color_mask[rover_mask] = (0,0,255) # Red for rover
            color_mask[obstacle_mask] = (255,0,0) # Blue for obstacles
            color_mask[floor_mask] = (0,255,0) # Green for floor

            #Save ONE combined mask per frame 
            mask_path = masks_dir / f"{frame_idx:05d}.png"
            ok = cv2.imwrite(str(mask_path), color_mask)
            
            if not ok:
                    print(f"[WARN] failed to write mask: {mask_path}")
            else:
                    written_masks += 1
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
