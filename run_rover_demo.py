import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIG ---
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_path = "video_output/close up little buddy dim light_converted.mp4"
output_dir = "video_output"
output_name = "rover_segmented.mp4"

# Point prompts on frame 0 for each class.
# You can use either:
#   - a single tuple: (x, y)
#   - a list of tuples: [(x1, y1), (x2, y2), ...]
# Set to None to use built-in defaults based on frame size.
class_points = {
    "rover": [(1157, 1546)],
    "wall": None,
    "floor": None,
}

class_obj_ids = {
    "rover": 1,
    "wall": 2,
    "floor": 3,
}

class_colors = {
    1: np.array([255, 0, 0]),   # rover: red
    2: np.array([0, 0, 255]),   # wall: blue
    3: np.array([0, 255, 0]),   # floor: green
}

# Draw lower priority first, higher priority last in overlap regions.
render_priority = [3, 2, 1]
# ---------------

os.makedirs(output_dir, exist_ok=True)

# Convert MOV to MP4 if needed (SAM2 video loader supports MP4)
base_name, ext = os.path.splitext(video_path)
if ext.lower() != ".mp4":
    converted_path = os.path.join(output_dir, f"{os.path.basename(base_name)}_converted.mp4")
    print(f"Converting {video_path} -> {converted_path} ...")
    cap_conv = cv2.VideoCapture(video_path)
    if not cap_conv.isOpened():
        raise RuntimeError(f"Could not open video for conversion: {video_path}")
    fps_conv = cap_conv.get(cv2.CAP_PROP_FPS) or 30
    width_conv = int(cap_conv.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_conv = int(cap_conv.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_conv = cv2.VideoWriter_fourcc(*"mp4v")
    out_conv = cv2.VideoWriter(converted_path, fourcc_conv, fps_conv, (width_conv, height_conv))
    while True:
        ret_conv, frame_conv = cap_conv.read()
        if not ret_conv:
            break
        out_conv.write(frame_conv)
    cap_conv.release()
    out_conv.release()
    video_path = converted_path

# Read first frame to determine size
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError(f"Could not read video: {video_path}")
height, width = first_frame.shape[:2]
cap.release()

default_points = {
    "rover": [(width // 2, height // 2)],
    "wall": [
        (width // 2, height // 5),
        (width // 4, height // 3),
        ((3 * width) // 4, height // 3),
    ],
    "floor": [
        (width // 2, (4 * height) // 5),
        (width // 4, (3 * height) // 4),
        ((3 * width) // 4, (3 * height) // 4),
    ],
}

resolved_points = {}
for class_name, point_data in class_points.items():
    if point_data is None:
        point_list = default_points[class_name]
    elif isinstance(point_data, tuple):
        point_list = [point_data]
    else:
        point_list = list(point_data)

    clipped_points = []
    for point_xy in point_list:
        x = int(np.clip(point_xy[0], 0, width - 1))
        y = int(np.clip(point_xy[1], 0, height - 1))
        clipped_points.append((x, y))

    if not clipped_points:
        clipped_points = default_points[class_name]

    resolved_points[class_name] = clipped_points

print("Using class prompts:")
for class_name in ("rover", "wall", "floor"):
    print(f"  {class_name}: {resolved_points[class_name]}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Building SAM2 video predictor on {device}...")
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

with torch.inference_mode():
    print("Initializing state...")
    state = predictor.init_state(video_path=video_path)

    for class_name, obj_id in class_obj_ids.items():
        positive_points = resolved_points[class_name]
        negative_points = [
            point_xy
            for other_class_name, point_list in resolved_points.items()
            if other_class_name != class_name
            for point_xy in point_list
        ]

        all_points = positive_points + negative_points
        all_labels = ([1] * len(positive_points)) + ([0] * len(negative_points))

        point = np.array(all_points, dtype=np.float32)
        label = np.array(all_labels, dtype=np.int32)

        print(
            f"Adding {class_name} prompts on frame 0: "
            f"{len(positive_points)} positive, {len(negative_points)} negative"
        )
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            points=point,
            labels=label,
        )

    print("Propagating mask through video...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

# Read full video for overlay
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = os.path.join(output_dir, output_name)

out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    overlay = overlay.astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    out_video.write(overlay_bgr)

    frame_idx += 1
    if frame_idx % 10 == 0:
        print(f"Processed {frame_idx} frames")

cap.release()
out_video.release()

print(f"Saved: {output_path}")
os.system(f'start "" "{output_path}"')
