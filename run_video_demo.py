import os
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIG ---
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_path = "notebooks/videos/bedroom.mp4"
output_dir = "video_output"
# ---------------

# Create output directory
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Building SAM2 video predictor on {device}...")
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

# Read video frames
print(f"Loading video: {video_path}")
cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
cap.release()

print(f"Loaded {len(frames)} frames")

# Get video properties
height, width = frames[0].shape[:2]
fps = 30  # default fps

# Initialize inference state
with torch.inference_mode():
    state = predictor.init_state(video_path=video_path)
    
    # Add prompts on the first frame
    # Add two points to mark the kids jumping
    # Adjust these coordinates based on where the kids are in the first frame
    points = np.array([
        [width // 3, height // 2],      # First kid
        [2 * width // 3, height // 2],  # Second kid
    ], dtype=np.float32)
    
    labels = np.array([1, 1], dtype=np.int32)  # 1 = foreground
    
    print("Adding prompts for the kids...")
    # Add prompts on frame 0
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=points[0:1],
        labels=labels[0:1],
    )
    
    # Add second kid
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=2,
        points=points[1:2],
        labels=labels[1:2],
    )
    
    print("Propagating masks through video...")
    # Propagate masks through the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

print("Creating output video with masks...")

# Define colors for different objects
colors = {
    1: np.array([255, 0, 0]),    # Red for first kid
    2: np.array([0, 255, 0]),    # Green for second kid
}

# Create video writer
output_path = os.path.join(output_dir, "bedroom_segmented.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame
for frame_idx, frame in enumerate(frames):
    overlay = frame.copy().astype(float)
    
    if frame_idx in video_segments:
        for obj_id, mask in video_segments[frame_idx].items():
            # Remove extra dimensions if needed
            mask = mask.squeeze()
            if mask.ndim > 2:
                mask = mask[0]
            
            mask_bool = mask.astype(bool)
            color = colors.get(obj_id, np.array([255, 255, 0]))
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
    
    # Convert back to uint8 and BGR for video writer
    overlay = overlay.astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    out_video.write(overlay_bgr)
    
    if (frame_idx + 1) % 10 == 0:
        print(f"Processed {frame_idx + 1}/{len(frames)} frames")

out_video.release()
print(f"\nVideo saved to: {output_path}")
print("Opening output video...")
os.system(f'start "" "{output_path}"')
