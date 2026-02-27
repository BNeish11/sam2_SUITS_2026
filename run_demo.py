import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIG ---
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"
image_path = "notebooks/images/cars.jpg"   # Use a real demo image from the repo
output     = "demo_output.png"
# ---------------

# Build model (forces CPU)
model = build_sam2(model_cfg, checkpoint, device="cpu")

# Create predictor
predictor = SAM2ImagePredictor(model)

# Load image
img = Image.open(image_path).convert("RGB")
img_array = np.array(img)
predictor.set_image(img_array)

height, width = img_array.shape[:2]

# Define points for the two cars (left car and right car)
# Adjust these coordinates based on where the cars are in the image
left_car_point = np.array([[width // 4, height // 2]])      # Left side, middle height
right_car_point = np.array([[3 * width // 4, height // 2]])  # Right side, middle height

input_points = np.vstack([left_car_point, right_car_point])
input_labels = np.array([1, 1])  # 1 = foreground for both points

# Run prediction with point prompts
print("Segmenting vehicles at marked points...")
with torch.inference_mode():
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,  # single best mask per point
    )

# Create visualization
overlay = img_array.copy().astype(float)

# Overlay both vehicle masks with different colors
colors = [
    [255, 0, 0],    # red for left car
    [0, 255, 0],    # green for right car
]

for idx, mask in enumerate(masks):
    mask_bool = mask.astype(bool)
    color = colors[idx]
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5

# Save overlay
Image.fromarray(overlay.astype(np.uint8)).save(output)

print(f"Segmentation saved as {output}")
print(f"Highlighted 2 vehicles: left car (red) and right car (green)")

