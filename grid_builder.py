import cv2
import numpy as np
from pathlib import Path

def _load_mask(mask_dir: Path, frame_idx: int, class_id: int) -> np.ndarray | None:
    """Load a mask using either class-suffixed or prefixed filenames."""
    candidates = [
        mask_dir / f"{frame_idx:05d}_{class_id}.png",
        mask_dir / f"frame_{frame_idx:05d}.png",
        mask_dir / f"{frame_idx:05d}.png",
    ]

    for path in candidates:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask
    return None

def build_occupancy_grid(mask_dir, frame_idx, grid_size=(64, 64)):
    """
    Convert SAM2 masks into a small occupancy grid for A*
    """

    mask_dir = Path(mask_dir)

    # Load masks for this frame
    # Load masks for this frame.
    # If you only have rover masks saved, wall/floor will be missing and the
    # occupancy grid will only reflect rover pixels.
    floor = _load_mask(mask_dir, frame_idx, 3)
    wall = _load_mask(mask_dir, frame_idx, 2)
    rover = _load_mask(mask_dir, frame_idx, 1)

    if floor is None and wall is None and rover is None:
        raise RuntimeError(f"No masks found for frame {frame_idx} in {mask_dir}")

    if floor is None:
        floor = np.zeros_like(wall if wall is not None else rover)
    if wall is None:
        wall = np.zeros_like(floor)
    if rover is None:
        rover = np.zeros_like(floor)

    # Create obstacle map
    obstacles = (wall > 0) | (rover > 0)

    # Resize to smaller grid (for fast A*)
    obstacles_small = cv2.resize(
        obstacles.astype(np.uint8),
        grid_size,
        interpolation=cv2.INTER_NEAREST
    )

    return obstacles_small