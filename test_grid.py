from grid_builder import build_occupancy_grid

grid = build_occupancy_grid("video_output/masks", frame_idx=0)

print("Grid shape:", grid.shape)
print("Unique values:", set(grid.flatten()))