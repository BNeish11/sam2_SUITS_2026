from grid_builder import build_occupancy_grid
from astar import astar

grid = build_occupancy_grid("video_output/masks", frame_idx=0)

start = (60, 32)   # bottom-ish (rover area)
goal  = (5, 32)    # top-ish (goal area)

path = astar(grid, start, goal)

if path is None:
    print("No path found")
else:
    print(f"Path length: {len(path)}")
    print("First 5 points:", path[:5])