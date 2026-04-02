#!/usr/bin/env python3
"""Multi-frame point picker using Tkinter GUI - pick points across different video regions."""

import cv2
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Label, Button, Frame
from PIL import Image, ImageTk
import numpy as np

frames_dir = Path("training/HighQualityHololensFootage_frames")
output_file = Path("picked_hololens_points_multiframe.txt")

frame_files = sorted(
    [f for f in frames_dir.glob("*.jpg")],
    key=lambda x: int(x.stem)
)

if not frame_files:
    print(f"No JPEG frames found in {frames_dir}")
    exit(1)

# Class definitions with BGR colors
CLASSES = {
    0: {"name": "ROVER",     "color": (0, 0, 255), "hex": "#FF0000", "key": "1"},      # RED
    1: {"name": "OBSTACLE",  "color": (255, 0, 0), "hex": "#0000FF", "key": "2"},      # BLUE
    2: {"name": "FLOOR",     "color": (0, 255, 0), "hex": "#00FF00", "key": "3"},      # GREEN
}

# Store points organized by frame
picked_points_by_frame = {}
current_frame_idx = 0
current_class_idx = 0

# Create Tkinter window
root = tk.Tk()
root.title("HoloLens Multi-Frame Point Picker - Pick across 30-frame jumps")
root.geometry("700x750")

# Top instruction panel
top_panel = tk.Frame(root, bg="#1a1a1a", height=140)
top_panel.pack(fill=tk.X, padx=10, pady=5)

# Frame navigation info
frame_label = Label(top_panel, text="FRAME 0 / 696", font=("Arial", 16, "bold"), 
                    bg="#1a1a1a", fg="#00FF00")
frame_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=5)

# Current selection indicator (HUGE)
current_label = Label(top_panel, text="NOW PICKING:\nROVER", font=("Arial", 20, "bold"), 
                      bg=CLASSES[0]["hex"], fg="white", width=25, height=4)
current_label.pack(side=tk.LEFT, padx=10, pady=5)

# Stats on the right
stats_frame = tk.Frame(top_panel, bg="#1a1a1a")
stats_frame.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)

class_stats = {}
for i, cls_info in CLASSES.items():
    lbl = Label(stats_frame, text=f"[{cls_info['key']}] {cls_info['name']}: 0", 
                font=("Arial", 11), bg="#1a1a1a", fg=cls_info["hex"])
    lbl.pack(anchor=tk.W)
    class_stats[i] = lbl

# Canvas for image
canvas = Canvas(root, bg="gray20", cursor="crosshair")
canvas.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

def load_and_display_frame(idx):
    """Load a frame and display it on the canvas."""
    global current_frame_idx
    
    if idx < 0 or idx >= len(frame_files):
        return False
    
    current_frame_idx = idx
    frame_cv = cv2.imread(str(frame_files[idx]))
    if frame_cv is None:
        return False
    
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    update_display(frame_rgb)
    return True

def update_display(frame_rgb):
    """Redraw the canvas with all points for current frame."""
    global photo
    display_array = frame_rgb.copy()
    
    # Get points for current frame (if they exist)
    frame_points = picked_points_by_frame.get(current_frame_idx, {i: [] for i in range(len(CLASSES))})
    
    # Draw all points from all classes on this frame
    for class_id in range(len(CLASSES)):
        color_bgr = CLASSES[class_id]["color"]
        for (x, y) in frame_points.get(class_id, []):
            # Draw circle
            cv2.circle(display_array, (x, y), 8, color_bgr, -1)
            cv2.circle(display_array, (x, y), 8, (255, 255, 255), 2)
    
    pil_image = Image.fromarray(display_array)
    photo = ImageTk.PhotoImage(pil_image)
    canvas.create_image(0, 0, anchor="nw", image=photo)
    canvas.image = photo
    
    # Update frame label
    frame_label.config(text=f"FRAME {current_frame_idx} / {len(frame_files) - 1}")
    
    # Update stats for current frame
    if current_frame_idx not in picked_points_by_frame:
        picked_points_by_frame[current_frame_idx] = {i: [] for i in range(len(CLASSES))}
    
    for i in range(len(CLASSES)):
        count = len(picked_points_by_frame[current_frame_idx][i])
        class_stats[i].config(text=f"[{CLASSES[i]['key']}] {CLASSES[i]['name']}: {count}")

def on_canvas_click(event):
    """Handle canvas clicks."""
    if current_frame_idx not in picked_points_by_frame:
        picked_points_by_frame[current_frame_idx] = {i: [] for i in range(len(CLASSES))}
    
    picked_points_by_frame[current_frame_idx][current_class_idx].append((event.x, event.y))
    class_name = CLASSES[current_class_idx]["name"]
    count = len(picked_points_by_frame[current_frame_idx][current_class_idx])
    print(f"Frame {current_frame_idx} - ✓ {class_name} point #{count}: ({event.x}, {event.y})")
    
    # Reload frame to show new point
    frame_cv = cv2.imread(str(frame_files[current_frame_idx]))
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    update_display(frame_rgb)

def switch_class(idx):
    """Switch to a different class."""
    global current_class_idx
    current_class_idx = idx
    class_name = CLASSES[idx]["name"]
    color_hex = CLASSES[idx]["hex"]
    current_label.config(text=f"NOW PICKING:\n{class_name}", bg=color_hex)
    print(f"\n→→→ SWITCHED TO: {class_name.upper()} ←←←")

def jump_frames(delta):
    """Jump by delta frames."""
    new_idx = current_frame_idx + delta
    if 0 <= new_idx < len(frame_files):
        if load_and_display_frame(new_idx):
            print(f"\n→ Jumped to frame {new_idx}")
        else:
            print(f"Could not load frame {new_idx}")
    else:
        print(f"Frame {new_idx} out of range (0-{len(frame_files)-1})")

def undo_point():
    """Remove last point from current class on current frame."""
    if current_frame_idx not in picked_points_by_frame:
        return
    
    if picked_points_by_frame[current_frame_idx][current_class_idx]:
        removed = picked_points_by_frame[current_frame_idx][current_class_idx].pop()
        class_name = CLASSES[current_class_idx]["name"]
        print(f"↶ Removed last {class_name} point from frame {current_frame_idx}: {removed}")
        
        # Reload frame to update display
        frame_cv = cv2.imread(str(frame_files[current_frame_idx]))
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        update_display(frame_rgb)
    else:
        print("! No points to undo for this class on this frame")

def save_and_exit():
    """Save points and close."""
    if any(any(pts for pts in frame_data.values()) for frame_data in picked_points_by_frame.values()):
        with open(output_file, "w") as f:
            for frame_idx in sorted(picked_points_by_frame.keys()):
                frame_data = picked_points_by_frame[frame_idx]
                f.write(f"# FRAME {frame_idx}\n")
                for class_id in range(len(CLASSES)):
                    class_name = CLASSES[class_id]["name"]
                    if frame_data[class_id]:
                        f.write(f"  # {class_name}\n")
                        for x, y in frame_data[class_id]:
                            f.write(f"  ({x}, {y})\n")
        
        print("\n" + "="*90)
        print(f"✓ SAVED: {output_file}")
        print("="*90)
        total_frames = len(picked_points_by_frame)
        total_points = sum(len(pts) for frame_data in picked_points_by_frame.values() 
                          for pts in frame_data.values())
        print(f"Total frames with points: {total_frames}")
        print(f"Total points picked: {total_points}")
        for frame_idx in sorted(picked_points_by_frame.keys()):
            frame_data = picked_points_by_frame[frame_idx]
            frame_total = sum(len(pts) for pts in frame_data.values())
            if frame_total > 0:
                print(f"  Frame {frame_idx}: {frame_total} points")
        print("="*90 + "\n")
    
    root.destroy()

def quit_no_save():
    """Quit without saving."""
    print("\n[QUIT] Discarding all picks...\n")
    root.destroy()

# Canvas click handler
canvas.bind("<Button-1>", on_canvas_click)

# Bottom control panel
bottom_panel = tk.Frame(root, bg="#2a2a2a")
bottom_panel.pack(fill=tk.X, padx=10, pady=5)

# Navigation buttons for frame jumping
nav_frame = tk.Frame(bottom_panel, bg="#2a2a2a")
nav_frame.pack(side=tk.TOP, anchor=tk.W, pady=5)

Button(nav_frame, text="[A] ←60 Frames", command=lambda: jump_frames(-60), 
       bg="#444444", fg="white", font=("Arial", 9), width=16).pack(side=tk.LEFT, padx=2)
Button(nav_frame, text="[D] -30 Frames", command=lambda: jump_frames(-30), 
       bg="#666666", fg="white", font=("Arial", 9), width=16).pack(side=tk.LEFT, padx=2)
Button(nav_frame, text="[SPACE] +30 Frames", command=lambda: jump_frames(30), 
       bg="#006600", fg="white", font=("Arial", 9, "bold"), width=18).pack(side=tk.LEFT, padx=2)
Button(nav_frame, text="[W] +60 Frames", command=lambda: jump_frames(60), 
       bg="#444444", fg="white", font=("Arial", 9), width=16).pack(side=tk.LEFT, padx=2)

# Class buttons
class_frame = tk.Frame(bottom_panel, bg="#2a2a2a")
class_frame.pack(side=tk.TOP, anchor=tk.W, pady=5)

Button(class_frame, text="[1] ROVER (Red)", command=lambda: switch_class(0), 
       bg="#FF0000", fg="white", font=("Arial", 10, "bold"), width=18).pack(side=tk.LEFT, padx=5)
Button(class_frame, text="[2] OBSTACLE (Blue)", command=lambda: switch_class(1), 
       bg="#0000FF", fg="white", font=("Arial", 10, "bold"), width=18).pack(side=tk.LEFT, padx=5)
Button(class_frame, text="[3] FLOOR (Green)", command=lambda: switch_class(2), 
       bg="#00FF00", fg="black", font=("Arial", 10, "bold"), width=18).pack(side=tk.LEFT, padx=5)

# Action buttons
action_frame = tk.Frame(bottom_panel, bg="#2a2a2a")
action_frame.pack(side=tk.TOP, anchor=tk.W, pady=5)

Button(action_frame, text="[U] UNDO Last Point", command=undo_point, 
       bg="#666666", fg="white", font=("Arial", 10), width=20).pack(side=tk.LEFT, padx=5)
Button(action_frame, text="[S] SAVE & EXIT", command=save_and_exit, 
       bg="#00AA00", fg="white", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT, padx=5)
Button(action_frame, text="[Q] QUIT (no save)", command=quit_no_save, 
       bg="#AA0000", fg="white", font=("Arial", 10), width=20).pack(side=tk.LEFT, padx=5)

# Instructions
instructions = tk.Label(bottom_panel, text=
    "SPACEBAR: Jump +30 frames  |  [A]=−60  [D]=−30  [W]=+60  |  CLICK to pick points",
    font=("Arial", 9), bg="#2a2a2a", fg="#CCCCCC")
instructions.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=3)

# Keyboard bindings
root.bind('1', lambda e: switch_class(0))
root.bind('2', lambda e: switch_class(1))
root.bind('3', lambda e: switch_class(2))
root.bind('u', lambda e: undo_point())
root.bind('s', lambda e: save_and_exit())
root.bind('q', lambda e: quit_no_save())
root.bind('space', lambda e: jump_frames(30))
root.bind('a', lambda e: jump_frames(-60))
root.bind('d', lambda e: jump_frames(-30))
root.bind('w', lambda e: jump_frames(60))

print("\n" + "="*90)
print(" HoloLens Multi-Frame Point Picker")
print("="*90)
print(f"\nTotal frames: {len(frame_files)}")
print("\nKEY MAPPINGS:")
print("  [SPACEBAR] = Jump +30 frames (main navigation)")
print("  [A] = Jump −60 frames")
print("  [D] = Jump −30 frames")
print("  [W] = Jump +60 frames")
print("  [1] = Pick ROVER points (RED)")
print("  [2] = Pick OBSTACLE points (BLUE)")
print("  [3] = Pick FLOOR points (GREEN)")
print("  [U] = Undo last point")
print("  [S] = Save points and exit")
print("  [Q] = Quit without saving")
print("\nWORKFLOW:")
print("  1. Pick points on frame 0 (all 3 classes)")
print("  2. Press SPACEBAR to jump to frame 30")
print("  3. Pick more points if needed")
print("  4. Continue jumping through video")
print("  5. Press [S] to save and exit")
print("\n" + "="*90 + "\n")

# Load first frame
if load_and_display_frame(0):
    root.mainloop()


