#!/usr/bin/env python3
"""Interactive point picker for HoloLens video frames with CLEAR COLOR CODING."""

import cv2
from pathlib import Path

frames_dir = Path("training/HighQualityHololensFootage_frames")
output_file = Path("picked_hololens_points.txt")

frame_files = sorted(
    [f for f in frames_dir.glob("*.jpg")],
    key=lambda x: int(x.stem)
)

if not frame_files:
    print(f"No JPEG frames found in {frames_dir}")
    exit(1)

# Load first frame
frame = cv2.imread(str(frame_files[0]))
if frame is None:
    print(f"Could not read {frame_files[0]}")
    exit(1)

h, w = frame.shape[:2]

# Class definitions with BGR colors (OpenCV uses BGR not RGB)
CLASSES = {
    0: {"name": "ROVER",     "color": (0, 0, 255), "key": "1"},      # RED
    1: {"name": "OBSTACLE",  "color": (255, 0, 0), "key": "2"},      # BLUE
    2: {"name": "FLOOR",     "color": (0, 255, 0), "key": "3"},      # GREEN
}

picked_points = {i: [] for i in range(len(CLASSES))}
current_class_idx = 0

def draw_ui(img):
    """Draw UI with HUGE color indicator showing current selection."""
    display = img.copy()
    h, w = display.shape[:2]
    
    # Draw all picked points from all classes
    for class_id in range(len(CLASSES)):
        color = CLASSES[class_id]["color"]
        for (x, y) in picked_points[class_id]:
            cv2.circle(display, (x, y), 8, color, -1)
            cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
    
    current_class = CLASSES[current_class_idx]
    current_color = current_class["color"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # ==================================================
    # TOP SECTION - CURRENT CLASS INDICATOR (HUGE)
    # ==================================================
    cv2.rectangle(display, (0, 0), (w, 180), (20, 20, 20), -1)
    
    # HUGE colored rectangle showing what you're currently picking
    color_box_h = 140
    color_box_y = 20
    cv2.rectangle(display, (20, color_box_y), (200, color_box_y + color_box_h), 
                  current_color, -1)
    cv2.rectangle(display, (20, color_box_y), (200, color_box_y + color_box_h), 
                  (255, 255, 255), 5)
    
    # Text next to box: CURRENT SELECTION
    cv2.putText(display, "NOW PICKING:", (220, 50), font, 1.2, (200, 200, 200), 2)
    cv2.putText(display, current_class['name'], (220, 110), font, 2.0, current_color, 4)
    count = len(picked_points[current_class_idx])
    cv2.putText(display, f"({count} points)", (220, 155), font, 1.0, (150, 255, 150), 2)
    
    # ==================================================
    # CLASS SELECTOR BUTTONS (showing all 3 classes)
    # ==================================================
    button_y = 165
    cv2.putText(display, "SELECT CLASS:", (20, button_y), font, 0.7, (180, 180, 180), 1)
    
    for i, cls_info in CLASSES.items():
        is_selected = (i == current_class_idx)
        x_pos = 20 + i * (w // 3)
        
        # Button background
        bg_color = cls_info["color"] if is_selected else (60, 60, 60)
        cv2.rectangle(display, (x_pos + 5, 185), (x_pos + (w // 3) - 15, 210), bg_color, -1)
        if is_selected:
            cv2.rectangle(display, (x_pos + 5, 185), (x_pos + (w // 3) - 15, 210), (255, 255, 255), 3)
        
        # Button text
        text = f"[{cls_info['key']}] {cls_info['name']}"
        cv2.putText(display, text, (x_pos + 10, 205), font, 0.6, (0, 0, 0) if is_selected else cls_info["color"], 1)
    
    # ==================================================
    # BOTTOM INSTRUCTIONS
    # ==================================================
    cv2.rectangle(display, (0, h - 120), (w, h), (20, 20, 20), -1)
    
    cv2.putText(display, "CONTROLS:", (20, h - 95), font, 0.7, (200, 200, 200), 1)
    cv2.putText(display, "[1] ROVER   [2] OBSTACLE   [3] FLOOR   |   LEFT-CLICK to pick point", 
                (20, h - 70), font, 0.65, (150, 200, 255), 1)
    cv2.putText(display, "[U] Undo last point   |   [S] Save & Exit   |   [Q] Quit (no save)", 
                (20, h - 45), font, 0.65, (150, 150, 150), 1)
    
    # Point summary on side
    summary_x = w - 280
    cv2.putText(display, "POINT SUMMARY:", (summary_x, h - 95), font, 0.65, (180, 180, 180), 1)
    for i, cls_info in CLASSES.items():
        count = len(picked_points[i])
        text = f"{cls_info['name']}: {count}"
        cv2.putText(display, text, (summary_x, h - 70 + i * 25), font, 0.6, cls_info["color"], 1)
    
    return display

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to pick points."""
    if event == cv2.EVENT_LBUTTONDOWN:
        picked_points[current_class_idx].append((x, y))
        class_name = CLASSES[current_class_idx]["name"]
        count = len(picked_points[current_class_idx])
        print(f"✓ {class_name} point #{count}: ({x}, {y})")
        cv2.imshow("HoloLens Point Picker", draw_ui(frame))

# Main loop
print("\n" + "="*90)
print(" HoloLens Point Picker - COLOR CODED")
print("="*90)
print(f"\nFrame size: {w}x{h}")
print("\nHOW TO USE:")
print("  1. Press [1], [2], or [3] to SELECT which class to pick points for")
print("  2. The HUGE COLOR BOX at the top shows your CURRENT SELECTION")
print("  3. LEFT-CLICK on the video frame to pick a point")
print("  4. All your picked points appear as colored circles")
print("  5. Point summary shows how many points you've picked for each class")
print("\nKEY MAPPINGS:")
print("  [1] = RED box = ROVER")
print("  [2] = BLUE box = OBSTACLE")
print("  [3] = GREEN box = FLOOR")
print("  [U] = Undo the last point")
print("  [S] = Save points and exit")
print("  [Q] = Quit without saving")
print("\n" + "="*90 + "\n")

cv2.imshow("HoloLens Point Picker", draw_ui(frame))
cv2.setMouseCallback("HoloLens Point Picker", mouse_callback)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('1'):  # Select Rover
        current_class_idx = 0
        print(f"\n→→→ SWITCHED TO: ROVER (RED) ←←←")
        cv2.imshow("HoloLens Point Picker", draw_ui(frame))
    
    elif key == ord('2'):  # Select Obstacle
        current_class_idx = 1
        print(f"\n→→→ SWITCHED TO: OBSTACLE (BLUE) ←←←")
        cv2.imshow("HoloLens Point Picker", draw_ui(frame))
    
    elif key == ord('3'):  # Select Floor
        current_class_idx = 2
        print(f"\n→→→ SWITCHED TO: FLOOR (GREEN) ←←←")
        cv2.imshow("HoloLens Point Picker", draw_ui(frame))
    
    elif key == ord('u'):  # Undo
        if picked_points[current_class_idx]:
            removed = picked_points[current_class_idx].pop()
            class_name = CLASSES[current_class_idx]["name"]
            print(f"↶ Removed last {class_name} point: {removed}")
            cv2.imshow("HoloLens Point Picker", draw_ui(frame))
        else:
            print("No points to undo for this class")
    
    elif key == ord('s'):  # Save and exit
        print("\n[SAVE] Saving picks...")
        break
    
    elif key == ord('q'):  # Quit without saving
        print("\n[QUIT] Discarding all picks...")
        picked_points = {i: [] for i in range(len(CLASSES))}
        break

cv2.destroyAllWindows()

# Save the picked points
if any(picked_points.values()):
    with open(output_file, "w") as f:
        for class_id in range(len(CLASSES)):
            class_name = CLASSES[class_id]["name"]
            if picked_points[class_id]:
                f.write(f"# {class_name}\n")
                for x, y in picked_points[class_id]:
                    f.write(f"({x}, {y})\n")
    
    print("\n" + "="*90)
    print(f"✓ SAVED: {output_file}")
    print("="*90)
    for class_id in range(len(CLASSES)):
        count = len(picked_points[class_id])
        if count > 0:
            class_name = CLASSES[class_id]["name"]
            color_info = {
                0: "RED",
                1: "BLUE",
                2: "GREEN"
            }[class_id]
            print(f"  [{color_info}] {class_name}: {count} points")
    print("="*90 + "\n")
else:
    print("\n! No points picked.\n")
