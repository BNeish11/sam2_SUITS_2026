import cv2

# --- CONFIG ---
video_path = "LittleBuddy.mp4"
# ---------------

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Could not read video: {video_path}")

print("Click on the rover in the first frame. Press ESC to quit.")

# Fit the frame to the screen (keeps aspect ratio)
screen_w, screen_h = 1280, 720
h, w = frame.shape[:2]
scale = min(screen_w / w, screen_h / h)
disp_w, disp_h = int(w * scale), int(h * scale)
frame_disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

clicked = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        clicked.append((orig_x, orig_y))
        print(f"Clicked: (x={orig_x}, y={orig_y})")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", on_mouse)

while True:
    display = frame_disp.copy()
    for (x, y) in clicked[-5:]:
        dx, dy = int(x * scale), int(y * scale)
        cv2.circle(display, (dx, dy), 6, (0, 255, 0), -1)
    cv2.imshow("Frame", display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

# Save clicked points to file
if clicked:
    with open("picked_points.txt", "w") as f:
        for x, y in clicked:
            f.write(f"({x}, {y})\n")
    print(f"\nSaved {len(clicked)} points to picked_points.txt")
    print(f"Last point: ({clicked[-1][0]}, {clicked[-1][1]})")
else:
    print("No points clicked")