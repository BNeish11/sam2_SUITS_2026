from __future__ import annotations

from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


IMAGES_DIR = Path("rover pictures")
OUTPUT_FILE = Path("picked_rover_image_points_with_neg.txt")


def load_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


class PointPicker:
    def __init__(self, root: tk.Tk, image_paths: list[Path]) -> None:
        self.root = root
        self.image_paths = image_paths
        self.index = 0
        self.mode = "pos"
        self.points: dict[str, dict[str, list[tuple[int, int]]]] = {
            p.name: {"pos": [], "neg": []} for p in image_paths
        }

        self.root.title("Rover Point Picker")
        self.root.geometry("1280x900")

        top = tk.Frame(root, bg="#1f1f1f")
        top.pack(fill=tk.X)

        self.info = tk.Label(top, text="", font=("Arial", 14, "bold"), fg="white", bg="#1f1f1f")
        self.info.pack(side=tk.LEFT, padx=12, pady=8)

        self.mode_label = tk.Label(top, text="", font=("Arial", 14, "bold"), fg="white")
        self.mode_label.pack(side=tk.RIGHT, padx=12, pady=8)

        self.canvas = tk.Canvas(root, bg="gray20", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        bottom = tk.Frame(root, bg="#1f1f1f")
        bottom.pack(fill=tk.X)

        tk.Button(bottom, text="Prev", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(bottom, text="Next", command=self.next_image, width=10).pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(bottom, text="Save", command=self.save, width=10).pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(bottom, text="Save & Exit", command=self.save_and_exit, width=12).pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(bottom, text="Undo", command=self.undo, width=10).pack(side=tk.LEFT, padx=4, pady=6)

        help_text = (
            "Keys: 1=POS, 2=NEG, ←/→=navigate, U=undo, S=save, Q=save & exit. "
            "Click to add a point."
        )
        tk.Label(bottom, text=help_text, bg="#1f1f1f", fg="#cccccc").pack(side=tk.RIGHT, padx=10)

        self.canvas.bind("<Button-1>", self.on_click)
        root.bind("1", lambda _e: self.set_mode("pos"))
        root.bind("2", lambda _e: self.set_mode("neg"))
        root.bind("u", lambda _e: self.undo())
        root.bind("s", lambda _e: self.save())
        root.bind("q", lambda _e: self.save_and_exit())
        root.bind("Left", lambda _e: self.prev_image())
        root.bind("Right", lambda _e: self.next_image())

        self.photo = None
        self.display_scale = 1.0
        self.current_image_rgb = None
        self.display_width = 0
        self.display_height = 0
        self.update_view()

    @property
    def current_path(self) -> Path:
        return self.image_paths[self.index]

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update_view()

    def current_points(self) -> dict[str, list[tuple[int, int]]]:
        return self.points[self.current_path.name]

    def load_current_image(self) -> None:
        bgr = cv2.imread(str(self.current_path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {self.current_path}")
        self.current_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def update_view(self) -> None:
        self.load_current_image()
        assert self.current_image_rgb is not None

        img = self.current_image_rgb.copy()
        h, w = img.shape[:2]
        max_w, max_h = 1220, 720
        self.display_scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = int(w * self.display_scale), int(h * self.display_scale)
        self.display_width = disp_w
        self.display_height = disp_h
        disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        for x, y in self.current_points()["pos"]:
            cx, cy = int(x * self.display_scale), int(y * self.display_scale)
            cv2.circle(disp, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(disp, (cx, cy), 8, (255, 255, 255), 2)
        for x, y in self.current_points()["neg"]:
            cx, cy = int(x * self.display_scale), int(y * self.display_scale)
            cv2.circle(disp, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(disp, (cx, cy), 8, (255, 255, 255), 2)

        self.photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.info.config(text=f"{self.index + 1}/{len(self.image_paths)}  {self.current_path.name}")
        self.mode_label.config(
            text=f"MODE: {'POS' if self.mode == 'pos' else 'NEG'}",
            bg="#006600" if self.mode == "pos" else "#880000",
        )

    def on_click(self, event: tk.Event) -> None:
        if event.x < 0 or event.y < 0:
            return
        if event.x >= self.display_width or event.y >= self.display_height:
            return

        assert self.current_image_rgb is not None
        h, w = self.current_image_rgb.shape[:2]
        x = int(np.clip(event.x / self.display_scale, 0, w - 1))
        y = int(np.clip(event.y / self.display_scale, 0, h - 1))
        self.current_points()[self.mode].append((x, y))
        self.update_view()

    def undo(self) -> None:
        bucket = self.current_points()[self.mode]
        if bucket:
            bucket.pop()
            self.update_view()

    def prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.update_view()

    def next_image(self) -> None:
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.update_view()

    def save(self) -> None:
        lines: list[str] = []
        for image_path in self.image_paths:
            data = self.points[image_path.name]
            if not data["pos"] and not data["neg"]:
                continue
            lines.append(f"# IMAGE {image_path.name}")
            if data["pos"]:
                lines.append("# POS")
                for x, y in data["pos"]:
                    lines.append(f"({x}, {y})")
            if data["neg"]:
                lines.append("# NEG")
                for x, y in data["neg"]:
                    lines.append(f"({x}, {y})")
            lines.append("")

        OUTPUT_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"Saved {OUTPUT_FILE}")

    def save_and_exit(self) -> None:
        self.save()
        self.root.destroy()


def main() -> None:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGES_DIR}")

    image_paths = load_images(IMAGES_DIR)
    if not image_paths:
        raise RuntimeError(f"No images found in: {IMAGES_DIR}")

    root = tk.Tk()
    PointPicker(root, image_paths)
    root.mainloop()


if __name__ == "__main__":
    main()