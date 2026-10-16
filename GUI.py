"""Black & White Image Colorization GUI.
Run from any working directory: python GUI.py
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PROTO_PATH = MODEL_DIR / "colorization_deploy_v2.prototxt"
MODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"
POINTS_PATH = BASE_DIR / "pts_in_hull.npy"

class ColorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("B&W Image Colorization")
        self.root.geometry("1100x700")
        self.root.minsize(800, 550)
        self.input_path = None
        self.input_photo = None
        self.output_photo = None
        self.output_path = None

        header = tk.Label(root, text="Black & White Image Colorization", font=("Segoe UI", 20, "bold"))
        header.pack(pady=(14, 6))
        tk.Label(root, text="Upload a grayscale photo and let the AI estimate its colors.", font=("Segoe UI", 10)).pack()

        controls = tk.Frame(root)
        controls.pack(pady=12)
        tk.Button(controls, text="1. Upload Image", width=18, command=self.upload_image).pack(side="left", padx=6)
        self.color_button = tk.Button(controls, text="2. Colorize Image", width=18, command=self.colorize, state="disabled")
        self.color_button.pack(side="left", padx=6)
        self.save_button = tk.Button(controls, text="Save Result As...", width=16, command=self.save_result, state="disabled")
        self.save_button.pack(side="left", padx=6)

        panels = tk.Frame(root)
        panels.pack(fill="both", expand=True, padx=14, pady=8)
        left = tk.LabelFrame(panels, text="Original Image", padx=8, pady=8)
        right = tk.LabelFrame(panels, text="Colorized Result", padx=8, pady=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 7))
        right.pack(side="left", fill="both", expand=True, padx=(7, 0))
        self.input_label = tk.Label(left, text="Choose an image to begin", bg="#eeeeee", anchor="center")
        self.input_label.pack(fill="both", expand=True)
        self.output_label = tk.Label(right, text="Your result will appear here", bg="#eeeeee", anchor="center")
        self.output_label.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="Ready. Model files are stored in the models folder.")
        tk.Label(root, textvariable=self.status, anchor="w", relief="sunken", padx=8).pack(fill="x", side="bottom")

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select a photo",
            initialdir=str(BASE_DIR),
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with Image.open(path) as im:
                im.load()
                preview = im.convert("RGB")
            self.input_path = path
            self._show_image(preview, self.input_label, "input_photo")
            self.output_label.configure(image="", text="Your result will appear here")
            self.output_photo = None
            self.save_button.configure(state="disabled")
            self.color_button.configure(state="normal")
            self.status.set(f"Selected: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))

    def _show_image(self, image, label, attr):
        width = max(label.winfo_width(), 450)
        height = max(label.winfo_height(), 380)
        image = image.copy()
        image.thumbnail((width - 20, height - 20), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        setattr(self, attr, photo)

    def colorize(self):
        if not self.input_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        missing = [str(p) for p in (PROTO_PATH, MODEL_PATH, POINTS_PATH) if not p.is_file()]
        if missing:
            message = "Required model file(s) are missing:\n\n" + "\n".join(missing)
            message += "\n\nRun download_model.bat or: python download_model.py"
            messagebox.showerror("Model files missing", message)
            return
        self.color_button.configure(state="disabled")
        self.status.set("Colorizing image... please wait.")
        threading.Thread(target=self._colorize_worker, daemon=True).start()

    def _colorize_worker(self):
        try:
            frame = cv2.imread(self.input_path)
            if frame is None:
                raise ValueError("OpenCV could not read this image. Try another JPG or PNG file.")
            points = np.load(str(POINTS_PATH))
            net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))
            points = points.transpose().reshape(2, 313, 1, 1)
            net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
            net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            resized_l = cv2.resize(l_channel, (224, 224)) - 50
            net.setInput(cv2.dnn.blobFromImage(resized_l))
            ab = net.forward()[0].transpose((1, 2, 0))
            ab = cv2.resize(ab, (rgb.shape[1], rgb.shape[0]))
            lab_out = np.concatenate((l_channel[:, :, None], ab), axis=2)
            rgb_out = np.clip(cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB), 0, 1)
            result = Image.fromarray((rgb_out * 255).astype(np.uint8))
            out_path = BASE_DIR / "result.png"
            result.save(out_path)
            self.root.after(0, self._colorize_done, result, str(out_path), None)
        except Exception as exc:
            self.root.after(0, self._colorize_done, None, None, str(exc))

    def _colorize_done(self, result, out_path, error):
        self.color_button.configure(state="normal")
        if error:
            self.status.set("Colorization failed.")
            messagebox.showerror("Colorization error", error)
            return
        self.output_path = out_path
        self._show_image(result, self.output_label, "output_photo")
        self.save_button.configure(state="normal")
        self.status.set(f"Done. Result saved to: {out_path}")
        messagebox.showinfo("Finished", f"Colorized image saved to:\n{out_path}")

    def save_result(self):
        if not self.output_path or not Path(self.output_path).exists():
            return
        path = filedialog.asksaveasfilename(
            title="Save colorized image", defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")],
            initialfile="colorized_image.png"
        )
        if path:
            try:
                Image.open(self.output_path).save(path)
                self.status.set(f"Saved: {path}")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizationApp(root)
    root.mainloop()
