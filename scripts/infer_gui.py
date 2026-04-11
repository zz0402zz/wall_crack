from __future__ import annotations

import sys
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dam_crack_unet.inference import run_inference


DEFAULT_CHECKPOINT = ROOT / "runs/unetpp_b2_v2/checkpoints/best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/infer_results"
VIEW_KEYS = {
    "Mask": "mask_path",
    "Overlay": "overlay_path",
    "Outline": "outline_path",
    "Compare": "compare_path",
}


class InferGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Dam Crack Inference")
        self.geometry("1200x900")

        self.selected_image = tk.StringVar()
        self.checkpoint_path = tk.StringVar(value=str(DEFAULT_CHECKPOINT))
        self.output_dir = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.status_text = tk.StringVar(value="Select an image, then click Run Inference.")
        self.view_name = tk.StringVar(value="Overlay")

        self.last_report: dict[str, str | int | float] | None = None
        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        top = ttk.LabelFrame(container, text="Inputs", padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Image").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.selected_image, width=90).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Choose", command=self.choose_image).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="Checkpoint").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.checkpoint_path, width=90).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(top, text="Choose", command=self.choose_checkpoint).grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Label(top, text="Output Dir").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.output_dir, width=90).grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(top, text="Choose", command=self.choose_output_dir).grid(row=2, column=2, padx=4, pady=(8, 0))

        ttk.Button(top, text="Run Inference", command=self.run_clicked).grid(row=3, column=0, columnspan=3, pady=(12, 0), sticky="ew")
        top.columnconfigure(1, weight=1)

        view_box = ttk.LabelFrame(container, text="Views", padding=12)
        view_box.pack(fill="x", pady=(12, 0))
        for idx, name in enumerate(VIEW_KEYS):
            ttk.Radiobutton(view_box, text=name, value=name, variable=self.view_name, command=self.refresh_view).grid(
                row=0, column=idx, padx=8, sticky="w"
            )

        ttk.Label(container, textvariable=self.status_text).pack(anchor="w", pady=(12, 6))

        image_frame = ttk.Frame(container)
        image_frame.pack(fill="both", expand=True)
        self.image_label = ttk.Label(image_frame, anchor="center")
        self.image_label.pack(fill="both", expand=True)

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
            initialdir=str(ROOT / "data"),
        )
        if path:
            self.selected_image.set(path)

    def choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt")],
            initialdir=str(ROOT / "runs"),
        )
        if path:
            self.checkpoint_path.set(path)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output directory", initialdir=str(ROOT / "outputs"))
        if path:
            self.output_dir.set(path)

    def run_clicked(self) -> None:
        image_path = Path(self.selected_image.get()).expanduser()
        checkpoint = Path(self.checkpoint_path.get()).expanduser()
        output_dir = Path(self.output_dir.get()).expanduser()

        if not image_path.exists():
            messagebox.showerror("Missing image", "Please choose a valid image file.")
            return
        if not checkpoint.exists():
            messagebox.showerror("Missing checkpoint", "Please choose a valid checkpoint file.")
            return

        self.status_text.set("Running inference...")
        self.update_idletasks()

        try:
            self.last_report = run_inference(
                checkpoint=checkpoint,
                image_path=image_path,
                output_dir=output_dir,
            )
        except Exception as exc:
            messagebox.showerror("Inference failed", str(exc))
            self.status_text.set("Inference failed.")
            return

        self.status_text.set(
            f"Done. positive_ratio={self.last_report['positive_ratio']:.4f}  view={self.view_name.get()}"
        )
        self.refresh_view()

    def refresh_view(self) -> None:
        if not self.last_report:
            return

        path = Path(str(self.last_report[VIEW_KEYS[self.view_name.get()]]))
        if not path.exists():
            self.status_text.set(f"Missing output: {path}")
            return

        image = Image.open(path).convert("RGB")
        max_w = max(self.image_label.winfo_width(), 800)
        max_h = max(self.image_label.winfo_height(), 600)
        image.thumbnail((max_w, max_h))
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo)


def main() -> None:
    app = InferGui()
    app.mainloop()


if __name__ == "__main__":
    main()
