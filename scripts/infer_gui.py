from __future__ import annotations

import sys
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
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
}


class InferGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("大坝裂缝智能识别系统")
        self.geometry("1440x920")
        self.minsize(1240, 780)
        self.configure(bg="#EEF3F8")
        self.font_family = self._resolve_font_family()

        self.selected_image = tk.StringVar()
        self.view_name = tk.StringVar(value="Overlay")
        self.damage_text = tk.StringVar(value="未识别")
        self.ratio_text = tk.StringVar(value="--")
        self.summary_text = tk.StringVar(value="识别结论：待检测")

        self.last_report: dict[str, str | int | float] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._original_photo: ImageTk.PhotoImage | None = None

        self._setup_style()
        self._build_ui()

    def _resolve_font_family(self) -> str:
        available = set(tkfont.families(self))
        for candidate in ("PingFang SC", "Microsoft YaHei", "Heiti SC", "Arial Unicode MS", "Arial"):
            if candidate in available:
                return candidate
        return "TkDefaultFont"

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#EEF3F8")
        style.configure("Panel.TFrame", background="#FFFFFF")
        style.configure("Card.TLabelframe", background="#FFFFFF", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background="#FFFFFF", foreground="#102A43", font=(self.font_family, 12, "bold"))
        style.configure("Title.TLabel", background="#EEF3F8", foreground="#102A43", font=(self.font_family, 20, "bold"))
        style.configure("SubTitle.TLabel", background="#EEF3F8", foreground="#486581", font=(self.font_family, 10))
        style.configure("Info.TLabel", background="#FFFFFF", foreground="#243B53", font=(self.font_family, 10))
        style.configure(
            "TButton",
            font=(self.font_family, 11, "bold"),
            padding=(10, 10),
            foreground="#FFFFFF",
            background="#1D4ED8",
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "TButton",
            background=[("active", "#1E40AF"), ("pressed", "#1E3A8A")],
            foreground=[("disabled", "#D1D5DB"), ("!disabled", "#FFFFFF")],
        )
        style.configure("Run.TButton", font=(self.font_family, 11, "bold"), padding=(10, 10))
        style.configure("View.TRadiobutton", background="#FFFFFF", foreground="#102A43", font=(self.font_family, 10, "bold"))

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=14, style="App.TFrame")
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="大坝裂缝智能识别系统", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="支持单图识别、结果切换查看和破损程度可视化展示", style="SubTitle.TLabel").pack(
            anchor="w", pady=(4, 0)
        )

        content = ttk.Frame(container, style="App.TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.configure(width=430)
        left.grid_propagate(False)

        input_box = ttk.LabelFrame(left, text="输入配置", padding=12, style="Card.TLabelframe")
        input_box.pack(fill="x")

        ttk.Button(input_box, text="选择图片", command=self.choose_image).pack(fill="x")

        ttk.Button(input_box, text="开始识别", command=self.run_clicked, style="Run.TButton").pack(fill="x", pady=(12, 0))

        original_box = ttk.LabelFrame(left, text="原图预览", padding=12, style="Card.TLabelframe")
        original_box.pack(fill="x", pady=(12, 0))
        original_canvas = tk.Frame(original_box, bg="#FFFFFF", width=390, height=320)
        original_canvas.pack(fill="both", expand=True)
        original_canvas.pack_propagate(False)
        self.original_image_label = ttk.Label(original_canvas, anchor="center", background="#FFFFFF")
        self.original_image_label.pack(fill="both", expand=True)
        self.original_image_label.configure(text="请选择图片", font=(self.font_family, 14), foreground="#64748B")

        result_box = ttk.LabelFrame(left, text="识别结果", padding=12, style="Card.TLabelframe")
        result_box.pack(fill="x", pady=(12, 0))

        self.damage_badge = tk.Label(
            result_box,
            textvariable=self.damage_text,
            bg="#94A3B8",
            fg="white",
            font=(self.font_family, 16, "bold"),
            padx=14,
            pady=10,
        )
        self.damage_badge.pack(fill="x")

        info_panel = tk.Frame(result_box, bg="#FFFFFF")
        info_panel.pack(fill="x", pady=(12, 0))
        self._make_info_row(info_panel, "面积占比", self.ratio_text)
        self._make_info_row(info_panel, "识别结论", self.summary_text)

        right = ttk.Frame(content, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        view_box = ttk.LabelFrame(right, text="结果视图", padding=12, style="Card.TLabelframe")
        view_box.grid(row=0, column=0, sticky="ew")
        for idx, name in enumerate(VIEW_KEYS):
            ttk.Radiobutton(
                view_box,
                text=name,
                value=name,
                variable=self.view_name,
                command=self.refresh_view,
                style="View.TRadiobutton",
            ).grid(
                row=0, column=idx, padx=8, sticky="w"
            )

        image_frame = ttk.Frame(right, style="Panel.TFrame")
        image_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        result_canvas = tk.Frame(image_frame, bg="#FFFFFF", width=900, height=700)
        result_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        result_canvas.pack_propagate(False)
        self.image_label = ttk.Label(result_canvas, anchor="center", background="#FFFFFF")
        self.image_label.pack(fill="both", expand=True)

    def _make_info_row(self, parent: tk.Frame, title: str, value_var: tk.StringVar) -> None:
        row = tk.Frame(parent, bg="#FFFFFF")
        row.pack(fill="x", pady=4)
        tk.Label(row, text=f"{title}:", bg="#FFFFFF", fg="#486581", font=(self.font_family, 10, "bold")).pack(side="left")
        tk.Label(row, textvariable=value_var, bg="#FFFFFF", fg="#102A43", font=(self.font_family, 11)).pack(side="right")

    def _update_badge_color(self, damage_level: str) -> None:
        color_map = {
            "轻度": "#16A34A",
            "中度": "#EA580C",
            "重度": "#DC2626",
        }
        self.damage_badge.configure(bg=color_map.get(damage_level, "#64748B"))

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
            initialdir=str(ROOT / "data"),
        )
        if path:
            self.selected_image.set(path)
            self._show_original_preview(Path(path))

    def run_clicked(self) -> None:
        image_path = Path(self.selected_image.get()).expanduser()
        checkpoint = DEFAULT_CHECKPOINT
        output_dir = DEFAULT_OUTPUT_DIR

        if not image_path.exists():
            messagebox.showerror("缺少图片", "请选择有效的图片文件。")
            return
        if not checkpoint.exists():
            messagebox.showerror("缺少权重", "请选择有效的模型权重文件。")
            return

        self.update_idletasks()

        try:
            self.last_report = run_inference(
                checkpoint=checkpoint,
                image_path=image_path,
                output_dir=output_dir,
            )
        except Exception as exc:
            messagebox.showerror("识别失败", str(exc))
            return

        damage_level = str(self.last_report["damage_level"])
        positive_ratio = float(self.last_report["positive_ratio"]) * 100
        self.damage_text.set(f"破损程度：{damage_level}")
        self.ratio_text.set(f"{positive_ratio:.2f}%")
        self.summary_text.set("检测到疑似裂缝区域" if positive_ratio > 0 else "未检测到明显裂缝区域")
        self._update_badge_color(damage_level)
        self.refresh_view()

    def _show_original_preview(self, path: Path) -> None:
        if not path.exists():
            return
        image = Image.open(path).convert("RGB")
        max_w = 390
        max_h = 320
        image.thumbnail((max_w, max_h))
        self._original_photo = ImageTk.PhotoImage(image)
        self.original_image_label.configure(image=self._original_photo, text="")

    def refresh_view(self) -> None:
        if not self.last_report:
            return

        path = Path(str(self.last_report[VIEW_KEYS[self.view_name.get()]]))
        if not path.exists():
            return

        image = Image.open(path).convert("RGB")
        max_w = max(self.image_label.winfo_width(), 900)
        max_h = max(self.image_label.winfo_height(), 700)
        image.thumbnail((max_w, max_h))
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo)


def main() -> None:
    app = InferGui()
    app.mainloop()


if __name__ == "__main__":
    main()
