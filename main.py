import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


WEIGHTS_PAL_NTSC = (0.299, 0.587, 0.114)
WEIGHTS_HDTV = (0.2126, 0.7152, 0.0722)

FORMULA_PAL_NTSC = "Y' = 0.299 R + 0.587 G + 0.114 B"
FORMULA_HDTV = "Y' = 0.2126 R + 0.7152 G + 0.0722 B"


def rgb_to_gray(rgb, weights):
    red_weight, green_weight, blue_weight = weights
    intensity = (
        rgb[..., 0] * red_weight
        + rgb[..., 1] * green_weight
        + rgb[..., 2] * blue_weight
    )
    return np.clip(np.floor(intensity + 0.5), 0, 255).astype(np.uint8)


def process_rgb(rgb):
    gray_pal_ntsc = rgb_to_gray(rgb, WEIGHTS_PAL_NTSC)
    gray_hdtv = rgb_to_gray(rgb, WEIGHTS_HDTV)
    difference = np.abs(
        gray_pal_ntsc.astype(np.int16) - gray_hdtv.astype(np.int16)
    ).astype(np.uint8)
    return {
        "gray_pal_ntsc": gray_pal_ntsc,
        "gray_hdtv": gray_hdtv,
        "difference": difference,
        "hist_pal_ntsc": np.bincount(gray_pal_ntsc.ravel(), minlength=256),
        "hist_hdtv": np.bincount(gray_hdtv.ravel(), minlength=256),
    }


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def to_photo(array, max_width, max_height):
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    image.thumbnail((max(max_width, 2), max(max_height, 2)), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def draw_histogram(canvas, counts, title):
    canvas.delete("all")
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    if width < 40 or height < 40:
        return

    left, right, top, bottom = 52, 12, 28, 28
    plot_width = width - left - right
    plot_height = height - top - bottom
    if plot_width < 2 or plot_height < 2:
        return

    peak = max(int(counts.max()), 1)
    x1, y1 = left, top
    x2, y2 = left + plot_width, top + plot_height
    canvas.create_rectangle(x1, y1, x2, y2, outline="#b0b0b0")
    canvas.create_text(width / 2, 14, text=title)

    bins = len(counts)
    for index, count in enumerate(counts):
        bar_x0 = left + index / bins * plot_width
        bar_x1 = left + (index + 1) / bins * plot_width
        bar_y = top + (peak - int(count)) / peak * plot_height
        canvas.create_rectangle(bar_x0, bar_y, bar_x1, y2, fill="#4a4a4a", outline="")

    canvas.create_text(x1, y2 + 14, text="0", anchor=tk.W)
    canvas.create_text((x1 + x2) / 2, y2 + 14, text="128")
    canvas.create_text(x2, y2 + 14, text="255", anchor=tk.E)
    canvas.create_text(x1 - 6, y1, text=str(peak), anchor=tk.E)
    canvas.create_text(x1 - 6, y2, text="0", anchor=tk.E)


class ImagePanel(ttk.Frame):
    def __init__(self, master, title):
        super().__init__(master, padding=4)
        ttk.Label(self, text=title, anchor=tk.CENTER, justify=tk.CENTER).pack(fill=tk.X)
        self.view = tk.Label(self, bg="#e6e6e6")
        self.view.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.photo = None

    def show(self, array, max_width, max_height):
        self.photo = to_photo(array, max_width, max_height)
        self.view.configure(image=self.photo, text="")

    def clear(self, text):
        self.photo = None
        self.view.configure(image="", text=text)


class GrayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная работа 2")
        self.root.geometry("1180x760")
        self.root.minsize(900, 620)
        self.resize_job = None
        self.rgb = None
        self.result = None

        controls = ttk.Frame(root, padding=(8, 8, 8, 0))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Открыть изображение", command=self.open_image).pack(
            side=tk.LEFT
        )
        ttk.Button(controls, text="Сохранить", command=self.save_results).pack(
            side=tk.LEFT, padx=8
        )
        self.path_label = ttk.Label(controls, text="изображение не выбрано")
        self.path_label.pack(side=tk.LEFT, padx=8)

        self.images_frame = ttk.Frame(root, padding=(8, 8, 8, 0))
        self.images_frame.pack(fill=tk.BOTH, expand=True)
        for column in range(4):
            self.images_frame.columnconfigure(column, weight=1)
        self.images_frame.rowconfigure(0, weight=1)

        self.panel_source = ImagePanel(self.images_frame, "Исходное RGB")
        self.panel_pal = ImagePanel(self.images_frame, "PAL / NTSC\n" + FORMULA_PAL_NTSC)
        self.panel_hdtv = ImagePanel(self.images_frame, "HDTV\n" + FORMULA_HDTV)
        self.panel_diff = ImagePanel(self.images_frame, "Разность")
        self.panel_source.grid(row=0, column=0, sticky="nsew")
        self.panel_pal.grid(row=0, column=1, sticky="nsew")
        self.panel_hdtv.grid(row=0, column=2, sticky="nsew")
        self.panel_diff.grid(row=0, column=3, sticky="nsew")

        self.hists_frame = ttk.Frame(root, padding=8)
        self.hists_frame.pack(fill=tk.BOTH, expand=True)
        self.hists_frame.columnconfigure(0, weight=1)
        self.hists_frame.columnconfigure(1, weight=1)
        self.hists_frame.rowconfigure(0, weight=1)

        self.hist_pal = tk.Canvas(self.hists_frame, bg="white", highlightthickness=0)
        self.hist_hdtv = tk.Canvas(self.hists_frame, bg="white", highlightthickness=0)
        self.hist_pal.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.hist_hdtv.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        self.root.bind("<Configure>", self.on_resize)
        self.root.after(100, self.redraw)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Открыть изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        try:
            rgb = load_rgb(path)
        except Exception as error:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл.\n{error}")
            return
        self.rgb = rgb
        self.result = process_rgb(rgb)
        self.path_label.configure(text=path)
        self.redraw()

    def save_results(self):
        if self.result is None:
            messagebox.showinfo("Сохранение", "Сначала откройте изображение.")
            return
        directory = filedialog.askdirectory(title="Куда сохранить результаты")
        if not directory:
            return
        try:
            Image.fromarray(self.result["gray_pal_ntsc"], mode="L").save(
                os.path.join(directory, "gray_pal_ntsc.png")
            )
            Image.fromarray(self.result["gray_hdtv"], mode="L").save(
                os.path.join(directory, "gray_hdtv.png")
            )
            Image.fromarray(self.result["difference"], mode="L").save(
                os.path.join(directory, "difference.png")
            )
        except Exception as error:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файлы.\n{error}")
            return
        messagebox.showinfo(
            "Сохранено",
            "Записаны gray_pal_ntsc.png, gray_hdtv.png и difference.png.",
        )

    def redraw(self):
        self.root.update_idletasks()
        if self.rgb is None:
            for panel in (self.panel_source, self.panel_pal, self.panel_hdtv, self.panel_diff):
                panel.clear("Откройте изображение")
            self.hist_pal.delete("all")
            self.hist_hdtv.delete("all")
            return

        panel_width = max(self.images_frame.winfo_width() // 4 - 16, 2)
        panel_height = max(self.images_frame.winfo_height() - 64, 2)
        self.panel_source.show(self.rgb, panel_width, panel_height)
        self.panel_pal.show(self.result["gray_pal_ntsc"], panel_width, panel_height)
        self.panel_hdtv.show(self.result["gray_hdtv"], panel_width, panel_height)
        self.panel_diff.show(self.result["difference"], panel_width, panel_height)
        draw_histogram(
            self.hist_pal,
            self.result["hist_pal_ntsc"],
            "Гистограмма интенсивности, PAL / NTSC",
        )
        draw_histogram(
            self.hist_hdtv,
            self.result["hist_hdtv"],
            "Гистограмма интенсивности, HDTV",
        )

    def on_resize(self, event):
        if event.widget is not self.root:
            return
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self.redraw)


if __name__ == "__main__":
    root = tk.Tk()
    GrayApp(root)
    root.mainloop()
