import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def extract_channels(rgb):
    names = ("R", "G", "B")
    channels = {}
    histograms = {}
    for i, name in enumerate(names):
        channel = rgb[..., i].copy()
        channels[name] = channel
        histograms[name] = np.bincount(channel.ravel(), minlength=256)
    return channels, histograms


def channel_as_color(rgb, index):
    result = np.zeros_like(rgb)
    result[..., index] = rgb[..., index]
    return result


def to_photo(array, max_width, max_height):
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    image.thumbnail(
        (max(max_width, 2), max(max_height, 2)),
        Image.Resampling.LANCZOS,
    )
    return ImageTk.PhotoImage(image)


def draw_histogram(canvas, counts, title, bar_color="#4a4a4a"):
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
        canvas.create_rectangle(
            bar_x0, bar_y, bar_x1, y2,
            fill=bar_color, outline="",
        )

    canvas.create_text(x1, y2 + 14, text="0", anchor=tk.W)
    canvas.create_text((x1 + x2) / 2, y2 + 14, text="128")
    canvas.create_text(x2, y2 + 14, text="255", anchor=tk.E)
    canvas.create_text(x1 - 6, y1, text=str(peak), anchor=tk.E)
    canvas.create_text(x1 - 6, y2, text="0", anchor=tk.E)


class ImagePanel(ttk.Frame):
    def __init__(self, master, title, bg="#e6e6e6"):
        super().__init__(master, padding=4)
        ttk.Label(
            self, text=title, anchor=tk.CENTER, justify=tk.CENTER
        ).pack(fill=tk.X)
        self.view = tk.Label(self, bg=bg)
        self.view.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.photo = None

    def show(self, array, max_width, max_height):
        self.photo = to_photo(array, max_width, max_height)
        self.view.configure(image=self.photo, text="")

    def clear(self, text):
        self.photo = None
        self.view.configure(image="", text=text)


class ChannelsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная работа 2 — пункт 2: каналы R, G, B")
        self.root.geometry("1180x760")
        self.root.minsize(900, 620)

        self.rgb = None
        self.channels = None
        self.histograms = None
        self.resize_job = None

        controls = ttk.Frame(root, padding=(8, 8, 8, 0))
        controls.pack(fill=tk.X)

        ttk.Button(
            controls,
            text="Открыть изображение",
            command=self.open_image,
        ).pack(side=tk.LEFT)

        ttk.Button(
            controls,
            text="Сохранить каналы",
            command=self.save_channels,
        ).pack(side=tk.LEFT, padx=8)

        self.path_label = ttk.Label(controls, text="изображение не выбрано")
        self.path_label.pack(side=tk.LEFT, padx=8)

        top = ttk.Frame(root, padding=(8, 8, 8, 0))
        top.pack(fill=tk.BOTH, expand=True)
        for col in range(3):
            top.columnconfigure(col, weight=1)
        top.rowconfigure(0, weight=1)

        self.panel_r = ImagePanel(top, "Канал R", bg="#ffe0e0")
        self.panel_g = ImagePanel(top, "Канал G", bg="#e0ffe0")
        self.panel_b = ImagePanel(top, "Канал B", bg="#e0e0ff")
        self.panel_r.grid(row=0, column=0, sticky="nsew", padx=4)
        self.panel_g.grid(row=0, column=1, sticky="nsew", padx=4)
        self.panel_b.grid(row=0, column=2, sticky="nsew", padx=4)

        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill=tk.BOTH, expand=True)
        for col in range(3):
            bottom.columnconfigure(col, weight=1)
        bottom.rowconfigure(0, weight=1)

        self.hist_r = tk.Canvas(bottom, bg="white", highlightthickness=0)
        self.hist_g = tk.Canvas(bottom, bg="white", highlightthickness=0)
        self.hist_b = tk.Canvas(bottom, bg="white", highlightthickness=0)
        self.hist_r.grid(row=0, column=0, sticky="nsew", padx=4)
        self.hist_g.grid(row=0, column=1, sticky="nsew", padx=4)
        self.hist_b.grid(row=0, column=2, sticky="nsew", padx=4)

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
        self.channels, self.histograms = extract_channels(rgb)
        self.path_label.configure(text=path)
        self.redraw()

    def save_channels(self):
        if self.channels is None:
            messagebox.showinfo("Сохранение", "Сначала откройте изображение.")
            return
        directory = filedialog.askdirectory(title="Куда сохранить каналы")
        if not directory:
            return
        try:
            for name, array in self.channels.items():
                Image.fromarray(array, mode="L").save(
                    os.path.join(directory, f"channel_{name}.png")
                )
        except Exception as error:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файлы.\n{error}")
            return
        messagebox.showinfo(
            "Сохранено",
            "Записаны channel_R.png, channel_G.png и channel_B.png.",
        )

    def redraw(self):
        self.root.update_idletasks()

        if self.rgb is None:
            for panel in (self.panel_r, self.panel_g, self.panel_b):
                panel.clear("Откройте изображение")
            self.hist_r.delete("all")
            self.hist_g.delete("all")
            self.hist_b.delete("all")
            return

        panel_width = max(self.root.winfo_width() // 3 - 24, 2)
        panel_height = max(self.root.winfo_height() // 2 - 80, 2)

        self.panel_r.show(channel_as_color(self.rgb, 0), panel_width, panel_height)
        self.panel_g.show(channel_as_color(self.rgb, 1), panel_width, panel_height)
        self.panel_b.show(channel_as_color(self.rgb, 2), panel_width, panel_height)

        draw_histogram(self.hist_r, self.histograms["R"],
                       "Гистограмма канала R", bar_color="#c0392b")
        draw_histogram(self.hist_g, self.histograms["G"],
                       "Гистограмма канала G", bar_color="#27ae60")
        draw_histogram(self.hist_b, self.histograms["B"],
                       "Гистограмма канала B", bar_color="#2c3e99")

    def on_resize(self, event):
        if event.widget is not self.root:
            return
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self.redraw)


if __name__ == "__main__":
    root = tk.Tk()
    ChannelsApp(root)
    root.mainloop()