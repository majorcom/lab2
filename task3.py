import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk


# ----------------------------------------------------------------------
# Преобразования RGB <-> HSV
# Реализовано вручную по формулам из презентации (слайды 25–26).
# Входные и выходные массивы — float32 в диапазоне [0, 1],
# тон H — в градусах [0, 360).
# ----------------------------------------------------------------------

def rgb_to_hsv(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    max_c = np.max(rgb, axis=-1)
    min_c = np.min(rgb, axis=-1)
    diff = max_c - min_c

    h = np.zeros_like(max_c)

    # MAX = R
    mask_r = (max_c == r) & (diff != 0)
    mask_r_ge = mask_r & (g >= b)
    mask_r_lt = mask_r & (g < b)
    h[mask_r_ge] = 60.0 * ((g[mask_r_ge] - b[mask_r_ge]) / diff[mask_r_ge])
    h[mask_r_lt] = 60.0 * ((g[mask_r_lt] - b[mask_r_lt]) / diff[mask_r_lt]) + 360.0

    # MAX = G
    mask_g = (max_c == g) & (diff != 0)
    h[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120.0

    # MAX = B
    mask_b = (max_c == b) & (diff != 0)
    h[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240.0

    s = np.zeros_like(max_c)
    mask_max = max_c != 0
    s[mask_max] = 1.0 - min_c[mask_max] / max_c[mask_max]

    v = max_c

    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv):
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h_i = np.floor(h / 60.0).astype(np.int32) % 6
    f = h / 60.0 - np.floor(h / 60.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

    m0 = h_i == 0
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]

    m1 = h_i == 1
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]

    m2 = h_i == 2
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]

    m3 = h_i == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]

    m4 = h_i == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]

    m5 = h_i == 5
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]

    return np.stack([r, g, b], axis=-1)


# ----------------------------------------------------------------------
# Вспомогательные функции (в стиле main.py и task2_channels.py)
# ----------------------------------------------------------------------

def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def rgb_to_float(rgb_uint8):
    return rgb_uint8.astype(np.float32) / 255.0


def float_to_uint8(rgb_float):
    return np.clip(rgb_float * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_hsv_shift(rgb_float, dh, ds, dv):
    hsv = rgb_to_hsv(rgb_float)
    hsv[..., 0] = (hsv[..., 0] + dh) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * ds, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * dv, 0.0, 1.0)
    return hsv_to_rgb(hsv)


def hsv_channels_as_rgb(hsv):
    """Три псевдоцветных изображения: H — как тон, S и V — как grayscale."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    h_rgb = hsv_to_rgb(np.stack([h, np.ones_like(s), np.ones_like(v)], axis=-1))
    s_rgb = np.stack([s, s, s], axis=-1)
    v_rgb = np.stack([v, v, v], axis=-1)

    return h_rgb, s_rgb, v_rgb


def to_photo(array, max_width, max_height):
    if array.dtype != np.uint8:
        array = float_to_uint8(array)
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


# ----------------------------------------------------------------------
# Основное приложение
# ----------------------------------------------------------------------

class HsvApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная работа 2 — пункт 3: RGB ↔ HSV")
        self.root.geometry("1180x900")
        self.root.minsize(980, 760)

        self.rgb_original = None        # uint8, (H, W, 3)
        self.rgb_original_float = None  # float32, (H, W, 3), [0, 1]
        self.hsv_original = None        # float32, (H, W, 3)
        self.result_rgb = None          # float32, (H, W, 3), [0, 1]
        self.resize_job = None

        # -------- Кнопки управления --------
        controls = ttk.Frame(root, padding=(8, 8, 8, 0))
        controls.pack(fill=tk.X)

        ttk.Button(
            controls,
            text="Открыть изображение",
            command=self.open_image,
        ).pack(side=tk.LEFT)

        ttk.Button(
            controls,
            text="Сбросить",
            command=self.reset_sliders,
        ).pack(side=tk.LEFT, padx=8)

        ttk.Button(
            controls,
            text="Сохранить результат",
            command=self.save_result,
        ).pack(side=tk.LEFT)

        self.path_label = ttk.Label(controls, text="изображение не выбрано")
        self.path_label.pack(side=tk.LEFT, padx=8)

        # -------- Ползунки H, S, V --------
        sliders = ttk.LabelFrame(root, text="Коррекция HSV", padding=8)
        sliders.pack(fill=tk.X, padx=8, pady=8)

        self.hue_var = tk.DoubleVar(value=0.0)
        self.sat_var = tk.DoubleVar(value=1.0)
        self.val_var = tk.DoubleVar(value=1.0)

        self._add_slider(sliders, "Оттенок (H), °", self.hue_var, -180.0, 180.0)
        self._add_slider(sliders, "Насыщенность (S), ×", self.sat_var, 0.0, 2.0)
        self._add_slider(sliders, "Яркость (V), ×", self.val_var, 0.0, 2.0)

        # -------- Верхний ряд: исходное и результат --------
        top = ttk.Frame(root, padding=(8, 8, 8, 0))
        top.pack(fill=tk.BOTH, expand=True)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)

        self.panel_source = ImagePanel(top, "Исходное RGB")
        self.panel_result = ImagePanel(top, "Результат (HSV → RGB)")
        self.panel_source.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.panel_result.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # -------- Средний ряд: каналы H, S, V --------
        middle = ttk.Frame(root, padding=(8, 8, 8, 0))
        middle.pack(fill=tk.BOTH, expand=True)
        for col in range(3):
            middle.columnconfigure(col, weight=1)
        middle.rowconfigure(0, weight=1)

        self.panel_h = ImagePanel(middle, "Канал H (тон)")
        self.panel_s = ImagePanel(middle, "Канал S (насыщенность)")
        self.panel_v = ImagePanel(middle, "Канал V (яркость)")
        self.panel_h.grid(row=0, column=0, sticky="nsew", padx=4)
        self.panel_s.grid(row=0, column=1, sticky="nsew", padx=4)
        self.panel_v.grid(row=0, column=2, sticky="nsew", padx=4)

        # -------- Нижний ряд: гистограммы H, S, V --------
        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill=tk.BOTH, expand=True)
        for col in range(3):
            bottom.columnconfigure(col, weight=1)
        bottom.rowconfigure(0, weight=1)

        self.hist_h = tk.Canvas(bottom, bg="white", highlightthickness=0)
        self.hist_s = tk.Canvas(bottom, bg="white", highlightthickness=0)
        self.hist_v = tk.Canvas(bottom, bg="white", highlightthickness=0)
        self.hist_h.grid(row=0, column=0, sticky="nsew", padx=4)
        self.hist_s.grid(row=0, column=1, sticky="nsew", padx=4)
        self.hist_v.grid(row=0, column=2, sticky="nsew", padx=4)

        self.root.bind("<Configure>", self.on_resize)
        self.root.after(100, self.redraw)

    def _add_slider(self, parent, label, variable, from_, to):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text=label, width=24, anchor=tk.W).pack(side=tk.LEFT)

        scale = ttk.Scale(
            row,
            from_=from_,
            to=to,
            variable=variable,
            orient=tk.HORIZONTAL,
            command=lambda _=None: self.update_result(),
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        value_label = ttk.Label(row, width=8, anchor=tk.E)
        value_label.pack(side=tk.LEFT)

        def refresh_label(*_):
            value_label.configure(text=f"{variable.get():.2f}")

        variable.trace_add("write", refresh_label)
        refresh_label()

    # -------- Действия --------

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

        self.rgb_original = rgb
        self.rgb_original_float = rgb_to_float(rgb)
        self.hsv_original = rgb_to_hsv(self.rgb_original_float)
        self.path_label.configure(text=path)
        self.reset_sliders()

    def reset_sliders(self):
        self.hue_var.set(0.0)
        self.sat_var.set(1.0)
        self.val_var.set(1.0)
        self.update_result()

    def update_result(self):
        if self.rgb_original_float is None:
            return
        self.result_rgb = apply_hsv_shift(
            self.rgb_original_float,
            dh=self.hue_var.get(),
            ds=self.sat_var.get(),
            dv=self.val_var.get(),
        )
        self.redraw()

    def save_result(self):
        if self.result_rgb is None:
            messagebox.showinfo("Сохранение", "Сначала откройте изображение.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not path:
            return
        try:
            Image.fromarray(float_to_uint8(self.result_rgb), mode="RGB").save(path)
        except Exception as error:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл.\n{error}")
            return
        messagebox.showinfo("Сохранено", f"Файл записан:\n{path}")

    # -------- Отрисовка --------

    def redraw(self):
        self.root.update_idletasks()

        if self.rgb_original is None:
            for panel in (self.panel_source, self.panel_result,
                          self.panel_h, self.panel_s, self.panel_v):
                panel.clear("Откройте изображение")
            for canvas in (self.hist_h, self.hist_s, self.hist_v):
                canvas.delete("all")
            return

        width_top = max(self.panel_source.winfo_width() - 16, 2)
        height_top = max(self.panel_source.winfo_height() - 32, 2)

        self.panel_source.show(self.rgb_original, width_top, height_top)
        self.panel_result.show(self.result_rgb, width_top, height_top)

        # HSV-каналы результата
        hsv_result = rgb_to_hsv(self.result_rgb)
        h_rgb, s_rgb, v_rgb = hsv_channels_as_rgb(hsv_result)

        width_mid = max(self.panel_h.winfo_width() - 16, 2)
        height_mid = max(self.panel_h.winfo_height() - 32, 2)

        self.panel_h.show(h_rgb, width_mid, height_mid)
        self.panel_s.show(s_rgb, width_mid, height_mid)
        self.panel_v.show(v_rgb, width_mid, height_mid)

        # Гистограммы H, S, V
        h_uint8 = np.clip(hsv_result[..., 0] / 360.0 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        s_uint8 = np.clip(hsv_result[..., 1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        v_uint8 = np.clip(hsv_result[..., 2] * 255.0 + 0.5, 0, 255).astype(np.uint8)

        draw_histogram(
            self.hist_h,
            np.bincount(h_uint8.ravel(), minlength=256),
            "Гистограмма тона (H)",
            bar_color="#8e44ad",
        )
        draw_histogram(
            self.hist_s,
            np.bincount(s_uint8.ravel(), minlength=256),
            "Гистограмма насыщенности (S)",
            bar_color="#27ae60",
        )
        draw_histogram(
            self.hist_v,
            np.bincount(v_uint8.ravel(), minlength=256),
            "Гистограмма яркости (V)",
            bar_color="#4a4a4a",
        )

    def on_resize(self, event):
        if event.widget is not self.root:
            return
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self.redraw)


if __name__ == "__main__":
    root = tk.Tk()
    HsvApp(root)
    root.mainloop()