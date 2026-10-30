#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BackgroundRemover - десктоп програма за махане на фон от снимки.

Стартиране от изходен код:   python app.py
Тест на GUI-то без обработка: python app.py --smoke
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from typing import List, Optional

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk

import core
from core import DEFAULT_MODEL, MODELS

POLL_MS = 100
PREVIEW_PADDING = 8

IMAGE_TYPES = [
    ("Снимки", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("Всички файлове", "*.*"),
]


def checkerboard(width: int, height: int, cell: int = 14) -> Image.Image:
    """Шахматен фон, за да се вижда прозрачността."""
    yy, xx = np.indices((height, width))
    mask = ((xx // cell + yy // cell) % 2).astype(bool)
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[:] = (255, 255, 255)
    arr[mask] = (198, 198, 198)
    return Image.fromarray(arr, "RGB")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("BackgroundRemover - махане на фон от снимки")
        root.geometry("1120x720")
        root.minsize(900, 560)

        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.items: List[dict] = []
        self.running = False
        self.preview_index: Optional[int] = None
        self.preview_mode = tk.StringVar(value="orig")  # orig | result
        self.model_choice = tk.StringVar(value=MODELS[DEFAULT_MODEL].label)
        self.status_text = tk.StringVar(value="Готово. Отворете снимки, за да започнете.")
        self.progress_value = tk.DoubleVar(value=0.0)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._preview_job: Optional[str] = None
        self.model_key = DEFAULT_MODEL

        self._build_toolbar(root)
        self._build_body(root)
        self._build_statusbar(root)

        root.bind("<Configure>", self._on_configure)
        root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self, parent: tk.Tk) -> None:
        bar = ttk.Frame(parent, padding=(8, 8, 8, 4))
        bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(bar, text="Отвори снимки...", command=self.on_open)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_run = ttk.Button(bar, text="Премахни фона", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT, padx=(6, 0))

        self.btn_save = ttk.Button(bar, text="Запиши резултатите...", command=self.on_save)
        self.btn_save.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        ttk.Label(bar, text="Модел:").pack(side=tk.LEFT)
        box = ttk.Combobox(
            bar,
            state="readonly",
            width=46,
            textvariable=self.model_choice,
            values=[m.label for m in MODELS.values()],
        )
        box.pack(side=tk.LEFT, padx=(6, 0))
        box.bind("<<ComboboxSelected>>", self._on_model_change)

    def _build_body(self, parent: tk.Tk) -> None:
        body = ttk.Frame(parent, padding=(8, 4))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # --- списък със снимки ---
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)

        ttk.Label(left, text="Снимки").grid(row=0, column=0, sticky="w")
        self.listbox = tk.Listbox(left, selectmode=tk.EXTENDED, width=38, activestyle="none")
        self.listbox.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        scroll.grid(row=1, column=1, sticky="ns", pady=(4, 0))
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        btns = ttk.Frame(left)
        btns.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btns, text="Премахни от списъка", command=self.on_remove).pack(
            side=tk.LEFT
        )
        ttk.Button(btns, text="Изчисти", command=self.on_clear).pack(side=tk.LEFT, padx=(6, 0))

        # --- превю ---
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        top = ttk.Frame(right)
        top.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Radiobutton(
            top, text="Оригинал", value="orig", variable=self.preview_mode,
            command=self.refresh_preview,
        ).pack(side=tk.LEFT)
        self.rb_result = ttk.Radiobutton(
            top, text="Без фон (прозрачен)", value="result", variable=self.preview_mode,
            command=self.refresh_preview, state=tk.DISABLED,
        )
        self.rb_result.pack(side=tk.LEFT, padx=(10, 0))
        self.size_label = ttk.Label(top, text="")
        self.size_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(right, highlightthickness=1, highlightbackground="#b0b0b0",
                                background="#4a4a4a")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _build_statusbar(self, parent: tk.Tk) -> None:
        bar = ttk.Frame(parent, padding=(8, 4, 8, 8))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.pbar = ttk.Progressbar(bar, variable=self.progress_value, maximum=100.0)
        self.pbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(bar, textvariable=self.status_text, anchor="w").pack(fill=tk.X)

    def _on_configure(self, _event) -> None:
        # преоразмеряването праща много събития - обработваме ги накуп
        if self._preview_job is not None:
            try:
                root_after_cancel = self.root.after_cancel
                root_after_cancel(self._preview_job)
            except tk.TclError:
                pass
        self._preview_job = self.root.after(120, self.refresh_preview)

    # ------------------------------------------------------------ действия
    def on_open(self) -> None:
        paths = filedialog.askopenfilenames(title="Изберете снимки", filetypes=IMAGE_TYPES)
        if not paths:
            return
        added = 0
        existing = {it["path"].lower() for it in self.items}
        for p in paths:
            if p.lower() in existing:
                continue
            self.items.append(
                {"path": p, "name": os.path.basename(p), "status": "заредена", "result": None}
            )
            self.listbox.insert(tk.END, f"{os.path.basename(p)}  -  заредена")
            added += 1
        self.status_text.set(f"Добавени снимки: {added}.")
        if self.preview_index is None and self.items:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self._on_select()

    def on_remove(self) -> None:
        if self.running:
            return
        sel = list(self.listbox.curselection())
        for idx in reversed(sel):
            self.listbox.delete(idx)
            del self.items[idx]
        if self.preview_index is not None and (not self.items or self.preview_index >= len(self.items)):
            self.preview_index = None
            self.canvas.delete("all")
        self.rb_result.configure(state=tk.DISABLED)

    def on_clear(self) -> None:
        if self.running:
            return
        self.listbox.delete(0, tk.END)
        self.items.clear()
        self.preview_index = None
        self.preview_mode.set("orig")
        self.rb_result.configure(state=tk.DISABLED)
        self.canvas.delete("all")
        self.status_text.set("Списъкът е изчистен.")

    def on_run(self) -> None:
        if self.running:
            return
        if not self.items:
            messagebox.showinfo("BackgroundRemover", "Първо отворете снимки.")
            return

        sel = list(self.listbox.curselection())
        pool = sel if sel else list(range(len(self.items)))
        queue_idx = [i for i in pool if self.items[i]["result"] is None]
        if not queue_idx:
            messagebox.showinfo(
                "BackgroundRemover",
                "Избраните снимки вече са обработени. Сменете модела, за да се обработат наново.",
            )
            return

        self.running = True
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_open.configure(state=tk.DISABLED)
        self.progress_value.set(0.0)
        self.status_text.set("Обработка...")
        threading.Thread(
            target=self._worker, args=(queue_idx, self.model_key), daemon=True
        ).start()

    def on_save(self) -> None:
        done = [(i, it) for i, it in enumerate(self.items) if it["result"] is not None]
        if not done:
            messagebox.showinfo(
                "BackgroundRemover", "Няма обработени снимки. Натиснете \"Премахни фона\"."
            )
            return
        folder = filedialog.askdirectory(title="Къде да се запишат резултатите?")
        if not folder:
            return
        written = 0
        for idx, item in done:
            stem = os.path.splitext(item["name"])[0]
            out = os.path.join(folder, stem + "_без_фон.png")
            try:
                item["result"].save(out, "PNG")
                item["status"] = "записана"
                self._set_row(idx, item)
                written += 1
            except OSError as exc:
                self.status_text.set(f"Грешка при запис: {exc}")
        self.status_text.set(f"Записани файлове: {written} в {folder}")

    # ------------------------------------------------------------ worker
    def _worker(self, indices: List[int], model: str) -> None:
        total = len(indices)
        first = True

        def model_progress(frac: float, text: str) -> None:
            self.events.put(("model_progress", frac * 100.0, text))

        for n, idx in enumerate(indices):
            item = self.items[idx]
            self.events.put(("status", idx, "обработва се..."))
            try:
                with Image.open(item["path"]) as im:
                    im.load()
                    im = ImageOps.exif_transpose(im)
                    result = core.remove_background(
                        im, model, model_progress if first else None
                    )
                first = False
                item["result"] = result
                item["status"] = "готова"
                self.events.put(("status", idx, "готова"))
                self.events.put(("preview", idx))
            except Exception:
                traceback.print_exc()
                msg = "грешка"
                self.events.put(("status", idx, msg))
                self.events.put(("error", item["name"]))
            self.events.put(("progress", (n + 1) * 100.0 / total))

        self.events.put(("done", total))

    def _poll(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "status":
                    _, idx, text = ev
                    if 0 <= idx < len(self.items):
                        self.items[idx]["status"] = text
                        self._set_row(idx, self.items[idx])
                elif kind == "progress":
                    self.progress_value.set(ev[1])
                    self.status_text.set(
                        f"Обработка... {int(ev[1])}%"
                    )
                elif kind == "model_progress":
                    self.progress_value.set(ev[1])
                    self.status_text.set(ev[2])
                elif kind == "preview":
                    idx = ev[1]
                    if self.preview_index is None:
                        self.listbox.selection_clear(0, tk.END)
                        self.listbox.selection_set(idx)
                        self.preview_index = idx
                    if self.preview_index == idx:
                        self.preview_mode.set("result")
                        self.rb_result.configure(state=tk.NORMAL)
                        self.refresh_preview()
                elif kind == "error":
                    self.status_text.set(f"Грешка при обработка на {ev[1]}.")
                elif kind == "done":
                    self.running = False
                    self.btn_run.configure(state=tk.NORMAL)
                    self.btn_open.configure(state=tk.NORMAL)
                    ok = sum(1 for it in self.items if it["result"] is not None)
                    self.progress_value.set(100.0)
                    self.status_text.set(
                        f"Готово. Обработени снимки: {ok} от {len(self.items)}. "
                        "Натиснете \"Запиши резултатите...\"."
                    )
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------ помощни
    def _set_row(self, idx: int, item: dict) -> None:
        if 0 <= idx < self.listbox.size():
            self.listbox.delete(idx)
            self.listbox.insert(idx, f"{item['name']}  -  {item['status']}")

    def _on_model_change(self, _event=None) -> None:
        label = self.model_choice.get()
        for key, info in MODELS.items():
            if info.label == label:
                if key != self.model_key:
                    self.model_key = key
                    # старите резултати са с друг модел - маркираме ги за нанова обработка
                    for i, item in enumerate(self.items):
                        if item["result"] is not None:
                            item["result"] = None
                            item["status"] = "заредена"
                            self._set_row(i, item)
                    self.rb_result.configure(state=tk.DISABLED)
                    self.preview_mode.set("orig")
                    self.refresh_preview()
                    self.status_text.set(f"Избран модел: {info.label}")
                break

    def _on_select(self, _event=None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        self.preview_index = sel[0]
        item = self.items[self.preview_index]
        if item["result"] is None:
            self.preview_mode.set("orig")
            self.rb_result.configure(state=tk.DISABLED)
        else:
            self.rb_result.configure(state=tk.NORMAL)
            self.preview_mode.set("result")
        self.refresh_preview()

    def refresh_preview(self) -> None:
        if self.preview_index is None or self.preview_index >= len(self.items):
            return
        item = self.items[self.preview_index]
        show_result = self.preview_mode.get() == "result" and item["result"] is not None

        try:
            with Image.open(item["path"]) as im:
                im.load()
                base = ImageOps.exif_transpose(im).convert("RGBA")
        except Exception:
            return

        if show_result:
            base = item["result"].convert("RGBA")
        self.size_label.configure(
            text=f"{base.width} x {base.height} px" + ("  •  прозрачно" if show_result else "")
        )

        cw = max(self.canvas.winfo_width(), 64) - 2 * PREVIEW_PADDING
        ch = max(self.canvas.winfo_height(), 64) - 2 * PREVIEW_PADDING
        if cw <= 0 or ch <= 0:
            return

        scale = min(cw / base.width, ch / base.height, 1.0)
        disp = base if scale == 1.0 else base.resize(
            (max(int(base.width * scale), 1), max(int(base.height * scale), 1)),
            Image.Resampling.LANCZOS,
        )

        if show_result:
            bg = checkerboard(disp.width, disp.height)
            bg.paste(disp, (0, 0), disp)
            disp = bg

        self.canvas.delete("all")
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self._photo,
            anchor=tk.CENTER,
        )


def _selftest() -> int:
    """Проверка на замръзналия exe: вграден модел + ONNX изчисление + PNG запис."""
    import tempfile

    from PIL import ImageDraw

    tmp = tempfile.mkdtemp(prefix="bgremover-selftest-")
    os.environ["BACKGROUNDREMOVER_MODELS"] = os.path.join(tmp, "models")

    src = os.path.join(tmp, "input.png")
    im = Image.new("RGB", (320, 240), (30, 120, 200))
    ImageDraw.Draw(im).ellipse((90, 40, 230, 200), fill=(245, 210, 40))
    im.save(src)

    out = core.process_file(src)

    def log(msg: str) -> None:
        if sys.stdout is not None:
            print(msg)

    with Image.open(out) as res:
        if res.mode != "RGBA" or res.size != (320, 240):
            log("SELFTEST FAIL: лош формат")
            return 1
        alpha = np.asarray(res.getchannel("A"))
        if alpha.min() != 0 or alpha.max() != 255:
            log("SELFTEST FAIL: маската не е двойствена")
            return 1
    log("SELFTEST OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        try:
            return _selftest()
        except Exception:
            traceback.print_exc()
            return 1

    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    if "--smoke" in sys.argv:
        root.after(400, root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
