#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BackgroundRemover - ядро за махане на фон от снимки (ONNX модели U2Net).

Модулът е самостоятелен (без GUI) и може да се тества от командния ред:

    python core.py input.jpg -o out.png --model u2netp
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

APP_NAME = "BackgroundRemover"

# Нормализация по ImageNet - същата като при оригиналните U2Net сесии.
MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
INPUT_SIZE = (320, 320)

ProgressFn = Optional[Callable[[float, str], None]]


class DownloadError(Exception):
    """Неуспешно сваляне или невалиден модел."""


class ModelInfo:
    def __init__(self, key: str, label: str, url: str, md5: str, bundled: bool) -> None:
        self.key = key
        self.label = label
        self.url = url
        self.md5 = md5
        self.bundled = bundled

    @property
    def filename(self) -> str:
        return f"{self.key}.onnx"


# Достъпните модели. "bundled" = копирани са вътре в exe-то и работят офлайн.
MODELS: Dict[str, ModelInfo] = {
    "u2netp": ModelInfo(
        key="u2netp",
        label="Бързо (вградено, работи офлайн)",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        md5="8e83ca70e441ab06c318d82300c84806",
        bundled=True,
    ),
    "u2net": ModelInfo(
        key="u2net",
        label="Качествено (u2net, сваля се веднъж - 176 MB)",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        md5="60024c5c889badc19c04ad937298a77b",
        bundled=False,
    ),
}

DEFAULT_MODEL = "u2netp"


def resource_path(*parts: str) -> str:
    """Път към ресурс, вграден в exe-то (или до изходния файл при стартиране от код)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def models_dir() -> str:
    """Постоянна папка за моделите (до 100 MB - не стои в Temp)."""
    override = os.environ.get("BACKGROUNDREMOVER_MODELS")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), APP_NAME)
    path = os.path.join(root, APP_NAME, "models")
    os.makedirs(path, exist_ok=True)
    return path


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(info: ModelInfo, dest: str, progress: ProgressFn) -> None:
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(info.url, headers={"User-Agent": APP_NAME})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        frac = (done / total) if total else 0.0
                        progress(
                            min(frac, 1.0),
                            f"Сваляне на модела... {done // (1 << 20)} / "
                            f"{total // (1 << 20) if total else '?'} MB",
                        )
        if info.md5 and _md5(tmp) != info.md5:
            raise DownloadError("Контролната сума на сваления модел не съвпада.")
        shutil.move(tmp, dest)
    except Exception as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        if isinstance(exc, DownloadError):
            raise
        raise DownloadError(f"Моделът не можа да се свали: {exc}") from exc


def ensure_model(key: str, progress: ProgressFn = None) -> str:
    """Връща готов път до .onnx модела, сваляйки/копирайки го при нужда."""
    if key not in MODELS:
        raise DownloadError(f"Непознат модел: {key}")
    info = MODELS[key]

    dest = os.path.join(models_dir(), info.filename)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest

    bundled = resource_path("models", info.filename)
    if os.path.isfile(bundled) and os.path.getsize(bundled) > 0:
        if progress:
            progress(1.0, "Разопаковане на вградения модел...")
        shutil.copyfile(bundled, dest)
        return dest

    if progress:
        progress(0.0, f"Сваляне на модела {info.key}...")
    _download(info, dest, progress)
    if progress:
        progress(1.0, "Моделът е готов.")
    return dest


# ---------------------------------------------------------------------------
# Инференция
# ---------------------------------------------------------------------------

_SESSIONS: Dict[str, ort.InferenceSession] = {}


def get_session(key: str, progress: ProgressFn = None) -> ort.InferenceSession:
    session = _SESSIONS.get(key)
    if session is None:
        path = ensure_model(key, progress)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        session = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        _SESSIONS[key] = session
    return session


def _preprocess(img: Image.Image) -> np.ndarray:
    im = img.convert("RGB").resize(INPUT_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float64)
    arr = arr / max(float(np.max(arr)), 1e-6)
    out = np.zeros_like(arr)
    for c in range(3):
        out[:, :, c] = (arr[:, :, c] - MEAN[c]) / STD[c]
    out = out.transpose((2, 0, 1))
    return np.expand_dims(out, 0).astype(np.float32)


def _mask_from_output(raw: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    pred = raw[0, 0, :, :] if raw.ndim == 4 else raw[0]
    lo, hi = float(np.min(pred)), float(np.max(pred))
    pred = (pred - lo) / max(hi - lo, 1e-6)
    mask = Image.fromarray((pred.clip(0, 1) * 255).astype("uint8"), mode="L")
    return mask.resize(size, Image.Resampling.LANCZOS)


def remove_background(
    img: Image.Image, model: str = DEFAULT_MODEL, progress: ProgressFn = None
) -> Image.Image:
    """Премахва фона и връща RGBA изображение с прозрачност."""
    src = ImageOps.exif_transpose(img).convert("RGBA")

    session = get_session(model, progress)
    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: _preprocess(src)})[0]
    mask = _mask_from_output(raw, src.size)

    alpha = np.asarray(mask, dtype=np.uint8)
    original_alpha = np.asarray(src.getchannel("A"), dtype=np.uint16)
    alpha = (original_alpha * alpha.astype(np.uint16) // 255).astype("uint8")

    out = src.copy()
    out.putalpha(Image.fromarray(alpha, mode="L"))
    return out


def process_file(
    src_path: str, dst_path: Optional[str] = None, model: str = DEFAULT_MODEL,
    progress: ProgressFn = None,
) -> str:
    """Обработва файл и записва PNG с прозрачност. Връща пътя на резултата."""
    with Image.open(src_path) as im:
        im.load()
        result = remove_background(im, model, progress)

    if dst_path is None:
        stem, _ = os.path.splitext(src_path)
        dst_path = stem + "_без_фон.png"
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    result.save(dst_path, "PNG")
    return dst_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Махане на фон от снимка.")
    ap.add_argument("input", help="Входна снимка")
    ap.add_argument("-o", "--output", help="Изходен PNG файл")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                    help="Модел за сегментация")
    args = ap.parse_args()

    def cb(frac: float, text: str) -> None:
        print(f"[{frac * 100:5.1f}%] {text}")

    out = process_file(args.input, args.output, args.model, cb)
    print(f"Записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
