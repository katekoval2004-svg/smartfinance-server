#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Покращення вирізок чека перед Claude (OpenCV): прибираємо тіні й нерівне
освітлення, «проявляємо» вицвілий термодрук, прибираємо шум, додаємо різкість.

YOLO отримує ОРИГІНАЛ: на обробленому фото вона знаходить менше полів/товарів,
бо тренувалась на звичайних кольорових фото. Вирівнювання перспективи робить
сканер документів у застосунку."""
import cv2
import numpy as np
from PIL import Image


def enhance(bgr):
    """Рівне освітлення + контраст + різкість, результат — 3-канальний сірий."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Фон паперу: розширюємо (прибирає текст) і сильно розмиваємо -> ділимо на нього.
    ksize = max(15, (min(gray.shape) // 40) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
    bg = cv2.medianBlur(bg, 21)
    flat = cv2.divide(gray, bg, scale=255)
    # Злегка затемнюємо «вицвілий» термодрук, не чіпаючи білий папір.
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)
    lut = np.clip(255 * (np.arange(256) / 255.0) ** 1.3, 0, 255).astype(np.uint8)
    flat = cv2.LUT(flat, lut)
    flat = cv2.fastNlMeansDenoising(flat, None, h=7, templateWindowSize=7, searchWindowSize=15)
    blur = cv2.GaussianBlur(flat, (0, 0), 1.2)
    sharp = cv2.addWeighted(flat, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def enhance_pil(img: Image.Image) -> Image.Image:
    """enhance() для PIL-вирізки (те, що бачить Claude)."""
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return Image.fromarray(cv2.cvtColor(enhance(bgr), cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    # python preprocess.py фото.jpg [папка] — зберегти покращену версію для перегляду
    import sys, os
    from PIL import ImageOps
    try:
        import pillow_heif; pillow_heif.register_heif_opener()
    except Exception:
        pass
    src, out_dir = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ".")
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    name = os.path.splitext(os.path.basename(src))[0]
    enhance_pil(im).save(os.path.join(out_dir, f"{name}_enhanced.jpg"), quality=92)
