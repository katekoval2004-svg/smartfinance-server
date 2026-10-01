#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Детектор-first: YOLO обводить поля/товари -> читаємо кожен бокс -> збираємо JSON.
CLI зберігає фото з боксами; функція scan_receipt() використовується сервером."""
import os, io, sys, json, glob, base64, argparse
from PIL import Image, ImageDraw, ImageFont, ImageOps
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except Exception:
    pass
from ultralytics import YOLO
import anthropic
from preprocess import enhance_pil

BASE = os.path.dirname(os.path.abspath(__file__))

# Беремо ключ саме з ai.env (а не зі старого ключа в оточенні терміналу)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, "ai.env"), override=True)
except Exception:
    pass
CATEGORIES = [
    "Автомобіль", "Банк", "Благодійність", "Держава", "Діти", "Дім", "Домашні тварини",
    "Здоров'я", "Інше", "Їда не вдома", "Краса", "Мобільний зв'язок", "Одяг і взуття",
    "Освіта", "Подарунки", "Подорожі", "Послуги для бізнесу", "Продукти харчування",
    "Розваги", "Техніка", "Транспорт", "Їжа", "Напої", "М'ясо", "Молочні продукти",
    "Хліб", "Солодощі", "Побутова хімія", "Комунальні послуги", "Спорт", "Підписки",
    "Поштові послуги", "Оплата за кредитом",
]
SCALAR = {"Title": "merchant", "Address": "location", "Date": "date",
          "TotalPrice": "total", "Tax": "vat", "OrderId": "order_number", "Subtotal": "subtotal"}
COLORS = {"Title": "#e6194B", "Address": "#4363d8", "Date": "#f58231", "TotalPrice": "#f032e6",
          "Tax": "#bfef45", "OrderId": "#469990", "Subtotal": "#42d4f4", "Item": "#000075"}

DEFAULT_CONF = 0.1
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
# Покращувати вирізки перед Claude (тіні, вицвілий термодрук). YOLO бачить оригінал:
# на обробленому фото вона знаходить менше товарів, бо тренувалась на звичайних фото.
ENHANCE_CROPS = os.environ.get("RECEIPT_ENHANCE", "1") != "0"

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua else 0

def nms(items, thr=0.45):
    items = sorted(items, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in items:
        if all(iou(d["box"], k["box"]) < thr for k in keep):
            keep.append(d)
    return keep

def b64_png(im):
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def latest_weights():
    return os.path.join(BASE, "best.pt")

# YOLO вантажимо один раз (важливо для сервера)
_model = None
def get_model(weights=None):
    global _model
    if _model is None:
        _model = YOLO(weights or latest_weights())
    return _model

def detect(img, conf=DEFAULT_CONF, weights=None):
    model = get_model(weights)
    res = model(img, conf=conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls = model.names[int(b.cls)]
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        dets.append({"cls": cls, "conf": round(float(b.conf), 2), "box": [x1, y1, x2, y2]})
    return dets

def organize(dets):
    """Найкращий бокс на кожне скалярне поле + відсортовані товари (без дублів)."""
    right = max(d["box"][2] for d in dets)
    scal = {}
    for d in dets:
        if d["cls"] in SCALAR and (d["cls"] not in scal or d["conf"] > scal[d["cls"]]["conf"]):
            scal[d["cls"]] = d
    items = nms([d for d in dets if d["cls"] == "Item"])
    items.sort(key=lambda d: d["box"][1])
    return scal, items, right

def build_content(img, scal, items, right, enhance=ENHANCE_CROPS):
    """Список кропів (полів і товарів) для Claude. ПОВНЕ фото не додаємо."""
    W, H = img.size
    content = []
    def add_crop(box, label, to_right=False, pad=6, extra_down=0):
        x1, y1, x2, y2 = box
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(W, (right+pad) if to_right else x2+pad); y2 = min(H, y2+pad+extra_down)
        crop = img.crop((x1, y1, x2, y2))
        if crop.width < 300:
            s = 300/crop.width; crop = crop.resize((int(crop.width*s), int(crop.height*s)), Image.LANCZOS)
        if enhance:
            crop = enhance_pil(crop)
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_png(crop)}})
        content.append({"type": "text", "text": label})
    for cls, d in scal.items():
        add_crop(d["box"], f"^ поле: {SCALAR[cls]}")
    for i, d in enumerate(items):
        # розширюємо кроп вниз ~на висоту рядка — щоб зловити ціну, якщо назва
        # товару перенеслась на другий рядок (напр. "...Dream Vio / let 59.90")
        h = d["box"][3] - d["box"][1]
        add_crop(d["box"], f"^ товар {i+1} (назва зліва, ціна справа; може бути 2 рядки)",
                 to_right=True, extra_down=int(h * 1.3))
    content.append({"type": "text", "text": f"""Вище — вирізки з чека, кожна підписана (поле або товар).
Прочитай текст кожної й поверни ЛИШЕ JSON:
{{
  "merchant": str, "location": str|null, "date": "YYYY-MM-DD"|null,
  "total": number|null, "vat": number|null, "order_number": str|null,
  "items": [ {{"name": str, "qty": number, "price": number, "category": str}} ]
}}
Правила:
- merchant — торгова назва магазину (напр. "EVA"), не юридична особа. Бренди латиницею як надруковано.
- Для кожного товару: name зліва, price (ціна за позицію) справа у тому ж рядку.
- qty: якщо у вирізці видно множник "N шт X ціна" — це N; інакше qty = 1. "Nшт" у назві — фасування, не кількість.
- Виправляй лише явні помилки OCR/переноси; не вигадуй нечіткі слова. Без штрих-кодів і цін усередині назви.
- category — рівно одне зі списку: {', '.join(CATEGORIES)}.
- Чого немає — null (для items — [])."""})
    return content

def read_with_claude(content, model_name=DEFAULT_MODEL):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Немає ANTHROPIC_API_KEY — перевір ai.env поряд з infer.py")
    msg = anthropic.Anthropic(api_key=api_key).messages.create(
        model=model_name, max_tokens=8192,
        messages=[{"role": "user", "content": content}])
    text = "".join(x.text for x in msg.content if x.type == "text").strip()
    s = text[text.find("{"): text.rfind("}")+1]
    return json.loads(s)

# ---- ФУНКЦІЯ ДЛЯ СЕРВЕРА ----
def scan_receipt(image_bytes, conf=DEFAULT_CONF, claude_model=DEFAULT_MODEL, enhance=ENHANCE_CROPS):
    """Байти зображення -> JSON чека (детекція YOLO + читання кропів Claude)."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")  # застосувати орієнтацію з телефона
    dets = detect(img, conf=conf)
    empty = {"merchant": None, "location": None, "date": None, "total": None,
             "vat": None, "order_number": None, "items": []}
    if not dets:
        empty["_error"] = "no_boxes"
        return empty
    scal, items, right = organize(dets)
    content = build_content(img, scal, items, right, enhance=enhance)
    try:
        data = read_with_claude(content, claude_model)
    except Exception as e:
        empty["_error"] = str(e)
        return empty
    data.setdefault("items", [])
    for k in ("merchant", "location", "date", "total", "vat", "order_number"):
        data.setdefault(k, None)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--annotated", default="annotated.png", help="куди зберегти фото з боксами")
    a = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Немає ключа: перевір ai.env поряд з infer.py")

    img = Image.open(a.image); img = ImageOps.exif_transpose(img).convert("RGB"); W, H = img.size
    dets = detect(img, conf=a.conf, weights=a.weights)
    if not dets:
        sys.exit("Модель не знайшла жодного боксу.")
    scal, items, right = organize(dets)

    # намалювати бокси й зберегти (для перегляду)
    vis = img.copy(); dr = ImageDraw.Draw(vis)
    try: fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", max(12, W//80))
    except Exception: fnt = ImageFont.load_default()
    def draw(box, label, color):
        dr.rectangle(box, outline=color, width=3)
        dr.text((box[0]+2, max(0, box[1]-16)), label, fill=color, font=fnt)
    for cls, d in scal.items(): draw(d["box"], cls, COLORS.get(cls, "#f00"))
    # малюємо РЕАЛЬНУ вирізку товару (до правого краю + вниз) — саме її бачить Claude
    for i, d in enumerate(items):
        x1, y1, x2, y2 = d["box"]; h = y2 - y1
        draw([x1, y1, min(W, right + 6), min(H, int(y2 + h * 1.3))], f"Item{i+1}", COLORS["Item"])
    os.makedirs(os.path.dirname(a.annotated) or ".", exist_ok=True); vis.save(a.annotated)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"Ключ: ...{api_key[-6:]}")
    content = build_content(img, scal, items, right)
    try:
        data = read_with_claude(content, a.model)
    except Exception as e:
        sys.exit(str(e))
    data["_boxes"] = {"scalar": list(scal.keys()), "items": len(items)}
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\n(бокси намальовано у {a.annotated})", file=sys.stderr)

if __name__ == "__main__":
    main()