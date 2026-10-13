import os
from flask import Flask, request, jsonify
import anthropic
from dotenv import load_dotenv
from flask_cors import CORS
from datetime import datetime
import threading

load_dotenv('ai.env')

# ── шляхи до моделей (файли лежать поряд із цим server.py) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CARD_DET_WEIGHTS", os.path.join(_HERE, "card_detector.pt"))
os.environ.setdefault("CARD_OCR_WEIGHTS", os.path.join(_HERE, "card_ocr_crnn.pt"))
os.environ.setdefault("RECEIPT_DET_WEIGHTS", os.path.join(_HERE, "best.pt"))  # YOLO чека

# 280 с × 2 спроби = 560 с < 600 с тайм-ауту gunicorn: повільний запит закінчиться помилкою, а не вбитим воркером
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=280, max_retries=1)

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """
Ти - SmartFinance AI, персональний фінансовий помічник.

Твої основні задачі:
1. Аналіз витрат користувача — допомагай розуміти, на що йдуть гроші, виявляй надмірні витрати, пропонуй де можна заощадити.
2. Фінансові цілі — допомагай ставити реалістичні цілі (накопичення, погашення боргів, великі покупки), розраховуй терміни та необхідні суми.
3. Фінансові консультації — давай конкретні практичні поради щодо бюджетування, заощаджень, планування витрат.

Якщо у запиті є розділ "Контекст користувача" — ОБОВ'ЯЗКОВО використовуй ці дані:
- Проаналізуй рахунки (wallets): баланси, витрати та доходи по категоріях за весь час.
- Якщо є financialGoal — відстежуй прогрес до цілі, порівнюй поточні заощадження з необхідними, пропонуй конкретні кроки.
- Звертай увагу на незвичні витрати, перевитрати по категоріях, можливості для економії.
- Якщо бачиш витрати на алкоголь, тютюн — можеш тактовно вказати на їх вплив на бюджет.

Як працювати з користувачем:
- Якщо користувач описує свої витрати — проаналізуй їх, визнач категорії, вкажи де витрачається найбільше і що можна оптимізувати.
- Якщо користувач описує фінансову ціль — допоможи скласти план досягнення з конкретними цифрами і кроками.
- Якщо користувач просить пораду — дай чіткі, практичні рекомендації з урахуванням його ситуації.
- Завжди задавай уточнюючі питання, якщо бракує даних для якісної відповіді.

Відповідай українською мовою. Будь конкретним і практичним, уникай загальних фраз. Не використовуй жирний шрифт в жодному разі!!!
Якщо питання не стосується фінансів — ввічливо поверни розмову до фінансових тем.
"""


# Застосунок шле: <усі рахунки й транзакції> + цей маркер + питання користувача.
QUESTION_MARKER = "=== ЗАПИТ КОРИСТУВАЧА ==="


def _split_for_cache(user_text):
    """Історію (усі транзакції) кешуємо: поки дані не змінились, наступні питання читають її
    з кешу Claude — швидше й ~10× дешевше. Питання йде окремим блоком після точки кешу."""
    idx = user_text.rfind(QUESTION_MARKER)
    if idx <= 0:
        return user_text
    return [
        {"type": "text", "text": user_text[:idx], "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": user_text[idx:]},
    ]


def get_claude_response(user_text):
    """Отримання фінансової відповіді від Claude. Помилки пробрасуються — /chat поверне 500."""
    try:
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=7000,
            # Для чату «medium» помітно швидше за типовий «high» і достатньо для порад
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _split_for_cache(user_text)}
            ]
        )
    except Exception as e:
        print(f"Помилка запиту до Claude: {e}")
        raise
    u = message.usage
    print(f"   Claude: кеш прочитано={u.cache_read_input_tokens or 0}, записано={u.cache_creation_input_tokens or 0}, "
          f"без кешу={u.input_tokens}, відповідь={u.output_tokens}", flush=True)
    text = "".join(block.text for block in message.content if block.type == "text").strip()
    if not text:
        raise RuntimeError(f"Claude повернув порожню відповідь (stop_reason={message.stop_reason})")
    return text


@app.route('/chat', methods=['POST'])
def chat():
    request_timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n📨 [{request_timestamp}] Отримано новий запит на /chat")

    try:
        if request.is_json:
            data = request.get_json()
            user_text = data.get('text', '')
        else:
            user_text = request.form.get('text', '')

        print(f"📝 Текст запиту: '{user_text[:80]}{'...' if len(user_text) > 80 else ''}'")

        if not user_text.strip():
            return jsonify({
                "status": "error",
                "error": "Порожній запит",
                "request_timestamp": request_timestamp
            }), 400

        print("🤖 Відправляємо запит до Claude...")
        ai_response = get_claude_response(user_text)

        response_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"✅ [{response_timestamp}] Відповідь надіслано")

        return jsonify({
            "status": "success",
            "response": ai_response,
            "request_timestamp": request_timestamp,
            "response_timestamp": response_timestamp
        })

    except Exception as e:
        error_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"❌ [{error_timestamp}] Помилка в ендпоінті /chat: {e}")
        return jsonify({
            "status": "error",
            "error": f"Помилка сервера: {str(e)}",
            "request_timestamp": request_timestamp,
            "error_timestamp": error_timestamp
        }), 500


# ==================== СКАНУВАННЯ КАРТКИ (локально, без Claude) ====================
@app.route('/scan-card', methods=['POST'])
def scan_card_route():
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n💳 [{ts}] Запит на /scan-card")
    f = request.files.get('image')
    if f is None:
        return jsonify({"status": "error", "error": "Немає файлу 'image'"}), 400
    try:
        from card_ocr import scan_card   # ліниво: моделі вантажаться при першому виклику
        result = scan_card(f.read())
        print(f"   -> {result}")
        return jsonify({"status": "success", **result})
    except Exception as e:
        print(f"❌ scan-card помилка: {e}")
        return jsonify({"status": "error", "error": f"Помилка сервера: {str(e)}"}), 500

@app.route('/scan-receipt', methods=['POST'])
def scan_receipt_route():
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n🧾 [{ts}] Запит на /scan-receipt")
    f = request.files.get('image')
    if f is None:
        return jsonify({"status": "error", "error": "Немає файлу 'image'"}), 400
    try:
        from infer import scan_receipt   # твоя YOLO (infer.py) + Claude по кропах
        result = scan_receipt(f.read())
        print(f"   -> merchant={result.get('merchant')} items={len(result.get('items', []))}"
              f"{' error=' + str(result['_error']) if result.get('_error') else ''}", flush=True)
        return jsonify({"status": "success", **result})
    except Exception as e:
        print(f"❌ scan-receipt помилка: {e}")
        return jsonify({"status": "error", "error": f"Помилка сервера: {str(e)}"}), 500


def _warmup():
    """Вантажимо YOLO одразу після старту, а не на першому запиті (інакше він може не вкластися в тайм-аут застосунку)."""
    try:
        from infer import get_model
        get_model()
        print("✅ YOLO чека завантажено", flush=True)
    except Exception as e:
        print(f"⚠️ Не вдалося завантажити YOLO заздалегідь: {e}", flush=True)

threading.Thread(target=_warmup, daemon=True).start()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "claude_configured": bool(os.getenv("ANTHROPIC_API_KEY"))
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)