import os
from flask import Flask, request, jsonify
import anthropic
from dotenv import load_dotenv
from flask_cors import CORS
from datetime import datetime

load_dotenv('ai.env')

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
Якщо питання не стосується фінансів — ввічливо поверни розмову до фінансових тем.89
"""


def get_claude_response(user_text):
    """Отримання фінансової відповіді від Claude"""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_text}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Помилка запиту до Claude: {e}")
        return f"На жаль, сталася помилка при обробці запиту: {str(e)}"


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

        print(f"🤖 Відправляємо запит до Claude...")
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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "claude_configured": bool(os.getenv("ANTHROPIC_API_KEY"))
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)