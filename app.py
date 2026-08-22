import os
import re

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "deepseek-ai/DeepSeek-V3-0324"
PROVIDER = "deepinfra"

# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    provider=PROVIDER,
    api_key=HF_TOKEN,
    timeout=90
)

# ============================================================
# EMOJI VOCABULARY
# ============================================================

EMOJI_LIST = """
😀 😃 😄 😁 😆 😅 😂 🤣 😊 😇 🙂 🙃 😉 😌
😍 🥰 😘 😗 😙 😚 😋 😛 😝 😜 🤪 🤨 🧐
🤓 😎 🤩 🥳 😏 😒 😞 😔 😟 😕 🙁 ☹️
😣 😖 😫 😩 🥺 😢 😭 😤 😠 😡 🤬 🤯 😳
🥵 🥶 😱 😨 😰 😥 😓 🤗 🤔 🫣 🤭 🤫 🤥
😶 😐 😑 😬 🙄 😯 😦 😧 😮 😲 🥱 😴 🤤
😪 😵 🤐 🤢 🤮 🤧 😷 🤒 🤕 🤑 🤠

❤️ 🧡 💛 💚 💙 💜 🖤 🤍 🤎
💔 ❤️‍🔥 ❣️ 💕 💞 💓 💗 💖 💘 💝 💟

👍 👎 👌 ✌️ 🤞 🤟 🤘 🤙 👋
🖐️ ✋ 🤚 🖖 👏 🙌 👐 🤝 🙏
💪 🫶 👀 👁️ 🧠 👑

🎉 🎊 🎂 🎁 🎈 🏆 🥇 🥈 🥉
🔥 ⭐ 🌟 ✨ 💫 ⚡ 💥 💯 🚀

☀️ 🌤️ ⛅ 🌥️ ☁️ 🌧️ ⛈️ 🌩️ ❄️
☃️ 🌈 🌙 🌕 🌑 🌸 🌺 🌻 🌹 🌷
🌱 🌿 🌳 🌴 🍀

🍎 🍊 🍋 🍉 🍇 🍓 🍒 🥭 🍍
🍕 🍔 🍟 🌭 🍿 🍩 🍪 🎂 🍰
🍫 🍭 🍬 🍦 ☕ 🫖 🥤 🍹
🍜 🍝 🍚 🍛 🍣 🍱

⚽ 🏀 🏈 ⚾ 🎾 🏐 🏸 🏏
🏓 🥊 🏋️ 🎮 🎸 🎹 🎤 🎧
📚 💻 📱 📷 🎬 🎨

🚗 🚕 🚌 🚆 🚇 ✈️ 🚀 🚲
🏠 🏫 🏢 🏥 🏖️ 🏝️ 🗺️

💡 🔑 🔒 🔓 💰 💎 🎯
📌 📍 📝 📖 ✏️ 🔔

✅ ❌ ❗ ❓ ⚠️ 💬 💭
✔️ ☑️ ❎
"""

# ============================================================
# EXTRACT ONE EMOJI FROM MODEL RESPONSE
# ============================================================

def extract_emoji(response_text):

    if not response_text:
        return None

    response_text = response_text.strip()

    # Check emojis from our vocabulary.
    emojis = EMOJI_LIST.split()

    # Longest first so combined emojis are checked correctly.
    emojis = sorted(emojis, key=len, reverse=True)

    for emoji in emojis:
        if emoji in response_text:
            return emoji

    return None


# ============================================================
# AI PREDICTION
# ============================================================

def generate_emoji(text):

    prompt = f"""
You are an intelligent AI emoji generator.

Read the user's sentence carefully.

Choose ONE emoji that best represents the overall meaning,
emotion, action, object, activity, or situation.

You have a large emoji vocabulary below:

{EMOJI_LIST}

Important rules:

- Return EXACTLY ONE emoji.
- Do not return any explanation.
- Do not return words.
- Do not return multiple emojis.
- Choose the emoji based on the meaning of the entire sentence.
- The answer does not have to be a facial expression.
- If the sentence describes an object, activity, food, drink,
  place, weather, celebration, sport, or other situation,
  choose an appropriate emoji for that meaning.

Examples:

"I am extremely happy today"
😊

"I am drinking tea"
☕

"I love you so much"
❤️

"I won the competition"
🏆

"It is raining outside"
🌧️

"I am going to sleep"
😴

"I am very angry"
😡

"I am confused about this"
😕

"I am eating pizza"
🍕

"I am going on vacation"
✈️

Now classify this sentence:

{text}

Return ONLY ONE emoji.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an AI emoji generator. Return exactly one emoji."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=10,
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    print("MODEL RESPONSE:", repr(result))

    emoji = extract_emoji(result)

    if emoji:
        return emoji

    raise ValueError(
        f"Model did not return a supported emoji: {result}"
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model": MODEL,
        "provider": PROVIDER,
        "token_configured": bool(HF_TOKEN)
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True) or {}

        text = data.get("text", "").strip()

        if not text:

            return jsonify({
                "error": "Please enter some text."
            }), 400

        print("\n========================================")
        print("NEW REQUEST")
        print("========================================")
        print("TEXT:", text)
        print("MODEL:", MODEL)
        print("PROVIDER:", PROVIDER)
        print("TOKEN:", "AVAILABLE" if HF_TOKEN else "MISSING")
        print("========================================")

        emoji = generate_emoji(text)

        print("FINAL EMOJI:", emoji)

        return jsonify({
            "emoji": emoji,
            "source": "AI"
        })

    except Exception as e:

        print("\n========================================")
        print("AI ERROR")
        print("========================================")
        print(type(e).__name__)
        print(str(e))
        print("========================================")

        return jsonify({
            "error": "Unable to generate emoji.",
            "details": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("\n========================================")
    print("        EMOJI GENERATOR AI")
    print("========================================")
    print("Model    :", MODEL)
    print("Provider :", PROVIDER)
    print("Token    :", "Configured" if HF_TOKEN else "MISSING")
    print("Port     :", port)
    print("========================================\n")

    app.run(
        host="0.0.0.0",
        port=port
    )