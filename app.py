import os
import re

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "Qwen/Qwen3-8B"

if not HF_TOKEN:
    print("WARNING: HF_TOKEN is missing.")

# Use Hugging Face automatic provider routing.
# Hugging Face chooses an available provider for the model.
client = InferenceClient(
    provider="auto",
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
😣 😖 😫 😩 🥺 😢 😭 😤 😠 😡 🤬 🤯
😳 🥵 🥶 😱 😨 😰 😥 😓 🤗 🤔 🫣 🤭 🤫 🤥
😶 😐 😑 😬 🙄 😯 😦 😧 😮 😲 🥱 😴
🤤 😪 😵 🤐 🤢 🤮 🤧 😷 🤒 🤕 🤑 🤠

❤️ 🧡 💛 💚 💙 💜 🖤 🤍 🤎
💔 ❤️‍🔥 ❣️ 💕 💞 💓 💗 💖 💘 💝 💟

👍 👎 👌 ✌️ 🤞 🤟 🤘 🤙 👋
🖐️ ✋ 🤚 🖖 👏 🙌 👐 🤝 🙏
💪 🫶 👀 👁️ 🧠 👑

🎉 🎊 🎂 🎁 🎈 🏆 🥇 🥈 🥉
🔥 ⭐ 🌟 ✨ 💫 ⚡ 💥 💯 🚀

☀️ 🌤️ ⛅ 🌥️ ☁️ 🌧️ ⛈️ 🌩️
❄️ ☃️ 🌈 🌙 🌕 🌑
🌸 🌺 🌻 🌹 🌷 🌱 🌿 🌳 🌴 🍀

🍎 🍊 🍋 🍉 🍇 🍓 🍒 🥭 🍍
🍕 🍔 🍟 🌭 🍿 🍩 🍪 🎂 🍰
🍫 🍭 🍬 🍦 ☕ 🫖 🥤 🧋
🍹 🍜 🍝 🍚 🍛 🍣 🍱

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
# CONVERT EMOJI LIST INTO PYTHON LIST
# ============================================================

SUPPORTED_EMOJIS = EMOJI_LIST.split()

# Sort longest first.
# This helps with combined emojis such as ❤️‍🔥.
SUPPORTED_EMOJIS = sorted(
    SUPPORTED_EMOJIS,
    key=len,
    reverse=True
)

# ============================================================
# EXTRACT EMOJI FROM MODEL RESPONSE
# ============================================================

def extract_emoji(text):

    if not text:
        return None

    text = text.strip()

    # First look for one of our supported emojis.
    for emoji in SUPPORTED_EMOJIS:
        if emoji in text:
            return emoji

    return None


# ============================================================
# AI EMOJI GENERATOR
# ============================================================

def generate_emoji(text):

    prompt = f"""
You are an AI emoji generator.

Understand the complete meaning of the user's sentence.

Choose ONE emoji that best represents the sentence.

You can choose from this large emoji vocabulary:

{EMOJI_LIST}

IMPORTANT RULES:

1. Return exactly ONE emoji.
2. Do not return words.
3. Do not explain your answer.
4. Do not return multiple emojis.
5. Choose based on the meaning of the entire sentence.
6. The emoji can represent an emotion, object, activity,
   food, drink, place, weather, celebration, sport,
   reaction, or situation.

Examples:

Sentence:
I am extremely happy today!

Answer:
😊

Sentence:
I want to drink some coffee.

Answer:
☕

Sentence:
I love you so much.

Answer:
❤️

Sentence:
I won the competition!

Answer:
🏆

Sentence:
It is raining outside.

Answer:
🌧️

Sentence:
I am going to sleep.

Answer:
😴

Sentence:
I am eating pizza.

Answer:
🍕

Sentence:
I am going on vacation.

Answer:
✈️

Sentence:
I am very angry.

Answer:
😡

Sentence:
I am confused about this.

Answer:
😕

Now analyze this sentence:

{text}

Return ONLY ONE emoji.
"""

    print("Sending request to Hugging Face...")
    print("Model:", MODEL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI emoji generator. "
                    "Return exactly one emoji and nothing else."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=10,
        temperature=0
    )

    result = response.choices[0].message.content

    print("RAW AI RESPONSE:", repr(result))

    emoji = extract_emoji(result)

    if emoji:
        return emoji

    raise ValueError(
        "The AI did not return a supported emoji. "
        f"AI response was: {result}"
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model": MODEL,
        "provider": "auto",
        "token_configured": bool(HF_TOKEN)
    })


# ============================================================
# PREDICT EMOJI
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Read request
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}

        text = data.get(
            "text",
            ""
        ).strip()

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not text:

            return jsonify({
                "error": "Please enter some text."
            }), 400

        # ----------------------------------------------------
        # Log request
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("NEW EMOJI REQUEST")
        print("=" * 60)

        print("TEXT:", text)
        print("MODEL:", MODEL)
        print(
            "HF TOKEN:",
            "CONFIGURED" if HF_TOKEN else "MISSING"
        )

        print("=" * 60)

        # ----------------------------------------------------
        # Generate emoji using AI
        # ----------------------------------------------------

        emoji = generate_emoji(text)

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print("FINAL EMOJI:", emoji)

        return jsonify({
            "emoji": emoji,
            "source": "AI"
        })

    except Exception as e:

        # ----------------------------------------------------
        # Print complete error to Render logs
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("AI ERROR")
        print("=" * 60)

        print("ERROR TYPE:")
        print(type(e).__name__)

        print()
        print("ERROR MESSAGE:")
        print(str(e))

        print("=" * 60)

        # ----------------------------------------------------
        # Send useful error to frontend
        # ----------------------------------------------------

        return jsonify({
            "error": "Unable to generate emoji.",
            "details": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("=" * 60)
    print("              EMOJI GENERATOR AI")
    print("=" * 60)

    print("Model    :", MODEL)
    print("Provider :", "Hugging Face Auto Routing")
    print(
        "HF Token :",
        "Configured" if HF_TOKEN else "MISSING"
    )
    print("Port     :", port)

    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=port
    )