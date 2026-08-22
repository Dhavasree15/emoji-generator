import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# CONFIG
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
# ALLOWED EMOJIS
# ============================================================

EMOJIS = [
    "😊",
    "😢",
    "😡",
    "😍",
    "😎",
    "😲",
    "😑",
    "🥰",
    "😕",
    "😴",
    "😱",
    "😐",
    "😂",
    "😭",
    "😉",
    "😘"
]

# ============================================================
# AI EMOJI PREDICTION
# ============================================================

def predict_emoji(text):

    prompt = f"""
You are an AI emoji classifier.

Analyze the meaning and emotion of this sentence.

Choose exactly ONE emoji from this list:

😊 😢 😡 😍 😎 😲 😑 🥰 😕 😴 😱 😐 😂 😭 😉 😘

Return ONLY ONE emoji.

Do not explain.
Do not return words.
Do not return multiple emojis.

Sentence:
{text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You classify sentences into one emoji. Return only one emoji."
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

    print("AI RAW RESPONSE:", repr(result))

    for emoji in EMOJIS:
        if emoji in result:
            return emoji

    raise ValueError(
        f"Model returned an unsupported response: {result}"
    )


# ============================================================
# HOME
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
        "hf_token": bool(HF_TOKEN)
    })


# ============================================================
# PREDICT API
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

        print()
        print("======================================")
        print("NEW EMOJI REQUEST")
        print("======================================")
        print("TEXT:", text)
        print("MODEL:", MODEL)
        print("PROVIDER:", PROVIDER)
        print("======================================")

        emoji = predict_emoji(text)

        print("FINAL EMOJI:", emoji)

        return jsonify({
            "emoji": emoji,
            "source": "AI"
        })

    except Exception as e:

        print()
        print("======================================")
        print("AI ERROR")
        print("======================================")
        print(type(e).__name__)
        print(str(e))
        print("======================================")

        return jsonify({
            "error": "Unable to generate emoji.",
            "details": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print()
    print("======================================")
    print("        EMOJI GENERATOR AI")
    print("======================================")
    print("Model    :", MODEL)
    print("Provider :", PROVIDER)
    print("Token    :", "Configured" if HF_TOKEN else "MISSING")
    print("Port     :", port)
    print("======================================")

    app.run(
        host="0.0.0.0",
        port=port
    )