import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# Current model with multiple Hugging Face inference providers
MODEL = "deepseek-ai/DeepSeek-V3.2"

if not HF_TOKEN:
    print("WARNING: HF_TOKEN is missing!")
else:
    print("HF_TOKEN loaded successfully.")

# Let Hugging Face automatically select an available provider.
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
    timeout=60
)

# ============================================================
# ALLOWED EMOJIS
# ============================================================

ALLOWED_EMOJIS = [
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
# AI PREDICTION
# ============================================================

def generate_emoji(text):

    prompt = f"""
You are an emoji classification AI.

Analyze the user's sentence carefully.

Choose the ONE emoji that best represents
the emotion or meaning of the sentence.

You MUST choose exactly one emoji from this list:

😊 😢 😡 😍 😎 😲 😑 🥰 😕 😴 😱 😐 😂 😭 😉 😘

Rules:
- Return ONLY the emoji.
- Return exactly ONE emoji.
- Do not return any words.
- Do not explain your answer.
- Do not return multiple emojis.

Sentence:
{text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an emoji classification AI. Always return exactly one emoji."
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

    print("AI RESPONSE:", repr(result))

    # Extract the emoji returned by the AI
    for emoji in ALLOWED_EMOJIS:
        if emoji in result:
            return emoji

    raise ValueError(
        "AI returned an invalid emoji: " + result
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
        "huggingface": (
            "configured"
            if HF_TOKEN
            else "missing"
        )
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(
        silent=True
    ) or {}

    text = data.get(
        "text",
        ""
    ).strip()

    if not text:

        return jsonify({
            "error": "Please enter some text."
        }), 400

    print("")
    print("==========================================")
    print("NEW REQUEST")
    print("==========================================")
    print("TEXT:", text)
    print("MODEL:", MODEL)
    print("PROVIDER: AUTO")
    print("==========================================")

    try:

        emoji = generate_emoji(text)

        print("FINAL EMOJI:", emoji)

        return jsonify({
            "emoji": emoji,
            "source": "AI"
        }), 200

    except Exception as e:

        print("")
        print("==========================================")
        print("AI ERROR")
        print("==========================================")
        print(str(e))
        print("==========================================")

        return jsonify({
            "error": "AI service is temporarily unavailable.",
            "details": str(e)
        }), 503


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

    print("")
    print("==========================================")
    print("          EMOJI GENERATOR AI")
    print("==========================================")
    print("MODEL    :", MODEL)
    print("PROVIDER :", "Hugging Face Auto")
    print("PORT     :", port)
    print(
        "HF TOKEN :",
        "Configured"
        if HF_TOKEN
        else "Missing"
    )
    print("==========================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=port
    )