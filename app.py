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

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROVIDER = "together"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN is missing.")
else:
    print("HF_TOKEN loaded successfully.")

# Create Hugging Face client
client = InferenceClient(
    provider=PROVIDER,
    api_key=HF_TOKEN,
    timeout=60
)

# ============================================================
# ALLOWED EMOJIS
# ============================================================

ALLOWED_EMOJIS = {
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
}

# ============================================================
# AI EMOJI PREDICTION
# ============================================================

def generate_emoji(text):

    prompt = f"""
You are an AI emoji classifier.

Analyze the meaning and emotion of the user's sentence.

Choose EXACTLY ONE emoji from this list:

😊 😢 😡 😍 😎 😲 😑 🥰 😕 😴 😱 😐 😂 😭 😉 😘

Use the emoji that best represents the overall meaning,
emotion, or situation described by the sentence.

Important rules:

1. Return ONLY ONE emoji.
2. The emoji MUST come from the provided list.
3. Do NOT return words.
4. Do NOT explain your answer.
5. Do NOT return multiple emojis.

User sentence:
{text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You classify sentences into exactly one emoji."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=5,
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    print("AI RAW RESPONSE:", repr(result))

    # Check whether Qwen returned one of our allowed emojis
    for emoji in ALLOWED_EMOJIS:
        if emoji in result:
            return emoji

    raise ValueError(
        f"AI returned an unsupported response: {result}"
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
        "huggingface": "configured" if HF_TOKEN else "missing"
    })


# ============================================================
# PREDICT EMOJI
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "Invalid request."
            }), 400

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        print("")
        print("==========================================")
        print("NEW EMOJI REQUEST")
        print("TEXT:", text)
        print("MODEL:", MODEL)
        print("PROVIDER:", PROVIDER)
        print("==========================================")

        # ----------------------------------------------------
        # QWEN AI PREDICTION
        # ----------------------------------------------------

        emoji = generate_emoji(text)

        print("FINAL AI EMOJI:", emoji)

        return jsonify({
            "emoji": emoji,
            "source": "Qwen AI"
        }), 200

    except Exception as e:

        print("")
        print("==========================================")
        print("AI ERROR")
        print(str(e))
        print("==========================================")

        return jsonify({
            "error": "Unable to generate emoji.",
            "details": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    print("")
    print("==========================================")
    print("          EMOJI GENERATOR AI")
    print("==========================================")
    print("Model    :", MODEL)
    print("Provider :", PROVIDER)
    print("Port     :", port)
    print(
        "HF Token :",
        "Configured" if HF_TOKEN else "Missing"
    )
    print("==========================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=port
    )