import os
import re

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# APP
# ============================================================

app = Flask(__name__, static_folder=".")
CORS(app)


# ============================================================
# HUGGING FACE CONFIG
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "Qwen/Qwen2.5-7B-Instruct"

if not HF_TOKEN:
    print("❌ HF_TOKEN is missing")
else:
    print("✅ HF_TOKEN is configured")


client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
    timeout=120
)


# ============================================================
# EMOJI EXTRACTION
# ============================================================

def extract_emoji(text):

    if not text:
        return None

    text = str(text).strip()

    # Unicode emoji ranges
    pattern = re.compile(
        "["
        "\U0001F000-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002300-\U000023FF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]"
    )

    emojis = pattern.findall(text)

    if emojis:
        return emojis[0]

    return None


# ============================================================
# AI EMOJI GENERATION
# ============================================================

def generate_emoji(text):

    system_prompt = """
You are an emoji recommendation AI.

Understand the complete meaning of the user's sentence.

Choose ONE single Unicode emoji that best represents
the sentence.

The emoji can represent:
- an emotion
- an action
- an object
- an animal
- food
- travel
- celebration
- weather
- work
- study
- relationships
- activities
- situations
- reactions
- or any other relevant concept.

Do NOT use keyword matching.

Understand the context.

Return EXACTLY ONE Unicode emoji.

Do not return:
- words
- explanations
- labels
- JSON
- punctuation
- multiple emojis

Your entire answer must contain ONE emoji only.
"""

    user_prompt = f"""
Sentence:

{text}

Return the single best emoji.
"""

    print()
    print("=" * 60)
    print("NEW LLM REQUEST")
    print("=" * 60)
    print("MODEL:", MODEL)
    print("TEXT:", text)

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        max_tokens=10,

        temperature=0.2,

        top_p=0.9
    )

    print()
    print("RAW RESPONSE:")
    print(response)

    content = response.choices[0].message.content

    print()
    print("MODEL CONTENT:")
    print(repr(content))

    emoji = extract_emoji(content)

    print()
    print("EXTRACTED EMOJI:")
    print(repr(emoji))

    if emoji:
        return emoji

    raise ValueError(
        f"Model did not return an emoji. "
        f"Raw content: {repr(content)}"
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# HEALTH
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

    try:

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

        if not HF_TOKEN:

            return jsonify({
                "error": "HF_TOKEN is missing on the server."
            }), 500

        emoji = generate_emoji(text)

        print()
        print("=" * 60)
        print("SUCCESS")
        print("EMOJI:", emoji)
        print("=" * 60)

        return jsonify({
            "emoji": emoji,
            "source": "LLM",
            "model": MODEL
        })

    except Exception as e:

        print()
        print("=" * 60)
        print("LLM ERROR")
        print("=" * 60)
        print("TYPE:", type(e).__name__)
        print("MESSAGE:", str(e))
        print("=" * 60)

        return jsonify({
            "error": "LLM failed to generate an emoji.",
            "details": str(e)
        }), 500


# ============================================================
# START
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
    print("             EMOJI GENERATOR")
    print("=" * 60)
    print("MODEL    :", MODEL)
    print("PROVIDER :", "auto")
    print(
        "HF TOKEN :",
        "Configured" if HF_TOKEN else "MISSING"
    )
    print("PORT     :", port)
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port
    )