import os
import re

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "Qwen/Qwen3-8B"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN is missing.")
else:
    print("HF_TOKEN is configured.")


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

    # Remove markdown/code formatting if present
    text = text.replace("```", "").strip()

    # Unicode emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F000-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002300-\U000023FF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]"
    )

    matches = emoji_pattern.findall(text)

    if matches:
        return matches[0]

    return None


# ============================================================
# AI EMOJI GENERATION
# ============================================================

def generate_emoji(text):

    system_prompt = """
You are an intelligent emoji generator.

Understand the meaning and context of the user's sentence.

Choose the SINGLE Unicode emoji that best represents
the meaning of the sentence.

You are NOT doing keyword matching.

Consider:
- emotion
- action
- object
- situation
- context
- intention
- tone

You may choose ANY appropriate Unicode emoji.

IMPORTANT:
Return EXACTLY ONE emoji.
Do not return words.
Do not explain.
Do not return JSON.
Do not return multiple emojis.
"""

    user_prompt = f"""
Convert this sentence into the most appropriate single emoji:

{text}

Return only one emoji.
"""

    print("\n" + "=" * 60)
    print("LLM REQUEST")
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

        # IMPORTANT:
        # Qwen3 normally thinks before answering.
        # Disable thinking because we only need one emoji.
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },

        max_tokens=20,

        temperature=0.7,

        top_p=0.8
    )

    print("\nRAW RESPONSE:")
    print(response)

    # --------------------------------------------------------
    # Get normal answer
    # --------------------------------------------------------

    content = response.choices[0].message.content

    print("\nMODEL CONTENT:")
    print(repr(content))

    # --------------------------------------------------------
    # Extract emoji
    # --------------------------------------------------------

    emoji = extract_emoji(content)

    print("\nEXTRACTED EMOJI:")
    print(repr(emoji))

    if emoji:
        return emoji

    raise ValueError(
        "Model did not return an emoji. "
        f"Model response: {repr(content)}"
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

    if not HF_TOKEN:

        return jsonify({
            "error": "HF_TOKEN is not configured."
        }), 500

    print("\n")
    print("=" * 60)
    print("NEW EMOJI REQUEST")
    print("=" * 60)
    print("TEXT:", text)

    try:

        emoji = generate_emoji(text)

        print("\nFINAL EMOJI:", emoji)
        print("=" * 60)

        return jsonify({
            "emoji": emoji,
            "source": "LLM",
            "model": MODEL
        })

    except Exception as e:

        print("\n")
        print("=" * 60)
        print("LLM ERROR")
        print("=" * 60)
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))
        print("=" * 60)

        return jsonify({
            "error": "LLM failed to generate an emoji.",
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

    print("")
    print("=" * 60)
    print("              EMOJI GENERATOR")
    print("=" * 60)
    print("Model    :", MODEL)
    print("Provider :", "auto")
    print(
        "HF Token :",
        "Configured" if HF_TOKEN else "MISSING"
    )
    print("Port     :", port)
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port
    )