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
# CONFIG
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "Qwen/Qwen3-8B"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN is missing")
else:
    print("HF_TOKEN loaded")

# Let Hugging Face choose an available provider
client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
    timeout=60
)


# ============================================================
# EXTRACT ONE EMOJI FROM MODEL RESPONSE
# ============================================================

def extract_emoji(text):

    if not text:
        return None

    # Remove markdown/code formatting
    text = text.replace("```", "").strip()

    # Unicode emoji ranges
    pattern = re.compile(
        r"[\U0001F000-\U0001FAFF"
        r"\U00002700-\U000027BF"
        r"\U00002300-\U000023FF"
        r"\u2600-\u26FF"
        r"\u2700-\u27BF]+"
    )

    matches = pattern.findall(text)

    if matches:
        return matches[0]

    return None


# ============================================================
# LLM EMOJI GENERATOR
# ============================================================

def generate_emoji(text):

    system_prompt = """
You are an intelligent semantic emoji generator.

Your job is to understand the COMPLETE meaning of the
user's sentence and select the SINGLE Unicode emoji
that best represents it.

Do NOT perform keyword matching.

Understand:
- emotion
- action
- object
- situation
- context
- intent
- tone

You can choose ANY Unicode emoji.

Examples:

User: My cat is sleeping on my laptop
Answer: 🐱

User: I can't believe I got the job
Answer: 🤩

User: I want to drink coffee
Answer: ☕

User: I want to drink tea
Answer: 🍵

User: I studied all night and finally understood it
Answer: 💡

User: I am completely exhausted
Answer: 😴

User: I love my family
Answer: ❤️

User: That movie scared me so much
Answer: 😱

User: I am furious right now
Answer: 😡

User: I am crying because I miss my friend
Answer: 😭

User: Look at this beautiful sunset
Answer: 🌅

User: My dog is playing in the park
Answer: 🐕

User: I am going on vacation tomorrow
Answer: ✈️

User: I just finished my exam
Answer: 📝

User: I won the competition
Answer: 🏆

User: I am hungry
Answer: 🍔

These are examples only.
Do not restrict yourself to these emojis.

IMPORTANT:
Return EXACTLY ONE emoji.

Never return:
- explanations
- words
- sentences
- JSON
- Markdown
- multiple emojis
"""

    user_prompt = f"""
Understand this sentence semantically and choose
the ONE emoji that best represents it.

Sentence:
{text}

Return only one emoji.
"""

    print("\n" + "=" * 60)
    print("SENDING TO LLM")
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
        temperature=0.3
    )

    print("\nRAW RESPONSE OBJECT:")
    print(response)

    result = response.choices[0].message.content

    print("\nRAW MODEL OUTPUT:")
    print(repr(result))

    emoji = extract_emoji(result)

    print("\nEXTRACTED EMOJI:")
    print(repr(emoji))

    if not emoji:
        raise ValueError(
            f"Model did not return a valid emoji. "
            f"Raw response: {repr(result)}"
        )

    return emoji


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
        "huggingface": bool(HF_TOKEN)
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
            "error": "HF_TOKEN is not configured on the server."
        }), 500

    try:

        emoji = generate_emoji(text)

        print("\nFINAL RESULT:", emoji)
        print("=" * 60)

        return jsonify({
            "emoji": emoji,
            "source": "LLM",
            "model": MODEL
        })

    except Exception as e:

        print("\n" + "=" * 60)
        print("LLM ERROR")
        print("=" * 60)

        print("TYPE:", type(e).__name__)
        print("ERROR:", str(e))

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

    print("\n")
    print("=" * 60)
    print("           EMOJI GENERATOR")
    print("=" * 60)
    print("Model    :", MODEL)
    print("Provider :", "auto")
    print("HF Token :", "Configured" if HF_TOKEN else "MISSING")
    print("Port     :", port)
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port
    )