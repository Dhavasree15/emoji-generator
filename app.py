import os
import re

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "Qwen/Qwen3-8B"

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model": MODEL,
        "llm": "Hugging Face"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True) or {}

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    if not HF_TOKEN:
        return jsonify({
            "error": "HF_TOKEN is missing."
        }), 500

    try:

        prompt = f"""
You are an intelligent emoji generator.

Read the user's sentence and understand its meaning,
emotion, action, object, situation and context.

Then choose the SINGLE emoji that best represents
the COMPLETE meaning of the sentence.

You are NOT doing keyword matching.

You must decide the emoji yourself.

Examples are only demonstrations of the task:

"I finally got selected after months of preparation"
→ 🎉

"I want to drink coffee"
→ ☕

"I want to drink tea"
→ 🍵

"I finally understand this difficult concept"
→ 💡

"I love you"
→ ❤️

"I am exhausted after studying all night"
→ 😴

"I can't believe this happened"
→ 😲

"I am so angry right now"
→ 😡

"I am feeling really sad"
→ 😢

"That joke was hilarious"
→ 😂

These examples do NOT limit the possible emojis.

You may use ANY appropriate Unicode emoji.

Important:
- Understand the entire sentence.
- Consider context.
- Do not simply match keywords.
- Choose the emoji yourself.
- Return ONLY ONE emoji.
- Do not return words.
- Do not explain your answer.
- Do not return JSON.
- Do not return Markdown.

User sentence:

{text}

Your answer:
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an emoji generation model. "
                        "Return exactly one Unicode emoji."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=20,
            temperature=0.2
        )

        result = response.choices[0].message.content.strip()

        print("=" * 60)
        print("USER:", text)
        print("LLM RAW:", repr(result))
        print("=" * 60)

        # Remove possible markdown
        result = result.replace("```", "").strip()

        # Extract emoji characters from model response.
        # This is NOT deciding the emoji.
        # It only cleans the model's response if Qwen adds text.
        emoji_pattern = re.compile(
            r"[\U0001F000-\U0001FAFF"
            r"\u2600-\u27BF"
            r"\uFE0F]+"
        )

        matches = emoji_pattern.findall(result)

        if not matches:
            raise ValueError(
                f"LLM did not return an emoji: {result}"
            )

        emoji = matches[0]

        return jsonify({
            "emoji": emoji,
            "source": "LLM"
        })

    except Exception as e:

        print("LLM ERROR:")
        print(type(e).__name__)
        print(str(e))

        return jsonify({
            "error": "Unable to generate emoji.",
            "details": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )