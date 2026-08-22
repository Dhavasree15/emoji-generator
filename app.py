import os
import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-3.6-flash"

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is missing")
else:
    print("✅ GEMINI_API_KEY is configured")

client = genai.Client(
    api_key=GEMINI_API_KEY
) if GEMINI_API_KEY else None


SYSTEM_INSTRUCTION = """
You are an emoji recommendation AI.

Read the user's sentence and understand its meaning and context.

Choose exactly ONE Unicode emoji that best represents the sentence.

Do NOT use keyword matching.
Do NOT use a hardcoded emoji mapping.
Do NOT explain your answer.

Return JSON only in this format:

{
  "emoji": "ONE_EMOJI"
}

The emoji must be selected based on the meaning of the user's input.
"""


def generate_emoji(text):

    if not client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    print("=" * 60)
    print("NEW GEMINI REQUEST")
    print("=" * 60)
    print("MODEL:", MODEL)
    print("TEXT:", text)

    response = client.models.generate_content(
        model=MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "emoji": {
                        "type": "STRING"
                    }
                },
                "required": ["emoji"]
            }
        )
    )

    print("RAW GEMINI RESPONSE:")
    print(response)

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    data = json.loads(response.text)

    emoji = data.get("emoji")

    if not emoji:
        raise ValueError(
            f"Gemini did not return an emoji: {response.text}"
        )

    emoji = emoji.strip()

    print("EMOJI:", emoji)
    print("=" * 60)

    return emoji


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "provider": "Google Gemini",
        "model": MODEL,
        "api_key": (
            "configured"
            if GEMINI_API_KEY
            else "missing"
        )
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True) or {}

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "Please enter some text"
            }), 400

        emoji = generate_emoji(text)

        return jsonify({
            "emoji": emoji
        })

    except Exception as e:

        print("=" * 60)
        print("❌ GEMINI ERROR")
        print("=" * 60)
        print(type(e).__name__)
        print(str(e))
        print("=" * 60)

        return jsonify({
            "error": "LLM failed to generate an emoji.",
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