from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import re

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found. "
        "Add HF_TOKEN to your .env file or Render Environment Variables."
    )

# ============================================================
# 2. FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# 3. HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)

# Explicitly route this model through Together
MODEL = "Qwen/Qwen2.5-7B-Instruct:together"

# ============================================================
# 4. SEMANTIC / CONTEXTUAL EMOJI PREDICTION
# ============================================================

def predict_emoji(text):

    prompt = f"""
You are an AI system that predicts the single best Unicode emoji
for a user's text.

Understand the COMPLETE meaning of the user's input.

Analyze:

- overall context
- semantic meaning
- main subject
- intent
- emotion
- actions
- objects
- people
- animals
- food
- places
- activities
- events
- situation
- tone

Do NOT perform simple keyword matching.

Do NOT use a predefined emoji dictionary.

Do NOT restrict yourself to a fixed set of emojis.

The model itself must decide which Unicode emoji best represents
the dominant meaning of the user's COMPLETE input.

For example, if the user gives a long sentence containing several
ideas, understand the sentence as a whole and select the emoji
that best represents the MAIN concept.

Important rules:

1. Return exactly ONE Unicode emoji.
2. The emoji can be any appropriate Unicode emoji.
3. Do not return multiple emojis.
4. Do not return emoji names.
5. Do not return explanations outside the JSON.
6. Do not use a hardcoded emoji mapping.
7. Base the prediction entirely on the meaning of the input.
8. Choose the dominant concept when multiple concepts exist.
9. Confidence must be a number between 0 and 1.
10. Reason must briefly explain why that emoji represents the
    dominant meaning.

User input:
{text}

Return ONLY this JSON:

{{
    "emoji": "ONE_UNICODE_EMOJI",
    "confidence": 0.0,
    "reason": "short explanation"
}}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a contextual emoji prediction AI. "
                        "Understand the complete meaning of the input "
                        "and return exactly one appropriate Unicode emoji "
                        "inside valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=150
        )

        answer = response.choices[0].message.content.strip()

        print("\n========================================")
        print("USER INPUT")
        print("========================================")
        print(text)

        print("\n========================================")
        print("MODEL RESPONSE")
        print("========================================")
        print(answer)

        # ----------------------------------------------------
        # Remove accidental Markdown code fences
        # ----------------------------------------------------

        answer = re.sub(
            r"```(?:json)?",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = answer.replace("```", "").strip()

        # ----------------------------------------------------
        # Extract JSON if model added extra text
        # ----------------------------------------------------

        json_match = re.search(
            r"\{.*\}",
            answer,
            re.DOTALL
        )

        if not json_match:
            raise ValueError(
                "Model did not return valid JSON."
            )

        result = json.loads(json_match.group())

        # ----------------------------------------------------
        # Extract values
        # ----------------------------------------------------

        emoji = str(result.get("emoji", "")).strip()
        confidence = result.get("confidence", 0)
        reason = str(result.get("reason", "")).strip()

        if not emoji:
            raise ValueError(
                "Model did not return an emoji."
            )

        # ----------------------------------------------------
        # Normalize confidence
        # ----------------------------------------------------

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "emoji": emoji,
            "confidence": confidence,
            "reason": reason
        }

    except Exception as e:

        print("\n========================================")
        print("HUGGING FACE ERROR")
        print("========================================")
        print(str(e))

        raise


# ============================================================
# 5. PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Invalid JSON request."
            }), 400

        text = data.get("text", "")

        if not isinstance(text, str):
            return jsonify({
                "error": "Text must be a string."
            }), 400

        text = text.strip()

        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        result = predict_emoji(text)

        return jsonify(result), 200

    except Exception as e:

        print("\nAPI ERROR:")
        print(str(e))

        return jsonify({
            "error": "Unable to generate emoji.",
            "details": str(e)
        }), 500


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "model": MODEL,
        "huggingface": "connected"
    })


# ============================================================
# 7. SERVE FRONTEND
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# 8. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("\n==========================================")
    print("          EMOJI GENERATOR AI")
    print("==========================================")
    print("🤗 Hugging Face : CONNECTED")
    print("🧠 Model        :", MODEL)
    print("🌐 Port         :", port)
    print("==========================================\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )