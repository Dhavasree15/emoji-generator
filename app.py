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
        "HF_TOKEN is missing. "
        "Add HF_TOKEN to your .env file locally "
        "and Render Environment Variables when deployed."
    )


# ============================================================
# 2. FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# 3. HUGGING FACE CLIENT
# ============================================================

# Hugging Face Router
#
# The :together suffix explicitly tells Hugging Face
# to use Together AI for this model.

client = InferenceClient(
    api_key=HF_TOKEN
)


MODEL = "Qwen/Qwen2.5-7B-Instruct:together"


# ============================================================
# 4. AI EMOJI PREDICTION
# ============================================================

def predict_emoji(text):

    prompt = f"""
You are an intelligent contextual emoji prediction AI.

Understand the COMPLETE meaning of the user's input.

Your job is to select ONE Unicode emoji that best represents
the dominant meaning of the COMPLETE input.

Analyze:

- context
- semantic meaning
- intent
- emotions
- objects
- people
- animals
- food
- places
- activities
- actions
- events
- situations
- tone

Do NOT perform simple keyword matching.

Do NOT use a predefined emoji dictionary.

Do NOT use hardcoded emoji mappings.

Do NOT select an emoji simply because one word appears
in the sentence.

If the input contains multiple concepts, understand the
whole sentence and select the emoji representing the
MOST IMPORTANT or DOMINANT concept.

The model itself must decide the emoji.

The emoji may be ANY valid Unicode emoji.

IMPORTANT RULES:

1. Return exactly ONE Unicode emoji.
2. Do not return multiple emojis.
3. Do not return emoji names.
4. Do not return words instead of an emoji.
5. Do not use a predefined emoji list.
6. Do not use keyword-to-emoji mappings.
7. Understand the complete input semantically.
8. Choose the dominant meaning.
9. Confidence must be between 0 and 1.
10. Return valid JSON only.

USER INPUT:

{text}

Return ONLY:

{{
    "emoji": "ONE_EMOJI",
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
                        "Understand the complete meaning of the "
                        "user input and return exactly one "
                        "Unicode emoji in valid JSON."
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


        print("\n==========================================")
        print("USER INPUT")
        print("==========================================")
        print(text)

        print("\n==========================================")
        print("MODEL")
        print("==========================================")
        print(MODEL)

        print("\n==========================================")
        print("MODEL RESPONSE")
        print("==========================================")
        print(answer)


        # ----------------------------------------------------
        # Remove Markdown fences
        # ----------------------------------------------------

        answer = re.sub(
            r"```json",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = answer.replace(
            "```",
            ""
        ).strip()


        # ----------------------------------------------------
        # Extract JSON
        # ----------------------------------------------------

        match = re.search(
            r"\{.*\}",
            answer,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "Model did not return valid JSON."
            )


        result = json.loads(
            match.group(0)
        )


        # ----------------------------------------------------
        # Extract result
        # ----------------------------------------------------

        emoji = str(
            result.get(
                "emoji",
                ""
            )
        ).strip()


        reason = str(
            result.get(
                "reason",
                ""
            )
        ).strip()


        confidence = result.get(
            "confidence",
            0
        )


        if not emoji:
            raise ValueError(
                "Model did not return an emoji."
            )


        try:

            confidence = float(
                confidence
            )

        except (ValueError, TypeError):

            confidence = 0.0


        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )


        return {
            "emoji": emoji,
            "confidence": confidence,
            "reason": reason
        }


    except Exception as e:

        print("\n==========================================")
        print("HUGGING FACE ERROR")
        print("==========================================")
        print(str(e))

        raise


# ============================================================
# 5. PREDICT API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        if not data:

            return jsonify({
                "error": "Invalid JSON request."
            }), 400


        text = data.get(
            "text",
            ""
        )


        if not isinstance(
            text,
            str
        ):

            return jsonify({
                "error": "Text must be a string."
            }), 400


        text = text.strip()


        if not text:

            return jsonify({
                "error": "Please enter some text."
            }), 400


        result = predict_emoji(
            text
        )


        return jsonify(
            result
        ), 200


    except Exception as e:

        print("\n==========================================")
        print("API ERROR")
        print("==========================================")
        print(str(e))


        return jsonify({
            "error": "Unable to generate emoji.",
            "details": str(e)
        }), 500


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "running",
        "model": MODEL,
        "provider": "together",
        "huggingface": "connected"
    })


# ============================================================
# 7. FRONTEND
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print("\n==================================================")
    print("          EMOJI GENERATOR AI")
    print("==================================================")
    print("🤗 Hugging Face : CONNECTED")
    print("☁️  Provider     : Together AI")
    print("🧠 Model        :", MODEL)
    print("🌐 Port         :", port)
    print("==================================================\n")


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )