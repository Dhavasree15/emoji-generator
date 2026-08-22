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
        "Add HF_TOKEN to your .env file locally "
        "and to Render Environment Variables when deployed."
    )


# ============================================================
# 2. FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# 3. HUGGING FACE INFERENCE CLIENT
# ============================================================

# We explicitly use Together AI as the inference provider.
#
# Hugging Face routes the request through its inference
# infrastructure using the HF token.
#
# The Qwen2.5-7B-Instruct model is currently available
# through Together AI for text generation.

client = InferenceClient(
    provider="together",
    api_key=HF_TOKEN
)


MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================================
# 4. AI EMOJI PREDICTION
# ============================================================

def predict_emoji(text):

    prompt = f"""
You are an intelligent contextual emoji prediction AI.

Your task is to understand the COMPLETE meaning of the user's
input and select the SINGLE Unicode emoji that best represents
the dominant meaning.

Do NOT use a predefined emoji dictionary.

Do NOT use keyword-to-emoji mappings.

Do NOT simply match individual words.

The emoji must be selected by understanding the overall meaning,
context, intent, situation, emotion, objects, activities,
people, animals, food, places, events, actions, and tone.

If the input contains multiple ideas, determine which concept
is most important in the COMPLETE sentence and choose the emoji
that best represents that dominant concept.

The emoji may be ANY valid Unicode emoji.

IMPORTANT RULES:

1. Return exactly ONE emoji.
2. The emoji must be a real Unicode emoji.
3. Do not return multiple emojis.
4. Do not return an emoji name.
5. Do not return words instead of an emoji.
6. Do not use a fixed emoji mapping.
7. Do not rely on keywords alone.
8. Understand the complete sentence.
9. Choose the dominant meaning.
10. Return valid JSON only.

The confidence value must be between 0 and 1.

The reason should briefly explain the semantic meaning that
caused the model to choose the emoji.

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

        # ----------------------------------------------------
        # Send request to Qwen through Hugging Face + Together
        # ----------------------------------------------------

        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a contextual emoji prediction AI. "
                        "Understand the complete meaning of the "
                        "user's input and return exactly one "
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


        # ----------------------------------------------------
        # Get model response
        # ----------------------------------------------------

        answer = response.choices[0].message.content.strip()


        print("\n==========================================")
        print("USER INPUT")
        print("==========================================")
        print(text)

        print("\n==========================================")
        print("MODEL RESPONSE")
        print("==========================================")
        print(answer)


        # ----------------------------------------------------
        # Remove Markdown code fences if the model adds them
        # ----------------------------------------------------

        answer = re.sub(
            r"```json",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = answer.replace("```", "").strip()


        # ----------------------------------------------------
        # Find JSON object
        # ----------------------------------------------------

        json_match = re.search(
            r"\{.*\}",
            answer,
            re.DOTALL
        )

        if not json_match:
            raise ValueError(
                "The AI did not return valid JSON."
            )


        json_text = json_match.group(0)


        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        result = json.loads(json_text)


        # ----------------------------------------------------
        # Extract prediction
        # ----------------------------------------------------

        emoji = str(
            result.get("emoji", "")
        ).strip()

        reason = str(
            result.get("reason", "")
        ).strip()

        confidence = result.get(
            "confidence",
            0
        )


        # ----------------------------------------------------
        # Validate emoji
        # ----------------------------------------------------

        if not emoji:

            raise ValueError(
                "The AI did not return an emoji."
            )


        # ----------------------------------------------------
        # Validate confidence
        # ----------------------------------------------------

        try:

            confidence = float(confidence)

        except (ValueError, TypeError):

            confidence = 0.0


        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )


        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

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
# 5. PREDICTION ENDPOINT
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


        if not isinstance(text, str):

            return jsonify({
                "error": "Text must be a string."
            }), 400


        text = text.strip()


        if not text:

            return jsonify({
                "error": "Please enter some text."
            }), 400


        # Call AI model
        result = predict_emoji(text)


        return jsonify({
            "emoji": result["emoji"],
            "confidence": result["confidence"],
            "reason": result["reason"]
        }), 200


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
# 7. SERVE FRONTEND
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
# 8. RUN SERVER
# ============================================================

if __name__ == "__main__":

    # Render provides PORT automatically.
    # Locally it will use 5000.

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
    print("🔌 Provider      : Together AI")
    print("🧠 Model         :", MODEL)
    print("🌐 Port          :", port)
    print("==================================================\n")


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )