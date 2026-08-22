from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import os
import json
import emoji


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "\nHF_TOKEN not found.\n\n"
        "Create a .env file and add:\n\n"
        "HF_TOKEN=hf_your_token_here\n"
    )


# ============================================================
# 2. FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# 3. HUGGING FACE
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================================
# 4. CONVERT SHORTCODE TO UNICODE
# ============================================================

def convert_to_unicode_emoji(value):

    if not value:
        return value

    value = str(value).strip()

    try:

        converted = emoji.emojize(
            value,
            language="alias"
        )

        return converted

    except Exception:

        return value


# ============================================================
# 5. CLEAN MODEL RESPONSE
# ============================================================

def clean_model_response(answer):

    answer = answer.strip()

    if answer.startswith("```json"):

        answer = answer[7:]

    elif answer.startswith("```"):

        answer = answer[3:]


    if answer.endswith("```"):

        answer = answer[:-3]


    return answer.strip()


# ============================================================
# 6. AI EMOJI PREDICTION
# ============================================================

def predict_emoji(text):

    prompt = f"""
You are an advanced semantic emoji recommendation AI.

Understand the COMPLETE meaning of the user's text.

The input can be:

- one word
- a phrase
- a sentence
- multiple sentences
- a long paragraph

Analyze the entire context.

Consider:

- emotions
- sentiment
- objects
- animals
- food
- activities
- events
- people
- relationships
- actions
- situations
- tone
- intent
- context

The goal is NOT keyword matching.

Do NOT use a predefined emoji dictionary.

Do NOT use keyword-to-emoji mappings.

Do NOT simply select an emoji because one word appears
in the sentence.

Understand the meaning of the complete input.

If the text describes an object, consider the object.

If the text describes an emotion, consider the emotion.

If the text describes an activity, consider the activity.

If the text contains multiple meanings, determine the
DOMINANT meaning of the complete text.

Return exactly THREE different emoji recommendations.

The FIRST emoji must be the BEST recommendation.

The second and third must be meaningful alternatives.

IMPORTANT OUTPUT RULE:

The "emoji" value must contain an actual Unicode emoji.

Correct:

"emoji": "🎉"

"emoji": "🎂"

"emoji": "🐶"

Incorrect:

"emoji": ":tada:"

"emoji": "tada"

"emoji": "party"

Do not return emoji names.

Return ONLY valid JSON.

Use this exact structure:

{{
    "predictions": [
        {{
            "emoji": "ONE_UNICODE_EMOJI",
            "reason": "short explanation"
        }},
        {{
            "emoji": "ONE_UNICODE_EMOJI",
            "reason": "short explanation"
        }},
        {{
            "emoji": "ONE_UNICODE_EMOJI",
            "reason": "short explanation"
        }}
    ]
}}

USER TEXT:

{text}
"""


    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You are an advanced semantic emoji "
                        "recommendation AI. "
                        "Understand the complete context "
                        "and return valid JSON only."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.1,

            max_tokens=300

        )


        answer = response.choices[0].message.content.strip()


        print("\n" + "=" * 60)
        print("USER INPUT")
        print("=" * 60)

        print(text)


        print("\n" + "=" * 60)
        print("RAW HUGGING FACE RESPONSE")
        print("=" * 60)

        print(answer)


        # ----------------------------------------------------
        # Clean Markdown
        # ----------------------------------------------------

        answer = clean_model_response(
            answer
        )


        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        result = json.loads(
            answer
        )


        predictions = result.get(
            "predictions",
            []
        )


        if not predictions:

            raise ValueError(
                "Model returned no predictions."
            )


        # ----------------------------------------------------
        # Convert to Unicode
        # ----------------------------------------------------

        cleaned_predictions = []


        for prediction in predictions[:3]:

            if not isinstance(
                prediction,
                dict
            ):

                continue


            model_emoji = prediction.get(
                "emoji",
                ""
            )


            reason = prediction.get(
                "reason",
                ""
            )


            unicode_emoji = (
                convert_to_unicode_emoji(
                    model_emoji
                )
            )


            if not unicode_emoji:

                continue


            cleaned_predictions.append({

                "emoji": unicode_emoji,

                "reason": str(
                    reason
                ).strip()

            })


        if not cleaned_predictions:

            raise ValueError(
                "No valid emojis were returned."
            )


        print("\n" + "=" * 60)
        print("FINAL PREDICTIONS")
        print("=" * 60)


        for prediction in cleaned_predictions:

            print(
                prediction["emoji"],
                "→",
                prediction["reason"]
            )


        return cleaned_predictions


    except json.JSONDecodeError as error:

        print("\nJSON ERROR:")
        print(error)

        print("\nMODEL OUTPUT:")
        print(answer)

        raise ValueError(
            "Hugging Face returned invalid JSON."
        )


    except Exception as error:

        print("\nHUGGING FACE ERROR:")
        print(error)

        raise error


# ============================================================
# 7. PREDICT API
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

                "error":
                "Invalid JSON request."

            }), 400


        text = data.get(
            "text",
            ""
        ).strip()


        if not text:

            return jsonify({

                "error":
                "Please enter some text."

            }), 400


        predictions = predict_emoji(
            text
        )


        best_emoji = (
            predictions[0]["emoji"]
        )


        return jsonify({

            "emoji":
            best_emoji,

            "predictions":
            predictions

        })


    except Exception as error:

        print("\nAPI ERROR:")
        print(error)


        return jsonify({

            "error":
            str(error)

        }), 500


# ============================================================
# 8. HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
        "running",

        "huggingface":
        "connected",

        "model":
        MODEL

    })


# ============================================================
# 9. SERVE FRONTEND
# ============================================================

@app.route("/")
def index():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# 10. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )


    print("\n" + "=" * 50)
    print("          EMOJI GENERATOR AI")
    print("=" * 50)

    print(
        "🤗 Hugging Face : CONNECTED"
    )

    print(
        "🧠 Model        :",
        MODEL
    )

    print(
        "🌐 Port         :",
        port
    )

    print("=" * 50 + "\n")


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )