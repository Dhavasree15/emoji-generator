import os
import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__, static_folder=".")
CORS(app)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-2.5-flash"


if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is NOT configured")
else:
    print("✅ GEMINI_API_KEY is configured")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are an intelligent emoji recommendation model.

Your task is to understand the meaning and context of a user's
sentence and select the SINGLE most appropriate Unicode emoji.

IMPORTANT:

Do NOT use keyword matching.

Understand the complete meaning of the sentence.

The emoji can represent:

- emotions
- activities
- objects
- food
- animals
- places
- celebrations
- work
- study
- travel
- relationships
- weather
- sports
- technology
- reactions
- everyday situations
- feelings
- events
- achievements
- humor
- sleep
- health
- nature
- transportation
- communication
- money
- shopping
- music
- movies
- games
- school
- coding
- programming
- friendship
- love
- family

You have access to the full Unicode emoji set.

Examples:

User:
"I want to drink coffee"

Good emoji:
☕

User:
"My cat is sleeping on my laptop"

Good emoji:
🐱

User:
"I finally got the job after months of preparation"

Good emoji:
🎉

User:
"I am studying for my exam"

Good emoji:
📚

User:
"I am extremely tired"

Good emoji:
😴

User:
"I love you"

Good emoji:
❤️

User:
"It is raining outside"

Good emoji:
🌧️

User:
"Let's go on vacation"

Good emoji:
✈️

User:
"I am hungry"

Good emoji:
🍔

User:
"I am going to the gym"

Good emoji:
🏋️

User:
"I received a gift"

Good emoji:
🎁

User:
"I am coding all night"

Good emoji:
💻

User:
"I am confused"

Good emoji:
🤔

User:
"I am getting married"

Good emoji:
💍

User:
"That movie was hilarious"

Good emoji:
😂

The examples above are only examples.
Do NOT limit your choices to those emojis.

For every new sentence, reason about its meaning and select
the most semantically appropriate emoji.

Return EXACTLY ONE emoji in the "emoji" field.

Never return:
- explanations
- sentences
- multiple emojis
- emoji lists
- Markdown
- code
- labels
"""


# ============================================================
# GENERATE EMOJI USING GEMINI
# ============================================================

def generate_emoji(text):

    if not client:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    print()
    print("=" * 60)
    print("NEW LLM REQUEST")
    print("=" * 60)

    print("MODEL:")
    print(MODEL)

    print()
    print("USER TEXT:")
    print(text)

    # --------------------------------------------------------
    # JSON schema
    # --------------------------------------------------------

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "emoji": {
                "type": "STRING",
                "description": (
                    "Exactly one Unicode emoji that best "
                    "represents the user's sentence."
                )
            }
        },
        "required": ["emoji"]
    }

    # --------------------------------------------------------
    # Gemini request
    # --------------------------------------------------------

    response = client.models.generate_content(

        model=MODEL,

        contents=text,

        config=types.GenerateContentConfig(

            system_instruction=SYSTEM_INSTRUCTION,

            temperature=0.2,

            max_output_tokens=20,

            response_mime_type="application/json",

            response_schema=response_schema
        )
    )

    # --------------------------------------------------------
    # Debug Gemini response
    # --------------------------------------------------------

    print()
    print("RAW GEMINI RESPONSE:")
    print(response)

    print()
    print("RESPONSE TEXT:")
    print(repr(response.text))

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    try:

        result = json.loads(
            response.text
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Gemini returned invalid JSON: "
            f"{response.text}"
        ) from e

    # --------------------------------------------------------
    # Get emoji
    # --------------------------------------------------------

    emoji = result.get("emoji")

    if not emoji:
        raise ValueError(
            "Gemini response does not contain an emoji."
        )

    emoji = str(emoji).strip()

    # --------------------------------------------------------
    # We expect ONE emoji.
    #
    # We are NOT maintaining a hardcoded list of emojis.
    # Gemini is responsible for selecting it.
    # --------------------------------------------------------

    print()
    print("MODEL SELECTED:")
    print(emoji)

    print("=" * 60)

    return emoji


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

        "provider": "Google Gemini",

        "model": MODEL,

        "gemini_api": (
            "configured"
            if GEMINI_API_KEY
            else "missing"
        )

    })


# ============================================================
# PREDICT EMOJI
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Read request
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        ) or {}

        text = data.get(
            "text",
            ""
        )

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        # ----------------------------------------------------
        # Validate text
        # ----------------------------------------------------

        if not text:

            return jsonify({

                "error":
                    "Please enter some text."

            }), 400

        # ----------------------------------------------------
        # Validate Gemini API
        # ----------------------------------------------------

        if not GEMINI_API_KEY:

            return jsonify({

                "error":
                    "GEMINI_API_KEY is not configured."

            }), 500

        # ----------------------------------------------------
        # Ask Gemini
        # ----------------------------------------------------

        emoji = generate_emoji(text)

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("✅ SUCCESS")
        print("INPUT :", text)
        print("EMOJI :", emoji)

        return jsonify({

            "emoji": emoji,

            "source": "Gemini LLM",

            "model": MODEL,

            "provider": "Google Gemini"

        })

    except Exception as e:

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("❌ GEMINI ERROR")
        print("=" * 60)

        print("ERROR TYPE:")
        print(type(e).__name__)

        print()
        print("ERROR MESSAGE:")
        print(str(e))

        print("=" * 60)

        return jsonify({

            "error":
                "Gemini failed to generate an emoji.",

            "details":
                str(e)

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

    print()
    print("=" * 60)
    print("              EMOJI GENERATOR")
    print("=" * 60)

    print()
    print("Provider : Google Gemini")
    print("Model    :", MODEL)

    print(
        "API Key  :",
        (
            "Configured"
            if GEMINI_API_KEY
            else "MISSING"
        )
    )

    print("Port     :", port)

    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=port
    )