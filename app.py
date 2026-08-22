import os
import re

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROVIDER = "nscale"

# ============================================================
# EMOJI LIST
# ============================================================
# The AI is allowed to choose from this complete list.
# We are NOT limiting it to only 10-15 emojis.

EMOJIS = [
    # Faces
    "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣",
    "😊", "😇", "🙂", "🙃", "😉", "😌", "😍", "🥰",
    "😘", "😗", "😙", "😚", "😋", "😛", "😝", "😜",
    "🤪", "🤨", "🧐", "🤓", "😎", "🥸", "🤩", "🥳",

    # Positive / neutral
    "😏", "😒", "😞", "😔", "😟", "😕", "🙁", "☹️",
    "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤",
    "😠", "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱",
    "😨", "😰", "😥", "😓", "🫣", "🤗", "🤔", "🫡",
    "🤭", "🫢", "🫣", "🤫", "🤥", "😶", "🫥", "😐",
    "😑", "😬", "🙄", "😯", "😦", "😧", "😮", "😲",
    "🥱", "😴", "🤤", "😪", "😵", "🤐", "🥴", "🤢",
    "🤮", "🤧", "😷", "🤒", "🤕",

    # Gestures
    "👍", "👎", "👌", "✌️", "🤞", "🤟", "🤘", "🤙",
    "👏", "🙌", "👐", "🤝", "🙏", "💪", "👋", "🤚",
    "✋", "🖐️", "☝️", "👇", "👆", "👉", "👈",

    # Hearts / love
    "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍",
    "🤎", "💔", "❣️", "💕", "💞", "💓", "💗", "💖",
    "💘", "💝", "💟",

    # Celebration
    "🎉", "🎊", "🥳", "🎈", "🎂", "🍾", "🥂",
    "🏆", "🥇", "🎁", "✨", "🌟", "⭐", "💫", "🔥",

    # Food / drinks
    "☕", "🍵", "🫖", "🍺", "🍻", "🍷", "🥤",
    "🍕", "🍔", "🍟", "🌮", "🍿", "🍫", "🍰",
    "🍩", "🍪", "🍎", "🍓", "🍉",

    # Weather / nature
    "☀️", "🌞", "🌤️", "⛅", "🌧️", "⛈️", "🌈",
    "❄️", "🌨️", "🌪️", "🌊", "🌙", "🌸", "🌹",
    "🌻", "🌺", "🌴", "🍀",

    # Animals
    "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼",
    "🐨", "🐯", "🦁", "🐮", "🐷", "🐸", "🐵", "🙈",
    "🙉", "🙊", "🐔", "🐧", "🐦", "🦄", "🐝", "🦋",

    # Objects / symbols
    "💡", "📚", "📖", "💻", "📱", "🎧", "🎵", "🎶",
    "🚀", "✈️", "🚗", "🏠", "💰", "💎", "🔑",
    "⚡", "💯", "❗", "❓", "‼️", "⁉️", "✅",
    "❌", "⚠️", "❤️‍🔥"
]

# Remove duplicates while preserving order
EMOJIS = list(dict.fromkeys(EMOJIS))

# ============================================================
# CREATE HUGGING FACE CLIENT
# ============================================================

client = None

if HF_TOKEN:
    client = InferenceClient(
        provider=PROVIDER,
        api_key=HF_TOKEN,
        timeout=60
    )

# ============================================================
# EMOJI EXTRACTION
# ============================================================

def extract_emoji(text):
    """
    Find the first supported emoji returned by the model.
    """

    if not text:
        return None

    text = str(text).strip()

    # Exact match first
    if text in EMOJIS:
        return text

    # Search for supported emoji inside response
    for emoji in sorted(EMOJIS, key=len, reverse=True):
        if emoji in text:
            return emoji

    return None


# ============================================================
# LOCAL FALLBACK
# ============================================================
# This is ONLY used if Hugging Face is temporarily unavailable.
# Normally the AI model makes the prediction.

def local_fallback(text):

    text = text.lower().strip()

    # Drinks / food
    if any(word in text for word in [
        "coffee", "tea", "drink", "juice", "cafe"
    ]):
        return "☕"

    # Love
    if any(word in text for word in [
        "love", "loving", "romantic", "crush",
        "boyfriend", "girlfriend", "kiss", "kissing"
    ]):
        return "😍"

    # Celebration / success
    if any(word in text for word in [
        "won", "winner", "selected", "success",
        "achievement", "passed", "got it",
        "finally", "celebrate", "celebration"
    ]):
        return "🎉"

    # Happy
    if any(word in text for word in [
        "happy", "happiness", "joy", "joyful",
        "excited", "great", "awesome", "amazing",
        "wonderful", "glad"
    ]):
        return "😊"

    # Sad
    if any(word in text for word in [
        "sad", "unhappy", "disappointed",
        "lonely", "hurt", "heartbroken",
        "cry", "crying"
    ]):
        return "😢"

    # Angry
    if any(word in text for word in [
        "angry", "anger", "furious", "mad",
        "hate", "annoyed", "frustrated"
    ]):
        return "😡"

    # Scared
    if any(word in text for word in [
        "scared", "afraid", "fear",
        "terrified", "danger", "worried",
        "anxious"
    ]):
        return "😱"

    # Tired
    if any(word in text for word in [
        "tired", "sleepy", "exhausted",
        "sleep", "fatigue", "drained"
    ]):
        return "😴"

    # Surprise
    if any(word in text for word in [
        "surprised", "surprise", "shocked",
        "shock", "unexpected", "wow"
    ]):
        return "😲"

    # Funny
    if any(word in text for word in [
        "funny", "joke", "laugh",
        "laughing", "hilarious", "lol"
    ]):
        return "😂"

    # Confused
    if any(word in text for word in [
        "confused", "confusion",
        "don't understand", "dont understand"
    ]):
        return "😕"

    # Bored
    if any(word in text for word in [
        "bored", "boring", "nothing to do"
    ]):
        return "😑"

    # Cool
    if any(word in text for word in [
        "cool", "stylish", "chill",
        "vibe", "swag"
    ]):
        return "😎"

    # Default
    return "🙂"


# ============================================================
# AI PREDICTION
# ============================================================

def predict_with_ai(text):

    if not client:
        raise RuntimeError("HF_TOKEN is missing.")

    # IMPORTANT:
    # We tell the model to output ONLY one emoji.
    # We do not ask it to explain its reasoning.

    system_prompt = f"""
You are an emoji prediction engine.

Your job is to understand the meaning, emotion,
activity, object, or situation in the user's sentence.

Choose the SINGLE emoji that best represents the sentence.

You may choose ANY emoji from this list:

{" ".join(EMOJIS)}

IMPORTANT RULES:

1. Return exactly ONE emoji.
2. Do not return words.
3. Do not explain your answer.
4. Do not return multiple emojis.
5. Do not return JSON.
6. Do not write a sentence.
7. Choose the emoji based on the MEANING of the sentence,
   not merely one keyword.

Examples:

"I am extremely happy today" -> 😊
"I finally got selected!" -> 🎉
"I love you so much" -> ❤️
"I am drinking coffee" -> ☕
"I am drinking tea" -> 🍵
"I am very tired" -> 😴
"I cannot believe this happened" -> 😲
"This is hilarious" -> 😂
"I am angry right now" -> 😡
"I am scared" -> 😱
"I miss you" -> 🥺
"I studied hard and now I understand" -> 💡
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=10,
        temperature=0.1,
        top_p=0.9
    )

    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    print("\n==========================================")
    print("AI REQUEST")
    print("MODEL:", MODEL)
    print("PROVIDER:", PROVIDER)
    print("TEXT:", text)

    print("RAW RESPONSE OBJECT:")
    print(response)

    # --------------------------------------------------------
    # Extract response
    # --------------------------------------------------------

    if not response.choices:
        raise ValueError("AI returned no choices.")

    message = response.choices[0].message

    if message is None:
        raise ValueError("AI returned no message.")

    content = message.content

    print("RAW AI CONTENT:", repr(content))

    if not content:
        raise ValueError("AI returned empty content.")

    emoji = extract_emoji(content)

    print("EXTRACTED EMOJI:", emoji)
    print("==========================================\n")

    if not emoji:
        raise ValueError(
            f"AI did not return a supported emoji. Response: {content}"
        )

    return emoji


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# ============================================================
# HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "model": MODEL,
        "provider": PROVIDER,
        "huggingface_token": (
            "configured" if HF_TOKEN else "missing"
        ),
        "emoji_count": len(EMOJIS)
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True) or {}

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        print("\n")
        print("==================================================")
        print("NEW EMOJI REQUEST")
        print("==================================================")
        print("TEXT:", text)
        print("MODEL:", MODEL)
        print("PROVIDER:", PROVIDER)
        print(
            "HF TOKEN:",
            "CONFIGURED" if HF_TOKEN else "MISSING"
        )

        # ====================================================
        # AI PREDICTION
        # ====================================================

        try:

            emoji = predict_with_ai(text)

            print("FINAL SOURCE: AI")
            print("FINAL EMOJI:", emoji)

            return jsonify({
                "success": True,
                "emoji": emoji,
                "source": "AI",
                "model": MODEL
            })

        except Exception as ai_error:

            print("\n")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("AI ERROR")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("ERROR TYPE:", type(ai_error).__name__)
            print("ERROR MESSAGE:", str(ai_error))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            # =================================================
            # TEMPORARY LOCAL FALLBACK
            # =================================================
            # The UI still works if the external provider fails.

            fallback = local_fallback(text)

            print("FALLBACK EMOJI:", fallback)

            return jsonify({
                "success": True,
                "emoji": fallback,
                "source": "fallback",
                "ai_error": str(ai_error)
            })

    except Exception as error:

        print("\nSERVER ERROR")
        print(type(error).__name__)
        print(str(error))

        return jsonify({
            "success": False,
            "error": "Unable to process the request."
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("")
    print("==================================================")
    print("              EMOJI GENERATOR AI")
    print("==================================================")
    print("Model    :", MODEL)
    print("Provider :", PROVIDER)
    print(
        "HF Token :",
        "CONFIGURED" if HF_TOKEN else "MISSING"
    )
    print("Emojis   :", len(EMOJIS))
    print("Port     :", port)
    print("==================================================")

    app.run(
        host="0.0.0.0",
        port=port
    )