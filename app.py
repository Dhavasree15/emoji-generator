import os
import json
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

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"

# Let Hugging Face automatically choose an available provider
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
    timeout=90
)

# ============================================================
# EMOJI LIST
# ============================================================

EMOJI_LIST = [
    "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣",
    "😊", "😇", "🙂", "🙃", "😉", "😌", "😍", "🥰",
    "😘", "😗", "😙", "😚", "😋", "😛", "😝", "😜",
    "🤪", "🤨", "🧐", "🤓", "😎", "🥸", "🤩", "🥳",
    "😏", "😒", "😞", "😔", "😟", "😕", "🙁", "☹️",
    "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤",
    "😠", "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱",
    "😨", "😰", "😥", "😓", "🤗", "🤔", "🤭", "🤫",
    "🤥", "😶", "😐", "😑", "😬", "🙄", "😯", "😦",
    "😧", "😮", "😲", "🥱", "😴", "🤤", "😪", "😵",
    "🤐", "🥴", "🤢", "🤮", "🤧", "😷", "🤒", "🤕",

    "👍", "👎", "👌", "✌️", "🤞", "🤟", "🤘", "🤙",
    "👏", "🙌", "👐", "🤝", "🙏", "💪", "👋", "🤚",
    "✋", "🖐️", "☝️", "👇", "👆", "👉", "👈",

    "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍",
    "🤎", "💔", "💕", "💞", "💓", "💗", "💖",
    "💘", "💝", "💟", "❤️‍🔥",

    "🎉", "🎊", "🎈", "🎂", "🍾", "🥂",
    "🏆", "🥇", "🎁", "✨", "🌟", "⭐", "💫", "🔥",
    "💯",

    "☕", "🍵", "🫖", "🍺", "🍻", "🍷", "🥤",
    "🍕", "🍔", "🍟", "🌮", "🍿", "🍫", "🍰",
    "🍩", "🍪", "🍎", "🍓", "🍉",

    "☀️", "🌞", "🌤️", "⛅", "🌧️", "⛈️", "🌈",
    "❄️", "🌨️", "🌪️", "🌊", "🌙", "🌸", "🌹",
    "🌻", "🌺", "🌴", "🍀",

    "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼",
    "🐨", "🐯", "🦁", "🐮", "🐷", "🐸", "🐵",
    "🙈", "🙉", "🙊", "🐔", "🐧", "🐦", "🦄",
    "🐝", "🦋",

    "💡", "📚", "📖", "💻", "📱", "🎧", "🎵", "🎶",
    "🚀", "✈️", "🚗", "🏠", "💰", "💎", "🔑",
    "⚡", "❗", "❓", "‼️", "⁉️", "✅", "❌", "⚠️"
]

EMOJI_LIST = list(dict.fromkeys(EMOJI_LIST))

# ============================================================
# EXTRACT EMOJI FROM MODEL RESPONSE
# ============================================================

def extract_emoji(response_text):

    if not response_text:
        return None

    response_text = str(response_text).strip()

    # Exact emoji
    if response_text in EMOJI_LIST:
        return response_text

    # Search for emoji
    for emoji in sorted(EMOJI_LIST, key=len, reverse=True):
        if emoji in response_text:
            return emoji

    return None


# ============================================================
# AI PREDICTION
# ============================================================

def predict_with_ai(text):

    prompt = f"""
You are an emoji prediction model.

Understand the meaning and emotion of the user's sentence.

Choose the SINGLE BEST emoji.

Return ONLY ONE emoji.
Do not return words.
Do not explain.
Do not return JSON.
Do not return multiple emojis.

Examples:

"I am extremely happy today" -> 😊
"I finally got selected" -> 🎉
"I studied hard and now I understand" -> 💡
"I want to drink coffee" -> ☕
"I want to drink tea" -> 🍵
"I love you" -> ❤️
"I am very tired" -> 😴
"I am angry" -> 😡
"I am scared" -> 😱
"This is hilarious" -> 😂
"I cannot believe this happened" -> 😲
"I miss you" -> 🥺
"I am confused" -> 😕
"I am bored" -> 😑
"You look cool" -> 😎

User sentence:
{text}
"""

    print("")
    print("============================================")
    print("AI REQUEST")
    print("MODEL:", MODEL)
    print("TEXT:", text)
    print("============================================")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You classify text into exactly one emoji."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=10,
        temperature=0.2
    )

    print("RAW RESPONSE:")
    print(response)

    if response is None:
        raise ValueError("Hugging Face returned None.")

    if not response.choices:
        raise ValueError("Hugging Face returned no choices.")

    message = response.choices[0].message

    print("MESSAGE:")
    print(message)

    content = getattr(message, "content", None)

    print("CONTENT:")
    print(repr(content))

    emoji = extract_emoji(content)

    print("PREDICTED EMOJI:", emoji)

    if emoji:
        return emoji

    raise ValueError(
        "Model did not return a recognizable emoji."
    )


# ============================================================
# LOCAL FALLBACK
# ============================================================

def fallback_emoji(text):

    text = text.lower()

    if any(x in text for x in [
        "coffee", "cafe"
    ]):
        return "☕"

    if any(x in text for x in [
        "tea", "chai"
    ]):
        return "🍵"

    if any(x in text for x in [
        "study", "studied", "learned",
        "understand", "understood",
        "got it", "finally"
    ]):
        return "💡"

    if any(x in text for x in [
        "happy", "joy", "excited",
        "awesome", "amazing",
        "wonderful", "great"
    ]):
        return "😊"

    if any(x in text for x in [
        "selected", "won", "success",
        "achievement", "passed"
    ]):
        return "🎉"

    if any(x in text for x in [
        "love", "loving", "romantic",
        "crush", "boyfriend",
        "girlfriend"
    ]):
        return "❤️"

    if any(x in text for x in [
        "sad", "unhappy", "lonely",
        "hurt", "cry", "crying"
    ]):
        return "😢"

    if any(x in text for x in [
        "angry", "furious", "mad",
        "hate", "annoyed", "frustrated"
    ]):
        return "😡"

    if any(x in text for x in [
        "scared", "afraid", "fear",
        "terrified", "danger",
        "worried"
    ]):
        return "😱"

    if any(x in text for x in [
        "tired", "sleepy",
        "exhausted", "drained"
    ]):
        return "😴"

    if any(x in text for x in [
        "funny", "joke", "laugh",
        "hilarious", "lol"
    ]):
        return "😂"

    if any(x in text for x in [
        "surprise", "surprised",
        "shocked", "wow"
    ]):
        return "😲"

    if any(x in text for x in [
        "confused", "confusion"
    ]):
        return "😕"

    if any(x in text for x in [
        "bored", "boring"
    ]):
        return "😑"

    if any(x in text for x in [
        "cool", "stylish",
        "chill", "swag"
    ]):
        return "😎"

    return "🙂"


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model": MODEL,
        "provider": "auto",
        "token": (
            "configured"
            if HF_TOKEN
            else "missing"
        ),
        "emoji_count": len(EMOJI_LIST)
    })


# ============================================================
# PREDICT API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True) or {}

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    print("")
    print("############################################")
    print("NEW REQUEST")
    print("TEXT:", text)
    print("############################################")

    # ========================================================
    # AI
    # ========================================================

    try:

        emoji = predict_with_ai(text)

        return jsonify({
            "success": True,
            "emoji": emoji,
            "source": "AI",
            "model": MODEL
        })

    except Exception as e:

        print("")
        print("############################################")
        print("AI FAILED")
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))
        print("############################################")

        # Still give user a result
        emoji = fallback_emoji(text)

        return jsonify({
            "success": True,
            "emoji": emoji,
            "source": "fallback",
            "ai_error": str(e)
        })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("")
    print("============================================")
    print("          EMOJI GENERATOR AI")
    print("============================================")
    print("MODEL:", MODEL)
    print("PROVIDER: auto")
    print(
        "HF TOKEN:",
        "CONFIGURED" if HF_TOKEN else "MISSING"
    )
    print("EMOJIS:", len(EMOJI_LIST))
    print("PORT:", port)
    print("============================================")

    app.run(
        host="0.0.0.0",
        port=port
    )