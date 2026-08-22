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
PROVIDER = "nscale"

# ============================================================
# ALL EMOJIS
# ============================================================

EMOJIS = [
    "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣",
    "😊", "😇", "🙂", "🙃", "😉", "😌", "😍", "🥰",
    "😘", "😗", "😙", "😚", "😋", "😛", "😝", "😜",
    "🤪", "🤨", "🧐", "🤓", "😎", "🥸", "🤩", "🥳",
    "😏", "😒", "😞", "😔", "😟", "😕", "🙁", "☹️",
    "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤",
    "😠", "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱",
    "😨", "😰", "😥", "😓", "🫣", "🤗", "🤔", "🫡",
    "🤭", "🫢", "🤫", "🤥", "😶", "🫥", "😐", "😑",
    "😬", "🙄", "😯", "😦", "😧", "😮", "😲", "🥱",
    "😴", "🤤", "😪", "😵", "🤐", "🥴", "🤢", "🤮",
    "🤧", "😷", "🤒", "🤕",

    "👍", "👎", "👌", "✌️", "🤞", "🤟", "🤘", "🤙",
    "👏", "🙌", "👐", "🤝", "🙏", "💪", "👋", "🤚",
    "✋", "🖐️", "☝️", "👇", "👆", "👉", "👈",

    "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍",
    "🤎", "💔", "❣️", "💕", "💞", "💓", "💗", "💖",
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

EMOJIS = list(dict.fromkeys(EMOJIS))

# ============================================================
# CLIENT
# ============================================================

if not HF_TOKEN:
    print("WARNING: HF_TOKEN is missing.")

client = InferenceClient(
    provider=PROVIDER,
    api_key=HF_TOKEN,
    timeout=90
)

# ============================================================
# EXTRACT EMOJI
# ============================================================

def extract_emoji(text):

    if not text:
        return None

    text = str(text).strip()

    # Exact match
    if text in EMOJIS:
        return text

    # Search longest emojis first
    for emoji in sorted(EMOJIS, key=len, reverse=True):

        if emoji in text:
            return emoji

    return None


# ============================================================
# AI
# ============================================================

def generate_emoji(text):

    prompt = """
You are an emoji prediction engine.

Understand the meaning of the user's sentence.

Choose ONE emoji that best represents the meaning.

Return ONLY ONE emoji.

Do NOT:
- explain
- give a sentence
- give multiple emojis
- give JSON
- give words

Examples:

I am extremely happy today
😊

I finally got selected
🎉

I love you
❤️

I am drinking coffee
☕

I want to drink tea
🍵

I studied hard and now I understand
💡

I am very tired
😴

I am angry
😡

I am scared
😱

This is hilarious
😂

I cannot believe this happened
😲

I miss you
🥺

User sentence:
""" + text

    print("")
    print("==========================================")
    print("SENDING TO HUGGING FACE")
    print("MODEL:", MODEL)
    print("PROVIDER:", PROVIDER)
    print("TEXT:", text)
    print("==========================================")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=10,
        temperature=0.7,
        top_p=0.8,

        # IMPORTANT FOR QWEN3
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    print("FULL RESPONSE:")
    print(response)

    if not response.choices:
        raise Exception("No choices returned by model.")

    message = response.choices[0].message

    print("MESSAGE:")
    print(message)

    # ========================================================
    # IMPORTANT:
    # Some providers may return content differently.
    # ========================================================

    content = None

    if message:

        content = getattr(
            message,
            "content",
            None
        )

        # Some provider responses can expose reasoning separately.
        if not content:
            content = getattr(
                message,
                "reasoning_content",
                None
            )

    print("CONTENT:", repr(content))

    emoji = extract_emoji(content)

    print("EXTRACTED EMOJI:", emoji)

    if emoji:
        return emoji

    raise Exception(
        "Model returned no supported emoji. "
        f"Raw response: {content}"
    )


# ============================================================
# FALLBACK
# ============================================================

def fallback_emoji(text):

    text = text.lower()

    # Drinks
    if any(x in text for x in [
        "coffee",
        "cafe",
        "tea",
        "drink",
        "juice"
    ]):
        if "tea" in text:
            return "🍵"

        return "☕"

    # Studying / understanding
    if any(x in text for x in [
        "understand",
        "understood",
        "learned",
        "learnt",
        "studied hard",
        "got it",
        "finally"
    ]):
        return "💡"

    # Love
    if any(x in text for x in [
        "love",
        "loving",
        "romantic",
        "crush",
        "boyfriend",
        "girlfriend"
    ]):
        return "❤️"

    # Happy
    if any(x in text for x in [
        "happy",
        "joy",
        "excited",
        "awesome",
        "amazing",
        "wonderful",
        "great"
    ]):
        return "😊"

    # Success
    if any(x in text for x in [
        "selected",
        "won",
        "success",
        "achievement",
        "passed"
    ]):
        return "🎉"

    # Sad
    if any(x in text for x in [
        "sad",
        "unhappy",
        "lonely",
        "hurt",
        "cry",
        "crying"
    ]):
        return "😢"

    # Angry
    if any(x in text for x in [
        "angry",
        "furious",
        "mad",
        "hate",
        "annoyed",
        "frustrated"
    ]):
        return "😡"

    # Scared
    if any(x in text for x in [
        "scared",
        "afraid",
        "fear",
        "terrified",
        "danger",
        "worried"
    ]):
        return "😱"

    # Tired
    if any(x in text for x in [
        "tired",
        "sleepy",
        "exhausted",
        "drained"
    ]):
        return "😴"

    # Funny
    if any(x in text for x in [
        "funny",
        "joke",
        "laugh",
        "hilarious",
        "lol"
    ]):
        return "😂"

    # Surprise
    if any(x in text for x in [
        "surprise",
        "surprised",
        "shocked",
        "wow",
        "unbelievable"
    ]):
        return "😲"

    # Confused
    if any(x in text for x in [
        "confused",
        "confusion",
        "don't understand"
    ]):
        return "😕"

    # Bored
    if any(x in text for x in [
        "bored",
        "boring"
    ]):
        return "😑"

    # Cool
    if any(x in text for x in [
        "cool",
        "stylish",
        "chill",
        "swag"
    ]):
        return "😎"

    return "🙂"


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
        "provider": PROVIDER,
        "token": (
            "configured"
            if HF_TOKEN
            else "missing"
        ),
        "emoji_count": len(EMOJIS)
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

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

        print("")
        print("##################################################")
        print("NEW EMOJI REQUEST")
        print("##################################################")
        print("TEXT:", text)

        # ====================================================
        # TRY AI
        # ====================================================

        try:

            emoji = generate_emoji(text)

            print("")
            print("AI SUCCESS")
            print("EMOJI:", emoji)

            return jsonify({
                "success": True,
                "emoji": emoji,
                "source": "AI",
                "model": MODEL
            })

        except Exception as e:

            print("")
            print("##################################################")
            print("AI ERROR")
            print("##################################################")
            print("TYPE:", type(e).__name__)
            print("MESSAGE:", str(e))
            print("##################################################")

            # Temporary safety fallback
            emoji = fallback_emoji(text)

            print("FALLBACK EMOJI:", emoji)

            return jsonify({
                "success": True,
                "emoji": emoji,
                "source": "fallback",
                "ai_error": str(e)
            })

    except Exception as e:

        print("SERVER ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("")
    print("==================================================")
    print("              EMOJI GENERATOR")
    print("==================================================")
    print("MODEL:", MODEL)
    print("PROVIDER:", PROVIDER)
    print(
        "HF TOKEN:",
        "CONFIGURED"
        if HF_TOKEN
        else "MISSING"
    )
    print(
        "EMOJIS:",
        len(EMOJIS)
    )
    print("PORT:", port)
    print("==================================================")

    app.run(
        host="0.0.0.0",
        port=port
    )