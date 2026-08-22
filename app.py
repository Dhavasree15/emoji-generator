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

HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================
# MODEL
# ============================================================

MODEL = "Qwen/Qwen3-8B"

# We use Hugging Face automatic provider routing.
# Qwen3-8B currently has an available inference provider.
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
    timeout=60
)

# ============================================================
# EMOJI LIST
# ============================================================

SUPPORTED_EMOJIS = [
    "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣",
    "😊", "😇", "🙂", "🙃", "😉", "😌", "😍", "🥰",
    "😘", "😗", "😙", "😚", "😋", "😛", "😝", "😜",
    "🤪", "🤨", "🧐", "🤓", "😎", "🤩", "🥳", "😏",
    "😒", "😞", "😔", "😟", "😕", "🙁", "☹️", "😣",
    "😖", "😫", "😩", "🥺", "😢", "😭", "😤", "😠",
    "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱", "😨",
    "😰", "😥", "😓", "🫣", "🤗", "🤔", "🫡", "🤭",
    "🤫", "🤥", "😶", "😐", "😑", "😬", "🙄", "😯",
    "😦", "😧", "😮", "😲", "🥱", "😴", "🤤", "😪",
    "😵", "🤐", "🥴", "🤢", "🤮", "🤧", "😷", "🤒",
    "🤕", "🤑", "🤠", "😈", "👿", "👹", "👺", "🤡",
    "💩", "👻", "💀", "☠️", "👽", "👾", "🤖", "🎃",
    "😺", "😸", "😹", "😻", "😼", "😽", "🙀", "😿",
    "😾",

    "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍",
    "🤎", "💔", "❣️", "💕", "💞", "💓", "💗", "💖",
    "💘", "💝", "💟",

    "👍", "👎", "👌", "✌️", "🤞", "🤟", "🤘", "🤙",
    "👏", "🙌", "👐", "🤲", "🙏", "💪", "👊", "✊",
    "🤝", "☝️", "👇", "👆", "👉", "👈", "✋", "🤚",
    "🖐️", "🖖", "👋", "🤏", "💅",

    "🔥", "✨", "⭐", "🌟", "💫", "⚡", "💥", "🎉",
    "🎊", "💯", "🚀", "🏆", "🥇", "🎯", "💡",

    "☕", "🍵", "🍺", "🍻", "🥂", "🍷", "🍕", "🍔",
    "🍟", "🌮", "🍰", "🎂", "🍫", "🍎", "🍓",

    "⚽", "🏀", "🏏", "🎾", "🏸", "🎮", "🎧", "🎵",
    "🎶", "🎬", "📚", "💻", "📱", "💰", "💸",
    "✈️", "🚗", "🏠", "🌍", "🌈", "☀️", "🌙", "🌧️"
]

# ============================================================
# FALLBACK EMOJI CLASSIFICATION
#
# This is ONLY used if Hugging Face is temporarily unavailable.
# The normal path is the Qwen AI model.
# ============================================================

def fallback_emoji(text):

    text = text.lower().strip()

    # --------------------------------------------------------
    # Food / drinks
    # --------------------------------------------------------

    if any(word in text for word in [
        "coffee",
        "tea",
        "drink",
        "cafe",
        "chai"
    ]):
        return "☕"

    # --------------------------------------------------------
    # Love
    # --------------------------------------------------------

    if any(word in text for word in [
        "love",
        "romantic",
        "romance",
        "crush",
        "boyfriend",
        "girlfriend",
        "kiss",
        "kissing",
        "heart"
    ]):
        return "😍"

    # --------------------------------------------------------
    # Happy
    # --------------------------------------------------------

    if any(word in text for word in [
        "happy",
        "happiness",
        "joy",
        "joyful",
        "excited",
        "great",
        "awesome",
        "amazing",
        "wonderful",
        "glad",
        "celebrate",
        "celebration",
        "success",
        "selected",
        "won",
        "victory",
        "achievement"
    ]):
        return "😊"

    # --------------------------------------------------------
    # Funny
    # --------------------------------------------------------

    if any(word in text for word in [
        "funny",
        "joke",
        "laugh",
        "laughing",
        "hilarious",
        "lol"
    ]):
        return "😂"

    # --------------------------------------------------------
    # Sad
    # --------------------------------------------------------

    if any(word in text for word in [
        "sad",
        "unhappy",
        "disappointed",
        "disappointment",
        "lonely",
        "alone",
        "hurt",
        "heartbroken",
        "lost",
        "cry",
        "crying"
    ]):
        return "😢"

    # --------------------------------------------------------
    # Angry
    # --------------------------------------------------------

    if any(word in text for word in [
        "angry",
        "anger",
        "furious",
        "mad",
        "hate",
        "annoyed",
        "annoying",
        "frustrated",
        "frustration"
    ]):
        return "😡"

    # --------------------------------------------------------
    # Scared / fear
    # --------------------------------------------------------

    if any(word in text for word in [
        "scared",
        "afraid",
        "fear",
        "terrified",
        "danger",
        "dangerous",
        "worried",
        "anxious",
        "anxiety"
    ]):
        return "😱"

    # --------------------------------------------------------
    # Tired
    # --------------------------------------------------------

    if any(word in text for word in [
        "tired",
        "sleepy",
        "exhausted",
        "fatigue",
        "drained"
    ]):
        return "😴"

    # --------------------------------------------------------
    # Surprise
    # --------------------------------------------------------

    if any(word in text for word in [
        "surprised",
        "surprise",
        "shocked",
        "shock",
        "unexpected",
        "unbelievable",
        "wow"
    ]):
        return "😲"

    # --------------------------------------------------------
    # Confused
    # --------------------------------------------------------

    if any(word in text for word in [
        "confused",
        "confusion",
        "don't understand",
        "dont understand",
        "what",
        "why",
        "how"
    ]):
        return "😕"

    # --------------------------------------------------------
    # Bored
    # --------------------------------------------------------

    if any(word in text for word in [
        "bored",
        "boring",
        "nothing to do"
    ]):
        return "😑"

    # --------------------------------------------------------
    # Cool
    # --------------------------------------------------------

    if any(word in text for word in [
        "cool",
        "stylish",
        "chill",
        "vibe",
        "swag"
    ]):
        return "😎"

    # --------------------------------------------------------
    # Shy
    # --------------------------------------------------------

    if any(word in text for word in [
        "shy",
        "embarrassed",
        "blushing",
        "blush"
    ]):
        return "🥰"

    # --------------------------------------------------------
    # Wink
    # --------------------------------------------------------

    if any(word in text for word in [
        "wink",
        "winked"
    ]):
        return "😉"

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "🙂"


# ============================================================
# EXTRACT EMOJI FROM AI RESPONSE
# ============================================================

def extract_emoji(response):

    if response is None:
        return None

    response = str(response).strip()

    print("RAW AI RESPONSE:", repr(response))

    # Direct exact match
    if response in SUPPORTED_EMOJIS:
        return response

    # Search for any supported emoji in the response
    for emoji in SUPPORTED_EMOJIS:
        if emoji in response:
            return emoji

    # Remove markdown/code formatting and search again
    cleaned = re.sub(r"[*`_\n\r]", " ", response)

    for emoji in SUPPORTED_EMOJIS:
        if emoji in cleaned:
            return emoji

    return None


# ============================================================
# QWEN AI EMOJI GENERATION
# ============================================================

def generate_with_ai(text):

    prompt = f"""
Classify the emotion or meaning of this sentence and return ONE
appropriate emoji.

IMPORTANT:
- Return ONLY ONE emoji.
- Do not explain.
- Do not return words.
- Do not return JSON.
- Do not use markdown.
- Do not think aloud.

Examples:

"I am extremely happy today" -> 😊
"I failed my exam" -> 😢
"I am very angry" -> 😡
"I love you so much" -> ❤️
"I want to drink coffee" -> ☕
"I am scared" -> 😱
"That is hilarious" -> 😂
"I am exhausted" -> 😴
"That is amazing!" -> 🤩
"I don't understand this" -> 😕
"Let's celebrate!" -> 🎉

User sentence:
{text}
"""

    print("==========================================")
    print("SENDING REQUEST TO HUGGING FACE")
    print("MODEL:", MODEL)
    print("TEXT:", text)
    print("==========================================")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an emoji classifier. "
                    "Return exactly one emoji and nothing else."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=20,
        temperature=0.7,

        # CRITICAL FOR QWEN3:
        # Disable thinking so the model directly returns
        # the emoji instead of spending output tokens
        # on reasoning.
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    print("FULL AI RESPONSE OBJECT:")
    print(response)

    # --------------------------------------------------------
    # Extract normal chat response
    # --------------------------------------------------------

    try:

        choice = response.choices[0]

        message = choice.message

        content = getattr(message, "content", None)

        print("AI MESSAGE CONTENT:", repr(content))

        emoji = extract_emoji(content)

        if emoji:
            print("AI EMOJI:", emoji)
            return emoji

    except Exception as e:

        print("ERROR READING AI RESPONSE:")
        print(type(e).__name__)
        print(str(e))

    # --------------------------------------------------------
    # Sometimes reasoning/content fields can differ.
    # Try text directly from choice if available.
    # --------------------------------------------------------

    try:

        text_value = getattr(response.choices[0], "text", None)

        print("AI CHOICE TEXT:", repr(text_value))

        emoji = extract_emoji(text_value)

        if emoji:
            return emoji

    except Exception as e:

        print("CHOICE TEXT ERROR:", str(e))

    return None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return send_from_directory(".", "index.html")


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model": MODEL,
        "provider": "auto",
        "huggingface": "configured" if HF_TOKEN else "missing"
    })


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True) or {}

    text = str(data.get("text", "")).strip()

    if not text:

        return jsonify({
            "error": "Please enter some text."
        }), 400

    print("")
    print("==========================================")
    print("NEW PREDICTION REQUEST")
    print("TEXT:", text)
    print("==========================================")

    # ========================================================
    # AI
    # ========================================================

    if HF_TOKEN:

        try:

            emoji = generate_with_ai(text)

            if emoji:

                print("FINAL RESULT FROM AI:", emoji)

                return jsonify({
                    "emoji": emoji,
                    "source": "Qwen AI"
                })

            print("AI returned no supported emoji.")

        except Exception as e:

            print("")
            print("==========================================")
            print("AI ERROR")
            print("==========================================")
            print("ERROR TYPE:", type(e).__name__)
            print("ERROR MESSAGE:", str(e))
            print("==========================================")

    else:

        print("HF_TOKEN is missing.")

    # ========================================================
    # FALLBACK
    # ========================================================

    print("Using local fallback.")

    emoji = fallback_emoji(text)

    print("FALLBACK RESULT:", emoji)

    return jsonify({
        "emoji": emoji,
        "source": "fallback"
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("")
    print("==========================================")
    print("          EMOJI GENERATOR AI")
    print("==========================================")
    print("Hugging Face : CONNECTED")
    print("Provider     : AUTO")
    print("Model        :", MODEL)
    print("Port         :", port)
    print("==========================================")

    app.run(
        host="0.0.0.0",
        port=port
    )