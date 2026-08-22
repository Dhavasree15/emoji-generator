import os
import json
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

MODEL = "Qwen/Qwen3-8B"

if not HF_TOKEN:
    print("WARNING: HF_TOKEN is missing.")

# IMPORTANT:
# Do NOT force Together.
# Hugging Face automatically chooses an available provider.
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
    timeout=90
)


# ============================================================
# LARGE EMOJI VOCABULARY
# ============================================================

EMOJI_LIST = {

    # -------------------------
    # PEOPLE / EMOTIONS
    # -------------------------

    "😀": "grinning face",
    "😃": "grinning face with big eyes",
    "😄": "grinning face with smiling eyes",
    "😁": "beaming face",
    "😆": "grinning squinting face",
    "😅": "grinning face with sweat",
    "🤣": "rolling on the floor laughing",
    "😂": "face with tears of joy",
    "🙂": "slightly smiling face",
    "🙃": "upside down face",
    "😉": "winking face",
    "😊": "smiling face with smiling eyes",
    "😇": "smiling face with halo",

    "🥰": "smiling face with hearts",
    "😍": "smiling face with heart eyes",
    "🤩": "star struck",
    "😘": "face blowing a kiss",
    "😗": "kissing face",
    "☺️": "smiling face",
    "😚": "kissing face with closed eyes",
    "😋": "face savoring food",
    "😛": "face with tongue",
    "😜": "winking face with tongue",
    "🤪": "zany face",
    "🤗": "hugging face",

    "🫶": "heart hands",
    "❤️": "red heart",
    "🧡": "orange heart",
    "💛": "yellow heart",
    "💚": "green heart",
    "💙": "blue heart",
    "💜": "purple heart",
    "🖤": "black heart",
    "🤍": "white heart",
    "🤎": "brown heart",
    "💔": "broken heart",
    "❤️‍🔥": "heart on fire",
    "❤️‍🩹": "mending heart",

    "😢": "crying face",
    "😭": "loudly crying face",
    "😞": "disappointed face",
    "😔": "pensive face",
    "😟": "worried face",
    "😕": "confused face",
    "🙁": "slightly frowning face",
    "☹️": "frowning face",
    "😣": "persevering face",
    "😖": "confounded face",
    "😫": "tired face",
    "😩": "weary face",
    "🥺": "pleading face",

    "😡": "enraged face",
    "😠": "angry face",
    "🤬": "face with symbols on mouth",
    "😤": "face with steam from nose",
    "💢": "anger symbol",

    "😱": "face screaming in fear",
    "😨": "fearful face",
    "😰": "anxious face with sweat",
    "😥": "sad but relieved face",
    "😓": "downcast face with sweat",
    "🫣": "face with peeking eye",

    "😲": "astonished face",
    "😮": "face with open mouth",
    "😯": "hushed face",
    "😳": "flushed face",
    "🤯": "exploding head",

    "😴": "sleeping face",
    "🥱": "yawning face",
    "😪": "sleepy face",
    "🫠": "melting face",

    "😐": "neutral face",
    "😑": "expressionless face",
    "😶": "face without mouth",
    "🙄": "face with rolling eyes",
    "😬": "grimacing face",
    "🤔": "thinking face",
    "🤨": "face with raised eyebrow",
    "🧐": "face with monocle",

    "😎": "smiling face with sunglasses",
    "🤓": "nerd face",
    "🥳": "partying face",
    "🤠": "cowboy hat face",
    "😏": "smirking face",
    "😌": "relieved face",

    # -------------------------
    # GESTURES
    # -------------------------

    "👍": "thumbs up",
    "👎": "thumbs down",
    "👌": "OK hand",
    "✌️": "victory hand",
    "🤞": "crossed fingers",
    "🤟": "love you gesture",
    "🤘": "sign of the horns",
    "🤙": "call me hand",
    "👏": "clapping hands",
    "🙌": "raising hands",
    "🙏": "folded hands",
    "💪": "flexed biceps",
    "👊": "fist",
    "✊": "raised fist",
    "🤝": "handshake",
    "👋": "waving hand",
    "☝️": "index pointing up",
    "👇": "backhand index pointing down",
    "👉": "backhand index pointing right",
    "👈": "backhand index pointing left",

    # -------------------------
    # FOOD / DRINK
    # -------------------------

    "☕": "hot coffee",
    "🍵": "tea",
    "🧋": "bubble tea",
    "🥤": "cup with straw",
    "🍺": "beer",
    "🍻": "clinking beer mugs",
    "🍷": "wine",
    "🥂": "clinking glasses",
    "🍹": "tropical drink",
    "🍸": "cocktail",
    "🥛": "glass of milk",

    "🍎": "red apple",
    "🍏": "green apple",
    "🍊": "orange",
    "🍋": "lemon",
    "🍌": "banana",
    "🍉": "watermelon",
    "🍇": "grapes",
    "🍓": "strawberry",
    "🍒": "cherries",
    "🍑": "peach",
    "🍍": "pineapple",
    "🥭": "mango",
    "🥥": "coconut",

    "🍕": "pizza",
    "🍔": "hamburger",
    "🍟": "french fries",
    "🌭": "hot dog",
    "🍿": "popcorn",
    "🍩": "doughnut",
    "🍪": "cookie",
    "🎂": "birthday cake",
    "🍰": "cake",
    "🧁": "cupcake",
    "🍫": "chocolate bar",
    "🍭": "lollipop",
    "🍦": "ice cream",

    # -------------------------
    # CELEBRATION / SUCCESS
    # -------------------------

    "🎉": "party popper, celebration, success, achievement, good news",
    "🎊": "confetti ball, celebration",
    "🏆": "trophy, winning, champion",
    "🥇": "gold medal, first place",
    "🥈": "silver medal, second place",
    "🥉": "bronze medal, third place",
    "🏅": "sports medal",
    "🎖️": "military medal",
    "🎁": "gift",
    "🎈": "balloon",
    "✨": "sparkles",
    "🌟": "glowing star",
    "⭐": "star",
    "💫": "dizzy star",
    "🔥": "fire, amazing, trending",
    "🚀": "rocket, growth, launch",
    "💯": "hundred points, perfect",
    "✅": "check mark, completed",
    "✔️": "check mark",
    "💡": "idea, realization, understanding, insight",

    # -------------------------
    # ACTIVITIES
    # -------------------------

    "⚽": "soccer",
    "🏀": "basketball",
    "🏏": "cricket",
    "🏸": "badminton",
    "🎾": "tennis",
    "🏆": "winning trophy",

    "🎮": "video game",
    "🎯": "target, goal",
    "🎵": "music",
    "🎶": "musical notes",
    "🎧": "headphones",
    "🎤": "microphone",
    "🎬": "movie",
    "📚": "books, studying",
    "📖": "open book",
    "✏️": "pencil",
    "📝": "writing",
    "💻": "laptop",
    "📱": "mobile phone",
    "⌨️": "keyboard",

    # -------------------------
    # WEATHER / NATURE
    # -------------------------

    "☀️": "sunny weather",
    "🌞": "sun with face",
    "🌈": "rainbow",
    "☁️": "cloud",
    "🌧️": "rain",
    "⛈️": "thunderstorm",
    "❄️": "snow",
    "🌨️": "snow weather",
    "🌪️": "tornado",
    "🌊": "wave",
    "🌙": "moon",
    "🌚": "new moon face",

    # -------------------------
    # ANIMALS
    # -------------------------

    "🐶": "dog",
    "🐱": "cat",
    "🐭": "mouse",
    "🐹": "hamster",
    "🐰": "rabbit",
    "🦊": "fox",
    "🐻": "bear",
    "🐼": "panda",
    "🐨": "koala",
    "🐯": "tiger",
    "🦁": "lion",
    "🐮": "cow",
    "🐷": "pig",
    "🐸": "frog",
    "🐵": "monkey",
    "🐔": "chicken",
    "🐧": "penguin",
    "🐦": "bird",
    "🦄": "unicorn",
    "🐝": "bee",
    "🦋": "butterfly",
    "🐢": "turtle",
    "🐍": "snake",
    "🐠": "fish",
    "🐬": "dolphin",
    "🐳": "whale",

    # -------------------------
    # TRAVEL
    # -------------------------

    "✈️": "airplane, travel",
    "🚗": "car",
    "🚕": "taxi",
    "🚌": "bus",
    "🚆": "train",
    "🚇": "metro",
    "🚲": "bicycle",
    "🏠": "home",
    "🏫": "school",
    "🏢": "office",
    "🏖️": "beach",
    "🏝️": "island",
    "🗺️": "map",
    "🧳": "luggage",

    # -------------------------
    # MONEY / WORK
    # -------------------------

    "💰": "money bag",
    "💵": "money",
    "💳": "credit card",
    "💎": "diamond",
    "💸": "money flying away",
    "💼": "briefcase, work",
    "📈": "growth chart",
    "📊": "bar chart",
    "💻": "computer, work",

    # -------------------------
    # SYMBOLS
    # -------------------------

    "❤️": "love",
    "❣️": "heart exclamation",
    "💕": "two hearts",
    "💞": "revolving hearts",
    "💓": "beating heart",
    "💗": "growing heart",
    "💖": "sparkling heart",
    "💘": "heart with arrow",
    "💝": "heart with ribbon",

    "⚡": "electricity, energy",
    "❗": "exclamation",
    "❓": "question",
    "‼️": "double exclamation",
    "⁉️": "exclamation question",
    "⚠️": "warning",
    "🚨": "emergency",
    "🔔": "notification",
    "🔒": "locked",
    "🔑": "key",
    "💬": "speech bubble",
    "💭": "thought bubble",

    # -------------------------
    # OTHER
    # -------------------------

    "🎓": "graduation, education",
    "🏫": "school",
    "🧑‍💻": "programmer",
    "👩‍💻": "woman programmer",
    "👨‍💻": "man programmer",
    "🧠": "brain, intelligence",
    "💭": "thinking",
    "🔍": "search",
    "🔬": "science",
    "🧪": "experiment",
    "🛠️": "tools, building",
    "🔧": "repair",
    "❤️‍🔥": "passion",
    "🕺": "dancing",
    "💃": "dancing",
}


# ============================================================
# LOCAL FALLBACK
# ============================================================

def fallback_emoji(text):

    text = text.lower().strip()

    # VERY SPECIFIC situations first
    if any(x in text for x in [
        "coffee",
        "cappuccino",
        "latte",
        "espresso"
    ]):
        return "☕"

    if any(x in text for x in [
        "tea",
        "chai"
    ]):
        return "🍵"

    if any(x in text for x in [
        "selected",
        "got selected",
        "got the job",
        "got a job",
        "dream job",
        "offer letter",
        "won",
        "winner",
        "winning",
        "passed",
        "success",
        "succeeded",
        "achievement",
        "achieved",
        "promoted"
    ]):
        return "🎉"

    if any(x in text for x in [
        "understand",
        "understood",
        "figured it out",
        "realized",
        "realise",
        "realized",
        "great idea",
        "new idea",
        "idea"
    ]):
        return "💡"

    if any(x in text for x in [
        "love",
        "loving",
        "romantic",
        "boyfriend",
        "girlfriend",
        "i love you"
    ]):
        return "❤️"

    if any(x in text for x in [
        "kiss",
        "kissing"
    ]):
        return "😘"

    if any(x in text for x in [
        "happy",
        "happiness",
        "joy",
        "joyful",
        "excited",
        "awesome",
        "amazing",
        "wonderful",
        "glad"
    ]):
        return "😊"

    if any(x in text for x in [
        "sad",
        "unhappy",
        "disappointed",
        "lonely",
        "heartbroken",
        "cry"
    ]):
        return "😢"

    if any(x in text for x in [
        "angry",
        "furious",
        "mad",
        "hate",
        "annoyed",
        "frustrated"
    ]):
        return "😡"

    if any(x in text for x in [
        "scared",
        "afraid",
        "terrified",
        "danger",
        "worried",
        "anxious"
    ]):
        return "😱"

    if any(x in text for x in [
        "tired",
        "sleepy",
        "exhausted",
        "drained"
    ]):
        return "😴"

    if any(x in text for x in [
        "surprised",
        "surprise",
        "shocked",
        "shock",
        "unexpected"
    ]):
        return "😲"

    if any(x in text for x in [
        "funny",
        "joke",
        "laugh",
        "laughing",
        "hilarious",
        "lol"
    ]):
        return "😂"

    if any(x in text for x in [
        "confused",
        "confusion",
        "don't understand",
        "dont understand"
    ]):
        return "😕"

    if any(x in text for x in [
        "bored",
        "boring",
        "nothing to do"
    ]):
        return "😑"

    if any(x in text for x in [
        "cool",
        "stylish",
        "chill",
        "swag"
    ]):
        return "😎"

    if any(x in text for x in [
        "study",
        "studying",
        "exam",
        "school",
        "college",
        "book",
        "learning"
    ]):
        return "📚"

    if any(x in text for x in [
        "cricket"
    ]):
        return "🏏"

    if any(x in text for x in [
        "badminton"
    ]):
        return "🏸"

    if any(x in text for x in [
        "football",
        "soccer"
    ]):
        return "⚽"

    if any(x in text for x in [
        "basketball"
    ]):
        return "🏀"

    return "🙂"


# ============================================================
# BUILD EMOJI CATALOG FOR AI
# ============================================================

def build_emoji_catalog():

    lines = []

    for emoji, meaning in EMOJI_LIST.items():
        lines.append(f"{emoji} = {meaning}")

    return "\n".join(lines)


EMOJI_CATALOG = build_emoji_catalog()


# ============================================================
# AI PREDICTION
# ============================================================

def predict_with_ai(text):

    prompt = f"""
You are an advanced emoji prediction engine.

Your task is to understand the COMPLETE meaning and intent of a
user's sentence and select the SINGLE BEST matching emoji.

DO NOT simply look for one keyword.

Understand:
- emotion
- action
- object
- situation
- context
- intention
- event
- activity
- relationship
- achievement
- food/drink
- travel
- sports
- study
- work

IMPORTANT:

Return ONLY valid JSON.

The JSON MUST have exactly these fields:

{{
  "emoji": "one emoji",
  "reason": "very short reason"
}}

The "emoji" MUST be exactly one emoji from the catalog below.

The "reason" must be short.

============================================================
SPECIAL CLASSIFICATION RULES
============================================================

SUCCESS / ACHIEVEMENT:

If the sentence says someone:
- got selected
- got a job
- got an offer
- won something
- passed an exam
- achieved a goal
- succeeded
- got promoted
- became a winner
- received good news

prefer:

🎉

Examples:

"I finally got selected after months of preparation"
-> 🎉

"I got selected for the company"
-> 🎉

"I finally got my dream job"
-> 🎉

"I passed my exam"
-> 🎉

"I won the competition"
-> 🎉

"I achieved my goal"
-> 🎉


UNDERSTANDING / IDEA:

If the sentence describes:
- understanding something
- realizing something
- discovering an idea
- figuring something out
- having an idea
- gaining insight

prefer:

💡

Examples:

"Now I understand the concept"
-> 💡

"I finally figured it out"
-> 💡

"I got a great idea"
-> 💡


FOOD AND DRINK:

If the sentence specifically asks for or describes drinking coffee:

☕

If the sentence specifically asks for or describes drinking tea:

🍵

Examples:

"I wanna drink some coffee"
-> ☕

"I need a cup of coffee"
-> ☕

"I wanna drink some tea"
-> 🍵

"I want chai"
-> 🍵


LOVE:

If the sentence expresses romantic love or affection:

❤️

Examples:

"I love you"
-> ❤️

"She is the love of my life"
-> ❤️


CELEBRATION:

If the sentence describes celebration, congratulations,
success or a major happy event:

🎉


STUDY:

If the main topic is studying, books, education or exams
WITHOUT a success/achievement context:

📚


============================================================
EMOJI CATALOG
============================================================

{EMOJI_CATALOG}

============================================================
USER SENTENCE
============================================================

{text}

============================================================
FINAL INSTRUCTION
============================================================

Understand the entire sentence.

Choose the SINGLE most appropriate emoji.

Return ONLY JSON in this exact structure:

{{
  "emoji": "emoji",
  "reason": "short reason"
}}
"""

    print("Sending request to Hugging Face...")
    print("Model:", MODEL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an emoji classification system. "
                    "Follow the user's output format exactly."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=150,
        temperature=0.1
    )

    # --------------------------------------------------------
    # Extract response safely
    # --------------------------------------------------------

    if not response:
        raise ValueError("Hugging Face returned an empty response.")

    if not response.choices:
        raise ValueError("Hugging Face returned no choices.")

    message = response.choices[0].message

    raw = message.content

    print("RAW AI RESPONSE:")
    print(repr(raw))

    if not raw:
        raise ValueError("AI returned empty content.")

    raw = raw.strip()

    # --------------------------------------------------------
    # Remove markdown code fences if model adds them
    # --------------------------------------------------------

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    raw = raw.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    data = None

    try:
        data = json.loads(raw)

    except json.JSONDecodeError:

        # Sometimes models put extra text around JSON.
        match = re.search(
            r'\{.*\}',
            raw,
            flags=re.DOTALL
        )

        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):

        # Last-resort extraction of emoji from response
        for emoji in EMOJI_LIST:

            if emoji in raw:

                print("Recovered emoji from raw response:", emoji)

                return {
                    "emoji": emoji,
                    "reason": "Recovered from model response."
                }

        raise ValueError(
            f"AI did not return valid JSON. Response: {raw}"
        )

    emoji = data.get("emoji", "").strip()
    reason = data.get("reason", "").strip()

    # --------------------------------------------------------
    # Validate emoji
    # --------------------------------------------------------

    if emoji not in EMOJI_LIST:

        # Try to find one valid emoji inside returned text
        for supported_emoji in EMOJI_LIST:

            if supported_emoji in emoji:

                emoji = supported_emoji
                break

    if emoji not in EMOJI_LIST:

        raise ValueError(
            f"AI returned unsupported emoji: {emoji}"
        )

    if not reason:
        reason = EMOJI_LIST[emoji]

    print("AI EMOJI:", emoji)
    print("AI REASON:", reason)

    return {
        "emoji": emoji,
        "reason": reason
    }


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
        "model": MODEL,
        "provider": "auto",
        "huggingface": (
            "configured"
            if HF_TOKEN
            else "missing"
        ),
        "emoji_count": len(EMOJI_LIST)
    })


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

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

    print("\n")
    print("=" * 70)
    print("NEW EMOJI REQUEST")
    print("=" * 70)
    print("TEXT:", text)
    print("MODEL:", MODEL)
    print(
        "HF TOKEN:",
        "CONFIGURED" if HF_TOKEN else "MISSING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # AI prediction
    # --------------------------------------------------------

    if HF_TOKEN:

        try:

            result = predict_with_ai(text)

            return jsonify({
                "emoji": result["emoji"],
                "reason": result["reason"],
                "source": "AI"
            })

        except Exception as e:

            print("=" * 70)
            print("AI ERROR")
            print("=" * 70)
            print("ERROR TYPE:", type(e).__name__)
            print("ERROR MESSAGE:", str(e))
            print("=" * 70)

    # --------------------------------------------------------
    # Local fallback
    # --------------------------------------------------------

    print("Using local fallback.")

    emoji = fallback_emoji(text)

    return jsonify({
        "emoji": emoji,
        "reason": EMOJI_LIST.get(
            emoji,
            "Local fallback prediction."
        ),
        "source": "fallback"
    })


# ============================================================
# RUN SERVER
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
    print("              EMOJI GENERATOR AI")
    print("=" * 60)
    print("Model      :", MODEL)
    print("Provider   : Hugging Face Auto")
    print("HF Token   :", "CONFIGURED" if HF_TOKEN else "MISSING")
    print("Emoji count:", len(EMOJI_LIST))
    print("Port       :", port)
    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=port
    )