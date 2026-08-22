# 🌟 Emoji Generator

An AI-powered contextual emoji generator that understands the meaning and context of user-provided text and recommends the most relevant Unicode emoji.

Unlike a traditional keyword-based emoji system, this project uses a Large Language Model through the Hugging Face Inference API to analyze the complete sentence, including its emotion, objects, activities, situations, and intent.

## ✨ Features

- 🧠 Contextual text understanding
- 🤖 AI-powered emoji prediction
- 🤗 Hugging Face Inference API integration
- 💬 Understands complete sentences and their context
- 🌍 Supports a wide range of Unicode emojis
- 🎯 Selects an emoji based on the dominant meaning of the text
- 🔐 Secure API key management using environment variables
- 🎨 Interactive web interface
- ⚡ Flask REST API backend
- 🚀 Deployable as a web service

## 🧠 How It Works

The application follows this pipeline:

```text
User enters text
       ↓
Frontend (HTML + JavaScript)
       ↓
Flask REST API
       ↓
Hugging Face Inference API
       ↓
Qwen/Qwen2.5-7B-Instruct
       ↓
Contextual analysis
       ↓
Most relevant emoji
       ↓
Displayed in the UI
