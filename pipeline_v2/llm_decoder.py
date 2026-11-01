"""
pipeline_v2/llm_decoder.py

Model 3b — Gemma 2B LLM Word Predictor (via Ollama)
=====================================================
This module runs AFTER the visual classifier has produced a recognized
sentence gloss (e.g. "help_me"). It queries the Gemma 2B LLM (served by
Ollama locally) to:

  1. Predict the NEXT WORD after the recognized sentence
     (the professor's "predict the word coming after the next word" requirement)

  2. Produce a fluent English translation/completion of the ISL gloss sequence

Why Ollama (not loading the model directly in Python):
  - Ollama runs Gemma 2B in 4-bit quantized form (~2 GB RAM)
  - It runs as a standalone background service — isolated from your Python env
  - Zero impact on your laptop when idle (no model weights in RAM)
  - Simple REST API; no HuggingFace downloads or GPU memory management needed

Setup (one-time, run in Terminal):
    brew install ollama
    ollama pull gemma2:2b
    ollama serve          # start the local server (keep this terminal open)

Then use this module in your code.
"""

import json
import requests
from typing import Optional

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"


# ─── ISL gloss → plain English sentence map (expand as needed) ────────────────
# These are the 97 sentence-level glosses from the corpus.
# The model uses these as context for its predictions.
GLOSS_TO_ENGLISH = {
    "he_is_going_into_the_room":          "He is going into the room",
    "this_place_is_beautiful":            "This place is beautiful",
    "are_you_free_today":                 "Are you free today?",
    "are_you_hiding_something":           "Are you hiding something?",
    "bring_water_for_me":                 "Bring water for me",
    "can_i_help_you":                     "Can I help you?",
    "can_you_repeat_that_please":         "Can you repeat that please?",
    "help_me":                            "Help me",
    "hi_how_are_you":                     "Hi, how are you?",
    "how_are_things":                     "How are things?",
    "i_am_afraid_of_that":               "I am afraid of that",
    "i_am_feeling_bored":                "I am feeling bored",
    "nice_to_meet_you":                  "Nice to meet you",
    "thank_you_so_much":                 "Thank you so much",
    "what_are_you_doing":                "What are you doing?",
    "where_are_you_from":                "Where are you from?",
    "who_are_you":                       "Who are you?",
    "why_are_you_crying":                "Why are you crying?",
    "you_are_welcome":                   "You are welcome",
    # ... add more as needed; the LLM handles unknown glosses gracefully
}


def is_ollama_running() -> bool:
    """Check if Ollama server is up."""
    try:
        r = requests.get("http://localhost:11434/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def predict_next_word(
    recognized_gloss: str,
    conversation_history: Optional[list[str]] = None,
    max_tokens: int = 20,
) -> dict:
    """
    Given the recognized ISL sentence gloss, predict the most likely
    next word a signer might sign next.

    Parameters
    ----------
    recognized_gloss      : slug from the label_map, e.g. "help_me"
    conversation_history  : list of previous English sentences (for context)
    max_tokens            : how many tokens to generate (keep small for speed)

    Returns
    -------
    dict with keys:
        recognized_english : str  — plain English of the recognized sign
        next_word          : str  — single predicted next word
        full_response      : str  — raw LLM output
    """
    english = GLOSS_TO_ENGLISH.get(
        recognized_gloss,
        recognized_gloss.replace("_", " ").capitalize()
    )

    # Build context
    history_str = ""
    if conversation_history:
        history_str = "\n".join(
            f"- {s}" for s in conversation_history[-5:]  # last 5 exchanges
        )
        history_str = f"Conversation so far:\n{history_str}\n\n"

    prompt = (
        f"You are helping with Indian Sign Language recognition. "
        f"A signer just signed the sentence: \"{english}\"\n"
        f"{history_str}"
        f"What is the single most likely NEXT WORD the signer will sign next? "
        f"Reply with ONLY that one word, nothing else."
    )

    if not is_ollama_running():
        return {
            "recognized_english": english,
            "next_word":          "[Ollama not running — start with: ollama serve]",
            "full_response":      "",
        }

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,   # low temp = deterministic
                "top_p":       0.9,
            }
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        next_word = data.get("response", "").strip().split()[0] if data.get("response") else ""
        return {
            "recognized_english": english,
            "next_word":          next_word,
            "full_response":      data.get("response", "").strip(),
        }
    except Exception as e:
        return {
            "recognized_english": english,
            "next_word":          f"[ERROR: {e}]",
            "full_response":      "",
        }


def translate_gloss_sequence(
    gloss_sequence: list[str],
    max_tokens: int = 100,
) -> str:
    """
    Given a sequence of recognized sentence glosses (e.g. from a longer
    signing session), ask Gemma to produce a fluent English translation.

    Parameters
    ----------
    gloss_sequence : list of recognized slugs in order
    max_tokens     : generation limit

    Returns
    -------
    str — fluent English translation
    """
    english_sentences = [
        GLOSS_TO_ENGLISH.get(g, g.replace("_", " ").capitalize())
        for g in gloss_sequence
    ]
    signs_str = " → ".join(english_sentences)

    prompt = (
        f"The following Indian Sign Language sentences were signed in order:\n"
        f"{signs_str}\n\n"
        f"Write a single fluent English paragraph that captures the meaning "
        f"of this signing sequence naturally. Be concise."
    )

    if not is_ollama_running():
        return "[Ollama not running — start with: ollama serve]"

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.5}
        }, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR: {e}]"


# ─── Demo / test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Checking Ollama server...")
    if not is_ollama_running():
        print("  ✗ Ollama is not running.")
        print("  Start it with: ollama serve")
        print("  Pull Gemma 2B with: ollama pull gemma2:2b")
    else:
        print("  ✓ Ollama is running\n")

        print("Test 1: Next-word prediction")
        result = predict_next_word("help_me")
        print(f"  Recognized: {result['recognized_english']}")
        print(f"  Predicted next word: {result['next_word']}\n")

        print("Test 2: Gloss sequence translation")
        seq = ["hi_how_are_you", "i_am_feeling_bored", "help_me"]
        translation = translate_gloss_sequence(seq)
        print(f"  Translation: {translation}")

