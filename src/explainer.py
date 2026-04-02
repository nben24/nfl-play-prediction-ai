"""
explainer.py

Generates natural-language explanations for run/pass predictions.

Architecture:
  1. _build_reasoning() converts the game situation into factual bullet points
     using deterministic rules — no LLM involved.
  2. explain_prediction() sends those bullet points to a local LLM whose only
     job is to paraphrase them into fluent analyst commentary.

This separation means the LLM cannot hallucinate game facts — it can only
rephrase what the rules engine provides.
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

OLLAMA_MODEL = "llama3.2:3b"
MAX_TOKENS = 120

SYSTEM_PROMPT = (
    "You are an NFL analyst. You will be given bullet points explaining why a play was predicted. "
    "Rewrite them as a single continuous paragraph of 2-3 sentences.\n\n"
    "Rules:\n"
    "- Write as one flowing paragraph — do NOT use line breaks between sentences.\n"
    "- Do NOT wrap sentences or the paragraph in quotation marks.\n"
    "- Do NOT add facts, stats, or reasoning not in the bullet points.\n"
    "- Do NOT use words like 'inevitable' or 'guaranteed' — outcomes are never certain.\n"
    "- Be direct and factual. This is broadcast commentary, not an essay."
)

# Prefixes the LLM sometimes adds before the actual commentary — strip them.
_LLM_PREAMBLES = (
    "Here's your commentary:",
    "Here's the commentary:",
    "Commentary:",
    "Here's my commentary:",
    "Sure!",
    "Sure,",
)


def _build_reasoning(
    down: int,
    ydstogo: int,
    yardline_100: int,
    score_diff: float,
    qtr: int,
    seconds: float,
    prediction: str,
) -> list[str]:
    """
    Build a list of factual bullet points explaining the prediction.

    All reasoning is rule-based and deterministic. The LLM only paraphrases
    the output of this function — it does not reason independently.
    """
    points: list[str] = []

    # --- Down and distance (primary driver of play-calling) ---
    if down == 1:
        if prediction == "run":
            points.append(f"1st and {ydstogo} is a balanced down where teams commonly run to establish the drive.")
        else:
            points.append(f"1st and {ydstogo} is a balanced down, and a pass opens up more yardage early.")
    elif down == 2:
        if ydstogo <= 3:
            points.append(f"2nd and {ydstogo} is short yardage — a run can pick up the conversion efficiently.")
        elif ydstogo <= 6:
            points.append(f"2nd and {ydstogo} is manageable distance, allowing flexible play-calling.")
        else:
            points.append(f"2nd and {ydstogo} is a long distance, creating pressure to pass and recover yards.")
    elif down in (3, 4):
        suffix = "rd" if down == 3 else "th"
        if ydstogo <= 2:
            points.append(
                f"{down}{suffix} and {ydstogo} is short yardage — a run or quick pass can pick up "
                f"the {ydstogo} yard(s) needed to convert."
            )
        elif ydstogo <= 5:
            points.append(f"{down}{suffix} and {ydstogo} is a must-convert down where a short pass or run is viable.")
        else:
            points.append(
                f"{down}{suffix} and {ydstogo} is a clear passing situation — "
                f"{ydstogo} yards is too far to reliably gain on the ground."
            )

    # --- Field position ---
    own_yard = 100 - yardline_100
    if yardline_100 <= 5:
        points.append(f"At the {yardline_100}-yard line, this is a goal-line situation — the end zone is right there.")
    elif yardline_100 <= 10:
        points.append("Inside the 10-yard line, the field compresses and passing lanes to the end zone open up.")
    elif yardline_100 <= 20:
        points.append(f"In the red zone ({yardline_100} yards out), the compressed field opens passing lanes to the end zone.")
    elif yardline_100 > 65:
        points.append(
            f"Deep in their own territory (own {own_yard}-yard line), "
            "teams protect the ball by avoiding risky throws."
        )

    # --- Score and time context ---
    pts = abs(int(score_diff))

    # Handle critical late-game combinations before general score logic
    if qtr == 4 and seconds <= 60 and -8 <= score_diff <= -1 and yardline_100 <= 10:
        points.append(
            f"The offense is losing by {pts} point(s) — not tied — with only {int(seconds)} seconds left; "
            "a touchdown here wins the game."
        )
    elif qtr == 4 and seconds <= 120 and score_diff < 0:
        points.append(
            f"With under 2 minutes left and trailing by {pts}, "
            "the offense is in hurry-up mode and must pass to have any chance."
        )
    elif score_diff <= -9:
        points.append(f"Trailing by {pts}, the offense must pass to have any realistic chance of catching up.")
    elif score_diff >= 9 and qtr >= 3:
        points.append(f"Leading by {pts} in the second half, running the ball helps drain the clock.")
    elif score_diff == 0 and qtr == 4 and seconds <= 300:
        points.append("Tied late in the 4th quarter, both teams are playing aggressively to take the lead.")

    if qtr == 2 and seconds <= 120:
        points.append("With the half ending, the offense may push for a quick score before halftime.")

    return points


def explain_prediction(play_context: dict, prediction: dict) -> str:
    """
    Generate a natural-language explanation for a run/pass prediction.

    Calls a local Ollama LLM to paraphrase rule-based bullet points into
    analyst commentary. Requires Ollama running at localhost:11434.
    """
    reasoning = _build_reasoning(
        down=play_context["down"],
        ydstogo=play_context["ydstogo"],
        yardline_100=play_context["yardline_100"],
        score_diff=play_context["score_differential"],
        qtr=play_context["qtr"],
        seconds=play_context["game_seconds_remaining"],
        prediction=prediction["prediction"],
    )
    bullets = "\n".join(f"- {r}" for r in reasoning)

    prompt = (
        f"Predicted play: {prediction['prediction'].upper()}\n\n"
        f"Key reasons:\n{bullets}\n\n"
        "Rewrite these points as a single flowing paragraph of 2-3 sentences. "
        "No line breaks between sentences. No quotation marks. "
        "Do not reference probabilities or model confidence. "
        "Do not add any new facts."
    )

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
    )

    text = response.choices[0].message.content.strip()

    for prefix in _LLM_PREAMBLES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    return text
