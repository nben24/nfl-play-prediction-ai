from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


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


def _build_reasoning(down: int, ydstogo: int, yardline_100: int,
                     score_diff: float, qtr: int, seconds: float,
                     prediction: str) -> list[str]:
    """Returns bullet points of factual reasoning. The LLM only paraphrases these."""
    points = []

    # Down and distance — always the primary reason
    if down == 1:
        if prediction == "run":
            points.append(f"1st and {ydstogo} is a balanced down where teams commonly run to set up the drive.")
        else:
            points.append(f"1st and {ydstogo} is a balanced down, and a pass opens up more yardage.")
    elif down == 2:
        if ydstogo <= 3:
            points.append(f"2nd and {ydstogo} is short yardage — a run can pick up the conversion efficiently.")
        elif ydstogo <= 6:
            points.append(f"2nd and {ydstogo} is manageable and allows flexible play-calling.")
        else:
            points.append(f"2nd and {ydstogo} is a long distance, creating pressure to pass and recover yards.")
    elif down in (3, 4):
        if ydstogo <= 2:
            points.append(f"{down}{'rd' if down == 3 else 'th'} and {ydstogo} is short yardage — a run or quick pass can pick up the {ydstogo} yard(s) needed to convert.")
        elif ydstogo <= 5:
            points.append(f"{down}{'rd' if down == 3 else 'th'} and {ydstogo} is a must-convert down where a short pass or run is viable.")
        else:
            points.append(f"{down}{'rd' if down == 3 else 'th'} and {ydstogo} is a clear passing situation — {ydstogo} yards is too far to reliably gain on the ground.")

    # Field position
    own_yard = 100 - yardline_100
    if yardline_100 <= 5:
        points.append(f"At the {yardline_100}-yard line, this is a goal-line situation — the end zone is right there.")
    elif yardline_100 <= 10:
        points.append(f"Inside the 10-yard line, the field compresses sharply and passing lanes to the end zone open up.")
    elif yardline_100 <= 20:
        points.append(f"In the red zone ({yardline_100} yards out), the compressed field opens passing lanes to the end zone.")
    elif yardline_100 > 65:
        points.append(f"Deep in their own territory (own {own_yard}-yard line), teams protect against turnovers by avoiding risky throws.")

    # Score + time — handle critical combinations first
    pts = abs(int(score_diff))
    is_late = qtr == 4 and seconds <= 300
    if qtr == 4 and seconds <= 60 and -8 <= score_diff <= -1 and yardline_100 <= 10:
        points.append(f"The offense is losing by {pts} point(s) — not tied — with only {int(seconds)} seconds left; a touchdown here wins the game.")
    elif qtr == 4 and seconds <= 120 and score_diff < 0:
        points.append(f"With under 2 minutes left and trailing by {pts}, the offense is in hurry-up mode and must pass to have any chance.")
    elif score_diff <= -9:
        points.append(f"Trailing by {pts}, the offense must pass to have any realistic chance of catching up.")
    elif score_diff >= 9 and qtr >= 3:
        points.append(f"Leading by {pts} in the second half, running the ball helps drain the clock.")
    elif score_diff == 0 and is_late:
        points.append("Tied late in the 4th quarter, both teams are playing aggressively to take the lead.")

    # Two-minute warning at end of first half
    if qtr == 2 and seconds <= 120:
        points.append("With the half ending, the offense may push for a quick score before halftime.")

    return points


def explain_prediction(play_context: dict, prediction: dict) -> str:
    qtr = play_context['qtr']
    down = play_context['down']
    ydstogo = play_context['ydstogo']
    yardline_100 = play_context['yardline_100']
    score_diff = play_context['score_differential']
    seconds = play_context['game_seconds_remaining']

    reasoning = _build_reasoning(down, ydstogo, yardline_100, score_diff, qtr, seconds, prediction['prediction'])
    bullets = "\n".join(f"- {r}" for r in reasoning)

    prompt = (
        f"Predicted play: {prediction['prediction'].upper()}\n\n"
        f"Key reasons:\n{bullets}\n\n"
        f"Rewrite these points as a single flowing paragraph of 2-3 sentences. "
        f"No line breaks between sentences. No quotation marks. Do not reference probabilities or model confidence. "
        f"Do not add any new facts."
    )

    response = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
    )

    text = response.choices[0].message.content.strip()
    # Strip any preamble the model adds before the actual commentary
    for prefix in ("Here's your commentary:", "Here's the commentary:", "Commentary:"
                   "Here's my commentary:", "Sure!", "Sure,"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text
