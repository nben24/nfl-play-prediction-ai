import sys

from predict import predict_play
from llm_explainer import explain_prediction


def main():
    print("=== NFL Play Predictor ===\n")

    try:
        qtr = int(input("Quarter (1-4): "))
        down = int(input("Down (1-4): "))
        ydstogo = int(input("Yards to go: "))
        yardline_100 = int(input("Yard line (distance from opponent's end zone, 1-99): "))
        score_differential = float(input("Score differential (offense - defense): "))
        game_seconds_remaining = float(input("Seconds remaining in game (max 3600): "))
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        sys.exit(1)

    errors = []
    if not 1 <= qtr <= 4:
        errors.append("Quarter must be between 1 and 4.")
    if not 1 <= down <= 4:
        errors.append("Down must be between 1 and 4.")
    if not 1 <= ydstogo <= 99:
        errors.append("Yards to go must be between 1 and 99.")
    if not 1 <= yardline_100 <= 99:
        errors.append("Yard line must be between 1 and 99.")
    if not 0 <= game_seconds_remaining <= 3600:
        errors.append("Seconds remaining must be between 0 and 3600.")
    if errors:
        for e in errors:
            print(e)
        sys.exit(1)

    play_context = {
        "qtr": qtr,
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "score_differential": score_differential,
        "game_seconds_remaining": game_seconds_remaining,
    }

    result = predict_play(**play_context)

    print(f"\nPrediction: {result['prediction'].upper()}")
    print(f"  Run probability:  {result['run_probability']:.1%}")
    print(f"  Pass probability: {result['pass_probability']:.1%}")

    print("\nExplanation (generating...)")
    explanation = explain_prediction(play_context, result)
    print(explanation)


if __name__ == "__main__":
    main()
