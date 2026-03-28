import os
import pickle

import pandas as pd

from preprocess import FEATURE_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/model.pkl")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_model = load_model()


def predict_play(down: int, ydstogo: int, yardline_100: int, qtr: int,
                 score_differential: float, game_seconds_remaining: float) -> dict:
    play = pd.DataFrame([{
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "qtr": qtr,
        "score_differential": score_differential,
        "game_seconds_remaining": game_seconds_remaining,
    }])[FEATURE_COLUMNS]

    proba = _model.predict_proba(play)[0]
    prediction = "pass" if proba[1] >= 0.5 else "run"

    return {
        "prediction": prediction,
        "run_probability": round(float(proba[0]), 3),
        "pass_probability": round(float(proba[1]), 3),
    }
