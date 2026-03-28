import pandas as pd


FEATURE_COLUMNS = [
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "score_differential",
    "game_seconds_remaining"
]


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Keep only run and pass plays
    df = df[df["play_type"].isin(["run", "pass"])]

    # Keep only needed columns
    keep_cols = FEATURE_COLUMNS + ["play_type"]
    df = df[keep_cols]

    # Drop missing values
    df = df.dropna()

    # Convert target to numeric
    df["target"] = df["play_type"].map({"run": 0, "pass": 1})

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    return X, y