import pandas as pd

season = 2025

url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"

cols = [
    "play_type",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "score_differential",
    "game_seconds_remaining",
    "posteam"
]

df = pd.read_csv(url, usecols=cols)
df.to_csv(f"nfl_pbp_{season}_modeling.csv", index=False)

print(df.head())
print(df.shape)