"""
get_data.py

Downloads NFL play-by-play data from nflverse and saves it as nfl_plays.csv.

Usage:
    python data/get_data.py [season]

    season  Optional. 4-digit year (default: 2025).

Data source: https://github.com/nflverse/nflverse-data
"""

import sys
import pandas as pd

COLUMNS = [
    "play_type",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "score_differential",
    "game_seconds_remaining",
    "posteam",
]

OUTPUT_PATH = "data/nfl_plays.csv"


def download(season: int = 2025) -> None:
    url = (
        f"https://github.com/nflverse/nflverse-data/releases/download/pbp/"
        f"play_by_play_{season}.csv.gz"
    )
    print(f"Downloading {season} play-by-play data...")
    df = pd.read_csv(url, usecols=COLUMNS, compression="gzip", low_memory=False)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    download(season)
