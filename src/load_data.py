import pandas as pd


def load_data(file_path: str = "data/nfl_plays.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df