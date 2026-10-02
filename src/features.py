"""Shared constants + loaders. Uditi's Stage 2 code can import from here too."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "olympics_features.csv"

# Columns that are lookups or targets, never features
NON_FEATURE_COLS = ["noc", "country", "medals", "won_medal"]

CSV_COLS = [
    "noc", "country", "year", "gdp", "gdp_per_capita", "gdp_growth",
    "pct_world_gdp", "population", "pct_world_pop", "pop_growth", "land_area",
    "athletes", "pct_athletes", "medals_last_games", "total_medals_awarded",
    "medals", "won_medal",
]

# every column except noc, country, medals, won_medal
FEATURE_COLS = [c for c in CSV_COLS if c not in NON_FEATURE_COLS]

TRAIN_YEARS = (1988, 2008)   # inclusive
VAL_YEAR = 2012
TEST_YEAR = 2016             # final test, owned by Stage 2 / Uditi. Never touch in Stage 1.


def load_data(path=CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
