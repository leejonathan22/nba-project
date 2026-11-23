import sqlite3
import pandas as pd
import numpy as np
from dagster import asset, Definitions, define_asset_job, ScheduleDefinition


def ml_to_prob(ml):
    """American moneyline = implied probability. Invalid odds = NaN."""
    try:
        ml = float(ml)
    except (ValueError, TypeError):
        return np.nan
    if ml <= -100:
        return -ml / (-ml + 100)
    if ml >= 100:
        return 100 / (ml + 100)
    return np.nan


# Step 1: Load raw CSVs
@asset
def raw_nba_data():
    games = pd.read_csv("data/nba_games_all.csv", dtype={"game_id": str})
    odds = pd.read_csv("data/nba_betting_money_line.csv", dtype={"game_id": str})
    return {"games": games, "odds": odds}


# Step 2: Clean data and build features
@asset
def cleaned_nba_features(raw_nba_data):
    games = raw_nba_data["games"].copy()
    odds = raw_nba_data["odds"].copy()

    # Sort games by date so rolling stats only use past games
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # Each team's win rate over its previous 10 games (home and away)
    games["win"] = (games["wl"] == "W").astype(int)
    games["roll_win_rate"] = (
        games.groupby("team_id")["win"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(0.5)
        .round(3)
    )

    # Keep one row per game: the home team's row
    home = games[games["is_home"].isin([1, "1", True, "t", "T"])].drop_duplicates("game_id")
    home = home.rename(columns={"win": "home_win", "roll_win_rate": "home_roll_win_rate"})

    # Odds: one row per sportsbook = average to one line per game
    # price1 = away team, price2 = home team
    odds["away_raw"] = odds["price1"].apply(ml_to_prob)
    odds["home_raw"] = odds["price2"].apply(ml_to_prob)
    odds = odds.groupby("game_id", as_index=False).agg(
        away_raw=("away_raw", "mean"),
        home_raw=("home_raw", "mean"),
        home_ml=("price2", "median"),
    )

    # Remove the vig so home + away probabilities sum to 1
    odds["mkt_home_prob"] = (odds["home_raw"] / (odds["home_raw"] + odds["away_raw"])).round(4)

    # Merge games with odds
    df = home[["game_id", "game_date", "season_year", "team_id", "home_win", "home_roll_win_rate"]].merge(
        odds[["game_id", "mkt_home_prob", "home_ml"]], on="game_id"
    )
    df = df.dropna(subset=["mkt_home_prob"])
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")

    print(f"Games after dedupe/merge: {len(df)}")
    return df


# Step 3: Save to SQLite
@asset
def nba_database(cleaned_nba_features):
    conn = sqlite3.connect("data/nba_analytics.db")
    cleaned_nba_features.to_sql("nba_game_features", conn, if_exists="replace", index=False)
    conn.close()
    return "Database updated in data/nba_analytics.db"


# Dagster job + weekly schedule
nba_job = define_asset_job(name="nba_pipeline_job")
weekly_schedule = ScheduleDefinition(job=nba_job, cron_schedule="0 8 * * 1")

defs = Definitions(
    assets=[raw_nba_data, cleaned_nba_features, nba_database],
    schedules=[weekly_schedule],
)