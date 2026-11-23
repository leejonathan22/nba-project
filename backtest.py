import sqlite3
import numpy as np
import pandas as pd

STAKE = 100   # dollars per bet
EDGE = 0.05   # bet when model prob beats market prob by this much

# Load held-out test games only
conn = sqlite3.connect("data/nba_analytics.db")
df = pd.read_sql_query("SELECT * FROM nba_game_features", conn)
conn.close()
if "model_home_prob" not in df.columns:
    raise SystemExit("Run train_model.py first.")
test = df[df["is_test"] == 1].copy()

# Home moneyline for each game; invalid odds (e.g. 0) fall back to no-vig fair odds
p = test["mkt_home_prob"]
fair_ml = np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)
bad = test["home_ml"].isna() | (test["home_ml"].abs() < 100)
test["bet_ml"] = np.where(bad, fair_ml, test["home_ml"])
if bad.any():
    print(f"Warning: {bad.sum()} games had no valid home moneyline; used fair odds.")

# Profit on a winning bet at American odds
ml = test["bet_ml"]
test["win_payout"] = np.where(ml > 0, STAKE * ml / 100, STAKE * 100 / ml.abs())

# How far the model disagrees with the market
edge = test["model_home_prob"] - test["mkt_home_prob"]
print(f"Edge range: {edge.min():+.3f} to {edge.max():+.3f}")

# Place bets where the model sees enough edge
bets = test[edge > EDGE].copy()
bets["profit"] = np.where(bets["home_win"] == 1, bets["win_payout"], -STAKE)

n = len(bets)
print("\n=== BACKTEST (test set only) ===")
print(f"Test games: {len(test)}")
print(f"Min edge: {EDGE:.0%}")
print(f"Bets placed: {n}")
if n:
    wins = int(bets["home_win"].sum())
    profit = bets["profit"].sum()
    breakeven = (STAKE / (STAKE + bets["win_payout"])).mean()
    print(f"Bets won: {wins} ({wins / n:.2%})")
    print(f"Break-even win rate: {breakeven:.2%}")
    print(f"Avg market prob: {bets['mkt_home_prob'].mean():.3f}")
    print(f"Total profit: ${profit:,.0f} on ${n * STAKE:,} wagered")
    print(f"ROI: {profit / (n * STAKE):.2%}")