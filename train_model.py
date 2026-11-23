import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Load data
conn = sqlite3.connect("data/nba_analytics.db")
df = pd.read_sql_query("SELECT * FROM nba_game_features", conn)
conn.close()

# Time-based split: train on earliest 80% of games, test on most recent 20%
df = df.sort_values("game_date").reset_index(drop=True)
cut = int(len(df) * 0.8)
df["is_test"] = (df.index >= cut).astype(int)
train, test = df[df.is_test == 0], df[df.is_test == 1]
print(f"Train: {train.game_date.min()} to {train.game_date.max()} ({len(train)} games)")
print(f"Test:  {test.game_date.min()} to {test.game_date.max()} ({len(test)} games)")

# Train on team stats only (market odds excluded so the model forms its own view)
features = ["home_roll_win_rate"]
model = LogisticRegression()
model.fit(train[features], train["home_win"])

# Compare model vs market on the test set
probs = model.predict_proba(test[features])[:, 1]
mkt = test["mkt_home_prob"].clip(0.01, 0.99)
print("\n=== TEST SET PERFORMANCE ===")
print(f"Model  accuracy: {accuracy_score(test.home_win, probs > 0.5):.4f}   log loss: {log_loss(test.home_win, probs):.4f}")
print(f"Market accuracy: {accuracy_score(test.home_win, mkt > 0.5):.4f}   log loss: {log_loss(test.home_win, mkt):.4f}")

# Save predictions
df["model_home_prob"] = model.predict_proba(df[features])[:, 1].round(4)
conn = sqlite3.connect("data/nba_analytics.db")
df.to_sql("nba_game_features", conn, if_exists="replace", index=False)
conn.close()
print("\nPredictions saved (is_test=1 marks held-out games)")