import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect("data/nba_analytics.db")

# Query 1: Total games count
df_count = pd.read_sql_query("SELECT COUNT(*) AS total_games FROM nba_game_features;", conn)
print("=== TOTAL GAMES PROCESSED ===")
print(df_count)

# Query 2: Aggregate market implied win probability vs actual win rate
query_market = """
SELECT 
    CASE 
        WHEN mkt_home_prob >= 0.70 THEN 'Heavy Favorite (70%+)'
        WHEN mkt_home_prob >= 0.55 THEN 'Favorite (55-70%)'
        WHEN mkt_home_prob >= 0.45 THEN 'Even Matchup (45-55%)'
        ELSE 'Underdog (<45%)'
    END AS market_tier,
    COUNT(*) AS total_games,
    ROUND(AVG(mkt_home_prob), 3) AS avg_market_prob,
    ROUND(AVG(home_win), 3) AS actual_win_rate
FROM nba_game_features
GROUP BY market_tier
ORDER BY avg_market_prob DESC;
"""

print("\n=== MARKET PROBABILITY VS ACTUAL WIN RATE ===")
print(pd.read_sql_query(query_market, conn))

conn.close()