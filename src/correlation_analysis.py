import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/spotify_tracks_cleaned.csv")

print("========== CORRELATION ANALYSIS ==========\n")

# Select numerical columns
features = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_min"
]

# Calculate correlation matrix
correlation_matrix = df[features].corr()

print("Correlation matrix:")
print(correlation_matrix.round(2))

# Correlation with popularity
popularity_correlation = (
    correlation_matrix["popularity"]
    .sort_values(ascending=False)
)

print("\nCorrelation with popularity:")
print(popularity_correlation.round(2))