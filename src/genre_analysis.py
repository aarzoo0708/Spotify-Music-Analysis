import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/spotify_tracks_cleaned.csv")

print("========== GENRE ANALYSIS ==========\n")

# Average popularity by genre
genre_popularity = (
    df.groupby("track_genre")["popularity"]
    .mean()
    .sort_values(ascending=False)
)

print("Top 10 genres by average popularity:")
print(genre_popularity.head(10).round(2))

# Average energy by genre
genre_energy = (
    df.groupby("track_genre")["energy"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 energetic genres:")
print(genre_energy.head(10).round(2))

# Average danceability by genre
genre_danceability = (
    df.groupby("track_genre")["danceability"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 most danceable genres:")
print(genre_danceability.head(10).round(2))

# Average duration by genre
genre_duration = (
    df.groupby("track_genre")["duration_min"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 genres by average duration:")
print(genre_duration.head(10).round(2))