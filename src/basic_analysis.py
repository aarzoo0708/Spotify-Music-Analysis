import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/spotify_tracks_cleaned.csv")

print("========== BASIC SPOTIFY ANALYSIS ==========\n")

# 1. Total number of tracks
total_tracks = df["track_id"].nunique()
print("Total unique tracks:", total_tracks)

# 2. Total number of artists
total_artists = df["artists"].nunique()
print("Total unique artists:", total_artists)

# 3. Total number of genres
total_genres = df["track_genre"].nunique()
print("Total genres:", total_genres)

# 4. Average popularity
average_popularity = np.mean(df["popularity"])
print("Average popularity:", round(average_popularity, 2))

# 5. Average song duration
average_duration = np.mean(df["duration_min"])
print("Average song duration:", round(average_duration, 2), "minutes")

# 6. Most popular song
most_popular_song = df.loc[df["popularity"].idxmax()]

print("\nMost popular song:")
print("Song:", most_popular_song["track_name"])
print("Artist:", most_popular_song["artists"])
print("Popularity:", most_popular_song["popularity"])

# 7. Top 10 most popular songs
# Load unique-track dataset for song-level analysis
unique_tracks = pd.read_csv(
    "data/spotify_unique_tracks.csv"
)

top_songs = unique_tracks.sort_values(
    by="popularity",
    ascending=False
)[
    ["track_name", "artists", "popularity"]
].head(10)

# 8. Most common genres
print("\nTop 10 genres:")

top_genres = df["track_genre"].value_counts().head(10)
print(top_genres)

# 9. Explicit vs non-explicit songs
print("\nExplicit song distribution:")

explicit_distribution = df["explicit"].value_counts()
print(explicit_distribution)

# 10. Average audio features
print("\nAverage audio features:")

audio_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

print(df[audio_features].mean().round(2))