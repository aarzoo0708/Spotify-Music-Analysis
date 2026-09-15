import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/spotify_tracks_cleaned.csv")

print("Original cleaned shape:", df.shape)

# Check duplicate track IDs
print(
    "Duplicate track IDs:",
    df["track_id"].duplicated().sum()
)

# Keep only one record per track_id
unique_tracks = df.drop_duplicates(
    subset=["track_id"],
    keep="first"
)

print(
    "Unique-track dataset shape:",
    unique_tracks.shape
)

# Check again
print(
    "Duplicate track IDs after cleaning:",
    unique_tracks["track_id"].duplicated().sum()
)

# Save unique-track dataset
unique_tracks.to_csv(
    "data/spotify_unique_tracks.csv",
    index=False
)

print("\nUnique-track dataset saved successfully!")