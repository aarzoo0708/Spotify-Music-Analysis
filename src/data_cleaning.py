import pandas as pd
import numpy as np

# Load the original dataset
df = pd.read_csv("data/spotify_tracks.csv")

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 1. Check missing values
# --------------------------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows where important information is missing
df = df.dropna(
    subset=["artists", "album_name", "track_name"]
)

# --------------------------------------------------
# 2. Remove duplicate rows
# --------------------------------------------------

print("\nDuplicate rows before cleaning:", df.duplicated().sum())

df = df.drop_duplicates()

# --------------------------------------------------
# 3. Convert duration from milliseconds to minutes
# --------------------------------------------------

df["duration_min"] = df["duration_ms"] / 60000

# Round duration to two decimal places
df["duration_min"] = df["duration_min"].round(2)

# --------------------------------------------------
# 4. Check invalid values
# --------------------------------------------------

print("\nPopularity range:")
print(df["popularity"].min(), "to", df["popularity"].max())

print("\nEnergy range:")
print(df["energy"].min(), "to", df["energy"].max())

print("\nDanceability range:")
print(df["danceability"].min(), "to", df["danceability"].max())

# --------------------------------------------------
# 5. Final dataset information
# --------------------------------------------------

print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

# --------------------------------------------------
# 6. Save cleaned dataset
# --------------------------------------------------

df.to_csv("data/spotify_tracks_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")