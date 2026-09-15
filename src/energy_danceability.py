import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/spotify_tracks_cleaned.csv")

genre_features = (
    df.groupby("track_genre")[["energy", "danceability"]]
    .mean()
    .sort_values("energy", ascending=False)
    .head(20)
)

plt.figure(figsize=(12, 7))

plt.scatter(
    genre_features["energy"],
    genre_features["danceability"]
)

for genre, row in genre_features.iterrows():
    plt.annotate(
        genre,
        (row["energy"], row["danceability"]),
        fontsize=8,
        alpha=0.7
    )

plt.title("Energy vs Danceability by Genre")
plt.xlabel("Average Energy")
plt.ylabel("Average Danceability")
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()