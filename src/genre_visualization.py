import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/spotify_tracks_cleaned.csv")

genre_popularity = (
    df.groupby("track_genre")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    genre_popularity.index,
    genre_popularity.values
)

plt.title("Top 10 Genres by Average Popularity")
plt.xlabel("Genre")
plt.ylabel("Average Popularity")

plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{height:.1f}",
        ha="center"
    )

plt.tight_layout()
plt.show()