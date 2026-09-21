import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/spotify_tracks_cleaned.csv")

plt.figure(figsize=(10, 6))

plt.hist(
    df["popularity"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Songs")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()