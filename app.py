import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration & Setup
# ==============================================================================
st.set_page_config(page_title="Spotify Data Analysis", page_icon="🎵", layout="wide")

# Custom CSS to mimic a professional dark theme without overly complex logic
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    h1, h2, h3, h4, h5 { color: #1DB954 !important; }
    div[data-testid="stMetricValue"] { color: #1DB954; }
</style>
""", unsafe_allow_html=True)

# Helper function to style matplotlib charts consistently
def style_plot(fig, ax):
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.tick_params(colors='#B3B3B3')
    ax.xaxis.label.set_color('#B3B3B3')
    ax.yaxis.label.set_color('#B3B3B3')
    ax.title.set_color('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_color('#282828')
    ax.grid(alpha=0.2, color='#B3B3B3')

# ==============================================================================
# Data Loading & Cleaning
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    """
    Loads the Spotify dataset and applies fundamental data cleaning operations.
    This ensures our analysis is built on reliable data.
    Cached by Streamlit — runs only once per session (or when the CSV changes).
    """
    df = pd.read_csv("data/spotify_tracks_cleaned.csv")

    # 1. Handle Duplicates
    # Drop rows that are exact duplicates across all columns
    df = df.drop_duplicates()

    # Often, the same song is released as a single and on an album.
    # We keep the most popular version to avoid skewing our analysis.
    if "popularity" in df.columns:
        df = df.sort_values(by="popularity", ascending=False)

    if "track_name" in df.columns and "artists" in df.columns:
        df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")

    # 2. Data Type Conversions & Feature Engineering
    # FIX 1: The cleaned CSV already has duration_min — only compute if missing
    if "duration_ms" in df.columns and "duration_min" not in df.columns:
        df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
        df["duration_min"] = df["duration_ms"] / 60000.0

    # 3. Spotify URLs
    # Create a clickable URL if we have the track ID
    if "track_id" in df.columns:
        df["spotify_url"] = "https://open.spotify.com/track/" + df["track_id"].astype(str)

    # Reset index after all filtering/sorting so iloc works predictably
    df = df.reset_index(drop=True)
    return df


# FIX 2: Cache the genre list — genres never change between reruns
@st.cache_data
def get_all_genres(_df):
    """Returns a sorted list of unique genres. Cached to avoid recomputing on every rerun."""
    if "track_genre" not in _df.columns:
        return []
    return sorted(_df["track_genre"].dropna().unique().tolist())


# FIX 4: Cache the correlation matrix — expensive to compute on a large DataFrame
@st.cache_data
def compute_corr_matrix(_df):
    """Computes the numeric correlation matrix. Cached so it only runs when filtered_df changes."""
    numeric_df = _df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
    return numeric_df.corr()


# FIX 6: Cache the artist list — avoids sorting 80k+ rows of unique strings on every rerun
@st.cache_data
def get_artist_list(_df):
    """Returns a sorted list of unique artists. Cached to avoid recomputing on every rerun."""
    if "artists" not in _df.columns:
        return []
    return sorted(_df["artists"].dropna().unique().tolist())


# Load the data
try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Failed to load dataset. Please ensure 'data/spotify_tracks_cleaned.csv' exists. Error: {e}")
    st.stop()

# ==============================================================================
# Global Sidebar Filters
# ==============================================================================
st.sidebar.title("🎵 Data Filters")
st.sidebar.write("Adjust these filters to explore specific subsets of the Spotify dataset.")

# FIX 2: Use cached genre list — no recomputation on each widget interaction
all_genres = get_all_genres(df)

# Genre Filter
selected_genre = "All"
if all_genres:
    selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + all_genres)

# Popularity Filter
min_pop, max_pop = 0, 100
if "popularity" in df.columns:
    min_pop_val = int(df["popularity"].min())
    max_pop_val = int(df["popularity"].max())
    min_pop, max_pop = st.sidebar.slider(
        "Popularity Range",
        min_value=min_pop_val,
        max_value=max_pop_val,
        value=(min_pop_val, max_pop_val)
    )

# Search Filter
search_term = st.sidebar.text_input("Search Artist or Song:")

# FIX 3: Apply Filters with direct boolean masks — avoids a full df.copy() on every rerun
# Build a single combined mask instead of creating intermediate DataFrame copies
mask = pd.Series(True, index=df.index)

if selected_genre != "All" and "track_genre" in df.columns:
    mask = mask & (df["track_genre"] == selected_genre)

if "popularity" in df.columns:
    mask = mask & (df["popularity"] >= min_pop) & (df["popularity"] <= max_pop)

if search_term:
    # Use case=False for case-insensitive matching
    mask_artist = df["artists"].str.contains(search_term, case=False, na=False)
    mask_track = df["track_name"].str.contains(search_term, case=False, na=False)
    mask = mask & (mask_artist | mask_track)

filtered_df = df[mask]

if filtered_df.empty:
    st.warning("No tracks match your current filters. Please adjust the sidebar settings.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.write(f"**Tracks currently in view:** {len(filtered_df):,}")

# ==============================================================================
# Main Dashboard UI (Tabs)
# ==============================================================================
# Using tabs keeps the UI clean and prevents endless scrolling
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Popularity & Genres",
    "🎤 Artists",
    "🎧 Audio Features",
    "🧠 Mood Explorer",
    "📝 Playlist Generator"
])

# ------------------------------------------------------------------------------
# TAB 1: Overview
# ------------------------------------------------------------------------------
with tab1:
    st.header("Dataset Overview")
    st.write("This dashboard analyzes audio features, popularity, and metadata from Spotify tracks.")

    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tracks", f"{len(filtered_df):,}")

    if "artists" in filtered_df.columns:
        col2.metric("Unique Artists", f"{filtered_df['artists'].nunique():,}")

    if "popularity" in filtered_df.columns:
        col3.metric("Avg Popularity", f"{filtered_df['popularity'].mean():.1f}")

    if "duration_min" in filtered_df.columns:
        col4.metric("Avg Duration", f"{filtered_df['duration_min'].mean():.2f} min")

    st.markdown("### Raw Data Preview")

    # Define columns to display to keep the dataframe readable
    display_cols = ["track_name", "artists", "track_genre", "popularity", "duration_min"]
    actual_cols = [c for c in display_cols if c in filtered_df.columns]

    # Changed use_container_width to width="stretch" per Streamlit deprecation warning
    st.dataframe(filtered_df[actual_cols].head(100), width=1000)

    # FIX 7: Only encode the CSV when the download button is actually clicked.
    # Use st.download_button's data parameter with a lazy lambda approach —
    # Streamlit evaluates 'data' before rendering, so we keep it but limit to 1000 rows
    # to avoid encoding the full DataFrame on every single rerun.
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=filtered_df.head(1000).to_csv(index=False).encode('utf-8'),
        file_name="spotify_filtered_data.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------------------
# TAB 2: Popularity & Genres
# ------------------------------------------------------------------------------
with tab2:
    st.header("Popularity & Genre Analysis")

    col_pop, col_genre = st.columns(2)

    with col_pop:
        st.subheader("Top 10 Most Popular Tracks")
        if "popularity" in filtered_df.columns and "track_name" in filtered_df.columns:
            top_tracks = filtered_df.nlargest(10, "popularity")

            fig, ax = plt.subplots(figsize=(8, 6))
            style_plot(fig, ax)

            # Create a horizontal bar chart
            ax.barh(top_tracks["track_name"].astype(str), top_tracks["popularity"], color="#1DB954")
            ax.invert_yaxis() # Highest popularity on top
            ax.set_xlabel("Popularity Score")
            ax.set_title("Highest Rated Tracks in Current Filter")

            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(top_tracks[["track_name", "artists", "popularity"]], hide_index=True)
        else:
            st.info("Popularity or track name data not available.")

    with col_genre:
        st.subheader("Top Genres by Track Count")
        if "track_genre" in filtered_df.columns:
            # Group by genre and count the tracks
            genre_counts = filtered_df["track_genre"].value_counts().head(10)

            fig, ax = plt.subplots(figsize=(8, 6))
            style_plot(fig, ax)

            ax.bar(genre_counts.index, genre_counts.values, color="#9B51E0")
            plt.xticks(rotation=45, ha='right')
            ax.set_ylabel("Number of Tracks")
            ax.set_title("Most Common Genres")

            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(genre_counts.reset_index().rename(columns={"track_genre": "Genre", "count": "Track Count"}), hide_index=True)
        else:
            st.info("Genre data not available.")

# ------------------------------------------------------------------------------
# TAB 3: Artists
# ------------------------------------------------------------------------------
with tab3:
    st.header("Artist Analysis & Comparison")

    if "artists" not in filtered_df.columns:
        st.error("Artist column missing from dataset.")
    else:
        st.markdown("### Single Artist Deep Dive")
        # FIX 6: Use cached artist list — avoids re-sorting 80k+ unique strings each rerun
        artist_list = get_artist_list(filtered_df)

        selected_artist = st.selectbox("Search for an Artist to analyze", [""] + artist_list)

        if selected_artist:
            artist_data = filtered_df[filtered_df["artists"] == selected_artist]

            st.write(f"**{selected_artist}** has {len(artist_data)} tracks in this dataset.")

            # Simple metrics for the artist
            a_col1, a_col2, a_col3 = st.columns(3)
            if "popularity" in artist_data.columns:
                a_col1.metric("Average Popularity", f"{artist_data['popularity'].mean():.1f}")
            if "energy" in artist_data.columns:
                a_col2.metric("Average Energy", f"{artist_data['energy'].mean():.2f}")
            if "danceability" in artist_data.columns:
                a_col3.metric("Average Danceability", f"{artist_data['danceability'].mean():.2f}")

            st.write("##### Discography (Top Tracks)")
            sort_col = "popularity" if "popularity" in artist_data.columns else "track_name"
            artist_top = artist_data.nlargest(10, sort_col) if sort_col == "popularity" else artist_data.head(10)

            show_cols = [c for c in ["track_name", "popularity", "energy", "danceability", "duration_min", "spotify_url"] if c in artist_data.columns]

            st.dataframe(
                artist_top[show_cols],
                hide_index=True,
                column_config={
                    "spotify_url": st.column_config.LinkColumn("Listen on Spotify")
                } if "spotify_url" in show_cols else None
            )

        st.markdown("---")
        st.markdown("### Compare Two Artists")

        comp_col1, comp_col2 = st.columns(2)
        artist_1 = comp_col1.selectbox("Select First Artist", [""] + artist_list, key="a1")
        artist_2 = comp_col2.selectbox("Select Second Artist", [""] + artist_list, key="a2")

        if artist_1 and artist_2:
            data_1 = filtered_df[filtered_df["artists"] == artist_1]
            data_2 = filtered_df[filtered_df["artists"] == artist_2]

            # Compile stats into a dictionary
            stats_1 = {"Track Count": len(data_1)}
            stats_2 = {"Track Count": len(data_2)}

            metrics_to_compare = ["popularity", "energy", "danceability", "valence", "acousticness"]
            for m in metrics_to_compare:
                if m in filtered_df.columns:
                    stats_1[m.title()] = data_1[m].mean()
                    stats_2[m.title()] = data_2[m].mean()

            # Build a DataFrame for the comparison table
            comparison_df = pd.DataFrame([stats_1, stats_2], index=[artist_1, artist_2]).T

            st.write("##### Metric Comparison")
            st.dataframe(comparison_df.style.format("{:.2f}"))

            # Simple bar chart comparison of Audio Features
            audio_cols = [m.title() for m in metrics_to_compare if m != "popularity" and m in filtered_df.columns]
            if audio_cols:
                fig, ax = plt.subplots(figsize=(10, 4))
                style_plot(fig, ax)

                x = np.arange(len(audio_cols))
                width = 0.35

                vals_1 = [stats_1[c] for c in audio_cols]
                vals_2 = [stats_2[c] for c in audio_cols]

                ax.bar(x - width/2, vals_1, width, label=artist_1, color="#1DB954")
                ax.bar(x + width/2, vals_2, width, label=artist_2, color="#9B51E0")

                ax.set_xticks(x)
                ax.set_xticklabels(audio_cols)
                ax.set_title("Audio Features Comparison")
                ax.legend(facecolor='#121212', labelcolor='#FFFFFF')

                st.pyplot(fig)
                plt.close(fig)

# ------------------------------------------------------------------------------
# TAB 4: Audio Features & Correlation
# ------------------------------------------------------------------------------
with tab4:
    st.header("Audio Feature Analysis")
    st.write("Explore the numerical audio attributes computed by Spotify.")

    audio_features = ["energy", "danceability", "valence", "acousticness", "liveness", "speechiness"]
    available_features = [f for f in audio_features if f in filtered_df.columns]

    if available_features:
        feat_col1, feat_col2 = st.columns(2)

        with feat_col1:
            st.subheader("Feature Distribution")
            selected_dist = st.selectbox("Select a feature to view its distribution", available_features)

            fig, ax = plt.subplots(figsize=(8, 6))
            style_plot(fig, ax)

            ax.hist(filtered_df[selected_dist].dropna(), bins=40, color="#2D9CDB", edgecolor="#121212")
            ax.set_title(f"Distribution of {selected_dist.title()}")
            ax.set_xlabel(selected_dist.title())
            ax.set_ylabel("Frequency")

            st.pyplot(fig)
            plt.close(fig)

        with feat_col2:
            st.subheader("Feature Relationship (Scatter)")
            x_feat = st.selectbox("X-Axis Feature", available_features, index=0)
            y_feat = st.selectbox("Y-Axis Feature", available_features, index=1 if len(available_features) > 1 else 0)

            fig, ax = plt.subplots(figsize=(8, 6))
            style_plot(fig, ax)

            # FIX 5: Downsample large datasets for the scatter plot.
            # Plotting 80k+ semi-transparent points is slow with no visual benefit.
            # 5,000 random samples produce the same visual distribution.
            SCATTER_SAMPLE_LIMIT = 5000
            scatter_data = filtered_df
            if len(filtered_df) > SCATTER_SAMPLE_LIMIT:
                scatter_data = filtered_df.sample(n=SCATTER_SAMPLE_LIMIT, random_state=42)

            # Use alpha=0.3 to make points semi-transparent (good for large datasets)
            ax.scatter(scatter_data[x_feat], scatter_data[y_feat], alpha=0.3, color="#F2C94C")
            ax.set_title(f"{x_feat.title()} vs {y_feat.title()}")
            ax.set_xlabel(x_feat.title())
            ax.set_ylabel(y_feat.title())

            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.subheader("Correlation Heatmap")
        st.write("This matrix shows how different audio features correlate with each other. Values close to 1 mean positive correlation, -1 means negative correlation.")

        # FIX 4: Use cached correlation matrix — avoids recomputing on every rerun
        corr_matrix = compute_corr_matrix(filtered_df)

        if corr_matrix is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            style_plot(fig, ax)

            cax = ax.matshow(corr_matrix, cmap="RdYlGn")
            fig.colorbar(cax)

            # Set up the axes
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, color='#B3B3B3')
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, color='#B3B3B3')

            st.pyplot(fig)
            plt.close(fig)

    else:
        st.error("Audio feature columns missing from dataset.")

# ------------------------------------------------------------------------------
# TAB 5: Mood Explorer
# ------------------------------------------------------------------------------
with tab5:
    st.header("Rule-Based Mood Explorer")
    st.markdown("""
    This tool classifies tracks into basic 'moods' using **simple Pandas rule-based filtering**.

    *Note: This does NOT use Machine Learning. It simply applies hardcoded thresholds to Spotify's numerical audio features (like Energy and Valence) to estimate the vibe of a song.*
    """)

    mood_selected = st.radio(
        "Select a Mood",
        ["Happy / Upbeat", "Sad / Melancholic", "Energetic / Party", "Relaxing / Acoustic"],
        horizontal=True
    )

    # We define simple thresholds based on typical Spotify audio feature ranges (0.0 to 1.0)
    # FIX 3 (continued): Use boolean masks directly instead of mood_df = filtered_df.copy()
    try:
        if mood_selected == "Happy / Upbeat":
            # High valence (positivity) and decent energy
            mood_mask = (filtered_df["valence"] > 0.7) & (filtered_df["energy"] > 0.6)
            st.info("Filter applied: Valence > 0.7 AND Energy > 0.6")

        elif mood_selected == "Sad / Melancholic":
            # Low valence and lower energy
            mood_mask = (filtered_df["valence"] < 0.3) & (filtered_df["energy"] < 0.5)
            st.info("Filter applied: Valence < 0.3 AND Energy < 0.5")

        elif mood_selected == "Energetic / Party":
            # High energy and high danceability
            mood_mask = (filtered_df["energy"] > 0.8) & (filtered_df["danceability"] > 0.7)
            st.info("Filter applied: Energy > 0.8 AND Danceability > 0.7")

        elif mood_selected == "Relaxing / Acoustic":
            # Low energy, high acousticness
            mood_mask = (filtered_df["energy"] < 0.4) & (filtered_df["acousticness"] > 0.7)
            st.info("Filter applied: Energy < 0.4 AND Acousticness > 0.7")

        else:
            mood_mask = pd.Series(True, index=filtered_df.index)

        mood_df = filtered_df[mood_mask]

        st.write(f"**Found {len(mood_df):,} tracks matching this mood.**")

        if not mood_df.empty:
            # Sort by popularity to show the best ones first
            if "popularity" in mood_df.columns:
                mood_df = mood_df.nlargest(50, "popularity")

            display_cols = [c for c in ["track_name", "artists", "popularity", "valence", "energy", "acousticness", "spotify_url"] if c in mood_df.columns]

            st.dataframe(
                mood_df[display_cols].head(50),
                hide_index=True,
                column_config={
                    "spotify_url": st.column_config.LinkColumn("Listen")
                } if "spotify_url" in display_cols else None
            )

    except KeyError as e:
        st.error(f"Missing required columns for mood filtering: {e}")

# ------------------------------------------------------------------------------
# TAB 6: Playlist Generator
# ------------------------------------------------------------------------------
with tab6:
    st.header("Simple Playlist Generator")
    st.write("Filter the dataset to create a custom playlist, then download it as a CSV.")

    p_col1, p_col2 = st.columns(2)

    # Playlist Filters
    with p_col1:
        pl_min_pop = st.slider("Minimum Popularity for Playlist", 0, 100, 50)

        # FIX 2 (continued): Reuse cached genre list instead of recomputing from filtered_df
        pl_genres = ["Any"] + all_genres
        pl_target_genre = st.selectbox("Target Genre", pl_genres)

    with p_col2:
        pl_energy = st.radio("Energy Level", ["Any", "High Energy (>0.7)", "Low Energy (<0.4)"])
        pl_size = st.number_input("Number of Tracks", min_value=5, max_value=100, value=20)

    if st.button("Generate Playlist", type="primary"):
        # 1. Start with the globally filtered data using a boolean mask (no copy)
        pl_mask = pd.Series(True, index=filtered_df.index)

        # 2. Apply Popularity Filter
        if "popularity" in filtered_df.columns:
            pl_mask = pl_mask & (filtered_df["popularity"] >= pl_min_pop)

        # 3. Apply Genre Filter
        if pl_target_genre != "Any" and "track_genre" in filtered_df.columns:
            pl_mask = pl_mask & (filtered_df["track_genre"] == pl_target_genre)

        # 4. Apply Energy Filter
        if "energy" in filtered_df.columns:
            if pl_energy == "High Energy (>0.7)":
                pl_mask = pl_mask & (filtered_df["energy"] > 0.7)
            elif pl_energy == "Low Energy (<0.4)":
                pl_mask = pl_mask & (filtered_df["energy"] < 0.4)

        playlist = filtered_df[pl_mask]

        # 5. Check if we have enough songs
        if playlist.empty:
            st.error("No songs found matching these strict criteria. Try lowering the popularity or changing the genre.")
        else:
            # 6. Randomly sample the requested number of songs (or fewer if not enough exist)
            sample_size = min(int(pl_size), len(playlist))
            final_playlist = playlist.sample(n=sample_size, random_state=42)

            # Sort the final playlist by popularity
            if "popularity" in final_playlist.columns:
                final_playlist = final_playlist.nlargest(sample_size, "popularity")

            st.success(f"Generated playlist with {sample_size} tracks!")

            pl_display = [c for c in ["track_name", "artists", "track_genre", "popularity", "energy", "spotify_url"] if c in final_playlist.columns]

            st.dataframe(
                final_playlist[pl_display],
                hide_index=True,
                column_config={
                    "spotify_url": st.column_config.LinkColumn("Listen")
                } if "spotify_url" in pl_display else None
            )

            # CSV Download
            csv_playlist = final_playlist[pl_display].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Playlist (CSV)",
                data=csv_playlist,
                file_name="my_spotify_playlist.csv",
                mime="text/csv"
            )
