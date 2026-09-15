"""Spotify Music Analytics. Run with: streamlit run app.py."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Spotify Music Analytics", page_icon="🟢", layout="wide")
DATA_FILE = Path("data/spotify_tracks_cleaned.csv")
NUMERIC = ["popularity", "energy", "danceability", "valence", "tempo", "loudness", "acousticness", "instrumentalness", "liveness", "speechiness"]

def add_styles():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .stApp, .stApp > header {background-color:#050B0D !important; color:#FFFFFF; font-family: 'Inter', Arial, sans-serif;} 
    header[data-testid="stHeader"] {background-color: transparent !important; display: none !important;}
    .block-container {padding-top: 1rem !important; padding-bottom: 2rem !important;}
    
    [data-testid="stSidebar"] {background:#081217 !important;}
    [data-testid="stSidebar"] * {color:#B3B3B3;} 
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:#FFFFFF !important;}
    
    /* Inputs */
    [data-testid="stTextInput"] input, div[data-baseweb="select"] > div {background-color: #202C31 !important; color: #FFFFFF !important; border: 1px solid #3A4A50 !important;}
    [data-testid="stTextInput"] input::placeholder, div[data-baseweb="select"] span {color: #B3B3B3 !important;}
    [data-testid="stTextInput"] input:focus, div[data-baseweb="select"] > div:focus-within {border-color: #1DB954 !important; box-shadow: none !important;}
    div[data-baseweb="popover"] {background-color: #202C31 !important;}
    div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {background-color: #202C31 !important; color: #FFFFFF !important;}
    div[data-baseweb="popover"] li:hover {background-color: #1DB954 !important;}
    
    /* Metrics */
    [data-testid="stMetric"] {background:#0D1A1E !important; border:1px solid #263238 !important; padding:12px !important; border-radius:8px !important;}
    [data-testid="stMetricValue"] {color:#FFFFFF !important; font-size: 24px !important; font-weight: 700 !important;}
    [data-testid="stMetricLabel"] {color:#B3B3B3 !important; font-weight: 600 !important; font-size: 14px !important;}
    
    /* Buttons */
    [data-testid="stDownloadButton"] button, .stButton button {background:#1DB954 !important; color:#FFFFFF !important; border:0 !important; font-weight:bold !important; border-radius:500px !important; transition: 0.2s !important;}
    [data-testid="stDownloadButton"] button:hover, .stButton button:hover {background:#1ED760 !important; color:#FFFFFF !important; border: 0 !important;}
    [data-testid="stDownloadButton"] button p, .stButton button p {color:#FFFFFF !important;} 
    
    /* Clear button special styling */
    button[kind="secondary"] {background: transparent !important; border: 1px solid #3A4A50 !important; color: #B3B3B3 !important;}
    button[kind="secondary"]:hover {background: #1DB954 !important; color: #FFFFFF !important; border-color: #1DB954 !important;}
    button[kind="secondary"] p {color: inherit !important;}
    
    /* Typography */
    h1, h2, h3, h4, h5 {color:#FFFFFF !important; font-weight: 700 !important;} 
    p, span, div {color:#B3B3B3;}
    .stMarkdown p {color: #B3B3B3;}
    .small-note {color:#B3B3B3; font-size: 0.85em;}
    hr {border-color: #263238 !important; margin: 1em 0;}
    
    /* Tables */
    [data-testid="stDataFrame"] {background-color: #0D1A1E !important;}
    
    /* Radio Nav */
    div[role="radiogroup"] > label {background-color: transparent !important; padding: 10px; border-radius: 5px; cursor: pointer;}
    div[role="radiogroup"] > label:hover {background-color: #0D1A1E !important;}
    div[role="radiogroup"] > label[data-checked="true"] {background-color: #1DB954 !important;}
    div[role="radiogroup"] > label[data-checked="true"] * {color: #FFFFFF !important;}
    
    /* Title highlight */
    .highlight {color: #1DB954;}
    
    /* Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>""", unsafe_allow_html=True)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    for column in NUMERIC + ["duration_min"]:
        if column in data: data[column] = pd.to_numeric(data[column], errors="coerce")
    if "explicit" in data:
        data["explicit"] = data["explicit"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    return data

def has(data, columns):
    missing = [c for c in columns if c not in data]
    if missing: st.info("This view needs: " + ", ".join(missing) + ".")
    return not missing

def histogram(data, column, title, x_label, color="#1DB954"):
    values = data[column].dropna()
    if values.empty: return
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0D1A1E')
    ax.set_facecolor('#0D1A1E')
    ax.hist(values, bins=30, color=color, edgecolor="#0D1A1E")
    ax.set_title(title, color="#FFFFFF", fontweight='bold', pad=10)
    ax.set_xlabel(x_label, color="#B3B3B3")
    ax.set_ylabel("Tracks", color="#B3B3B3")
    ax.tick_params(colors="#B3B3B3")
    ax.grid(axis="y", alpha=.1, color="#B3B3B3")
    for spine in ax.spines.values(): spine.set_color('#263238')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def scatter(data, x, y, title, color="#1DB954"):
    if not has(data, [x, y]): return
    plot = data[[x, y]].dropna()
    if plot.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#0D1A1E')
    ax.set_facecolor('#0D1A1E')
    ax.scatter(plot[x], plot[y], alpha=.3, color=color, edgecolors="none")
    ax.set_title(title, color="#FFFFFF", fontweight='bold', pad=10)
    ax.set_xlabel(x.replace("_", " ").title(), color="#B3B3B3")
    ax.set_ylabel(y.replace("_", " ").title(), color="#B3B3B3")
    ax.tick_params(colors="#B3B3B3")
    ax.grid(alpha=.1, color="#B3B3B3")
    for spine in ax.spines.values(): spine.set_color('#263238')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def horizontal_bar(data, x_col, y_col, title, color="#1DB954"):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0D1A1E')
    ax.set_facecolor('#0D1A1E')
    ax.barh(data[y_col], data[x_col], color=color)
    ax.set_title(title, color="#FFFFFF", fontweight='bold', pad=10)
    ax.set_xlabel(x_col.title(), color="#B3B3B3")
    ax.tick_params(colors="#B3B3B3")
    ax.invert_yaxis()
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(axis="x", alpha=.1, color="#B3B3B3")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def csv_bytes(data): return data.to_csv(index=False).encode("utf-8")

add_styles()

if not DATA_FILE.exists(): st.error("Dataset not found. Place spotify_tracks_cleaned.csv in the data folder."); st.stop()
try: df = load_data(str(DATA_FILE))
except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error: st.error(f"The dataset could not be loaded: {error}"); st.stop()
if df.empty: st.warning("The dataset is empty."); st.stop()


# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown("<h2>🟢 Spotify <br><span style='font-size:16px; font-weight:normal; color:#B3B3B3'>Music Analytics</span></h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
nav_options_map = {
    "Home": "🏠 Home", 
    "Explore Data": "📊 Explore Data", 
    "Top Songs": "🎵 Top Songs", 
    "Top Artists": "🎤 Top Artists", 
    "Genre Analysis": "🎸 Genre Analysis", 
    "Audio Features": "🎧 Audio Features", 
    "Trends": "📈 Trends", 
    "Recommendations": "💡 Recommendations", 
    "About": "ℹ️ About"
}
nav = st.sidebar.radio("Navigation", list(nav_options_map.keys()), format_func=lambda x: nav_options_map[x], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("<br><p class='small-note'>Built with Python | Pandas | NumPy | Streamlit</p>", unsafe_allow_html=True)

# ---------------- Session State for Filters ----------------
if "filter_key" not in st.session_state:
    st.session_state.filter_key = 0

# ---------------- Hero & Metrics (Home Only) ----------------
if nav == "Home":
    st.markdown("<h3 style='margin-bottom:5px; margin-top:0;'>Welcome to <span class='highlight'>Spotify</span> Music Analytics</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#B3B3B3; font-size:14px; margin-bottom:10px;'>Explore music trends, artists, genres, and audio features through data.</p>", unsafe_allow_html=True)
    
    # Metric Cards
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Total Tracks", f"{len(df):,}")
    
    unique_artists = df["artists"].nunique() if "artists" in df else "N/A"
    metrics_cols[1].metric("Total Artists", f"{unique_artists:,}" if isinstance(unique_artists, int) else unique_artists)
    
    unique_genres = df["track_genre"].nunique() if "track_genre" in df else "N/A"
    metrics_cols[2].metric("Genres", f"{unique_genres:,}" if isinstance(unique_genres, int) else unique_genres)
    
    avg_pop = df["popularity"].mean() if "popularity" in df else 0
    metrics_cols[3].metric("Average Popularity", f"{avg_pop:.0f}")
    
    st.markdown("<br>", unsafe_allow_html=True)


# ---------------- Search & Filters (Global) ----------------
with st.container():
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
    
    search_text = f_col1.text_input("Search Song / Artist", key=f"search_{st.session_state.filter_key}", placeholder="Type a song or artist...")
    
    genre_choice = "All"
    if "track_genre" in df:
        genres = sorted(df["track_genre"].dropna().astype(str).unique())
        genre_choice = f_col2.selectbox("Select Genre", ["All"] + genres, key=f"genre_{st.session_state.filter_key}")
        
    artist_choice = "All"
    if "artists" in df:
        artists = sorted(df["artists"].dropna().astype(str).unique())
        artist_choice = f_col3.selectbox("Select Artist", ["All"] + artists, key=f"artist_{st.session_state.filter_key}")
        
    pop_min, pop_max = 0.0, 100.0
    if "popularity" in df and not df["popularity"].empty:
        pop_min, pop_max = float(df["popularity"].min()), float(df["popularity"].max())
    pop_range = f_col4.slider("Popularity Range", pop_min, pop_max, (pop_min, pop_max), key=f"pop_{st.session_state.filter_key}")
    
    with f_col5:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Clear Filters", key=f"clear_{st.session_state.filter_key}", help="Reset all filters"):
            st.session_state.filter_key += 1
            st.rerun()
            
    st.markdown("<hr style='margin: 10px 0 20px 0; border-color: #263238;'>", unsafe_allow_html=True)

# Apply Filters
filtered_df = df.copy()
if search_text:
    mask = filtered_df.astype(str).apply(lambda s: s.str.contains(search_text, case=False, na=False)).any(axis=1)
    filtered_df = filtered_df[mask]
if genre_choice != "All" and "track_genre" in filtered_df:
    filtered_df = filtered_df[filtered_df["track_genre"].astype(str) == genre_choice]
if artist_choice != "All" and "artists" in filtered_df:
    filtered_df = filtered_df[filtered_df["artists"].astype(str) == artist_choice]
if "popularity" in filtered_df:
    filtered_df = filtered_df[filtered_df["popularity"].between(pop_range[0], pop_range[1])]

if filtered_df.empty:
    st.warning("No tracks match the current filters. Please adjust them to see results.")
    st.stop()


# ---------------- Page Views ----------------

if nav == "Home":
    # Main Charts Grid Row 1
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if has(filtered_df, ["track_name", "popularity"]):
            top_songs = filtered_df.sort_values("popularity", ascending=False).head(10)
            if not top_songs.empty:
                labels = top_songs["track_name"].fillna("Unknown track")
                if "artists" in top_songs: labels = labels + " - " + top_songs["artists"].fillna("Unknown")
                labels = labels + " [" + top_songs.index.astype(str) + "]" 
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('#0D1A1E'); ax.set_facecolor('#0D1A1E')
                ax.barh(labels, top_songs["popularity"], color="#1DB954")
                ax.set_title("Top 10 Most Popular Songs", color="#FFFFFF", fontweight='bold')
                ax.set_xlabel("Popularity", color="#B3B3B3")
                ax.tick_params(colors="#B3B3B3")
                ax.set_yticklabels([l.rsplit(' [', 1)[0] for l in labels])
                ax.invert_yaxis()
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.grid(axis="x", alpha=.1, color="#B3B3B3")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                
    with c2:
        if has(filtered_df, ["artists", "popularity"]):
            artist_pop = filtered_df.groupby("artists")["popularity"].mean().sort_values(ascending=False).head(10).reset_index()
            if not artist_pop.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('#0D1A1E'); ax.set_facecolor('#0D1A1E')
                ax.barh(artist_pop["artists"], artist_pop["popularity"], color="#9B51E0") 
                ax.set_title("Top 10 Artists by Popularity", color="#FFFFFF", fontweight='bold')
                ax.set_xlabel("Average Popularity", color="#B3B3B3")
                ax.tick_params(colors="#B3B3B3")
                ax.invert_yaxis()
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.grid(axis="x", alpha=.1, color="#B3B3B3")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                
    with c3:
        if has(filtered_df, ["track_genre"]):
            top_genres = filtered_df["track_genre"].value_counts().head(5)
            if not top_genres.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('#0D1A1E'); ax.set_facecolor('#0D1A1E')
                colors = ["#1DB954", "#1ED760", "#9B51E0", "#F2C94C", "#2F80ED"]
                wedges, texts, autotexts = ax.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=90, colors=colors, textprops=dict(color="#FFFFFF"))
                ax.set_title("Top Genres", color="#FFFFFF", fontweight='bold')
                centre_circle = plt.Circle((0,0),0.70,fc='#0D1A1E')
                fig.gca().add_artist(centre_circle)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Main Charts Grid Row 2
    c4, c5, c6 = st.columns(3)
    
    with c4:
        numeric = [c for c in NUMERIC if c in filtered_df and filtered_df[c].notna().any()][:8]
        if len(numeric) >= 2:
            correlation = filtered_df[numeric].corr()
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#0D1A1E'); ax.set_facecolor('#0D1A1E')
            image = ax.imshow(correlation, cmap="viridis", vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric)), [x.title() for x in numeric], rotation=45, ha="right", color="#B3B3B3", fontsize=8)
            ax.set_yticks(range(len(numeric)), [x.title() for x in numeric], color="#B3B3B3", fontsize=8)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_title("Audio Features Correlation", color="#FFFFFF", fontweight='bold')
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            
    with c5:
        if has(filtered_df, ["explicit"]):
            labels = filtered_df["explicit"].map({True:"Explicit", False:"Non-explicit"})
            val_counts = labels.value_counts()
            if not val_counts.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('#0D1A1E'); ax.set_facecolor('#0D1A1E')
                ax.bar(val_counts.index, val_counts.values, color=["#F2C94C", "#2F80ED"])
                ax.set_title("Explicit vs Non-explicit Tracks", color="#FFFFFF", fontweight='bold')
                ax.set_ylabel("Number of tracks", color="#B3B3B3")
                ax.tick_params(colors="#B3B3B3")
                for spine in ax.spines.values(): spine.set_visible(False)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            
    with c6:
        if has(filtered_df, ["duration_min"]): 
            histogram(filtered_df, "duration_min", "Track Duration Distribution", "Duration (minutes)", color="#1DB954")
            
    st.markdown("---")
    
    # Lower Section
    lc1, lc2, lc3 = st.columns(3)
    
    with lc1:
        st.subheader("Recently Viewed / Popular")
        st.markdown("<div style='background:#0D1A1E; padding:15px; border-radius:8px; border:1px solid #263238;'>", unsafe_allow_html=True)
        if has(filtered_df, ["track_name", "artists", "popularity"]):
            popular = filtered_df.sort_values("popularity", ascending=False).head(3)
            for _, row in popular.iterrows():
                st.markdown(f"<p style='margin:0; font-weight:bold; color:#FFFFFF;'>{row['track_name']}</p><p style='margin:0; font-size:12px; color:#B3B3B3; margin-bottom:10px;'>{row['artists']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with lc2:
        st.subheader("Key Insights")
        st.markdown("<div style='background:#0D1A1E; padding:15px; border-radius:8px; border:1px solid #263238; height: 100%;'>", unsafe_allow_html=True)
        if "track_genre" in filtered_df and not filtered_df["track_genre"].empty:
            top_g = filtered_df["track_genre"].value_counts().index[0]
            st.markdown(f"📈 **{top_g.title()}** is the most dominant genre.")
        if "duration_min" in filtered_df and not filtered_df["duration_min"].empty:
            avg_dur = filtered_df["duration_min"].mean()
            st.markdown(f"⏱️ Average duration is **{avg_dur:.2f}** min.")
        if "popularity" in filtered_df and not filtered_df["popularity"].empty:
            avg_p = filtered_df["popularity"].mean()
            st.markdown(f"⭐ Average popularity is **{avg_p:.0f}/100**.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with lc3:
        st.subheader("Recommended for You")
        st.markdown("<div style='background:#0D1A1E; padding:15px; border-radius:8px; border:1px solid #263238;'>", unsafe_allow_html=True)
        if has(filtered_df, ["track_name", "artists", "energy"]):
            # Use High Energy tracks as pseudo recommendations based on real dataset filters
            recs = filtered_df.sort_values("energy", ascending=False).head(3)
            for _, row in recs.iterrows():
                st.markdown(f"<p style='margin:0; font-weight:bold; color:#1DB954;'>{row['track_name']}</p><p style='margin:0; font-size:12px; color:#B3B3B3; margin-bottom:10px;'>{row['artists']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


elif nav == "Explore Data":
    st.title("Explore Data")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(filtered_df):,}")
    c2.metric("Columns", f"{len(filtered_df.columns):,}")
    c3.metric("Missing Values", f"{int(filtered_df.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(filtered_df.duplicated().sum()):,}")
    st.download_button("Download Filtered CSV", csv_bytes(filtered_df), "spotify_filtered_data.csv", "text/csv")


elif nav == "Top Songs":
    st.title("Top Songs")
    if has(filtered_df, ["track_name", "popularity"]):
        top_songs = filtered_df.sort_values("popularity", ascending=False).head(50)
        display = [c for c in ["track_name", "artists", "popularity", "energy", "danceability", "valence", "explicit"] if c in top_songs]
        st.dataframe(top_songs[display], use_container_width=True, hide_index=True)
        st.download_button("Download Top Tracks CSV", csv_bytes(top_songs), "top_spotify_tracks.csv", "text/csv")


elif nav == "Top Artists":
    st.title("Top Artists")
    if has(filtered_df, ["artists"]):
        artist_data = filtered_df.dropna(subset=["artists"]).copy()
        summary = artist_data.groupby("artists").agg(track_count=("artists", "size"))
        if "popularity" in artist_data: summary["avg_popularity"] = artist_data.groupby("artists")["popularity"].mean()
        
        c1, c2 = st.columns(2)
        with c1: 
            top_by_count = summary.sort_values("track_count", ascending=False).head(10).reset_index()
            horizontal_bar(top_by_count, "track_count", "artists", "Top Artists by Track Count", color="#1DB954")
        with c2:
            if "avg_popularity" in summary:
                top_by_pop = summary.sort_values("avg_popularity", ascending=False).head(10).reset_index()
                horizontal_bar(top_by_pop, "avg_popularity", "artists", "Top Artists by Avg Popularity", color="#9B51E0")
        
        st.dataframe(summary.reset_index().sort_values("track_count", ascending=False), use_container_width=True, hide_index=True)


elif nav == "Genre Analysis":
    st.title("Genre Analysis")
    if has(filtered_df, ["track_genre"]):
        genre_data = filtered_df["track_genre"].value_counts().reset_index()
        genre_data.columns = ["Genre", "Track Count"]
        st.dataframe(genre_data, use_container_width=True, hide_index=True)


elif nav == "Audio Features":
    st.title("Audio Features")
    if not filtered_df.empty:
        pairs = [("energy","danceability","Energy vs Danceability"),("energy","popularity","Energy vs Popularity"),("danceability","popularity","Danceability vs Popularity"),("valence","energy","Valence vs Energy")]
        valid_pairs = [p for p in pairs if p[0] in filtered_df and p[1] in filtered_df]
        for start in range(0, len(valid_pairs), 2):
            cols = st.columns(2)
            for col, (x, y, title) in zip(cols, valid_pairs[start:start+2]):
                with col: scatter(filtered_df, x, y, title, color="#1DB954")


elif nav == "Trends":
    st.title("Trends")
    st.caption("Distribution of audio features across the dataset.")
    if not filtered_df.empty:
        for col in ["tempo", "loudness", "acousticness", "liveness"]:
            if col in filtered_df: histogram(filtered_df, col, f"{col.replace('_',' ').title()} Distribution", col.title())


elif nav == "Recommendations":
    st.title("Recommendations")
    if has(filtered_df, ["track_name", "artists"]):
        cats = []
        if "energy" in filtered_df: cats.append(("High Energy Tracks", "energy", False))
        if "danceability" in filtered_df: cats.append(("Most Danceable Tracks", "danceability", False))
        if "valence" in filtered_df: cats.append(("Positive Mood Tracks", "valence", False))
        
        for start in range(0, len(cats), 2):
            cols = st.columns(2)
            for col, (title, sort_col, asc) in zip(cols, cats[start:start+2]):
                col.subheader(title)
                top_cat = filtered_df.sort_values(sort_col, ascending=asc).head(10)
                display_cols = ["track_name", "artists", sort_col]
                if "popularity" in top_cat: display_cols.append("popularity")
                col.dataframe(top_cat[display_cols], use_container_width=True, hide_index=True)


elif nav == "About":
    st.title("About")
    st.markdown("""
    **Spotify Music Analytics** is an interactive dashboard built using Python, Pandas, Matplotlib, and Streamlit.
    It empowers users to explore a massive dataset of music tracks, compare artists, investigate audio features, and uncover trends.
    """)
