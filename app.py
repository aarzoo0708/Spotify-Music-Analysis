"""Spotify Music Analytics. Run with: streamlit run app.py."""
from pathlib import Path
from io import BytesIO
from datetime import date
import matplotlib
matplotlib.use("Agg")  # Streamlit renders figures; no desktop GUI backend is needed.
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
    for column in NUMERIC + ["duration_min", "duration_ms"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "explicit" in data:
        data["explicit"] = data["explicit"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    return data


@st.cache_data
def clean_music_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate tracks once and add display-ready duration values.

    Track IDs are the preferred identity.  Rows without a usable ID fall back to
    a normalized track-name/artist pair, so similarly named tracks by different
    artists are not merged.  Sorting first retains the most popular duplicate.
    """
    cleaned = data.drop_duplicates().copy()  # Remove exact duplicate records.
    popularity = pd.Series(np.nan, index=cleaned.index)
    if "popularity" in cleaned:
        popularity = pd.to_numeric(cleaned["popularity"], errors="coerce")
    cleaned = cleaned.assign(_popularity_sort=popularity).sort_values(
        "_popularity_sort", ascending=False, na_position="last", kind="mergesort"
    )

    track_id = pd.Series("", index=cleaned.index, dtype="string")
    if "track_id" in cleaned:
        track_id = cleaned["track_id"].astype("string").str.strip()
        track_id = track_id.fillna("")

    artist_column = next((column for column in ("artists", "artist_name") if column in cleaned), None)
    track_name = pd.Series("", index=cleaned.index, dtype="string")
    artist_name = pd.Series("", index=cleaned.index, dtype="string")
    if "track_name" in cleaned:
        track_name = cleaned["track_name"].astype("string").fillna("")
    if artist_column:
        artist_name = cleaned[artist_column].astype("string").fillna("")

    # Use a fallback identity only when an ID is unavailable; this protects
    # genuinely different tracks that happen to share a title.
    normalise = lambda values: values.str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    fallback_key = normalise(track_name) + "|" + normalise(artist_name)
    has_fallback_key = normalise(track_name).ne("") & normalise(artist_name).ne("")
    identity = pd.Series(pd.NA, index=cleaned.index, dtype="string")
    identity.loc[track_id.ne("")] = "id:" + track_id.loc[track_id.ne("")]
    fallback_rows = track_id.eq("") & has_fallback_key
    identity.loc[fallback_rows] = "name_artist:" + fallback_key.loc[fallback_rows]
    cleaned = cleaned.assign(_track_identity=identity)
    identified = cleaned["_track_identity"].notna()
    cleaned = pd.concat(
        [cleaned.loc[identified].drop_duplicates("_track_identity", keep="first"), cleaned.loc[~identified]],
        axis=0,
    ).sort_values("_popularity_sort", ascending=False, na_position="last", kind="mergesort")

    # Convert valid millisecond durations for display while retaining duration_ms
    # for calculations and the existing duration_min histogram.
    if "duration_ms" in cleaned:
        duration_ms = pd.to_numeric(cleaned["duration_ms"], errors="coerce")
        valid_duration = duration_ms.notna() & duration_ms.ge(0)
        total_seconds = duration_ms.where(valid_duration).floordiv(1000)
        minutes = total_seconds.floordiv(60)
        seconds = total_seconds.mod(60)
        formatted = pd.Series(pd.NA, index=cleaned.index, dtype="string")
        formatted.loc[valid_duration] = (
            minutes.loc[valid_duration].astype("Int64").astype(str)
            + ":"
            + seconds.loc[valid_duration].astype("Int64").astype(str).str.zfill(2)
        )
        cleaned["duration_formatted"] = formatted
        cleaned["duration_min"] = duration_ms.div(60000).round(2)

    return cleaned.drop(columns=["_popularity_sort", "_track_identity"], errors="ignore")


def display_data(data: pd.DataFrame) -> pd.DataFrame:
    """Keep raw milliseconds internal and out of user-facing data tables."""
    return data.drop(columns=["duration_ms", "duration_min"], errors="ignore")


def format_duration(duration_ms):
    if duration_ms is None:
        return "N/A"
    try:
        total_seconds = int(duration_ms // 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
    except (TypeError, ValueError):
        return "N/A"


def clean_artist_data(data: pd.DataFrame, artist: str) -> pd.DataFrame:
    """Select one artist and reapply the shared unique-track cleaning rules."""
    artist_column = next((column for column in ("artists", "artist_name") if column in data), None)
    if not artist or not artist_column:
        return pd.DataFrame(columns=data.columns)
    selected = data.loc[data[artist_column].astype("string") == artist].copy()
    return clean_music_data(selected)


@st.cache_data
def get_feature_matrix(data: pd.DataFrame):
    """Prepare a standardized feature matrix for similarity calculations."""
    features = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness", "tempo", "loudness"]
    available = [f for f in features if f in data.columns]
    if not available:
        return pd.DataFrame(), np.array([])
    
    valid_data = data.dropna(subset=available).copy()
    if valid_data.empty:
        return pd.DataFrame(), np.array([])
        
    mat = valid_data[available].values
    means = np.mean(mat, axis=0)
    stds = np.std(mat, axis=0)
    stds[stds == 0] = 1.0  # Prevent division by zero
    standardized = (mat - means) / stds
    return valid_data, standardized


def get_recommendations(valid_data: pd.DataFrame, standardized: np.ndarray, target_idx: int, n: int = 10):
    """Calculate cosine similarity using pure NumPy."""
    target_vec = standardized[target_idx]
    
    # Cosine similarity: dot(A, B) / (norm(A) * norm(B))
    dot_products = np.dot(standardized, target_vec)
    norms = np.linalg.norm(standardized, axis=1) * np.linalg.norm(target_vec)
    norms[norms == 0] = 1.0
    similarities = dot_products / norms
    
    results = valid_data.copy()
    results["similarity"] = similarities
    
    target_row = valid_data.iloc[target_idx]
    # Exclude the target song itself based on uniqueness criteria
    target_name = target_row.get("track_name", "")
    target_artist = target_row.get("artists", "")
    
    # Filter out exactly matching track name + artist pairs
    mask = ~((results["track_name"] == target_name) & (results["artists"] == target_artist))
    
    top_n = results[mask].sort_values("similarity", ascending=False).head(n)
    return top_n


def get_mood_recommendations(df, mood, n=10):
    profiles = {
        "Happy": {"valence": 0.9, "energy": 0.8, "danceability": 0.7},
        "Sad": {"valence": 0.1, "energy": 0.2, "danceability": 0.3},
        "Energetic": {"energy": 0.9, "danceability": 0.7},
        "Relaxing": {"energy": 0.2, "acousticness": 0.8},
        "Party": {"valence": 0.8, "energy": 0.9, "danceability": 0.9},
        "Focus": {"instrumentalness": 0.9, "energy": 0.3, "speechiness": 0.1}
    }
    
    if mood not in profiles: return pd.DataFrame()
    profile = profiles[mood]
    
    available = [f for f in profile.keys() if f in df.columns]
    if not available: return pd.DataFrame()
    
    valid_data = df.dropna(subset=available).copy()
    if valid_data.empty: return pd.DataFrame()
    
    import numpy as np
    X = valid_data[available].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    
    range_diff = X_max - X_min
    range_diff[range_diff == 0] = 1.0
    X_norm = (X - X_min) / range_diff
    
    ideal_vec = np.array([profile[f] for f in available])
    distances = np.linalg.norm(X_norm - ideal_vec, axis=1)
    
    max_dist = np.sqrt(len(available))
    scores = np.clip(1 - (distances / max_dist), 0, 1) * 100
    
    valid_data["mood_score"] = scores
    
    if "artists" in valid_data:
        valid_data = valid_data.drop_duplicates(subset=["track_name", "artists"])
    else:
        valid_data = valid_data.drop_duplicates(subset=["track_name"])
        
    return valid_data.sort_values("mood_score", ascending=False).head(n)



@st.cache_data
def add_language_column(data: pd.DataFrame) -> pd.DataFrame:
    """Heuristically detects song language based on available metadata keywords."""
    df_lang = data.copy()
    if "detected_language" in df_lang.columns:
        return df_lang
        
    text_series = pd.Series("", index=df_lang.index)
    for col in ["track_genre", "artists", "album_name", "track_name"]:
        if col in df_lang:
            text_series += df_lang[col].fillna("").astype(str).str.lower() + " "
            
    keywords = {
        "Hindi": ["hindi", "bollywood", "desi", "indian pop", "arijit", "atif", "shreya", "neha", "jubin"],
        "Punjabi": ["punjabi", "bhangra", "gurbani", "sidhu", "diljit", "amrinder", "karan aujla", "ap dhillon", "hardy"],
        "Tamil": ["tamil", "kollywood", "anirudh", "yuvan"],
        "Telugu": ["telugu", "tollywood", "dsp", "thaman"],
        "Bengali": ["bengali", "bangla", "rabindra"],
        "Marathi": ["marathi"],
        "Malayalam": ["malayalam"],
        "Kannada": ["kannada"],
        "Gujarati": ["gujarati"],
        "English": ["english", "pop", "rock", "r-n-b", "country"]
    }
    
    def detect_lang(t):
        for lang, words in keywords.items():
            for word in words:
                if word in t:
                    return lang
        return "Other"
        
    df_lang["detected_language"] = text_series.apply(detect_lang)
    return df_lang

def calculate_artist_statistics(artist_data: pd.DataFrame) -> dict:
    """Calculate report metrics only from the artist's cleaned tracks."""
    statistics = {"unique_songs": len(artist_data)}
    popularity = pd.to_numeric(artist_data["popularity"], errors="coerce") if "popularity" in artist_data else pd.Series(dtype=float)
    if not popularity.dropna().empty:
        statistics.update({
            "average_popularity": popularity.mean(),
            "highest_popularity": popularity.max(),
            "lowest_popularity": popularity.min(),
        })
        top = artist_data.assign(_popularity=popularity).sort_values("_popularity", ascending=False, na_position="last").iloc[0]
        statistics["most_popular_song"] = str(top.get("track_name", "Unknown track"))
    duration = pd.to_numeric(artist_data["duration_ms"], errors="coerce") if "duration_ms" in artist_data else pd.Series(dtype=float)
    valid_duration = duration[duration.ge(0)].dropna()
    statistics["average_duration"] = format_duration(valid_duration.mean()) if not valid_duration.empty else "N/A"
    statistics["total_duration"] = format_duration(valid_duration.sum()) if not valid_duration.empty else "N/A"
    return statistics


def generate_artist_insights(statistics: dict, audio_averages: pd.Series) -> list[str]:
    """Create concise, data-driven observations for the artist report."""
    insights = [f"The artist has {statistics['unique_songs']:,} unique songs in the dataset."]
    if "average_popularity" in statistics:
        insights.append(f"The average popularity score is {statistics['average_popularity']:.1f}.")
        insights.append(f"The most popular song is {statistics['most_popular_song']}.")
    if statistics.get("average_duration") != "N/A":
        insights.append(f"The average song duration is {statistics['average_duration']}.")
    for feature, label in (("energy", "energy"), ("danceability", "danceability")):
        if feature in audio_averages and pd.notna(audio_averages[feature]):
            insights.append(f"The average {label} level is {audio_averages[feature]:.2f}.")
    return insights


@st.cache_data(show_spinner=False)
def create_artist_pdf(artist: str, artist_csv: bytes) -> bytes | None:
    """Build a professional PDF report; return None if ReportLab is unavailable."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError:
        return None

    artist_data = pd.read_csv(BytesIO(artist_csv))
    statistics = calculate_artist_statistics(artist_data)
    audio_columns = [column for column in ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"] if column in artist_data]
    audio_averages = artist_data[audio_columns].apply(pd.to_numeric, errors="coerce").mean() if audio_columns else pd.Series(dtype=float)
    insights = generate_artist_insights(statistics, audio_averages)
    top_songs = artist_data.sort_values("popularity", ascending=False, na_position="last").head(10) if "popularity" in artist_data else artist_data.head(10)

    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("SpotifyTitle", parent=styles["Title"], textColor=colors.HexColor("#1DB954"), fontSize=20, spaceAfter=10)
    heading = ParagraphStyle("SpotifyHeading", parent=styles["Heading2"], textColor=colors.HexColor("#1DB954"), spaceBefore=12, spaceAfter=6)
    story = [Paragraph("Spotify Music Analytics: Artist Report", title), Paragraph(f"Artist: <b>{artist}</b><br/>Generated: {date.today().isoformat()}", styles["BodyText"]), Spacer(1, 10)]
    overview = [["Metric", "Value"], ["Unique songs", str(statistics["unique_songs"])], ["Average popularity", f"{statistics.get('average_popularity', float('nan')):.1f}" if "average_popularity" in statistics else "N/A"], ["Highest popularity", str(statistics.get("highest_popularity", "N/A"))], ["Lowest popularity", str(statistics.get("lowest_popularity", "N/A"))], ["Most popular song", statistics.get("most_popular_song", "N/A")], ["Average song duration", statistics["average_duration"]], ["Total music duration", statistics["total_duration"]]]
    story += [Paragraph("Artist Overview", heading), report_table(overview, colors), Paragraph("Top Songs", heading)]
    song_rows = [["Song", "Artist", "Popularity", "Duration"]]
    artist_col = "artists" if "artists" in top_songs else "artist_name"
    for _, song in top_songs.iterrows():
        song_rows.append([str(song.get("track_name", "Unknown"))[:38], str(song.get(artist_col, artist))[:24], str(song.get("popularity", "N/A")), str(song.get("duration_formatted", format_duration(song.get("duration_ms"))))])
    story += [report_table(song_rows, colors, [2.5 * inch, 1.45 * inch, .75 * inch, .75 * inch])]
    if not audio_averages.empty:
        audio_rows = [["Audio feature", "Average"]] + [[feature.replace("_", " ").title(), f"{value:.3f}"] for feature, value in audio_averages.items() if pd.notna(value)]
        story += [Paragraph("Audio Feature Summary", heading), report_table(audio_rows, colors)]
    story += [Paragraph("Artist Insights", heading)]
    story.extend(Paragraph(f"• {insight}", styles["BodyText"]) for insight in insights)
    document.build(story)
    return output.getvalue()


def report_table(rows, colors, widths=None):
    """Apply the dashboard's Spotify-inspired palette to PDF tables."""
    # Imported here so ReportLab remains an optional dependency at app startup.
    from reportlab.platypus import Table, TableStyle
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1DB954")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#B3B3B3")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F4F4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table

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

def create_radar_chart(features_series, title, color="#1DB954"):
    features = list(features_series.index)
    values = list(features_series.values)
    
    features += [features[0]]
    values += [values[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(features) - 1, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0D1A1E')
    ax.set_facecolor('#0D1A1E')
    
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", " ").title() for f in features[:-1]], color="#B3B3B3", size=10)
    
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#263238')
    ax.grid(color='#263238')
    
    ax.set_title(title, color="#FFFFFF", fontweight="bold", pad=20)
    return fig

def csv_bytes(data): return data.to_csv(index=False).encode("utf-8")

add_styles()

if not DATA_FILE.exists(): st.error("Dataset not found. Place spotify_tracks_cleaned.csv in the data folder."); st.stop()
try: 
    raw_df = load_data(str(DATA_FILE))
    original_rows = len(raw_df)
    df = clean_music_data(raw_df)
    cleaned_rows = len(df)
    duplicates_removed = original_rows - cleaned_rows
except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error: st.error(f"The dataset could not be loaded: {error}"); st.stop()
if "playlist_history" not in st.session_state:
    st.session_state.playlist_history = []

if df.empty: st.warning("The dataset is empty."); st.stop()


# ---------------- Sidebar Navigation ----------------
st.sidebar.markdown("<h2>🟢 Spotify <br><span style='font-size:16px; font-weight:normal; color:#B3B3B3'>Music Analytics</span></h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
nav_options_map = {
    "Home": "🏠 Home", 
    "Explore Data": "📊 Explore Data", 
    "Top Songs": "🎵 Top Songs", 
    "Top Artists": "🎤 Top Artists", 
    "Artist Comparison": "⚖️ Artist Comparison",
    "Genre Analysis": "🎸 Genre Analysis", 
    "Audio Features": "🎧 Audio Features", 
    "Trends": "📈 Trends", 
    "Recommendations": "💡 Recommendations", 
    "Smart Playlist": "🧠 Smart Playlist",
    "About": "ℹ️ About"
}
nav = st.sidebar.radio("Navigation", list(nav_options_map.keys()), format_func=lambda x: nav_options_map[x], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("### Recently Generated Playlists")
if st.session_state.playlist_history:
    for idx, pl in enumerate(st.session_state.playlist_history):
        st.sidebar.markdown(f"**{idx+1}. {pl['name']}** ({pl['n']} songs)")
    if st.sidebar.button("Clear History", key="clear_hist"):
        st.session_state.playlist_history = []
        st.rerun()
else:
    st.sidebar.info("No generated playlists yet.")

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
        
    artist_search = ""
    if "artists" in df:
        artist_search = f_col3.text_input("Select Artist", value="", key=f"artist_{st.session_state.filter_key}", placeholder="Type to search (empty for All)")
        
    pop_min, pop_max = 0.0, 100.0
    if "popularity" in df and not df["popularity"].empty:
        pop_min, pop_max = float(df["popularity"].min()), float(df["popularity"].max())
    pop_range = f_col4.slider("Popularity Range", pop_min, pop_max, (pop_min, pop_max), key=f"pop_{st.session_state.filter_key}")
    
    with f_col5:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Clear Filters", key=f"clear_{st.session_state.filter_key}", help="Reset all filters"):
            st.session_state.filter_key += 1
            st.rerun()
            
    with st.expander("Advanced Filters"):
        adv_c1, adv_c2, adv_c3, adv_c4 = st.columns(4)
        
        eng_min, eng_max = 0.0, 1.0
        if "energy" in df and not df["energy"].empty: eng_min, eng_max = float(df["energy"].min()), float(df["energy"].max())
        eng_range = adv_c1.slider("Energy Range", eng_min, eng_max, (eng_min, eng_max), key=f"energy_{st.session_state.filter_key}")
        
        danc_min, danc_max = 0.0, 1.0
        if "danceability" in df and not df["danceability"].empty: danc_min, danc_max = float(df["danceability"].min()), float(df["danceability"].max())
        danc_range = adv_c2.slider("Danceability Range", danc_min, danc_max, (danc_min, danc_max), key=f"dance_{st.session_state.filter_key}")
        
        dur_min, dur_max = 0.0, 60.0
        if "duration_min" in df and not df["duration_min"].empty: dur_min, dur_max = float(df["duration_min"].min()), min(float(df["duration_min"].max()), 60.0)
        dur_range = adv_c3.slider("Duration Range (min)", dur_min, dur_max, (dur_min, dur_max), key=f"duration_{st.session_state.filter_key}")
        
        adv_c4.slider("Release Year", 1950, 2024, (1950, 2024), disabled=True, help="Release year data is not available in the current dataset.", key=f"year_{st.session_state.filter_key}")
        
    st.markdown("<hr style='margin: 10px 0 20px 0; border-color: #263238;'>", unsafe_allow_html=True)

# Apply Filters
filtered_df = df.copy()
if search_text:
    mask = filtered_df.astype(str).apply(lambda s: s.str.contains(search_text, case=False, na=False)).any(axis=1)
    filtered_df = filtered_df[mask]
if genre_choice != "All" and "track_genre" in filtered_df:
    filtered_df = filtered_df[filtered_df["track_genre"].astype(str) == genre_choice]
if artist_search and "artists" in filtered_df:
    filtered_df = filtered_df[filtered_df["artists"].astype(str).str.contains(artist_search, case=False, na=False)]
if "popularity" in filtered_df:
    filtered_df = filtered_df[filtered_df["popularity"].between(pop_range[0], pop_range[1])]
if "energy" in filtered_df:
    filtered_df = filtered_df[filtered_df["energy"].between(eng_range[0], eng_range[1])]
if "danceability" in filtered_df:
    filtered_df = filtered_df[filtered_df["danceability"].between(danc_range[0], danc_range[1])]
if "duration_min" in filtered_df:
    filtered_df = filtered_df[filtered_df["duration_min"].between(dur_range[0], dur_range[1])]

if filtered_df.empty:
    st.warning("No tracks match the current filters. Please adjust them to see results.")
    st.stop()


# ---------------- Page Views ----------------

if nav == "Home":
    # Main Charts Grid Row 1
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if has(filtered_df, ["track_name", "popularity"]):
            top_songs = filtered_df.sort_values("popularity", ascending=False)
            if "artists" in top_songs:
                top_songs = top_songs.drop_duplicates(subset=["track_name", "artists"])
            else:
                top_songs = top_songs.drop_duplicates(subset=["track_name"])
            top_songs = top_songs.head(10)
            
            if not top_songs.empty:
                labels = top_songs["track_name"].fillna("Unknown track")
                if "artists" in top_songs:
                    labels = labels + " - " + top_songs["artists"].fillna("Unknown")
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('#0D1A1E'); ax.set_facecolor('#0D1A1E')
                ax.barh(labels, top_songs["popularity"], color="#1DB954")
                ax.set_title("Top 10 Most Popular Songs", color="#FFFFFF", fontweight='bold')
                ax.set_xlabel("Popularity", color="#B3B3B3")
                ax.tick_params(colors="#B3B3B3")
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
    
    st.markdown("### Data Quality Overview")
    dq_c1, dq_c2, dq_c3, dq_c4, dq_c5 = st.columns(5)
    dq_c1.metric("Original Rows", f"{original_rows:,}")
    dq_c2.metric("Cleaned Rows", f"{cleaned_rows:,}")
    dq_c3.metric("Duplicates Removed", f"{duplicates_removed:,}")
    dq_c4.metric("Unique Songs", f"{cleaned_rows:,}")
    dq_c5.metric("Unique Artists", f"{df['artists'].nunique():,}" if "artists" in df else "N/A")
    st.markdown("---")
    
    st.dataframe(display_data(filtered_df), use_container_width=True, hide_index=True)
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
        display = [c for c in ["track_name", "artists", "duration_formatted", "popularity", "energy", "danceability", "valence", "explicit"] if c in top_songs]
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

        # Artist report uses the existing global artist selector and the shared
        # cleaning function, so duplicate tracks never affect report results.
        st.markdown("---")
        st.subheader("Artist Analysis Report")
        
        report_artist = None
        if not artist_search:
            st.warning("Please search for an artist in the top filter bar first.")
        else:
            matched_artists = sorted(filtered_df["artists"].dropna().astype(str).str.strip().unique())
            if not matched_artists:
                st.warning("No valid artists found for your search.")
            else:
                report_artist = st.selectbox("Select exact artist for the report", matched_artists, key="report_artist_selectbox")
        
        generate_report = st.button("📄 Generate Artist Report", key="generate_artist_report")
        if generate_report:
            if not report_artist:
                st.warning("Please select an artist first.")
            else:
                report_data = clean_artist_data(df, report_artist)
                if report_data.empty:
                    st.warning("No valid tracks are available for this artist.")
                else:
                    statistics = calculate_artist_statistics(report_data)
                    audio_columns = [column for column in ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"] if column in report_data]
                    audio_averages = report_data[audio_columns].apply(pd.to_numeric, errors="coerce").mean() if audio_columns else pd.Series(dtype=float)
                    report_top_songs = report_data.sort_values("popularity", ascending=False, na_position="last").head(10) if "popularity" in report_data else report_data.head(10)
                    report_display_columns = [column for column in ["track_name", "artists", "artist_name", "popularity", "duration_formatted", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"] if column in report_data]
                    artist_export = report_data[report_display_columns]
                    artist_csv = csv_bytes(artist_export)

                    st.success(f"Report prepared for {report_artist}.")
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Unique Songs", f"{statistics['unique_songs']:,}")
                    metric_cols[1].metric("Average Popularity", f"{statistics.get('average_popularity', 0):.1f}" if "average_popularity" in statistics else "N/A")
                    metric_cols[2].metric("Average Duration", statistics["average_duration"])
                    metric_cols[3].metric("Total Duration", statistics["total_duration"])
                    st.markdown("#### Top 10 Unique Songs")
                    st.dataframe(report_top_songs[[column for column in ["track_name", "artists", "artist_name", "popularity", "duration_formatted"] if column in report_top_songs]], use_container_width=True, hide_index=True)

                    chart_cols = st.columns(2)
                    with chart_cols[0]:
                        if "popularity" in report_top_songs and "track_name" in report_top_songs:
                            chart_data = report_top_songs.sort_values("popularity")
                            horizontal_bar(chart_data, "popularity", "track_name", "Top Songs by Popularity", color="#1DB954")
                    with chart_cols[1]:
                        if not audio_averages.dropna().empty:
                            chart_data = audio_averages.dropna().reset_index()
                            chart_data.columns = ["feature", "average"]
                            horizontal_bar(chart_data, "average", "feature", "Average Audio Features", color="#9B51E0")
                    if "popularity" in report_data:
                        histogram(report_data, "popularity", "Popularity Distribution", "Popularity", color="#1DB954")

                    audio_cols_radar = [c for c in ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness", "liveness"] if c in report_data]
                    if audio_cols_radar:
                        av_r = report_data[audio_cols_radar].apply(pd.to_numeric, errors="coerce").mean().dropna()
                        if not av_r.empty:
                            r_col1, r_col2 = st.columns([1, 2])
                            with r_col1:
                                fig_r = create_radar_chart(av_r, f"{report_artist} Audio Profile", color="#1DB954")
                                st.pyplot(fig_r); plt.close(fig_r)
                            with r_col2:
                                st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
                                st.info("This radar chart visualizes the average audio profile of the artist based on all available tracks in the dataset.")

                    st.markdown("#### Artist Career Timeline")
                    st.warning("Trend and timeline analysis requires release-year data, which is currently unavailable in the dataset.")

                    st.markdown("#### Artist Insights")
                    for insight in generate_artist_insights(statistics, audio_averages):
                        st.markdown(f"• {insight}")
                    downloads = st.columns(2)
                    with downloads[0]:
                        st.download_button("Download Artist CSV", artist_csv, f"{report_artist}_spotify_tracks.csv", "text/csv", key="artist_csv_download")
                    with downloads[1]:
                        try:
                            pdf_bytes = create_artist_pdf(report_artist, artist_csv)
                        except Exception as error:
                            pdf_bytes = None
                            st.error(f"The PDF report could not be generated: {error}")
                        if pdf_bytes:
                            st.download_button("Download Artist PDF", pdf_bytes, f"{report_artist}_artist_report.pdf", "application/pdf", key="artist_pdf_download")
                        else:
                            st.info("PDF export requires ReportLab. Install it with: pip install reportlab")

elif nav == "Artist Comparison":
    st.title("Artist Comparison")
    st.markdown("Select two artists to compare their profiles side-by-side.")
    
    comp_c1, comp_c2 = st.columns(2)
    artist1_search = comp_c1.text_input("Search Artist 1", key="a1_search")
    artist2_search = comp_c2.text_input("Search Artist 2", key="a2_search")
    
    a1_choices = [""]
    a2_choices = [""]
    if "artists" in df:
        if artist1_search:
            a1_choices = sorted(df[df["artists"].astype(str).str.contains(artist1_search, case=False, na=False)]["artists"].dropna().astype(str).str.strip().unique())
        if artist2_search:
            a2_choices = sorted(df[df["artists"].astype(str).str.contains(artist2_search, case=False, na=False)]["artists"].dropna().astype(str).str.strip().unique())
            
    a1 = comp_c1.selectbox("Select Exact Artist 1", a1_choices, key="a1_exact")
    a2 = comp_c2.selectbox("Select Exact Artist 2", a2_choices, key="a2_exact")
    
    compare_btn = st.button("⚖️ Compare Artists")
    
    if compare_btn:
        if not a1 or not a2:
            st.warning("Please select both artists to compare.")
        elif a1 == a2:
            st.warning("Please select two different artists.")
        else:
            d1 = clean_artist_data(df, a1)
            d2 = clean_artist_data(df, a2)
            if d1.empty or d2.empty:
                st.warning("One or both artists have no valid tracks.")
            else:
                s1 = calculate_artist_statistics(d1)
                s2 = calculate_artist_statistics(d2)
                
                st.markdown("### Comparison Metrics")
                m_cols = st.columns(2)
                
                with m_cols[0]:
                    st.markdown(f"#### {a1}", unsafe_allow_html=True)
                    st.markdown(f"<div style='background:#0D1A1E; padding:15px; border-radius:8px; border:1px solid #263238;'>", unsafe_allow_html=True)
                    st.markdown(f"**Unique Songs:** {s1['unique_songs']}<br>"
                                f"**Avg Popularity:** {s1.get('average_popularity', 0):.1f}<br>"
                                f"**Highest Popularity:** {s1.get('highest_popularity', 0)}<br>"
                                f"**Lowest Popularity:** {s1.get('lowest_popularity', 0)}<br>"
                                f"**Avg Duration:** {s1.get('average_duration', 'N/A')}<br>"
                                f"**Most Popular Song:** {s1.get('most_popular_song', 'N/A')}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with m_cols[1]:
                    st.markdown(f"#### {a2}", unsafe_allow_html=True)
                    st.markdown(f"<div style='background:#0D1A1E; padding:15px; border-radius:8px; border:1px solid #263238;'>", unsafe_allow_html=True)
                    st.markdown(f"**Unique Songs:** {s2['unique_songs']}<br>"
                                f"**Avg Popularity:** {s2.get('average_popularity', 0):.1f}<br>"
                                f"**Highest Popularity:** {s2.get('highest_popularity', 0)}<br>"
                                f"**Lowest Popularity:** {s2.get('lowest_popularity', 0)}<br>"
                                f"**Avg Duration:** {s2.get('average_duration', 'N/A')}<br>"
                                f"**Most Popular Song:** {s2.get('most_popular_song', 'N/A')}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("### Top Songs Comparison")
                ts_cols = st.columns(2)
                with ts_cols[0]:
                    st.markdown(f"**{a1} Top 5 Songs**")
                    top1 = d1.sort_values("popularity", ascending=False).head(5) if "popularity" in d1 else d1.head(5)
                    st.dataframe(top1[[c for c in ["track_name", "popularity", "duration_formatted", "track_genre"] if c in top1]], use_container_width=True, hide_index=True)
                with ts_cols[1]:
                    st.markdown(f"**{a2} Top 5 Songs**")
                    top2 = d2.sort_values("popularity", ascending=False).head(5) if "popularity" in d2 else d2.head(5)
                    st.dataframe(top2[[c for c in ["track_name", "popularity", "duration_formatted", "track_genre"] if c in top2]], use_container_width=True, hide_index=True)

                st.markdown("### Audio Profile & Metrics Chart")
                audio_cols = [c for c in ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness", "liveness", "tempo"] if c in df]
                
                if audio_cols:
                    ac1, ac2 = st.columns(2)
                    av1 = d1[audio_cols].apply(pd.to_numeric, errors="coerce").mean().dropna()
                    av2 = d2[audio_cols].apply(pd.to_numeric, errors="coerce").mean().dropna()
                    
                    with ac1:
                        chart_data = pd.DataFrame(columns=["Metric", a1, a2])
                        metrics_list = []
                        if "average_popularity" in s1:
                            metrics_list.append({"Metric": "Avg Popularity", a1: s1["average_popularity"]/100, a2: s2["average_popularity"]/100})
                        if "energy" in av1:
                            metrics_list.append({"Metric": "Avg Energy", a1: av1["energy"], a2: av2["energy"]})
                        if "danceability" in av1:
                            metrics_list.append({"Metric": "Avg Danceability", a1: av1["danceability"], a2: av2["danceability"]})
                        
                        if metrics_list:
                            chart_df = pd.DataFrame(metrics_list).set_index("Metric")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            fig.patch.set_facecolor('#0D1A1E')
                            ax.set_facecolor('#0D1A1E')
                            
                            x = np.arange(len(chart_df.index))
                            width = 0.35
                            
                            ax.bar(x - width/2, chart_df[a1], width, label=a1, color="#1DB954")
                            ax.bar(x + width/2, chart_df[a2], width, label=a2, color="#9B51E0")
                            
                            ax.set_title("Key Metrics Comparison (Normalized 0-1)", color="#FFFFFF", fontweight="bold")
                            ax.set_xticks(x)
                            ax.set_xticklabels(chart_df.index, color="#B3B3B3")
                            ax.tick_params(colors="#B3B3B3")
                            ax.legend(facecolor='#0D1A1E', labelcolor="#FFFFFF", edgecolor="#263238")
                            ax.grid(axis="y", alpha=0.1, color="#B3B3B3")
                            for spine in ax.spines.values(): spine.set_visible(False)
                            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

                    with ac2:
                        radar_features1 = av1[[c for c in ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness", "liveness"] if c in av1]]
                        if not radar_features1.empty:
                            fig1 = create_radar_chart(radar_features1, f"{a1} Audio Profile", color="#1DB954")
                            st.pyplot(fig1); plt.close(fig1)

elif nav == "Genre Analysis":
    st.title("Genre Explorer")
    if has(filtered_df, ["track_genre"]):
        genres = sorted(filtered_df["track_genre"].dropna().astype(str).unique())
        selected_genre = st.selectbox("Select a Genre to Explore", [""] + genres)
        
        if selected_genre:
            g_df = filtered_df[filtered_df["track_genre"].astype(str) == selected_genre]
            if not g_df.empty:
                g_c1, g_c2, g_c3, g_c4 = st.columns(4)
                g_c1.metric("Unique Songs", f"{len(g_df):,}")
                
                unique_a = g_df["artists"].nunique() if "artists" in g_df else "N/A"
                g_c2.metric("Unique Artists", f"{unique_a:,}" if isinstance(unique_a, int) else unique_a)
                
                avg_p = g_df["popularity"].mean() if "popularity" in g_df else 0
                g_c3.metric("Average Popularity", f"{avg_p:.1f}")
                
                avg_d = g_df["duration_ms"].mean() if "duration_ms" in g_df else None
                g_c4.metric("Average Duration", format_duration(avg_d))
                
                r_c1, r_c2 = st.columns([1, 1])
                with r_c1:
                    st.markdown("#### Top 10 Songs")
                    if "popularity" in g_df:
                        g_top = g_df.sort_values("popularity", ascending=False).head(10)
                        cols = [c for c in ["track_name", "artists", "popularity", "duration_formatted"] if c in g_top]
                        st.dataframe(g_top[cols], use_container_width=True, hide_index=True)
                with r_c2:
                    audio_cols = [c for c in ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness", "liveness"] if c in g_df]
                    if audio_cols:
                        av_g = g_df[audio_cols].apply(pd.to_numeric, errors="coerce").mean().dropna()
                        if not av_g.empty:
                            fig_g = create_radar_chart(av_g, f"{selected_genre.title()} Audio Profile", color="#9B51E0")
                            st.pyplot(fig_g); plt.close(fig_g)
        else:
            st.info("Select a genre above to see detailed metrics.")
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
    
    st.markdown("### Release-Year Trends")
    st.warning("Trend analysis over time requires release-year data, which is currently unavailable in the dataset.")
    
    st.markdown("### Feature Distributions")
    st.caption("Distribution of audio features across the dataset.")
    if not filtered_df.empty:
        for col in ["tempo", "loudness", "acousticness", "liveness"]:
            if col in filtered_df: histogram(filtered_df, col, f"{col.replace('_',' ').title()} Distribution", col.title())


elif nav == "Recommendations":
    st.title("Song Recommendation System")
    st.markdown("Search for a song below to find similar tracks based on dataset audio features.")
    
    if has(df, ["track_name", "artists"]):
        r_c1, r_c2, r_c3 = st.columns(3)
        rec_lang = r_c1.selectbox("Filter Language (Optional)", ["All Languages", "Hindi", "Punjabi", "English", "Tamil", "Telugu", "Bengali", "Marathi", "Malayalam", "Kannada", "Gujarati", "Other"], key="rec_lang")
        
        genres_opt = ["All Genres"]
        if "track_genre" in df.columns:
            genres_opt.extend(sorted([str(g) for g in df["track_genre"].dropna().unique()]))
        rec_genre = r_c2.selectbox("Filter Genre (Optional)", genres_opt, key="rec_genre")
        
        n_recs = r_c3.selectbox("Number of recommendations:", [5, 10, 15, 20], index=1)
        
        song_search = st.text_input("Search for a target song:", key="rec_search_input", placeholder="Type a song name...")
        
        song_choices = [""]
        song_lookup = {}
        if song_search:
            matches = df[df["track_name"].astype(str).str.contains(song_search, case=False, na=False)].head(50)
            if not matches.empty:
                for idx, row in matches.iterrows():
                    display_name = f"{row.get('track_name', 'Unknown')} - {row.get('artists', 'Unknown')}"
                    song_lookup[display_name] = idx
                song_choices = sorted(list(song_lookup.keys()))
                
        target_display = st.selectbox("Select exact song:", song_choices, key="rec_exact_song")
        
        if st.button("🎵 Find Similar Songs"):
            if not target_display:
                st.warning("Please search and select a target song first.")
            else:
                target_idx_df = song_lookup[target_display]
                
                with st.spinner("Analyzing audio features..."):
                    valid_data, standardized_mat = get_feature_matrix(df)
                    
                    if standardized_mat.size == 0:
                        st.warning("Not enough valid audio-feature data is available to generate recommendations.")
                    else:
                        if target_idx_df not in valid_data.index:
                            st.warning("The selected song does not have enough audio feature data to calculate similarities.")
                        else:
                            pos_idx = valid_data.index.get_loc(target_idx_df)
                            
                            # Get all recommendations sorted by similarity
                            recommendations_full = get_recommendations(valid_data, standardized_mat, pos_idx, n=len(valid_data))
                            
                            # Apply Language filter
                            if rec_lang != "All Languages":
                                recommendations_full = add_language_column(recommendations_full)
                                recommendations_full = recommendations_full[recommendations_full["detected_language"] == rec_lang]
                                
                            # Apply Genre filter
                            if rec_genre != "All Genres" and "track_genre" in recommendations_full.columns:
                                recommendations_full = recommendations_full[recommendations_full["track_genre"].astype(str) == rec_genre]
                                
                            recommendations = recommendations_full.head(n_recs).copy()
                            
                            if recommendations.empty:
                                st.warning("No similar songs could be found with the selected filters. Try removing the language or genre filter.")
                            else:
                                st.success("Recommendations found!")
                                st.markdown("<p class='small-note'>These songs were selected based on similarities in available audio features. This is a dataset-based numerical similarity score, not an official Spotify recommendation.</p>", unsafe_allow_html=True)
                                
                                recommendations["Similarity"] = (recommendations["similarity"] * 100).map("{:.1f}%".format)
                                
                                display_cols = []
                                for col in ["track_name", "artists", "popularity", "duration_formatted", "track_genre", "detected_language", "Similarity"]:
                                    if col in recommendations.columns:
                                        display_cols.append(col)
                                        
                                disp_df = recommendations[display_cols].copy()
                                disp_df.columns = [c.replace('_', ' ').title() if c not in ["duration_formatted", "detected_language"] else ("Duration" if c == "duration_formatted" else "Language") for c in disp_df.columns]
                                
                                st.dataframe(disp_df, use_container_width=True, hide_index=True)
                                
                                csv_cols = [c for c in ["track_name", "artists", "popularity", "duration_formatted", "track_genre", "detected_language", "similarity"] if c in recommendations.columns]
                                csv_data = csv_bytes(recommendations[csv_cols])
                                st.download_button(label="⬇️ Download Recommendations CSV", data=csv_data, file_name="recommendations.csv", mime="text/csv")

    st.markdown("---")
    st.title("Mood-Based Music Explorer")
    
    mood_c1, mood_c2, mood_c3 = st.columns([2, 2, 1])
    mood_choice = mood_c1.selectbox("Select a Mood", ["Happy", "Sad", "Energetic", "Relaxing", "Party", "Focus"])
    lang_choice = mood_c2.selectbox("Select Music Language", ["All Languages", "Hindi", "Punjabi", "English", "Tamil", "Telugu", "Bengali", "Marathi", "Malayalam", "Kannada", "Gujarati", "Other"])
    n_mood = mood_c3.selectbox("Number of songs", [5, 10, 15, 20], index=1, key="n_mood")
    
    st.caption("Language detection is approximate and depends on available dataset metadata.")
    
    mood_explanations = {
        "Happy": "Happy mood focuses on songs with higher valence, energy, and danceability.",
        "Sad": "Sad mood focuses on lower-energy songs with lower valence and danceability where available.",
        "Energetic": "Energetic mood focuses on high-energy, danceable tracks.",
        "Relaxing": "Relaxing mood focuses on lower-energy songs with higher acousticness where available.",
        "Party": "Party mood focuses on the highest energy, danceability, and positive valence.",
        "Focus": "Focus mood focuses on instrumental and lower-speechiness songs where available."
    }
    
    st.markdown(f"<p style='color:#B3B3B3; font-style:italic;'>{mood_explanations[mood_choice]}</p>", unsafe_allow_html=True)
    
    if st.button("🎧 Explore Mood"):
        with st.spinner(f"Curating {mood_choice} playlist..."):
            df_with_lang = add_language_column(df)
            
            if lang_choice != "All Languages":
                df_filtered = df_with_lang[df_with_lang["detected_language"] == lang_choice].copy()
            else:
                df_filtered = df_with_lang.copy()
                
            mood_results = get_mood_recommendations(df_filtered, mood_choice, n_mood)
            
            if mood_results.empty:
                st.warning("No songs found for this mood and language combination. Try another mood or language.")
            else:
                st.success(f"Generated {mood_choice} Playlist!")
                
                st.markdown("#### Playlist Summary")
                s_cols = st.columns(6)
                s_cols[0].metric("Songs", len(mood_results))
                
                avg_pop = mood_results["popularity"].mean() if "popularity" in mood_results else 0
                s_cols[1].metric("Avg Popularity", f"{avg_pop:.1f}")
                
                avg_e = mood_results["energy"].mean() if "energy" in mood_results else 0
                s_cols[2].metric("Avg Energy", f"{avg_e:.2f}")
                
                avg_d = mood_results["danceability"].mean() if "danceability" in mood_results else 0
                s_cols[3].metric("Avg Danceability", f"{avg_d:.2f}")
                
                avg_v = mood_results["valence"].mean() if "valence" in mood_results else "N/A"
                s_cols[4].metric("Avg Valence", f"{avg_v:.2f}" if isinstance(avg_v, float) else "N/A")
                
                tot_ms = mood_results["duration_ms"].sum() if "duration_ms" in mood_results else 0
                s_cols[5].metric("Total Duration", format_duration(tot_ms))
                
                st.info("This score represents similarity to the selected mood profile based on available audio features. It is not an official Spotify score.")
                
                mood_results["Mood Match Score"] = mood_results["mood_score"].map("{:.1f}%".format)
                
                disp_cols_mood = []
                for c in ["track_name", "artists", "album_name", "popularity", "duration_formatted", "track_genre", "detected_language", "Mood Match Score"]:
                    if c in mood_results.columns: disp_cols_mood.append(c)
                
                disp_df_m = mood_results[disp_cols_mood].copy()
                disp_df_m.columns = [c.replace('_', ' ').title() if c not in ["duration_formatted", "detected_language"] else ("Duration" if c == "duration_formatted" else "Language") for c in disp_df_m.columns]
                st.dataframe(disp_df_m, use_container_width=True, hide_index=True)
                
                csv_cols = [c for c in ["track_name", "artists", "album_name", "popularity", "duration_formatted", "track_genre", "detected_language", "energy", "danceability", "valence", "acousticness", "instrumentalness", "mood_score"] if c in mood_results.columns]
                csv_data = csv_bytes(mood_results[csv_cols])
                st.download_button(label="⬇️ Download Playlist CSV", data=csv_data, file_name=f"{mood_choice.lower()}_playlist.csv", mime="text/csv")

elif nav == "Smart Playlist":
    st.title("Smart Playlist Generator")
    st.markdown("Create a personalized playlist using multiple intelligent filters.")
    
    st.markdown("### Create Your Playlist")
    playlist_name = st.text_input("Enter Playlist Name", value="My Smart Playlist")
    
    c1, c2, c3 = st.columns(3)
    sp_mood = c1.selectbox("Select Mood", ["Any Mood", "Happy", "Sad", "Energetic", "Relaxing", "Party", "Focus"], key="sp_mood")
    sp_lang = c2.selectbox("Select Language", ["All Languages", "Hindi", "Punjabi", "English", "Tamil", "Telugu", "Bengali", "Marathi", "Malayalam", "Kannada", "Gujarati", "Other"], key="sp_lang")
    
    genres = ["All Genres"]
    if "track_genre" in df.columns:
        valid_g = df["track_genre"].dropna().unique().tolist()
        genres.extend(sorted([str(g) for g in valid_g]))
    sp_genre = c3.selectbox("Select Genre", genres, key="sp_genre")
    
    st.markdown("### Playlist Preferences")
    c4, c5 = st.columns(2)
    sp_pop = c4.slider("Minimum Popularity", 0, 100, 0, key="sp_pop") if "popularity" in df.columns else 0
    sp_n = c5.selectbox("Select Number of Songs", [5, 10, 20, 30, 50, 100], index=2, key="sp_n")
    
    audio_cols_ui = [col for col in ["energy", "danceability", "valence", "acousticness"] if col in df.columns]
    sp_audio_prefs = {}
    if audio_cols_ui:
        st.markdown("Optional Audio Preferences")
        a_cols = st.columns(len(audio_cols_ui))
        for idx, ac in enumerate(audio_cols_ui):
            sp_audio_prefs[ac] = a_cols[idx].selectbox(ac.title(), ["Any", "Low", "Medium", "High"], key=f"sp_a_{ac}")
            
    if st.button("Generate Smart Playlist", type="primary"):
        with st.spinner("Generating playlist..."):
            sp_df = df.copy()
            
            if sp_genre != "All Genres" and "track_genre" in sp_df.columns:
                sp_df = sp_df[sp_df["track_genre"].astype(str) == sp_genre]
                
            if "popularity" in sp_df.columns:
                sp_df = sp_df[pd.to_numeric(sp_df["popularity"], errors="coerce").fillna(0) >= sp_pop]
                
            if sp_lang != "All Languages":
                sp_df = add_language_column(sp_df)
                sp_df = sp_df[sp_df["detected_language"] == sp_lang]
                
            for ac, val in sp_audio_prefs.items():
                if val != "Any":
                    ac_series = pd.to_numeric(sp_df[ac], errors="coerce")
                    if val == "Low": sp_df = sp_df[ac_series <= 0.33]
                    elif val == "Medium": sp_df = sp_df[(ac_series > 0.33) & (ac_series <= 0.66)]
                    elif val == "High": sp_df = sp_df[ac_series > 0.66]
            
            if "track_name" in sp_df.columns:
                sp_df = sp_df.dropna(subset=["track_name"])
            
            if sp_df.empty:
                st.warning("No songs found for these playlist settings.\nTry changing the mood, language, genre, or popularity filter.")
            else:
                if sp_mood != "Any Mood":
                    mood_results = get_mood_recommendations(sp_df, sp_mood, sp_n)
                else:
                    if "popularity" in sp_df.columns:
                        mood_results = sp_df.sort_values("popularity", ascending=False)
                    else:
                        mood_results = sp_df.sample(frac=1, random_state=42)
                    subset_cols = ["track_name", "artists"] if "artists" in mood_results.columns else ["track_name"]
                    mood_results = mood_results.drop_duplicates(subset=subset_cols).head(sp_n)
                    
                if mood_results.empty:
                    st.warning("No songs found for these playlist settings.\nTry changing the mood, language, genre, or popularity filter.")
                else:
                    st.success("Playlist Generated Successfully!")
                    st.session_state.playlist_history.insert(0, {
                        "name": playlist_name,
                        "mood": sp_mood,
                        "lang": sp_lang,
                        "genre": sp_genre,
                        "n": len(mood_results),
                        "df": mood_results.copy()
                    })
                    if len(st.session_state.playlist_history) > 10:
                        st.session_state.playlist_history.pop()

    if st.session_state.playlist_history:
        st.markdown("---")
        st.markdown("### Generated Playlist")
        history_names = [f"{idx+1}. {pl['name']}" for idx, pl in enumerate(st.session_state.playlist_history)]
        selected_hist = st.selectbox("View Generated Playlist", history_names, index=0)
        selected_idx = int(selected_hist.split(".")[0]) - 1
        current_pl = st.session_state.playlist_history[selected_idx]
        
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        p_c1.metric("Songs", current_pl["n"])
        p_c2.metric("Mood", current_pl["mood"])
        p_c3.metric("Language", current_pl["lang"])
        p_c4.metric("Genre", current_pl["genre"])
        
        pdf = current_pl["df"]
        display_cols = []
        for c in ["track_name", "artists", "album_name", "popularity", "duration_formatted", "track_genre", "detected_language", "energy", "danceability", "valence"]:
            if c in pdf.columns: display_cols.append(c)
        
        st.dataframe(pdf[display_cols], use_container_width=True, hide_index=True)
        
        d_c1, d_c2 = st.columns([1, 4])
        csv_data = csv_bytes(pdf[display_cols])
        safe_name = "".join([c if c.isalnum() else "_" for c in current_pl['name']]).lower()
        d_c1.download_button(label="⬇️ Download CSV", data=csv_data, file_name=f"{safe_name}.csv", mime="text/csv")
        
        txt_lines = [f"Playlist: {current_pl['name']}\n"]
        for idx_t, row in pdf.iterrows():
            tn = row.get("track_name", "Unknown")
            ar = row.get("artists", "Unknown")
            txt_lines.append(f"{tn} - {ar}")
        txt_str = "\n".join(txt_lines)
        d_c2.download_button(label="⬇️ Download TXT", data=txt_str, file_name=f"{safe_name}.txt", mime="text/plain")
        
        st.markdown("### Playlist Insights")
        i_c1, i_c2 = st.columns(2)
        with i_c1:
            st.markdown("**Audio Profile**")
            acols = [c for c in ["energy", "danceability", "valence", "acousticness", "instrumentalness", "tempo"] if c in pdf.columns]
            if acols:
                avg_audio = pdf[acols].apply(pd.to_numeric, errors="coerce").mean().dropna()
                if not avg_audio.empty:
                    f = create_radar_chart(avg_audio, "Average Features", "#1DB954")
                    st.pyplot(f); plt.close(f)
        with i_c2:
            if "artists" in pdf.columns:
                st.markdown("**Top Artists**")
                st.dataframe(pdf["artists"].value_counts().head(5).reset_index(name="Count"), hide_index=True)
            if "track_genre" in pdf.columns:
                st.markdown("**Top Genres**")
                st.dataframe(pdf["track_genre"].value_counts().head(5).reset_index(name="Count"), hide_index=True)


elif nav == "About":
    st.title("About")
    st.markdown("""
    **Spotify Music Analytics** is an interactive dashboard built using Python, Pandas, Matplotlib, and Streamlit.
    It empowers users to explore a massive dataset of music tracks, compare artists, investigate audio features, and uncover trends.
    """)
