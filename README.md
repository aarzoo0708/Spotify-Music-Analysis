# 🎵 Spotify Music Analysis

An interactive **Spotify Music Analysis Dashboard** built with Python, Pandas, NumPy, Matplotlib, and Streamlit.

This project explores Spotify track data through data cleaning, exploratory data analysis, statistical summaries, interactive filtering, and visualizations. The goal is to turn a large Spotify dataset into an easy-to-use interactive dashboard.

---

## 📌 Project Overview

The Spotify dataset contains information about thousands of songs, artists, genres, popularity, duration, and audio features.

This project analyzes that data to answer questions such as:

* Which songs are the most popular?
* Which genres have the highest average popularity?
* How energetic or danceable are different genres?
* How does an individual artist perform across different metrics?
* How do two artists compare?
* What relationships exist between Spotify audio features?

The project presents these analyses through an interactive Streamlit dashboard.

---

## ✨ Features

### 📊 Dashboard Overview

The dashboard provides an overview of the Spotify dataset, including:

* Total number of songs
* Total number of artists
* Total number of genres
* Average song popularity
* Average song duration
* Basic dataset statistics

### 🧹 Data Cleaning

The project performs basic data preprocessing:

* Removes duplicate tracks
* Handles missing values where required
* Converts song duration from milliseconds
* Creates cleaned datasets
* Creates a dataset containing unique tracks

### 🎵 Popularity Analysis

Explore Spotify song popularity through:

* Top 10 popular songs
* Popularity statistics
* Popularity distribution
* Popularity-based filtering
* Popularity visualizations

### 🎼 Genre Analysis

Analyze different Spotify genres using:

* Number of songs per genre
* Average popularity by genre
* Genre-based filtering
* Genre visualizations
* Audio feature comparisons

### 🎤 Artist Analysis

Select an artist and explore:

* Artist's songs
* Number of songs
* Average popularity
* Popular songs
* Audio features

### ⚖️ Artist Comparison

Compare two artists using simple statistical measures such as:

* Number of songs
* Average popularity
* Average energy
* Average danceability

### 🎧 Audio Feature Analysis

The project explores Spotify audio features including:

* Energy
* Danceability
* Valence
* Acousticness
* Instrumentalness
* Speechiness
* Liveness
* Tempo
* Loudness

### 📈 Data Visualization

Matplotlib is used to create visualizations such as:

* Bar charts
* Histograms
* Comparison charts
* Feature distributions
* Correlation visualizations

### 🎭 Simple Mood Exploration

Songs can optionally be explored using simple **rule-based filtering** based on audio features.

For example:

* Higher valence → approximately happier songs
* Higher energy → approximately more energetic songs
* Higher acousticness and lower energy → approximately relaxing songs
* Lower valence → approximately sad songs

> Mood classification is only an approximate rule-based analysis. It is not a machine-learning prediction system.

### 🎶 Simple Playlist Generation

Users can generate a simple playlist by applying filters such as:

* Genre
* Mood
* Popularity
* Number of songs

The playlist is created using Pandas filtering and sampling rather than a machine-learning recommendation algorithm.

### 🔗 Spotify Search Links

Songs can be opened through Spotify search links using the track and artist information.

The application does not download or directly stream copyrighted music.

### 📥 Data Downloads

Users can download selected or generated data as CSV files for further analysis.

---

# 🛠️ Technologies Used

| Technology | Purpose                    |
| ---------- | -------------------------- |
| Python     | Main programming language  |
| Pandas     | Data cleaning and analysis |
| NumPy      | Numerical operations       |
| Matplotlib | Data visualization         |
| Streamlit  | Interactive web dashboard  |
| Git        | Version control            |
| GitHub     | Repository hosting         |

---

# 📂 Project Structure

```text
Spotify-Music-Analysis/
│
├── __pycache__/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── spotify_tracks_cleaned.csv
│   ├── spotify_tracks.csv
│   └── spotify_unique_tracks.csv
│
├── src/
│   ├── basic_analysis.py
│   ├── correlation_analysis.py
│   ├── create_unique_tracks.py
│   ├── data_cleaning.py
│   ├── energy_danceability.py
│   ├── genre_analysis.py
│   ├── genre_visualization.py
│   └── popularity_histogram.py
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

### 📁 Important Files

| File / Folder            | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `app.py`           | Simplified and beginner-friendly Streamlit application   |
| `data/`                  | Original, cleaned, and unique Spotify datasets           |
| `src/`                   | Individual Python scripts for analysis and visualization |
| `.streamlit/config.toml` | Streamlit configuration                                  |
| `requirements.txt`       | Required Python packages                                 |
| `.gitignore`             | Files excluded from Git                                  |

---

# 📊 Dataset

The project uses a Spotify tracks dataset containing information about songs, artists, genres, popularity, and audio characteristics.

Important columns include, where available:

```text
track_id
artists
album_name
track_name
popularity
duration_ms
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
track_genre
```

The project contains different versions of the dataset:

### `spotify_tracks.csv`

Original Spotify dataset.

### `spotify_tracks_cleaned.csv`

Dataset after basic data cleaning.

### `spotify_unique_tracks.csv`

Dataset containing unique tracks after duplicate removal.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/aarzoo0708/Spotify-Music-Analysis.git
```

## 2. Open the Project

```bash
cd Spotify-Music-Analysis
```

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Application

For the simplified data-analysis version:

```bash
streamlit run app_basic.py
```

The application will open automatically in your browser.

---

# 🖥️ Application Sections

The dashboard contains sections for exploring different aspects of the Spotify dataset.

### Overview

Provides basic information about the dataset and important statistics.

### Popularity Analysis

Explores the most popular tracks and popularity distributions.

### Genre Analysis

Analyzes song counts and average popularity across genres.

### Artist Analysis

Allows users to select an artist and explore their songs and statistics.

### Artist Comparison

Provides a simple comparison between two selected artists.

### Audio Features

Explores characteristics such as energy, danceability, valence, and acousticness.

### Correlation Analysis

Shows relationships between numerical Spotify features.

### Playlist Explorer

Allows users to filter songs and create a simple playlist from the available dataset.

---

# 📈 Example Analysis

Some examples of analysis performed in the project include:

```python
# Find the most popular songs
top_songs = (
    df.sort_values("popularity", ascending=False)
    .drop_duplicates("track_name")
    .head(10)
)
```

Average popularity by genre:

```python
genre_popularity = (
    df.groupby("track_genre")["popularity"]
    .mean()
    .sort_values(ascending=False)
)
```

Artist filtering:

```python
artist_df = df[df["artists"] == selected_artist]
```

These operations form the foundation of the project's data analysis.

---

# 🧠 Concepts Demonstrated

This project demonstrates practical use of:

### Python

* Functions
* Conditions
* Loops
* Modules
* File handling

### Pandas

* DataFrames
* Filtering
* Sorting
* GroupBy
* Aggregation
* Missing values
* Duplicate removal
* String operations
* Data transformation

### NumPy

* Numerical calculations
* Array operations
* Statistical calculations

### Matplotlib

* Bar charts
* Histograms
* Feature comparisons
* Data visualization

### Streamlit

* Interactive dashboards
* Select boxes
* Sliders
* Metrics
* Tabs
* DataFrames
* Download buttons
* Interactive filtering

---

# 🎯 Project Goals

The main goals of this project are:

1. Practice Python-based data analysis.
2. Understand real-world datasets.
3. Perform data cleaning and preprocessing.
4. Explore patterns in Spotify music data.
5. Create meaningful visualizations.
6. Build an interactive Streamlit dashboard.
7. Make data analysis accessible through a simple interface.

---

# ⚠️ Limitations

* Mood categories are approximate and rule-based.
* Results depend on the quality of the Spotify dataset.
* Language information may not be available or reliable for every track.
* Spotify links open search pages rather than directly streaming music.
* The project currently focuses on data analysis rather than machine-learning recommendations.

---

# 🔮 Future Improvements

Possible future improvements include:

* Machine-learning-based song recommendations
* More advanced mood classification
* Spotify API integration
* Personalized recommendations
* User accounts
* Persistent playlist storage
* Advanced language classification
* Interactive audio previews
* Deployment as a public web application

These features can be added later as additional learning modules.

---

# 📚 Learning Purpose

This project was developed as a practical learning project to understand how Python data-analysis libraries can be combined with Streamlit to create an interactive application.

The current version intentionally focuses on concepts that can be understood using:

**Python → NumPy → Pandas → Matplotlib → Streamlit**

More advanced machine-learning concepts can be added as a future extension.

---

# 👩‍💻 Author

**Aarzoo Bajaj**

GitHub:
https://github.com/aarzoo0708

Project Repository:
https://github.com/aarzoo0708/Spotify-Music-Analysis

---

# ⭐ Acknowledgement

This project was developed for educational and portfolio purposes while learning Python, data analysis, visualization, and Streamlit application development.

---

# 📄 License

This project is intended for educational and portfolio purposes.
