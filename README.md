# 🎵 Spotify Music Analysis

An interactive Spotify Music Analysis Dashboard built using **Python, Pandas, NumPy, Matplotlib, and Streamlit**.

This project analyzes Spotify music data and provides interactive insights into songs, artists, genres, popularity, audio features, moods, languages, and personalized playlists.

---

## 🚀 Project Overview

Music datasets contain valuable information about songs, artists, genres, popularity, and audio characteristics. However, understanding this data through raw CSV files can be difficult.

The **Spotify Music Analysis** project converts raw Spotify data into an interactive dashboard where users can explore music trends, compare artists, discover songs, and generate personalized playlists.

The application combines:

* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Song recommendation
* Mood-based music exploration
* Language-based filtering
* Smart playlist generation
* Spotify search integration
* Downloadable reports

---

## ✨ Features

### 📊 Interactive Dashboard

* View an overview of the Spotify dataset.
* Display the total number of songs.
* Explore artists and genres.
* View average popularity.
* Analyze average song duration.
* Explore important dataset statistics.

### 🧹 Data Cleaning

The project includes data-cleaning operations such as:

* Removing duplicate songs.
* Creating unique track datasets.
* Handling missing values.
* Cleaning and preparing data for analysis.
* Converting song duration from milliseconds into minutes and seconds.
* Preparing data for recommendation and playlist generation.

### 🎼 Genre Analysis

* Explore available music genres.
* Analyze the number of songs in each genre.
* Compare genre popularity.
* Visualize genre-related information.
* Explore audio features across genres.

### 📈 Popularity Analysis

* Analyze song popularity.
* Display popular songs.
* Explore popularity distributions.
* Generate popularity histograms.
* Compare popularity across different categories.

### 🎤 Artist Analysis

* Search and explore artists.
* View songs associated with an artist.
* Analyze artist popularity.
* Explore the most popular songs by an artist.
* Generate artist-related reports where supported.

### ⚖️ Artist Comparison

Compare two artists using available dataset information, such as:

* Number of songs.
* Average popularity.
* Average energy.
* Average danceability.
* Average valence.
* Other available audio features.

### 🎧 Song Recommendation System

The application provides song recommendations using available song metadata and audio features.

Possible recommendation features include:

* Danceability
* Energy
* Valence
* Acousticness
* Instrumentalness
* Tempo
* Loudness
* Popularity

The recommendation system also handles duplicate songs and missing data where possible.

### 🎭 Mood-Based Music Explorer

Users can explore songs according to approximate mood categories:

* Happy
* Sad
* Energetic
* Relaxing
* Party
* Focus

The mood-based system uses available audio features such as:

* Valence
* Energy
* Danceability
* Acousticness
* Instrumentalness
* Tempo
* Speechiness
* Liveness

> **Note:** Mood classification is approximate and is based on available audio features. It is not a scientifically validated mood prediction system.

### 🌍 Language-Based Music Selection

Users can filter songs according to language.

Supported language options include:

* All Languages
* Hindi
* Punjabi
* English
* Tamil
* Telugu
* Bengali
* Marathi
* Malayalam
* Kannada
* Gujarati
* Other

Language detection uses an existing language column when available. If a language column is not available, the application may use approximate metadata-based detection from fields such as genre, artist name, album name, or track name.

> **Note:** Language detection is approximate and depends on the metadata available in the dataset.

### 🎶 Smart Playlist Generator

Users can generate personalized playlists by selecting:

* Playlist name
* Mood
* Language
* Genre
* Minimum popularity
* Number of songs
* Available audio-feature preferences

Example:

```text
Playlist Name: Hindi Happy Songs
Mood: Happy
Language: Hindi
Genre: All Genres
Minimum Popularity: 40
Number of Songs: 20
```

The application applies the selected filters and generates a unique playlist from the available dataset.

### 📁 Playlist Downloads

Generated playlists can be downloaded in supported formats such as:

* CSV
* TXT
* PDF reports, where implemented

The downloaded files may contain:

* Track name
* Artist name
* Album name
* Genre
* Language
* Popularity
* Duration
* Audio features

### 🎧 Listen on Spotify

The application provides **Listen on Spotify** links for songs in recommendations and generated playlists.

The links open a Spotify search page using the track name and artist name.

> The application redirects users to Spotify search. It does not download songs or automatically start playback.

### 🕘 Recently Generated Playlists

Recently generated playlists can be stored using Streamlit session state.

Users can:

* View playlists generated during the current session.
* Select a previous playlist.
* Review its songs.
* Access Spotify search links.
* Clear playlist history where supported.

> Playlist history is session-based and may reset when the Streamlit session is restarted.

---

## 🛠️ Technologies Used

| Technology   | Purpose                             |
| ------------ | ----------------------------------- |
| Python       | Main programming language           |
| Streamlit    | Interactive web application         |
| Pandas       | Data cleaning and analysis          |
| NumPy        | Numerical operations                |
| Matplotlib   | Data visualization                  |
| Scikit-learn | Recommendation and similarity logic |
| ReportLab    | PDF report generation, where used   |
| Git          | Version control                     |
| GitHub       | Repository hosting                  |

---

## 📂 Project Structure

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
├── first_analysis.py
├── README.md
└── requirements.txt
```

### Folder Description

| File/Folder              | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `app.py`                 | Main Streamlit application                               |
| `first_analysis.py`      | Initial data analysis script                             |
| `data/`                  | Original, cleaned, and unique Spotify datasets           |
| `src/`                   | Python scripts for analysis, cleaning, and visualization |
| `.streamlit/config.toml` | Streamlit configuration                                  |
| `requirements.txt`       | Required Python dependencies                             |
| `.gitignore`             | Files and folders excluded from Git                      |

---

## 📊 Dataset

The project uses Spotify song data containing information about tracks, artists, genres, popularity, duration, and audio features.

Commonly used columns may include:

```text
track_name
artists
album_name
track_genre
playlist_genre
popularity
duration_ms
danceability
energy
valence
acousticness
instrumentalness
tempo
speechiness
liveness
loudness
```

The project contains multiple datasets:

| Dataset                      | Purpose                          |
| ---------------------------- | -------------------------------- |
| `spotify_tracks.csv`         | Original Spotify dataset         |
| `spotify_tracks_cleaned.csv` | Cleaned Spotify dataset          |
| `spotify_unique_tracks.csv`  | Dataset containing unique tracks |

The application checks for available columns before using them to reduce errors caused by missing features.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/aarzoo0708/Spotify-Music-Analysis.git
```

### 2. Open the project folder

```bash
cd Spotify-Music-Analysis
```

### 3. Create a virtual environment

For Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example Usage

### Explore music data

1. Open the dashboard.
2. View dataset statistics.
3. Explore genres and popularity.
4. Analyze audio features.

### Explore an artist

1. Open Artist Analysis.
2. Select an artist.
3. View songs and artist statistics.
4. Generate a report if available.

### Explore songs by mood and language

1. Open Mood-Based Music Explorer.
2. Select a mood.
3. Select a language.
4. Select the number of songs.
5. Click **Explore Mood**.

Example:

```text
Mood: Sad
Language: Punjabi
Number of Songs: 20
```

### Generate a smart playlist

1. Open Smart Playlist Generator.
2. Enter a playlist name.
3. Select mood, language, and genre.
4. Set the minimum popularity.
5. Select the number of songs.
6. Click **Generate Smart Playlist**.
7. Download the playlist or open songs on Spotify.

---

## 📸 Screenshots

### Dashboard

![alt text](<Screenshot (340).png>)

### Genre Analysis

![alt text](<Screenshot (341).png>)

### Artist Comparison

![alt text](<Screenshot (343).png>)

### Mood-Based Music Explorer

![alt text](<Screenshot (345).png>)

### Smart Playlist Generator

![alt text](<Screenshot (346).png>)

### Generated Playlist

![alt text](<Screenshot (347).png>)

---

## 🧠 Learning Outcomes

This project helped practice:

* Python programming.
* Pandas DataFrame operations.
* NumPy numerical calculations.
* Data cleaning and preprocessing.
* Exploratory Data Analysis.
* Data visualization.
* Feature-based recommendation logic.
* Streamlit application development.
* Session state management.
* CSV and PDF report generation.
* Git and GitHub.
* Handling missing dataset columns.
* Building interactive data applications.

---

## 🔮 Future Enhancements

Future improvements may include:

* Spotify API integration.
* Spotify account login.
* Direct playlist creation in a Spotify account.
* Real-time Spotify track information.
* Personalized recommendations using listening history.
* Improved language classification using machine learning.
* Advanced mood classification.
* Audio preview playback.
* Permanent playlist history.
* User accounts.
* Cloud database integration.
* Improved deployment and performance.

---

## ⚠️ Limitations

* Mood classification is approximate.
* Language detection depends on available metadata.
* Some audio features may not be available in every dataset.
* Spotify links generally open Spotify search.
* Playlist history is stored only during the active session.
* Recommendation quality depends on dataset quality.
* The application does not download or stream music directly.

---

## 🔐 Privacy and API Information

This project does not require Spotify login or Spotify API credentials for Spotify search links.

The application does not collect personal listening history.

If Spotify API integration is added in the future, API keys and secrets should be stored securely using environment variables or Streamlit secrets.

Never commit API keys or passwords to GitHub.

---

## 👩‍💻 Author

**Aarzoo Bajaj**

GitHub:
https://github.com/aarzoo0708

Project Repository:
https://github.com/aarzoo0708/Spotify-Music-Analysis

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is created for educational and portfolio purposes.

A suitable open-source license, such as the MIT License, can be added if you want others to use, modify, and distribute the project.
