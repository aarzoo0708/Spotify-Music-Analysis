# Soundify Analytics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_UI-FF4B4B)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data_Visualization-yellow)

**Soundify Analytics** is a professional, interactive Streamlit dashboard designed to explore, analyze, and visualize music track data. Inspired by the sleek aesthetics of modern music streaming platforms, it helps users discover songs based on audio characteristics, compare artists, inspect metrics like energy and danceability, and download filtered analysis results—without needing to write a single line of code.

### **“Discover the data behind your favorite music.”**

## 🌟 Problem Statement & Overview
Data in raw CSV formats is difficult to interpret. **Soundify** solves this by turning raw music datasets into an interactive, visually stunning analytical product. The dashboard answers key questions like:
- Which artists dominate the dataset in terms of average track popularity?
- Is there a correlation between a track's energy and its acousticness?
- How do explicit tracks compare to non-explicit tracks in overall volume and popularity?

Whether you're a data enthusiast, music lover, or analyst, Soundify provides the tools to slice the data seamlessly.

## ✨ Features
- **Dynamic Filtering**: Robust sidebar controls to filter by song name, artist, popularity, energy, danceability, valence, tempo, and explicit content.
- **Discover Engine**: Automatically highlights tracks matching specific profiles (e.g., "Most Danceable", "Chill / Sad", "Fast Tempo").
- **KPI Overview & Distributions**: High-level metrics (Total Tracks, Unique Artists) and aesthetic histograms visualizing data distributions.
- **Top Tracks & Artist Analytics**: Dive deep into the most popular songs and examine top-performing artists with distinct metrics for their catalogs.
- **Advanced Audio Analysis**: Scatter plots and a professional Correlation Heatmap to uncover relationships between audio features.
- **Data Explorer**: Transparent raw data view, duplicate row tracking, missing values info, and intelligent text-search functionality.
- **Data Exports**: Download filtered datasets, top tracks, artist summaries, and correlation matrices to CSV with one click.

## 💻 Tech Stack
- **Python**: Core logic and data processing.
- **Pandas**: Advanced dataframe manipulation, aggregation, and filtering.
- **Matplotlib**: Generation of customized, dark-mode compatible visualizations (histograms, scatter plots, horizontal bars).
- **Streamlit**: Web application framework providing a responsive, state-managed interactive UI.

## 📊 Dataset Context
The app relies on `spotify_tracks_cleaned.csv` located in the `data/` directory. 
Important audio features include `popularity`, `danceability`, `energy`, `valence` (mood positivity), `tempo` (BPM), and explicit flags.
> **Note on Data Limitations**: Collaborations stored as single, unsplit strings (e.g., "Artist A, Artist B") are treated as a distinct entity from solo works.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Spotify-Music-Analytics.git
   cd Spotify-Music-Analytics
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is in place:**
   Place your `spotify_tracks_cleaned.csv` inside the `data/` folder.

4. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📸 Screenshots

*(Add screenshots of your application here after launching it locally!)*
- **Overview Tab:** `![Overview](screenshots/overview.png)`
- **Discover Tab:** `![Discover](screenshots/discover.png)`
- **Audio Analysis:** `![Correlation Heatmap](screenshots/heatmap.png)`

## 🔮 Future Improvements
- **Time Series Analysis**: Add release-year metrics to view how track characteristics evolve over decades.
- **Artist Splitting**: Implement logic to safely split multi-artist credits so individual artists get credit for their collaborations.
- **Machine Learning**: Introduce clustering algorithms (like K-Means) to suggest similar tracks automatically based on audio fingerprints.

---
*Built for music data exploration with ❤️ using Python, Pandas, Matplotlib, and Streamlit.*
