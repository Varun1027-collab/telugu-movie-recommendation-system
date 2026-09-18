# 🎬 Telugu Movie Recommendation System

A Machine Learning based movie recommendation system that recommends movies similar to a selected Telugu movie.

## 📌 Project Overview

This project uses a **Content-Based Recommendation System** to find movies that are similar to a user's selected movie.

The system uses the movie's **Genre** and **Overview** to calculate similarity between movies.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

## 🤖 Machine Learning Techniques

### TF-IDF

TF-IDF converts the movie's text information into numerical features.

### Cosine Similarity

Cosine Similarity compares the numerical features of movies and finds movies with similar content.

## 📊 Dataset

The project uses a Telugu Movies dataset.

Current dataset coverage:

* Movies: **979**
* Years: **2000–2020**
* Features: Movie, Year, Certificate, Genre, Overview, Runtime, Rating and Number of Ratings

## ⚙️ How the System Works

1. Load the movie dataset.
2. Clean missing and duplicate data.
3. Combine **Genre** and **Overview**.
4. Convert the text into numerical features using **TF-IDF**.
5. Calculate similarity using **Cosine Similarity**.
6. Select a movie through the Streamlit interface.
7. Display the most similar movies.

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

## 🎯 Features

* 🔎 Search for movies
* 🎬 Select a movie
* 🍿 Get similar movie recommendations
* ⭐ View movie ratings
* 📅 View release years
* 🎭 View genres
* 🔗 View similarity percentage
* 💻 Interactive Streamlit interface

## 📷 Example

For example, selecting **Arjun Reddy** produces recommendations such as:

* Naa Peru Surya Na Illu India
* Aadi
* Narasimha Naidu
* Mr. Medhavi
* Love Today

The similarity percentage represents the **text similarity between movies**, based on their Genre and Overview.

## 👨‍💻 Project

**Project:** Movie Recommendation System
**Domain:** Machine Learning
**Recommendation Type:** Content-Based Filtering
**Algorithm:** TF-IDF + Cosine Similarity
**Interface:** Streamlit
