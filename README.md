# 🎬 Telugu Movie Recommendation System

A **Machine Learning-based content recommendation system** that recommends movies similar to a selected Telugu movie.

## 📌 Project Overview
This project uses a Content-Based Recommendation System to find movies that are similar to a user's selected movie.
The system uses the movie's Genre and Overview to calculate similarity between movies.
## 📸 Application Screenshots

### 🏠 Movie Recommender Interface

![Movie Recommender Interface](Screenshot%202026-09-19%20210711.png)

### 🍿 Movie Recommendations

![Movie Recommendations](Screenshot%202026-09-19%20210838.png)

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

## 🤖 Machine Learning Techniques

### TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts movie text information into numerical features.

### Cosine Similarity

**Cosine Similarity** compares the numerical features of movies and identifies movies with similar content.

## 📊 Dataset

The project uses a Telugu Movies dataset.

**Current dataset coverage:**

* 🎬 Movies: **979**
* 📅 Years: **2000–2020**
* 📋 Features: Movie, Year, Certificate, Genre, Overview, Runtime, Rating and Number of Ratings

## ⚙️ How the System Works

1. Load the movie dataset.
2. Clean missing and duplicate data.
3. Combine **Genre** and **Overview** information.
4. Convert the text into numerical features using **TF-IDF**.
5. Calculate movie similarity using **Cosine Similarity**.
6. Select a movie through the Streamlit interface.
7. Display the most similar movies.

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

## 📚 Learning Outcomes

Through this project, I gained practical experience with:

* Content-Based Recommendation Systems
* TF-IDF Vectorization
* Cosine Similarity
* Data Preprocessing
* Machine Learning
* Streamlit Application Development

## 👨‍💻 Project Details

**Project:** Telugu Movie Recommendation System
**Domain:** Machine Learning
**Recommendation Type:** Content-Based Filtering
**Technique:** TF-IDF + Cosine Similarity
**Interface:** Streamlit
