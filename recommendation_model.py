import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("telugu_movies_clean.csv")

# Handle missing values safely
df["Genre"] = df["Genre"].fillna("").astype(str)
df["Overview"] = df["Overview"].fillna("").astype(str)

# Combine Genre and Overview
df["content"] = df["Genre"] + " " + df["Overview"]

# Convert text into numerical features using TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])

# Calculate similarity between all movies
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(movie_name, number_of_movies=5):

    # Find the movie
    movie_indices = df[
        df["Movie"].str.lower() == movie_name.lower()
    ].index

    # If movie is not found
    if len(movie_indices) == 0:
        return ["Movie not found"]

    # Get movie index
    movie_index = movie_indices[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity[movie_index])
    )

    # Sort movies by similarity score
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Get recommended movies
    recommended_movies = []

    for index, score in similarity_scores[1:number_of_movies + 1]:
        recommended_movies.append(df.iloc[index]["Movie"])

    return recommended_movies


# Test the recommendation system
movie = "Eega"

print("Movie:", movie)
print("\nRecommended Movies:")

recommendations = recommend(movie)

for movie in recommendations:
    print("-", movie)