import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telugu Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.movie-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

.movie-title {
    font-size: 23px;
    font-weight: bold;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.footer {
    text-align: center;
    margin-top: 50px;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("telugu_movies_clean.csv")

    df["Genre"] = df["Genre"].fillna("").astype(str)
    df["Overview"] = df["Overview"].fillna("").astype(str)
    df["Movie"] = df["Movie"].fillna("").astype(str)

    return df


df = load_data()


# =========================================================
# CREATE CONTENT
# =========================================================

df["content"] = (
    df["Genre"] + " " + df["Overview"]
)


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def create_model(content):

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(content)

    similarity = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    return similarity


similarity = create_model(
    df["content"]
)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(
    movie_name,
    number_of_movies=5
):

    movie_indices = df[
        df["Movie"].str.lower()
        == movie_name.lower()
    ].index

    if len(movie_indices) == 0:
        return []

    movie_index = movie_indices[0]

    similarity_scores = list(
        enumerate(
            similarity[movie_index]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[
        1:number_of_movies + 1
    ]:

        recommendations.append({
            "Movie": df.iloc[index]["Movie"],
            "Year": int(df.iloc[index]["Year"]),
            "Genre": df.iloc[index]["Genre"],
            "Rating": df.iloc[index]["Rating"],
            "Similarity": round(score * 100, 2)
        })

    return recommendations


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🎬 Telugu Movie Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover movies similar to your favorite movies using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DATASET STATISTICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🎬 Total Movies",
        len(df)
    )

with col2:

    st.metric(
        "📅 From Year",
        int(df["Year"].min())
    )

with col3:

    st.metric(
        "📅 To Year",
        int(df["Year"].max())
    )

with col4:

    st.metric(
        "🤖 ML Method",
        "TF-IDF"
    )


st.divider()


# =========================================================
# MOVIE SEARCH
# =========================================================

st.markdown(
    '<div class="section-title">🍿 Find Similar Movies</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 1])


movie_list = sorted(
    df["Movie"].unique().tolist()
)


# =========================================================
# SEARCH BOX
# =========================================================

with col1:

    search_movie = st.text_input(
        "🔎 Search for a movie",
        placeholder="Example: Arjun Reddy"
    )

    if search_movie:

        matching_movies = [
            movie
            for movie in movie_list
            if search_movie.lower() in movie.lower()
        ]

        if matching_movies:

            selected_movie = st.selectbox(
                "Select a movie",
                matching_movies
            )

        else:

            st.warning(
                "No movie found. Try another name."
            )

            selected_movie = movie_list[0]

    else:

        selected_movie = st.selectbox(
            "Select a movie",
            movie_list
        )


# =========================================================
# NUMBER OF RECOMMENDATIONS
# =========================================================

with col2:

    number_of_movies = st.selectbox(
        "Recommendations",
        [3, 5, 7, 10],
        index=1
    )


# =========================================================
# RECOMMEND BUTTON
# =========================================================

if st.button(
    "🍿 Recommend Movies",
    use_container_width=True
):

    recommendations = recommend_movies(
        selected_movie,
        number_of_movies
    )

    if recommendations:

        st.success(
            f"Found {len(recommendations)} movies similar to "
            f"{selected_movie}!"
        )

        st.markdown(
            f"### 🎯 Movies similar to **{selected_movie}**"
        )

        for i, movie in enumerate(
            recommendations,
            start=1
        ):

            st.markdown(
                f"""
                <div class="movie-card">

                <div class="movie-title">
                {i}. {movie["Movie"]}
                </div>

                <br>

                📅 <b>Year:</b> {movie["Year"]}
                &nbsp;&nbsp;&nbsp;

                🎭 <b>Genre:</b> {movie["Genre"]}
                &nbsp;&nbsp;&nbsp;

                ⭐ <b>Rating:</b> {movie["Rating"]}
                &nbsp;&nbsp;&nbsp;

                🔗 <b>Similarity:</b> {movie["Similarity"]}%

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.error(
            "Movie not found."
        )


# =========================================================
# HOW THE SYSTEM WORKS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 How Does It Work?</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        ### 1️⃣ Movie Information

        The system uses the movie's
        **Genre** and **Overview**.
        """
    )

with col2:

    st.markdown(
        """
        ### 2️⃣ TF-IDF

        TF-IDF converts movie text into
        numerical features.
        """
    )

with col3:

    st.markdown(
        """
        ### 3️⃣ Cosine Similarity

        Cosine Similarity compares the
        movie features and finds similar movies.
        """
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎬 Movie Recommender")

st.sidebar.info(
    """
    **Project:** Movie Recommendation System

    **Domain:** Machine Learning

    **Algorithm:** TF-IDF + Cosine Similarity

    **Recommendation:** Content-Based

    **Dataset:** Telugu Movies

    **Period:** 2000–2020
    """
)

st.sidebar.divider()

st.sidebar.subheader(
    "📊 Dataset Details"
)

st.sidebar.write(
    f"🎬 Movies: {len(df)}"
)

st.sidebar.write(
    f"🎭 Genres: {df['Genre'].nunique()}"
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🎬 Telugu Movie Recommendation System

    <br>

    Built using Python • Pandas • Scikit-learn • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)