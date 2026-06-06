import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    color: #FF4B4B;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.movie-title {
    text-align: center;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 0.7rem;
    color: #FFFFFF;
}

.block-container {
    padding-top: 2rem;
}

.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #E63E3E;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Constants ----------------
TMDB_API_KEY = "77298ab6826b273a5ba9c39713b00810"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
FALLBACK_POSTER = "https://via.placeholder.com/500x750.png?text=No+Poster+Available"


# ---------------- Load Movie Data ----------------
@st.cache_data
def load_movies():
    with open("movie_dict.pkl", "rb") as file:
        movies_dict = pickle.load(file)

    return pd.DataFrame(movies_dict)


movies = load_movies()


# ---------------- Create Movie Vectors ----------------
@st.cache_resource
def create_vectors(tags):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(tags)
    return vectors


vectors = create_vectors(movies["tags"])


# ---------------- Fetch Poster from TMDB ----------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return POSTER_BASE_URL + poster_path

        return FALLBACK_POSTER

    except Exception:
        return FALLBACK_POSTER


# ---------------- Recommendation Function ----------------
def recommend(movie_title, number_of_recommendations=5):
    movie_matches = movies[movies["title"] == movie_title]

    if movie_matches.empty:
        return []

    movie_index = movie_matches.index[0]

    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    recommended_indices = similarity_scores.argsort()[::-1][1:number_of_recommendations + 1]

    recommendations = []

    for index in recommended_indices:
        movie_id = movies.iloc[index]["movie_id"]
        title = movies.iloc[index]["title"]
        poster = fetch_poster(movie_id)

        recommendations.append({
            "title": title,
            "poster": poster
        })

    return recommendations


# ---------------- App Header ----------------
st.markdown(
    '<div class="main-title">🎬 Movie Recommender System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover movies similar to the ones you already love.</div>',
    unsafe_allow_html=True
)


# ---------------- Movie Selection Section ----------------
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        selected_movie_name = st.selectbox(
            "Choose a movie",
            movies["title"].values,
            index=0
        )

        recommend_button = st.button(
            "Recommend Movies",
            use_container_width=True
        )


st.divider()


# ---------------- Display Recommendations ----------------
if recommend_button:
    with st.spinner("Finding the best recommendations for you..."):
        recommendations = recommend(selected_movie_name)

    if recommendations:
        st.subheader("Recommended Movies")

        columns = st.columns(5)

        for col, movie in zip(columns, recommendations):
            with col:
                st.image(movie["poster"], use_column_width=True)
                st.markdown(
                    f"""
                    <div class="movie-title">
                        {movie["title"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:
        st.warning("No recommendations found. Please try another movie.")
