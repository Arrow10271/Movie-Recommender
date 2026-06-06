import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=77298ab6826b273a5ba9c39713b00810'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

@st.cache_resource
def create_similarity_matrix(tags):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(tags).toarray()
    return cosine_similarity(vectors)

movies_similarity_score = create_similarity_matrix(movies['tags'])

def recommend(movie):
        movie_index = movies[movies['title'] == movie].index[0]
        distances = movies_similarity_score[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
        recommended = []
        recommended_posters = []
        for i in movie_list:
            movie_id = i[0]
            # Fetch movies poster with API
            recommended_posters.append(fetch_poster(movies.iloc[i[0]].movie_id))         
            recommended.append(movies.iloc[i[0]].title)
        return recommended, recommended_posters

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    "Select the type of movie you like and I will recommend you a movie",
    (movies['title'].values),
)

if st.button("Recommend"):
    names,posters = recommend(selected_movie_name)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])