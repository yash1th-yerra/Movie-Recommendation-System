import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from app import fetch_poster


# Compute cosine similarity between vectors
def compute_similarity(vectors):
    cosine_sim_matrix = cosine_similarity(vectors)

    # Keep only top-K similarities for each item to create a sparse matrix
    top_k = 10
    for idx in range(cosine_sim_matrix.shape[0]):
        top_k_indices = np.argsort(cosine_sim_matrix[idx])[-top_k:]
        cosine_sim_matrix[idx, np.setdiff1d(np.arange(cosine_sim_matrix.shape[1]), top_k_indices)] = 0

    # Convert to sparse format
    sparse_cosine_sim = csr_matrix(cosine_sim_matrix)
    return sparse_cosine_sim


# Recommend function to get top 5 similar movies
def recommend(movie, movies, similarity):
    """
    Recommends movies based on the given movie using cosine similarity.

    Parameters:
        movie (str): The title of the movie to base recommendations on.
        movies (DataFrame): DataFrame containing movie information.
        similarity (ndarray): Cosine similarity matrix.

    Returns:
        list: List of recommended movie titles.
        list: List of corresponding movie poster URLs.
    """
    # Find the index of the selected movie
    movie_index = movies[movies['title'] == movie].index[0]

    # Get similarity distances for the selected movie
    distances = similarity[movie_index].toarray().ravel()

    # Sort movies by similarity and get the top 9 recommendations
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]

    # Store recommended movie titles and posters
    recommended_movies = []
    recommended_movies_poster = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_poster