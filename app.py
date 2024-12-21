import gdown
import pandas as pd
import streamlit as st
import pickle as pkl
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Function to download file using requests from Google Drive
def download_from_drive(file_id, destination):
    """
    Downloads a file from Google Drive using its file ID and saves it locally.

    Parameters:
        file_id (str): The Google Drive file ID.
        destination (str): The local file path where the file will be saved.
    """
    # Construct the download URL
    url = f"https://drive.google.com/uc?id={file_id}&export=download"

    # Start a session
    session = requests.Session()

    # Send a request to initiate the download
    response = session.get(url, stream=True)

    # Handle potential redirection for large files (Google Drive confirmation)
    if 'confirmation' in response.text:
        confirm_url = f"https://drive.google.com/uc?export=download&confirm={response.cookies['confirm']}&id={file_id}"
        response = session.get(confirm_url, stream=True)

    # Write the content to the local file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)

    print(f"File downloaded successfully: {destination}")


# Example usage of the function to download both similarity and movie dictionary files
def download_files():
    # Set your Google Drive file IDs here
    similarity_file_id = os.getenv("SIMILARITY_FILE_ID")  # Example: '1Io3LPN4d82bL5VzBOYZUTA_UFzeO3xUV'
    movies_dict_file_id = os.getenv("MOVIES_DICT_FILE_ID")  # Example: '12Q5PDgfxt3uwetE2JZR6Or9iUfprCxZN'

    # Download the files
    download_from_drive(similarity_file_id, 'sparse_cosine_sim.pkl')
    download_from_drive(movies_dict_file_id, 'movie_dict.pkl')


# 1. Fetch movie poster using The Movie Database (TMDb) API
def fetch_poster(movie_id):
    """
    Fetches the poster URL for a given movie ID using TMDb API.

    Parameters:
        movie_id (int): The movie ID for which to fetch the poster.

    Returns:
        str: URL of the movie poster.
    """
    url = "https://api.themoviedb.org/3/movie/{}?language=en-US".format(movie_id)
    api_key = os.getenv('API_KEY')
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer {}".format(api_key)
    }

    response = requests.get(url, headers=headers)
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', '')


# 2. Recommend movies based on cosine similarity
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


# 3. Load model and data
def load_model_data():
    """
    Loads the similarity matrix and movies dictionary from pickle files.

    Returns:
        DataFrame: DataFrame containing movie information.
        ndarray: Cosine similarity matrix.
    """
    # Call the download_files function
    download_files()


    similarity = pkl.load(open('sparse_cosine_sim.pkl', 'rb'))
    movies_dict = pkl.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    return movies, similarity


# 4. Streamlit app
def build_streamlit_app():
    """
    Builds the movie recommendation system using Streamlit.
    """
    st.title("Movie Recommendation System")

    # Load the movie data and similarity matrix
    movies, similarity = load_model_data()

    # Movie selection dropdown
    selected_movie = st.selectbox("Search Movie", movies['title'].values)

    # Recommendation button and output
    if st.button("Recommend"):
        titles, posters = recommend(selected_movie, movies, similarity)

        # Display recommendations in a dynamic grid
        num_columns = len(titles)
        cols = st.columns(num_columns)

        # Display each recommended movie and its poster
        for i, (title, poster) in enumerate(zip(titles, posters)):
            with cols[i]:
                st.write(title)
                st.image(poster)


# 5. Main function to run the Streamlit app
if __name__ == '__main__':
    build_streamlit_app()
