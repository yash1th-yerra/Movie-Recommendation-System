import gdown

from data_processing.load_data import load_data, merge_datasets
from data_processing.clean_data import clean_data
from data_processing.process_data import (
    select_features, convert_list, convert_top3, fetch_director,
    clean_overview, clean_spaces, create_tags_column, apply_stemming, vectorize_tags
)
from data_processing.recommend import compute_similarity, recommend

# Function to download files from Google Drive
def download_from_drive(file_id, output_file):
    """
    Downloads a file from Google Drive using its file ID.

    Parameters:
        file_id (str): Google Drive file ID.
        output_file (str): Local path to save the downloaded file.

    Returns:
        str: Path to the downloaded file.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)
    return output_file

# Main function to execute the process
def main(credits_file_id, movies_file_id, movie_title):
    # Download datasets from Google Drive
    credits_file = download_from_drive(credits_file_id, "tmdb_5000_credits.csv")
    movies_file = download_from_drive(movies_file_id, "tmdb_5000_movies.csv")

    # Load and preprocess data
    credits, movies = load_data(credits_file, movies_file)
    movies = merge_datasets(credits, movies)
    movies = select_features(movies)
    movies = clean_data(movies)

    # Preprocess columns
    movies['genres'] = movies['genres'].apply(convert_list)
    movies['keywords'] = movies['keywords'].apply(convert_list)
    movies['cast'] = movies['cast'].apply(convert_top3)
    movies['production_companies'] = movies['production_companies'].apply(convert_top3)
    movies['crew'] = movies['crew'].apply(fetch_director)

    # Tokenize and clean the data
    movies = clean_overview(movies)
    movies = clean_spaces(movies)

    # Create tags and finalize the DataFrame
    new_movies_df = create_tags_column(movies)
    new_movies_df = apply_stemming(new_movies_df)

    # Vectorize and compute similarity
    vectors, _ = vectorize_tags(new_movies_df)
    similarity = compute_similarity(vectors)

    # Recommend movies
    recommend(movie_title, new_movies_df, similarity)


# Example usage
if __name__ == '__main__':
    # Replace these with your actual Google Drive file IDs
    CREDITS_FILE_ID = "18LBoZSogPTkCW3xRbzQYhd_3sS90VD9y"
    MOVIES_FILE_ID = "1HLf3j2U-CBAXpV9Qnxn096sgF2Db1UWy"

    main(CREDITS_FILE_ID, MOVIES_FILE_ID, "Avatar")
