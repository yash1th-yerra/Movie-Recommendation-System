from data_processing.load_data import load_data, merge_datasets
from data_processing.clean_data import clean_data
from data_processing.process_data import (
    select_features, convert_list, convert_top3, fetch_director,
    clean_overview, clean_spaces, create_tags_column, apply_stemming, vectorize_tags
)
from data_processing.recommend import compute_similarity, recommend


# Main function to execute the process
def main(credits_file, movies_file, movie_title):
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
    main('tmdb_5000_credits.csv', 'tmdb_5000_movies.csv', 'Avatar')
