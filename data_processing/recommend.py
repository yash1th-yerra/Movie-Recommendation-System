from sklearn.metrics.pairwise import cosine_similarity


# Compute cosine similarity between vectors
def compute_similarity(vectors):
    return cosine_similarity(vectors)


# Recommend function to get top 5 similar movies
def recommend(movie, new_movies_df, similarity):
    """
    Recommends top 5 movies based on cosine similarity.
    """
    movie_index = new_movies_df[new_movies_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print("Top 5 recommended movies:")
    for i in movies_list:
        print(new_movies_df.iloc[i[0]].title)
