def clean_data(movies):
    """
    Drops missing and duplicated data.
    """
    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)
    return movies
