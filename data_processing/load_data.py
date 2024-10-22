import pandas as pd

def load_data(credits_file, movies_file):
    """
    Loads the credits and movies datasets.
    """
    credits = pd.read_csv(credits_file)
    movies = pd.read_csv(movies_file)
    return credits, movies

def merge_datasets(credits, movies):
    """
    Merges the credits and movies datasets on 'title'.
    """
    return movies.merge(credits, on='title')
