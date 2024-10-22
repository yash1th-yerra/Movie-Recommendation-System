import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# Convert stringified lists to actual lists
def convert_list(object):
    l = []
    for i in ast.literal_eval(object):
        l.append(i['name'])
    return l


# Select important features
def select_features(movies):
    """
    Selects required features from the movies DataFrame.
    """
    features = ['movie_id', 'genres', 'keywords', 'overview', 'production_companies', 'title', 'cast', 'crew']
    return movies[features]


# Convert columns like cast, production_companies to top 3 items
def convert_top3(object):
    l = []
    counter = 0
    for i in ast.literal_eval(object):
        if counter < 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l


# Extract director from the crew
def fetch_director(object):
    for i in ast.literal_eval(object):
        if i['job'] == 'Director':
            return [i['name']]
    return []


# Tokenize the overview column and clean spaces
def clean_overview(movies):
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    return movies


def clean_spaces(movies):
    for column in ['genres', 'keywords', 'cast', 'crew', 'production_companies']:
        movies[column] = movies[column].apply(lambda x: [i.replace(" ", "") for i in x])
    return movies


# Create the 'tags' column by combining various features
def create_tags_column(movies):
    movies['tags'] = (movies['overview'] + movies['genres'] + movies['keywords'] +
                      movies['production_companies'] + movies['cast'] + movies['crew'])
    return movies[['movie_id', 'title', 'tags']]


# Stem the tags
def apply_stemming(movies_df):
    ps = PorterStemmer()

    def stem_text(text):
        return " ".join([ps.stem(word) for word in text.split()])

    movies_df['tags'] = movies_df['tags'].apply(lambda x: " ".join(x))  # Join the list into a string
    movies_df['tags'] = movies_df['tags'].apply(stem_text)  # Apply stemming
    return movies_df


# Vectorize the tags
def vectorize_tags(movies_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags']).toarray()
    return vectors, cv.get_feature_names_out()

