# Movie Recommendation System 🎬

A movie recommendation system is a type of recommendation engine specifically designed to suggest films to users based on their past viewing habits, preferences, and other relevant data. Such systems are integral to streaming platforms, such as Netflix and Amazon Prime, where they aim to provide personalized content to increase user engagement and satisfaction.

This project is a content-based movie recommendation system built with the TMDb 5000 Movies and Credits datasets. It processes and combines movie metadata, including genres, keywords, cast, crew, production companies, and overview descriptions, to recommend movies based on similarity.
### Types of Recommendation Approaches

There are three primary approaches for a movie recommendation system:

1.  **Content-Based Filtering**:
    
    -   **How It Works**: Content-based filtering recommends movies similar to those a user has liked or interacted with in the past. This is achieved by analyzing movie metadata (such as genre, cast, director) and matching it with user preferences.
    -   **Advantages**: No dependence on other users’ data; good for niche items. Works well even with new or infrequent users if enough metadata is available.
    -   **Limitations**: Limited to known preferences and metadata, often leading to recommendations that lack variety or novelty.
    -   **Example Techniques**:
        -   **TF-IDF and Cosine Similarity**: Compute similarity between movies based on their metadata.
        -   **Word Embeddings**: Represent words or metadata as vectors in a continuous vector space, making it possible to compute more nuanced similarities.
2.  **Collaborative Filtering**:
    
    -   **How It Works**: Collaborative filtering bases its recommendations on user-item interactions across the entire user base. There are two main types:
        -   **User-Based Collaborative Filtering**: Recommends items based on users with similar viewing patterns.
        -   **Item-Based Collaborative Filtering**: Recommends items similar to those the user has liked or interacted with.
    -   **Advantages**: Can uncover hidden patterns and recommend diverse content by leveraging user similarity.
    -   **Limitations**: Requires a substantial amount of user interaction data; struggles with the “cold start” problem for new users or movies.
    -   **Example Techniques**:
        -   **Matrix Factorization (e.g., SVD)**: Decomposes the user-item interaction matrix into lower-dimensional matrices, capturing latent factors representing user preferences and movie attributes.
        -   **Alternating Least Squares (ALS)**: Optimizes user and item matrices to predict missing values in the user-item matrix.
3.  **Hybrid Approaches**:
    
    -   **How It Works**: Hybrid methods combine both content-based and collaborative filtering to achieve better accuracy and cover each technique's weaknesses.
    -   **Types of Hybrids**:
        -   **Weighted Hybrid**: A weighted average of content and collaborative scores for each recommendation.
        -   **Switching Hybrid**: Switches between methods based on criteria like user activity or availability of metadata.
        -   **Meta-Level Hybrid**: Uses the output of one recommender as input for another.
    -   **Advantages**: Greater personalization and diversity, ability to balance recommendation novelty with relevance.
    -   **Limitations**: More complex to implement and computationally intensive.
    -   **Example Techniques**:
        -   **Deep Learning Models**: Neural Collaborative Filtering, recurrent neural networks (RNNs) for sequential recommendations, or transformers to capture more sophisticated patterns in user-item interactions.
In this Project we have developed Content filtering based Movie Recommendation System
![](https://miro.medium.com/v2/resize:fit:281/1*qw5w1ClAW0DEdGzI5kUs2g.png)
Content-Based filtering doesn’t involve other users, but based on our preference, the algorithm will simply pick items with similar content to generate recommendations for us.

#### TF-IDF Vectorizer
**TF-IDF** (Term Frequency-Inverse Document Frequency) is a commonly used text vectorization technique in natural language processing (NLP) and information retrieval to represent documents and words in a way that highlights the most important words in each document within a corpus. It measures the relevance of a term in a document by balancing two factors:

1.  **Term Frequency (TF)**: How often a word appears in a document. A higher count increases the word's relevance within that document. For term ttt in document ddd, TF is calculated as:
    
  TF(t,d)=Total number of terms in d/Number of times t appears in d​
  
2.  **Inverse Document Frequency (IDF)**: How common or rare a word is across the entire corpus of documents. Words that appear in many documents have lower IDF scores, while unique words have higher scores, helping to identify terms that are particularly significant to certain documents. For term t in a corpus of N documents where df​ documents contain the term t:
    
   IDF(t)=log(N/df​+1​)
    
    Adding 1 to df​ avoids division by zero.
    

### TF-IDF Calculation

The final **TF-IDF score** for a term in a document is the product of the term’s TF and IDF scores:

TF-IDF(t,d)=TF(t,d)×IDF(t)
![Introduction to Natural Language Processing — TF-IDF | by Kinder Chen ...](https://miro.medium.com/v2/resize:fit:816/1*1pTLnoOPJKKcKIcRi3q0WA.jpeg)

**Cosine Similarity**

Cosine Similarity can be defined as a method to measure the difference between two non-zero vectors. In our case, the film title and the key movie features represent the coordinates of a movie vector. Thus, in order to calculate the similarity between the two movies, if we know the film title and key features of both the movies, we just need to calculate the difference between the two movie vectors.

The cosine similarity formula can be mathematically described as shown below.

![](https://miro.medium.com/v2/resize:fit:429/1*0n2aHtmTwDolc5Y7dSm-OA.png)

**Fig 3.2. Cosine Similarity formula**

A.B = Dot product between the two movies vectors,

||A||||B|| = Product of the magnitudes of the two movie vectors

![](https://miro.medium.com/v2/resize:fit:345/1*4ub83DgxaqIWtNtyWxGchQ.png)

**Fig 3.3.**  **Movie vectors representation**

## Project Structure

``
├── app.py                    	# Main Streamlit app file
├── data_processing/        	  # Directory for data processing scripts
│   ├── load_data.py       	   # Loads and merges datasets
	│   ├── clean_data.py         # Cleans and preprocesses data
│   ├── process_data.py      	 # Contains feature engineering functions
│   └── recommend.py          # Computes similarity and recommends movies
├── tmdb_5000_credits.csv     # TMDb credits dataset
├── tmdb_5000_movies.csv      # TMDb movies dataset
├── movie_dict.pkl            # Pickle file with movie data
├── sparse_cosine_sim.pkl     # Pickle file with sparse cosine similarity matrix
├── requirements.txt          # Lists all required libraries and dependencies
└── README.md                 # Project README file`

## Features

-   **Genres**: Movie genres such as Action, Drama, Comedy, etc.
-   **Keywords**: Keywords associated with the movie plot.
-   **Cast**: Top 3 cast members.
-   **Crew**: Director information.
-   **Overview**: Movie synopsis.
-   **Production Companies**: Top 3 production companies involved in the movie.
## Installation

1.  Clone this repository.
2.  Install the required packages:
    
    bash
    
    Copy code
    
    `pip install -r requirements.txt` 
    
3.  Add your TMDb API key to a `.env` file:
    
    plaintext
    
    Copy code
    
    `API_KEY=your_tmdb_api_key`
## Datasets

The datasets used in this project are:

-   `tmdb_5000_credits.csv`: Contains cast and crew information for each movie.
-   `tmdb_5000_movies.csv`: Contains movie metadata such as genres, keywords, and overview.


## Data Processing

The data processing pipeline is modularized into different scripts:

1.  **Load Data**: Reads and merges the `tmdb_5000_credits` and `tmdb_5000_movies` datasets.
2.  **Clean Data**: Removes missing values and duplicates.
3.  **Feature Engineering**:
    -   Extracts and processes fields like genres, keywords, cast, crew, and production companies.
    -   Combines these features into a single `tags` column for easier vectorization.
4.  **Text Processing**:
    -   Stemming: Reduces words to their root forms for consistency.
    -   Vectorization: Uses TF-IDF on the `tags` field to create a matrix of features.
5.  **Cosine Similarity**: Calculates cosine similarity between movies to find the most similar movies for recommendations.


### Key Data Processing Functions

-   `convert_list`: Extracts names from JSON-like fields for genres and keywords.
-   `convert_top3`: Extracts the top 3 entries for cast and production companies.
-   `fetch_director`: Extracts director information from the crew.
-   `create_tags_column`: Combines features into a single `tags` column for each movie.
-   `apply_stemming`: Applies stemming to the `tags` field.
-   `vectorize_tags`: Vectorizes the `tags` using TF-IDF.
##  Files

-   **movie_dict.pkl**: Dictionary of movies and their associated data.
-   **sparse_cosine_sim.pkl**: Sparse cosine similarity matrix for recommendations.
 
 
## Recommendation System

The recommendation system is built using cosine similarity. Here’s how it works:

1.  **Cosine Similarity Matrix**: Measures similarity between movies based on the `tags` column.
2.  **Sparse Matrix**: Only the top 10 similar movies are retained for efficiency.
3.  **Recommendation Function**: Returns the titles and posters of the top 9 most similar movies.
3.  **Saving Model**: The processed data (`movie_dict.pkl`) and similarity matrix (`sparse_cosine_sim.pkl`) are saved for future use.



## Streamlit Web App

The Streamlit app serves as the frontend for the recommendation system:

1.  **Movie Selection**: Users select a movie title from a dropdown menu.
2.  **Recommendations**: Upon clicking "Recommend," the app displays the top 9 recommended movies along with their posters.

### Streamlit App Components

-   `fetch_poster(movie_id)`: Fetches the movie poster from TMDb API using the movie ID.
-   `recommend(movie, movies, similarity)`: Generates recommendations based on cosine similarity.
-   `load_model_data()`: Loads the similarity matrix and movie data from pickle files.
-   `build_streamlit_app()`: Constructs the Streamlit interface.

### Running the App

To run the Streamlit app:

bash

Copy code

`streamlit run app.py`


## Main Scripts and Usage

The `main` function in `app.py` runs the data loading, preprocessing, and recommendation flow:

python

Copy code

`main('tmdb_5000_credits.csv', 'tmdb_5000_movies.csv', 'Avatar')` 

This script loads the data, processes it, calculates similarities, and recommends movies based on the provided movie title.
Here’s an updated `README.md` file that includes all the sections from your code:

----------

# Movie Recommendation System

This project is a content-based movie recommendation system that uses The Movie Database (TMDb) datasets to suggest movies based on metadata, such as genres, cast, crew, and keywords. The system includes both data preprocessing functions and a Streamlit web app for user interaction.

## Project Structure

The project is organized into different modules for data processing, model training, and serving the recommendation system via a Streamlit app.

## Datasets

The following TMDb datasets are used:

-   **`tmdb_5000_credits.csv`**: Contains cast and crew information for each movie.
-   **`tmdb_5000_movies.csv`**: Contains metadata such as genres, keywords, and overview descriptions.

## Features

-   **Genres**: Movie genres such as Action, Drama, Comedy, etc.
-   **Keywords**: Keywords associated with the movie plot.
-   **Cast**: Top 3 cast members.
-   **Crew**: Director information.
-   **Overview**: Movie synopsis.
-   **Production Companies**: Top 3 production companies involved in the movie.

## Installation

1.  Clone this repository.
2.  Install the required packages:
    
    bash
    
    Copy code
    
    `pip install -r requirements.txt` 
    
3.  Add your TMDb API key to a `.env` file:
    
    plaintext
    
    Copy code
    
    `API_KEY=your_tmdb_api_key` 
    

## Data Processing

The data processing pipeline is modularized into different scripts:

1.  **Load Data**: Reads and merges the `tmdb_5000_credits` and `tmdb_5000_movies` datasets.
2.  **Clean Data**: Removes missing values and duplicates.
3.  **Feature Engineering**:
    -   Extracts and processes fields like genres, keywords, cast, crew, and production companies.
    -   Combines these features into a single `tags` column for easier vectorization.
4.  **Text Processing**:
    -   Stemming: Reduces words to their root forms for consistency.
    -   Vectorization: Uses TF-IDF on the `tags` field to create a matrix of features.
5.  **Cosine Similarity**: Calculates cosine similarity between movies to find the most similar movies for recommendations.

### Key Data Processing Functions

-   `convert_list`: Extracts names from JSON-like fields for genres and keywords.
-   `convert_top3`: Extracts the top 3 entries for cast and production companies.
-   `fetch_director`: Extracts director information from the crew.
-   `create_tags_column`: Combines features into a single `tags` column for each movie.
-   `apply_stemming`: Applies stemming to the `tags` field.
-   `vectorize_tags`: Vectorizes the `tags` using TF-IDF.

## Recommendation System

The recommendation system is built using cosine similarity. Here’s how it works:

1.  **Cosine Similarity Matrix**: Measures similarity between movies based on the `tags` column.
2.  **Sparse Matrix**: Only the top 10 similar movies are retained for efficiency.
3.  **Recommendation Function**: Returns the titles and posters of the top 9 most similar movies.

## Streamlit Web App

The Streamlit app serves as the frontend for the recommendation system:

1.  **Movie Selection**: Users select a movie title from a dropdown menu.
2.  **Recommendations**: Upon clicking "Recommend," the app displays the top 9 recommended movies along with their posters.

### Streamlit App Components

-   `fetch_poster(movie_id)`: Fetches the movie poster from TMDb API using the movie ID.
-   `recommend(movie, movies, similarity)`: Generates recommendations based on cosine similarity.
-   `load_model_data()`: Loads the similarity matrix and movie data from pickle files.
-   `build_streamlit_app()`: Constructs the Streamlit interface.

### Running the App

To run the Streamlit app:

bash

Copy code

`streamlit run app.py` 

## Main Scripts and Usage

The `main` function in `app.py` runs the data loading, preprocessing, and recommendation flow:

python

Copy code

`main('tmdb_5000_credits.csv', 'tmdb_5000_movies.csv', 'Avatar')` 

This script loads the data, processes it, calculates similarities, and recommends movies based on the provided movie title.

## Files

-   **`sparse_cosine_sim.pkl`**: Pickle file containing the sparse cosine similarity matrix.
-   **`movie_dict.pkl`**: Pickle file containing the movie data dictionary.

## Example Usage

python

Copy code

`# Fetch recommendations for a movie
recommend('Avatar', new_movies_df, similarity)``


## Requirements

-   **Python** 3.7+
-   **Libraries**: `pandas`, `numpy`, `requests`, `streamlit`, `scikit-learn`, `python-dotenv `, `scipy``nltk`
-   **TMDb API Key**: Required to fetch movie posters.


## Future Enhancements

1.  **Collaborative Filtering**: Combine with collaborative filtering for better recommendations.
2.  **Improved NLP**: Implement lemmatization and additional preprocessing.
3.  **Enhanced UI**: Add more filtering options, such as genre-based recommendations.

-   **GitHub Repository**: [Movie Recommendation System](https://github.com/yash1th-yerra/Movie-Recommendation-System)
-   **Website**: [Live Demo](https://movie-recc-sys.streamlit.app/)
