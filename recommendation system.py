import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommend_movies(movie_title, movies_df, top_n=5):
    # Combine relevant features into a single text
    movies_df['combined_features'] = movies_df['genres'] + " " + movies_df['description']

    # Convert text data into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies_df['combined_features'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get index of the selected movie
    movie_index = movies_df[movies_df['title'] == movie_title].index[0]

    # Get similarity scores and sort them
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top N similar movies
    top_movies_indices = [i[0] for i in similarity_scores[1:top_n + 1]]

    return movies_df.iloc[top_movies_indices][['title', 'genres']]


# Sample movie data
data = {
    'title': ['Inception', 'Interstellar', 'The Matrix', 'The Dark Knight', 'Avatar'],
    'genres': ['Sci-Fi', 'Sci-Fi', 'Sci-Fi', 'Action', 'Sci-Fi'],
    'description': [
        'A thief enters dreams to steal secrets.',
        'Explorers travel through a wormhole in space.',
        'A computer hacker learns about reality.',
        'A vigilante fights crime in Gotham.',
        'A marine on an alien planet fights for survival.'
    ]
}

movies_df = pd.DataFrame(data)

# Example usage
recommended_movies = recommend_movies('Inception', movies_df)
print(recommended_movies)
