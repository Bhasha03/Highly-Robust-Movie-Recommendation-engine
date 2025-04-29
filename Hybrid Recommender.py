#!/usr/bin/env python
# coding: utf-8

"""
Hybrid Recommender System

This script implements a hybrid movie recommender system that combines:
1. Content-based filtering (using cosine similarity)
2. Collaborative filtering (using SVD)

The hybrid approach leverages the strengths of both methods to provide
more accurate and diverse movie recommendations.
"""

import os
import numpy as np
import pandas as pd
from surprise import SVD, Reader, Dataset

# No special imports needed for CSV

# Define data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')

# Find the similarity file in CSV format
def find_similarity_file():
    # Check for CSV file in main data directory
    csv_path = os.path.join(DATA_DIR, 'Data', 'metadata_cosine_sim_sparse.csv')
    if os.path.exists(csv_path):
        return csv_path
        
    # Check for CSV file in hybrid data directory
    hybrid_csv_path = os.path.join(os.path.dirname(DATA_DIR), 'data', 'cosine_sim.csv')
    if os.path.exists(hybrid_csv_path):
        return hybrid_csv_path
    
    # If no file found, use the default path (may not exist)
    return os.path.join(os.path.dirname(DATA_DIR), 'data', 'cosine_sim.csv')

# Find the similarity file
SIMILARITY_FILE = find_similarity_file()

# Set the directory containing the similarity file
DATA_DIR = os.path.dirname(SIMILARITY_FILE)


def load_data():
    """Load all necessary data files for the recommender system.
    
    Returns:
        tuple: Contains all loaded data components:
            - cosine_sim: Content-based similarity matrix
            - cosine_sim_map: Mapping from titles to matrix indices
            - svd_model: Trained SVD model for collaborative filtering
            - id_to_title: Mapping from movie IDs to titles
            - title_to_id: Mapping from movie titles to IDs
            - movie_metadata: DataFrame with movie metadata
    """
    # Load content-based similarity matrix from CSV file
    print(f"Loading similarity data from {SIMILARITY_FILE}...")
    # Load the CSV file with similarity data in the format: movie_idx, similar_idx, similarity
    try:
        similarity_df = pd.read_csv(SIMILARITY_FILE)
        
        # Convert to a dictionary for faster lookups
        cosine_sim = {}
        for _, row in similarity_df.iterrows():
            movie_idx = str(int(row['movie_idx']))
            similar_idx = str(int(row['similar_idx']))
            similarity = row['similarity']
            
            if movie_idx not in cosine_sim:
                cosine_sim[movie_idx] = {}
                
            cosine_sim[movie_idx][similar_idx] = similarity
    except FileNotFoundError:
        print(f"Warning: {SIMILARITY_FILE} not found. Using empty similarity matrix.")
        cosine_sim = {}
    
    # Find and load the mapping file
    map_file = os.path.join(DATA_DIR, 'cosine_sim_map.csv')
    try:
        print(f"Loading title mapping from {map_file}...")
        cosine_sim_map = pd.read_csv(map_file, header=None)
        cosine_sim_map = cosine_sim_map.set_index(0)[1]  # Convert to Series for easier lookup
    except FileNotFoundError:
        print(f"Warning: {map_file} not found. Creating empty mapping.")
        cosine_sim_map = pd.Series(dtype='int64')
    
    # Load movie metadata
    print("Loading movies_metadata.csv...")
    movie_metadata = pd.read_csv(os.path.join(DATA_DIR, 'movies_metadata.csv'), low_memory=False)
    
    # Ensure id column is string type for consistent lookups
    movie_metadata['id'] = movie_metadata['id'].astype(str)
    
    # Create ID mappings
    print("Creating ID mappings...")
    # Map from ID to title
    id_to_title = movie_metadata[['id', 'title']].set_index('id')
    # Map from title to ID
    title_to_id = movie_metadata[['title', 'id']].set_index('title')
    
    # Build and train SVD model
    svd_model = train_svd_model()
    
    return cosine_sim, cosine_sim_map, svd_model, id_to_title, title_to_id, movie_metadata


def train_svd_model():
    """Train an SVD model for collaborative filtering.
    
    Returns:
        SVD: Trained SVD model
    """
    # Load ratings data
    print("Loading ratings data...")
    try:
        ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings_small.csv'), low_memory=False)
        
        # Prepare data for Surprise library
        print("Training SVD model...")
        reader = Reader()
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        
        # Train SVD model
        svd = SVD()
        trainset = data.build_full_trainset()
        svd.fit(trainset)  # Using fit instead of train (fit is the newer API)
        
        print("SVD model training complete.")
        return svd
    except FileNotFoundError:
        print("Warning: ratings_small.csv not found. Creating a dummy SVD model.")
        # Create a dummy SVD model that returns a constant rating
        class DummySVD:
            def predict(self, user_id, movie_id):
                class DummyPrediction:
                    def __init__(self):
                        self.est = 3.5  # Return a middle-of-the-road rating
                return DummyPrediction()
        
        return DummySVD()


def get_similar_movies(title, cosine_sim, cosine_sim_map, n=25):
    """Find movies similar to the given title using content-based filtering.
    
    Args:
        title (str): Title of the reference movie
        cosine_sim (dict): Dictionary of movie similarities
        cosine_sim_map (Series): Mapping from titles to matrix indices
        n (int): Number of similar movies to return
        
    Returns:
        tuple: (movie_indices, similarity_scores) - Indices and scores of similar movies
    """
    try:
        # Get the index of the movie in the similarity matrix
        if title not in cosine_sim_map:
            print(f"Movie '{title}' not found in the mapping")
            return [], []
            
        idx = cosine_sim_map[title]
        idx_str = str(int(idx))
        
        # Handle the dictionary format
        if idx_str in cosine_sim:
            # Get the similarities for this movie
            similarities = cosine_sim[idx_str]
            
            # Sort by similarity value (descending)
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Get the indices and scores (limit to n)
            similar_indices = [int(i[0]) for i in sorted_similarities[:n]]
            similarity_scores = [i[1] for i in sorted_similarities[:n]]
            
            return similar_indices, similarity_scores
        else:
            print(f"Movie index {idx} not found in similarity matrix")
            return [], []
    except Exception as e:
        print(f"Error finding similar movies for '{title}': {str(e)}")
        return [], []


def predict_ratings(user_id, movie_indices, movie_metadata, svd_model, id_to_title):
    """Predict user ratings for a list of movies using collaborative filtering.
    
    Args:
        user_id (int): User ID for whom to make predictions
        movie_indices (list): Indices of movies in the metadata DataFrame
        movie_metadata (DataFrame): Movie metadata
        svd_model (SVD): Trained SVD model
        id_to_title (DataFrame): Mapping from movie IDs to internal IDs
        
    Returns:
        DataFrame: Movies with predicted ratings
    """
    # Extract metadata for the selected movies
    try:
        # Get the movies at the specified indices
        movies = movie_metadata.iloc[movie_indices].copy()
        
        # Select only the columns we need
        columns_to_keep = ['title', 'id']
        for col in ['vote_count', 'vote_average', 'year']:
            if col in movies.columns:
                columns_to_keep.append(col)
                
        movies = movies[columns_to_keep]
        
        # Predict ratings using SVD model
        def predict_movie_rating(movie_id):
            try:
                # Use the movie ID directly for prediction
                return svd_model.predict(user_id, movie_id).est
            except Exception as e:
                print(f"Warning: Could not predict rating for movie ID {movie_id}: {str(e)}")
                return 3.0  # Default rating if prediction fails
        
        movies['est'] = movies['id'].apply(predict_movie_rating)
    except Exception as e:
        print(f"Error in predict_ratings: {str(e)}")
        # Return an empty DataFrame with the expected columns
        movies = pd.DataFrame(columns=['title', 'vote_count', 'vote_average', 'year', 'id', 'est'])
    
    # Sort by predicted rating
    return movies.sort_values('est', ascending=False)


def hybrid_recommend(user_id, title, n_recommendations=10):
    """Generate hybrid recommendations for a user based on a movie they liked.
    
    This function combines content-based and collaborative filtering approaches
    to provide personalized movie recommendations.
    
    Args:
        user_id (int): User ID for whom to make recommendations
        title (str): Title of a movie the user likes
        n_recommendations (int): Number of recommendations to return
        
    Returns:
        DataFrame: Top N recommended movies with metadata and predicted ratings
    """
    # Load all required data
    cosine_sim, cosine_sim_map, svd_model, id_to_title, title_to_id, movie_metadata = load_data()
    
    # Check if the title exists in our data
    if title not in cosine_sim_map.index:
        print(f"Movie '{title}' not found in the dataset. Please check the title and try again.")
        # Find similar titles to suggest
        try:    
            all_titles = movie_metadata['title'].tolist()
            import difflib
            close_matches = difflib.get_close_matches(title, all_titles, n=5, cutoff=0.6)
            if close_matches:
                print("Did you mean one of these movies?")
                for i, match in enumerate(close_matches, 1):
                    print(f"  {i}. {match}")
        except TypeError:
            print("Error finding similar movies.")
        return pd.DataFrame()
    
    # Get content-based similar movies
    print(f"Finding movies similar to '{title}' for user {user_id}...")
    movie_indices, similarity_scores = get_similar_movies(title, cosine_sim, cosine_sim_map)
    
    if not movie_indices:
        print(f"No similar movies found for '{title}'.")
        return pd.DataFrame()
        
    # Get predicted ratings for similar movies
    print("Predicting ratings for similar movies...")
    cf_recommendations = predict_ratings(user_id, movie_indices, movie_metadata, svd_model, id_to_title)
    
    if cf_recommendations.empty:
        print("Could not generate recommendations. Please try a different movie.")
        return pd.DataFrame()
    
    # Combine content and collaborative scores
    print("Combining content-based and collaborative filtering scores...")
    
    # Ensure the dataframe has the same length as similarity_scores
    if len(cf_recommendations) == len(similarity_scores):
        cf_recommendations['similarity'] = similarity_scores
    else:
        print("Warning: Length mismatch between recommendations and similarity scores.")
        # Use only the available similarity scores or pad with zeros
        cf_recommendations['similarity'] = similarity_scores[:len(cf_recommendations)] + [0] * (len(cf_recommendations) - len(similarity_scores))
    
    # Calculate hybrid score
    cf_recommendations['hybrid_score'] = (
        cf_recommendations['similarity'] + 
        cf_recommendations['est'] / 5.0  # Normalize ratings to 0-1 scale
    )
    
    # Sort by hybrid score
    recommendations = cf_recommendations.sort_values('hybrid_score', ascending=False)
    
    return recommendations.head(n_recommendations)


def main():
    """Demonstrate the hybrid recommender system with example users and movies."""
    print("Hybrid Recommender System Demo\n")
    
    # Example 1: Recommendations for user 1 based on 'Avatar'
    
    ip = input("Enter the movie title: ")
    print(f"Recommendations for User who liked '{ip}':\n")
    recommendations1 = hybrid_recommend(1, ip)
    print(recommendations1)
    
    print("\n" + "-"*50 + "\n")
    
    """
    # Example 2: Recommendations for user 2 based on 'Avatar'
    print("Recommendations for User 2 who liked 'Avatar':\n")
    recommendations2 = hybrid_recommend(2, 'Avatar')
    print(recommendations2)
    """


if __name__ == "__main__":
    main()
