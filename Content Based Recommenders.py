#!/usr/bin/env python
# coding: utf-8

"""
Content-Based Movie Recommender Systems

This script implements two types of content-based recommender systems:
1. Plot Description Based: Uses TF-IDF on movie overviews
2. Metadata Based: Uses movie metadata like cast, director, keywords, and genres

Both systems recommend movies similar to a given movie based on content features.
"""

import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import os


# File paths
DATA_DIR = r'C:\Users\bhava\Desktop\Python\Hands-On-Recommendation-Systems-with-Python\Data'
METADATA_CLEAN_PATH = os.path.join(DATA_DIR, 'metadata_clean.csv')
MOVIES_METADATA_PATH = os.path.join(DATA_DIR, 'movies_metadata.csv')
CREDITS_PATH = os.path.join(DATA_DIR, 'credits.csv')
KEYWORDS_PATH = os.path.join(DATA_DIR, 'keywords.csv')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
# Create output directory for processed data
OUTPUT_DIR = os.path.join(DATA_DIR, 'Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set a limit for the number of movies to process to avoid memory issues
MAX_MOVIES = 2000  # Adjust this number based on your system's memory capacity


def load_data():
    """Load and prepare movie data for content-based recommendation.
    
    Returns:
        tuple: (plot_df, metadata_df, indices1, indices2, cosine_sim1, cosine_sim2)
            - plot_df: DataFrame for plot-based recommendations
            - metadata_df: DataFrame for metadata-based recommendations
            - indices1: Series mapping movie titles to indices for plot-based system
            - indices2: Series mapping movie titles to indices for metadata-based system
            - cosine_sim1: Cosine similarity matrix for plot-based system
            - cosine_sim2: Cosine similarity matrix for metadata-based system
    """
    print("Loading data...")
    
    # Load the cleaned metadata file
    df = pd.read_csv(METADATA_CLEAN_PATH)
    
    # Limit the number of movies to prevent memory issues
    if len(df) > MAX_MOVIES:
        print(f"  Limiting dataset to {MAX_MOVIES} movies to prevent memory issues")
        df = df.sort_values('vote_count', ascending=False).head(MAX_MOVIES)
        print(f"  Selected top {MAX_MOVIES} movies by vote count")
    
    # Load the original file to get movie overviews
    orig_df = pd.read_csv(MOVIES_METADATA_PATH, low_memory=False)
    
    # Add overview and id to the cleaned dataframe
    df.loc[:, 'overview'] = orig_df['overview']
    df.loc[:, 'id'] = orig_df['id']
    
    # Create a copy for plot-based recommendations
    plot_df = df.copy()
    
    # Create plot-based similarity matrix
    print("Building plot-based recommendation system...")
    cosine_sim1, indices1 = build_plot_based_recommender(plot_df)
    
    # Create metadata-based similarity matrix
    print("Building metadata-based recommendation system...")
    metadata_df, cosine_sim2, indices2 = build_metadata_based_recommender(df)
    
    return plot_df, metadata_df, indices1, indices2, cosine_sim1, cosine_sim2


def build_plot_based_recommender(df):
    """Build a plot description based recommender system.
    
    Args:
        df (pandas.DataFrame): DataFrame with movie data including overviews
        
    Returns:
        tuple: (cosine_sim, indices)
            - cosine_sim: Cosine similarity matrix
            - indices: Series mapping movie titles to indices
    """
    # Define TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Replace NaN with empty string
    df.loc[:, 'overview'] = df['overview'].fillna('')
    
    # Construct TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(f"  Cosine similarity matrix shape: {cosine_sim.shape}")
    
    # Save top similarities instead of the full matrix to avoid memory issues
    print("  Saving top similarities instead of full matrix to avoid memory issues...")
    
    # Get the top 100 similar movies for each movie (adjust this number as needed)
    n_movies = cosine_sim.shape[0]
    top_n = min(100, n_movies-1)  # Limit to 100 or fewer if we have fewer movies
    
    # Process in batches to avoid memory issues
    batch_size = 1000  # Process this many movies at a time
    
    # Create a dictionary to store movie similarities
    similarity_dict = {}
    
    # Process movies in batches
    for batch_start in range(0, n_movies, batch_size):
        batch_end = min(batch_start + batch_size, n_movies)
        print(f"  Processing movies {batch_start} to {batch_end-1} of {n_movies}...")
        
        # Process each movie in the batch
        for i in range(batch_start, batch_end):
            # Create a list of (index, similarity) tuples for all other movies
            sims = [(j, float(cosine_sim[i, j])) for j in range(n_movies) if i != j]
            
            # Sort by similarity (descending) and take top N
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]
            
            # Store as a dictionary of index -> similarity
            similarity_dict[str(i)] = {str(j): score for j, score in sims}
    
    # Convert to DataFrame for easier saving
    print("  Converting to DataFrame...")
    sim_df = pd.DataFrame.from_dict(similarity_dict, orient='index')
    
    # Save to CSV format - only save the top similarities to keep file size manageable
    # Convert the sparse dictionary format to a more CSV-friendly format
    rows = []
    for movie_idx, similar_movies in similarity_dict.items():
        for similar_idx, score in similar_movies.items():
            rows.append({
                'movie_idx': int(movie_idx),
                'similar_idx': int(similar_idx),
                'similarity': score
            })
    
    # Create a DataFrame with only the necessary data
    csv_df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'plot_cosine_sim_sparse.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    
    # Save indices mapping
    pd.Series(df.index, index=df['title']).drop_duplicates().to_csv(os.path.join(OUTPUT_DIR, 'plot_indices.csv'), header=False)
    
    # Construct reverse mapping of indices and movie titles
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return cosine_sim, indices


def clean_ids(x):
    """Convert value to integer or return NaN if not possible.
    
    Args:
        x: Value to convert
        
    Returns:
        int or np.nan: Converted integer or NaN
    """
    try:
        return int(x)
    except:
        return np.nan


def get_director(crew):
    """Extract director name from crew list.
    
    Args:
        crew (list): List of crew members
        
    Returns:
        str or np.nan: Director name or NaN if not found
    """
    for crew_member in crew:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan


def generate_list(x, max_items=3):
    """Extract names from a list of dictionaries, limiting to max_items.
    
    Args:
        x (list): List of dictionaries with 'name' key
        max_items (int): Maximum number of items to include
        
    Returns:
        list: List of names, limited to max_items
    """
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Limit to max_items if list is longer
        if len(names) > max_items:
            names = names[:max_items]
        return names
    # Return empty list for missing/malformed data
    return []


def sanitize(x):
    """Sanitize data by removing spaces and converting to lowercase.
    
    Args:
        x (list or str): Data to sanitize
        
    Returns:
        list or str: Sanitized data
    """
    if isinstance(x, list):
        # Strip spaces and convert to lowercase for lists
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Handle string or other types
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def create_soup(x):
    """Create a metadata soup by combining keywords, cast, director, and genres.
    
    Args:
        x (pandas.Series): Row of the DataFrame with metadata columns
        
    Returns:
        str: Combined metadata string
    """
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def build_metadata_based_recommender(df):
    """Build a metadata-based recommender system.
    
    Args:
        df (pandas.DataFrame): DataFrame with movie data
        
    Returns:
        tuple: (df, cosine_sim, indices)
            - df: Processed DataFrame
            - cosine_sim: Cosine similarity matrix
            - indices: Series mapping movie titles to indices
    """
    # Load additional data
    cred_df = pd.read_csv(CREDITS_PATH)
    key_df = pd.read_csv(KEYWORDS_PATH)
    
    # Clean and convert IDs
    df.loc[:, 'id'] = df['id'].apply(clean_ids)
    df = df[df['id'].notnull()]  # Filter rows with null IDs
    df.loc[:, 'id'] = df['id'].astype('int')
    key_df.loc[:, 'id'] = key_df['id'].astype('int')
    cred_df.loc[:, 'id'] = cred_df['id'].astype('int')
    
    # Merge dataframes
    df = df.merge(cred_df, on='id')
    df = df.merge(key_df, on='id')
    
    # Convert stringified objects to Python objects
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df.loc[:, feature] = df[feature].apply(literal_eval)
    
    # Extract director from crew
    df.loc[:, 'director'] = df['crew'].apply(get_director)
    
    # Process cast and keywords
    df.loc[:, 'cast'] = df['cast'].apply(generate_list)
    df.loc[:, 'keywords'] = df['keywords'].apply(generate_list)
    
    # Limit genres to top 3
    df.loc[:, 'genres'] = df['genres'].apply(lambda x: x[:3])
    
    # Sanitize features
    for feature in ['cast', 'director', 'genres', 'keywords']:
        df.loc[:, feature] = df[feature].apply(sanitize)
    
    # Create metadata soup
    df.loc[:, 'soup'] = df.apply(create_soup, axis=1)
    
    # Create count vectors
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    print(f"  Count matrix shape: {count_matrix.shape}")
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    print(f"  Cosine similarity matrix shape: {cosine_sim.shape}")
    
    # Save top similarities instead of the full matrix to avoid memory issues
    print("  Saving top similarities instead of full matrix to avoid memory issues...")
    
    # Get the top 100 similar movies for each movie (adjust this number as needed)
    n_movies = cosine_sim.shape[0]
    top_n = min(100, n_movies-1)  # Limit to 100 or fewer if we have fewer movies
    
    # Process in batches to avoid memory issues
    batch_size = 1000  # Process this many movies at a time
    
    # Create a dictionary to store movie similarities
    similarity_dict = {}
    
    # Process movies in batches
    for batch_start in range(0, n_movies, batch_size):
        batch_end = min(batch_start + batch_size, n_movies)
        print(f"  Processing movies {batch_start} to {batch_end-1} of {n_movies}...")
        
        # Process each movie in the batch
        for i in range(batch_start, batch_end):
            # Create a list of (index, similarity) tuples for all other movies
            sims = [(j, float(cosine_sim[i, j])) for j in range(n_movies) if i != j]
            
            # Sort by similarity (descending) and take top N
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]
            
            # Store as a dictionary of index -> similarity
            similarity_dict[str(i)] = {str(j): score for j, score in sims}
    
    # Convert to DataFrame for easier saving
    print("  Converting to DataFrame...")
    sim_df = pd.DataFrame.from_dict(similarity_dict, orient='index')
    
    # Save to CSV format - only save the top similarities to keep file size manageable
    # Convert the sparse dictionary format to a more CSV-friendly format
    rows = []
    for movie_idx, similar_movies in similarity_dict.items():
        for similar_idx, score in similar_movies.items():
            rows.append({
                'movie_idx': int(movie_idx),
                'similar_idx': int(similar_idx),
                'similarity': score
            })
    
    # Create a DataFrame with only the necessary data
    csv_df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'metadata_cosine_sim_sparse.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    
    # Save indices mapping
    pd.Series(df.index, index=df['title']).to_csv(os.path.join(OUTPUT_DIR, 'metadata_indices.csv'), header=False)
    
    # Reset index and create mapping
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['title'])
    
    return df, cosine_sim, indices


def get_recommendations(title, cosine_sim, df, indices, num_recommendations=10):
    """Get movie recommendations based on similarity to a given movie.
    
    Args:
        title (str): Title of the movie to get recommendations for
        cosine_sim (numpy.ndarray): Cosine similarity matrix
        df (pandas.DataFrame): DataFrame with movie data
        indices (pandas.Series): Series mapping movie titles to indices
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        pandas.Series: Series of recommended movie titles
    """
    # Input validation
    if not isinstance(title, str) or not title.strip():
        print("Please provide a valid movie title.")
        return pd.Series([])
        
    if not isinstance(num_recommendations, int) or num_recommendations <= 0:
        print("Number of recommendations must be a positive integer.")
        return pd.Series([])
    
    try:
        # Check if title exists in indices
        if title not in indices:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.Series([])
            
        # Get the index of the movie that matches the title
        idx = indices[title]
        
        # Handle case where idx is not a single integer (e.g., Series due to duplicate titles)
        if not isinstance(idx, (int, np.integer)):
            if hasattr(idx, 'iloc') and len(idx) > 0:
                idx = idx.iloc[0]  # Take first occurrence if it's a Series
            else:
                print(f"Invalid index for movie '{title}'.")
                return pd.Series([])
        
        # Validate idx is within bounds of cosine_sim matrix
        if idx < 0 or idx >= len(cosine_sim):
            print(f"Index {idx} for movie '{title}' is out of bounds.")
            return pd.Series([])
        def suggest_movies(df, n_per_genre=2, genres=('Action', 'Comedy', 'Drama', 'Animation')):
            suggestions = []
            for genre in genres:
                genre_movies = df[df['genres'].apply(lambda g: genre.lower() in [x.lower() for x in g] if isinstance(g, list) else False)]
                suggestions.extend(genre_movies['title'].sample(min(n_per_genre, len(genre_movies))).tolist())
            print("\nSample movies you can try:")
            for movie in suggestions:
                print(f"  - {movie}")
        try:
            # Get pairwise similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # Sort movies by similarity score
            sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)
            
            # Get scores for most similar movies (excluding the movie itself)
            sim_scores = sim_scores[1:num_recommendations+1]
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Validate movie indices are within DataFrame bounds
            valid_indices = [i for i in movie_indices if i >= 0 and i < len(df)]
            if not valid_indices:
                print(f"No valid recommendations found for '{title}'.")
                return pd.Series([])
                
            # Return movie titles
            return df['title'].iloc[valid_indices]
        except Exception as e:
            print(f"Error processing recommendations for '{title}': {str(e)}")
            return pd.Series([])
            
    except KeyError:
        print(f"Movie '{title}' not found in the dataset.")
        return pd.Series([])
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return pd.Series([])


def main():
    """Main function to demonstrate the recommender systems."""
    # Load data and build recommenders
    plot_df, metadata_df, indices1, indices2, cosine_sim1, cosine_sim2 = load_data()
    
    # Save files specifically for Hybrid Recommender
    print("\nSaving files for Hybrid Recommender...")
    # Create hybrid recommender data directory if it doesn't exist
    hybrid_data_dir = os.path.join(os.path.dirname(DATA_DIR), 'data')
    os.makedirs(hybrid_data_dir, exist_ok=True)
    
    # Create a sparse representation for the hybrid recommender
    print("  Creating sparse representation for hybrid recommender...")
    
    # Get the top 50 similar movies for each movie (for hybrid recommender)
    n_movies = cosine_sim2.shape[0]
    top_n = min(50, n_movies-1)  # Use fewer movies for hybrid to save space
    
    # Process in batches to avoid memory issues
    batch_size = 1000  # Process this many movies at a time
    
    # Create a dictionary to store movie similarities
    similarity_dict = {}
    
    # Process movies in batches
    for batch_start in range(0, n_movies, batch_size):
        batch_end = min(batch_start + batch_size, n_movies)
        print(f"  Processing movies {batch_start} to {batch_end-1} of {n_movies}...")
        
        # Process each movie in the batch
        for i in range(batch_start, batch_end):
            # Create a list of (index, similarity) tuples for all other movies
            sims = [(j, float(cosine_sim2[i, j])) for j in range(n_movies) if i != j]
            
            # Sort by similarity (descending) and take top N
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]
            
            # Store as a dictionary of index -> similarity
            similarity_dict[str(i)] = {str(j): score for j, score in sims}
    
    # Convert to DataFrame for easier saving
    print("  Converting to DataFrame...")
    sim_df = pd.DataFrame.from_dict(similarity_dict, orient='index')
    
    # Save to CSV format - only save the top similarities to keep file size manageable
    # Convert the sparse dictionary format to a more CSV-friendly format
    rows = []
    for movie_idx, similar_movies in similarity_dict.items():
        for similar_idx, score in similar_movies.items():
            rows.append({
                'movie_idx': int(movie_idx),
                'similar_idx': int(similar_idx),
                'similarity': score
            })
    
    # Create a DataFrame with only the necessary data
    csv_df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(hybrid_data_dir, 'cosine_sim.csv')
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved cosine_sim.csv to {hybrid_data_dir}")
    
    # Save the metadata-based indices mapping
    indices2.to_csv(os.path.join(hybrid_data_dir, 'cosine_sim_map.csv'), header=False)
    print(f"  Saved cosine_sim_map.csv to {hybrid_data_dir}")
    
    # Example movie to get recommendations for
    example_movie = 'The Lion King'
    
    # Get recommendations using plot-based system
    print("\nPlot Description Based Recommendations:")
    plot_recommendations = get_recommendations(example_movie, cosine_sim1, plot_df, indices1)
    print(f"Movies similar to '{example_movie}' based on plot:")
    for i, movie in enumerate(plot_recommendations, 1):
        print(f"  {i}. {movie}")
    
    # Get recommendations using metadata-based system
    print("\nMetadata Based Recommendations:")
    metadata_recommendations = get_recommendations(example_movie, cosine_sim2, metadata_df, indices2)
    print(f"Movies similar to '{example_movie}' based on metadata (cast, director, keywords, genres):")
    for i, movie in enumerate(metadata_recommendations, 1):
        print(f"  {i}. {movie}")
    
    # Interactive mode
    print("\nWould you like to get recommendations for another movie? (y/n)")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            while True:
                print("\nEnter a movie title (or 'q' to quit):")
                movie_title = input().strip()
                if movie_title.lower() == 'q':
                    break
                    
                print("\nChoose recommendation type:\n1. Plot-based\n2. Metadata-based\n3. Both")
                rec_type = input().strip()
                
                if rec_type == '1' or rec_type == '3':
                    plot_recs = get_recommendations(movie_title, cosine_sim1, plot_df, indices1)
                    if not plot_recs.empty:
                        print(f"\nPlot-based recommendations for '{movie_title}':")
                        for i, movie in enumerate(plot_recs, 1):
                            print(f"  {i}. {movie}")
                
                if rec_type == '2' or rec_type == '3':
                    meta_recs = get_recommendations(movie_title, cosine_sim2, metadata_df, indices2)
                    if not meta_recs.empty:
                        print(f"\nMetadata-based recommendations for '{movie_title}':")
                        for i, movie in enumerate(meta_recs, 1):
                            print(f"  {i}. {movie}")
    except (EOFError, KeyboardInterrupt):
        print("\nInteractive mode exited.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
