#!/usr/bin/env python
# coding: utf-8

"""
Simple Movie Recommender

This script implements a simple movie recommender system based on IMDB's weighted rating formula.
It filters movies by runtime and popularity, then ranks them according to a weighted score
that balances average rating with number of votes.
"""

import pandas as pd
import numpy as np


def load_movie_data(file_path):
    """Load movie data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded movie data
    """
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path, low_memory=False)


def filter_movies(df, min_runtime=45, max_runtime=300, min_votes_percentile=0.8):
    """Filter movies by runtime and minimum number of votes.
    
    Args:
        df (pandas.DataFrame): Original movie dataframe
        min_runtime (int): Minimum runtime in minutes
        max_runtime (int): Maximum runtime in minutes
        min_votes_percentile (float): Percentile cutoff for minimum votes
        
    Returns:
        tuple: (filtered dataframe, vote count threshold, average rating)
    """
    # Calculate vote threshold (m) - minimum votes needed for consideration
    m = df['vote_count'].quantile(min_votes_percentile)
    print(f"Vote count threshold (m): {m:.2f}")
    
    # Calculate global mean rating (C)
    C = df['vote_average'].mean()
    print(f"Global mean rating (C): {C:.2f}")
    
    # Filter by runtime
    runtime_filtered = df[(df['runtime'] >= min_runtime) & (df['runtime'] <= max_runtime)]
    print(f"After runtime filtering: {runtime_filtered.shape[0]} movies")
    
    # Filter by vote count
    qualified = runtime_filtered[runtime_filtered['vote_count'] >= m]
    print(f"After vote count filtering: {qualified.shape[0]} movies")
    
    return qualified, m, C


def weighted_rating(x, m, C):
    """Calculate the IMDB weighted rating for a movie.
    
    Formula: (v/(v+m) * R) + (m/(m+v) * C)
    
    Args:
        x (pandas.Series): Movie data row
        m (float): Vote count threshold
        C (float): Global mean rating
        
    Returns:
        float: Weighted rating score
    """
    v = x['vote_count']  # Number of votes for the movie
    R = x['vote_average']  # Average rating of the movie
    
    # The formula balances the movie's own average (R) with the global average (C)
    # based on the number of votes (v) relative to the threshold (m)
    return (v/(v+m) * R) + (m/(m+v) * C)


def rank_movies(df, m, C, top_n=25):
    """Rank movies based on the weighted rating formula.
    
    Args:
        df (pandas.DataFrame): Filtered movie dataframe
        m (float): Vote count threshold
        C (float): Global mean rating
        top_n (int): Number of top movies to return
        
    Returns:
        pandas.DataFrame: Top ranked movies
    """
    # Create a copy to avoid modifying the original dataframe
    ranked_df = df.copy()
    
    # Apply the weighted rating formula to each movie
    ranked_df['score'] = ranked_df.apply(lambda x: weighted_rating(x, m, C), axis=1)
    
    # Sort by score in descending order
    ranked_df = ranked_df.sort_values('score', ascending=False)
    
    # Return only relevant columns for the top N movies
    return ranked_df[['title', 'vote_count', 'vote_average', 'score', 'runtime']].head(top_n)


def main():
    """Main function to execute the simple recommender workflow."""
    # Load movie data
    file_path = r'C:\Users\bhava\Desktop\Python\Hands-On-Recommendation-Systems-with-Python\Data\movies_metadata.csv'
    df = load_movie_data(file_path)
    
    # Filter movies by runtime and vote count
    qualified_movies, vote_threshold, avg_rating = filter_movies(df)
    
    # Rank movies using the weighted rating formula
    top_movies = rank_movies(qualified_movies, vote_threshold, avg_rating)
    
    # Display results
    print("\nTop 25 Movies by Weighted Rating:")
    print(top_movies)


if __name__ == "__main__":
    main()
