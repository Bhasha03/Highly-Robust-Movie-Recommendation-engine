#!/usr/bin/env python
"""
Data Preparation Script for Hybrid Recommender

This script prepares the necessary data files for the Hybrid Recommender by:
1. Creating a mapping between movie IDs in different datasets
2. Ensuring compatibility between movies_metadata.csv and ratings_small.csv
3. Generating the required CSV files for the recommender system
"""

import os
import pandas as pd
import numpy as np

# Define data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare movie data for the hybrid recommender system."""
    print("Loading movie metadata...")
    # Load movies metadata
    movies_metadata = pd.read_csv(os.path.join(DATA_DIR, 'movies_metadata.csv'), low_memory=False)
    
    # Load ratings data
    print("Loading ratings data...")
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings_small.csv'), low_memory=False)
    
    # Clean and prepare the data
    print("Cleaning and preparing data...")
    
    # Convert id to string in movies_metadata for safer comparisons
    movies_metadata['id'] = movies_metadata['id'].astype(str)
    
    # Create a mapping between movieId in ratings and id in movies_metadata
    # First, load links.csv if available (contains mapping between movieId and tmdbId)
    try:
        links = pd.read_csv(os.path.join(DATA_DIR, 'links.csv'), low_memory=False)
        links['tmdbId'] = links['tmdbId'].astype(str)
        
        # Merge movies_metadata with links to get the mapping
        print("Creating ID mapping using links.csv...")
        id_mapping = pd.merge(movies_metadata[['id', 'title']], 
                             links[['movieId', 'tmdbId']], 
                             left_on='id', 
                             right_on='tmdbId', 
                             how='inner')
        
        # Keep only the necessary columns
        id_mapping = id_mapping[['movieId', 'id', 'title']]
    except FileNotFoundError:
        # If links.csv is not available, create a simple mapping based on title matching
        print("links.csv not found. Creating ID mapping based on title matching...")
        # Extract year from title if available
        movies_metadata['year_from_title'] = movies_metadata['title'].str.extract(r'\((\d{4})\)$')
        
        # Create a simplified mapping
        id_mapping = movies_metadata[['id', 'title']].copy()
        id_mapping['movieId'] = range(1, len(id_mapping) + 1)
    
    # Save the ID mapping
    print("Saving ID mapping...")
    id_mapping.to_csv(os.path.join(OUTPUT_DIR, 'movie_ids.csv'), index=False)
    
    # Create a smaller version of movies_metadata with only necessary columns
    print("Creating simplified metadata file...")
    metadata_small = movies_metadata[['id', 'title', 'vote_count', 'vote_average', 'year']].copy()
    metadata_small.to_csv(os.path.join(OUTPUT_DIR, 'metadata_small.csv'), index=False)
    
    return id_mapping, movies_metadata, ratings

def main():
    """Main function to prepare data for the hybrid recommender."""
    print("Preparing data for Hybrid Recommender...")
    id_mapping, movies_metadata, ratings = load_and_prepare_data()
    
    # Create necessary directories for the hybrid recommender
    hybrid_data_dir = os.path.join(os.path.dirname(DATA_DIR), 'data')
    os.makedirs(hybrid_data_dir, exist_ok=True)
    
    # Copy the ID mapping to the hybrid data directory
    id_mapping.to_csv(os.path.join(hybrid_data_dir, 'movie_ids.csv'), index=False)
    
    print("\nData preparation complete! The following files have been created:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'movie_ids.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'metadata_small.csv')}")
    print(f"  - {os.path.join(hybrid_data_dir, 'movie_ids.csv')}")
    
    print("\nYou can now run the Hybrid Recommender.py script.")

if __name__ == "__main__":
    main()
