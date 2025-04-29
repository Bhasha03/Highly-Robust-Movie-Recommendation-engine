#!/usr/bin/env python
# coding: utf-8

"""
Knowledge-Based Movie Recommender

This script implements a knowledge-based movie recommender system that filters movies
based on user preferences for genre, runtime, and release year. It also cleans and
preprocesses movie metadata for use in recommendation systems.
"""

import pandas as pd
import numpy as np
from ast import literal_eval
import os


def load_movie_data(file_path):
    """Load movie data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded movie data
    """
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path, low_memory=False)


def select_features(df, features):
    """Select only the required features from the dataframe.
    
    Args:
        df (pandas.DataFrame): Original dataframe
        features (list): List of column names to keep
        
    Returns:
        pandas.DataFrame: Dataframe with only the selected features
    """
    return df[features]


def extract_year(df):
    """Convert release_date to datetime and extract year.
    
    Args:
        df (pandas.DataFrame): Dataframe with release_date column
        
    Returns:
        pandas.DataFrame: Modified dataframe with year column added
    """
    # Convert to datetime format
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # Extract year from datetime
    df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if pd.notna(x) else np.nan)
    
    # Convert year to integer, with 0 for invalid values
    df['year'] = df['year'].apply(convert_int)
    
    # Drop the original release_date column
    df = df.drop('release_date', axis=1)
    
    return df


def convert_int(x):
    """Convert a value to integer, returning 0 for invalid values.
    
    Args:
        x: Value to convert
        
    Returns:
        int: Converted integer value or 0 if conversion fails
    """
    try:
        return int(x)
    except:
        return 0


def process_genres(df):
    """Process the genres column from JSON strings to lists of genre names.
    
    Args:
        df (pandas.DataFrame): Dataframe with genres column
        
    Returns:
        pandas.DataFrame: Modified dataframe with processed genres
    """
    # Convert NaN values to empty list strings
    df['genres'] = df['genres'].fillna('[]')
    
    # Convert string representations to Python objects
    df['genres'] = df['genres'].apply(literal_eval)
    
    # Extract genre names from dictionaries and convert to lowercase
    df['genres'] = df['genres'].apply(lambda x: [i['name'].lower() for i in x] if isinstance(x, list) else [])
    
    return df


def explode_genres(df):
    """Create a new dataframe with one row per genre for each movie.
    
    Args:
        df (pandas.DataFrame): Dataframe with processed genres column
        
    Returns:
        pandas.DataFrame: New dataframe with exploded genres
    """
    # Stack the genres lists into a Series
    s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'  # Name the Series
    
    # Create a new dataframe by joining the Series
    gen_df = df.drop('genres', axis=1).join(s)
    
    return gen_df


def get_available_genres(gen_df):
    """Get a list of all available genres in the dataset.
    
    Args:
        gen_df (pandas.DataFrame): Dataframe with exploded genres
        
    Returns:
        list: Sorted list of unique genres
    """
    # Get unique genres
    unique_genres = gen_df['genre'].unique()
    
    # Filter out non-string values and convert to strings
    string_genres = []
    for genre in unique_genres:
        if isinstance(genre, str):
            string_genres.append(genre)
        elif not pd.isna(genre):
            # Convert non-NaN values to strings
            string_genres.append(str(genre))
    
    # Sort the list of string genres
    return sorted(string_genres)


def get_year_range(gen_df):
    """Get the range of years available in the dataset.
    
    Args:
        gen_df (pandas.DataFrame): Dataframe with year column
        
    Returns:
        tuple: (min_year, max_year)
    """
    # Ensure year is numeric before finding min/max
    numeric_years = pd.to_numeric(gen_df['year'], errors='coerce')
    min_year = numeric_years.min()
    max_year = numeric_years.max()
    
    # Handle NaN or invalid values
    if pd.isna(min_year) or min_year <= 0:
        min_year = 1900
    if pd.isna(max_year):
        max_year = 2023  # Current year as fallback
        
    return (int(min_year), int(max_year))


def get_runtime_range(gen_df):
    """Get the range of runtimes available in the dataset.
    
    Args:
        gen_df (pandas.DataFrame): Dataframe with runtime column
        
    Returns:
        tuple: (min_runtime, max_runtime)
    """
    # Ensure runtime is numeric and filter out extreme values (0 or very large)
    numeric_runtimes = pd.to_numeric(gen_df['runtime'], errors='coerce')
    valid_runtimes = numeric_runtimes[(numeric_runtimes > 0) & (numeric_runtimes < 1000)]
    
    if len(valid_runtimes) == 0:
        # Default values if no valid runtimes found
        return (60, 180)
    
    min_runtime = valid_runtimes.min()
    max_runtime = valid_runtimes.max()
    
    # Handle NaN values
    if pd.isna(min_runtime):
        min_runtime = 60  # Default minimum runtime
    if pd.isna(max_runtime):
        max_runtime = 180  # Default maximum runtime
        
    return (int(min_runtime), int(max_runtime))


def build_chart(gen_df, percentile=0.8, use_input=True, params=None):
    """Build a chart of recommended movies based on user preferences.
    
    Args:
        gen_df (pandas.DataFrame): Dataframe with exploded genres
        percentile (float): Percentile for vote count threshold
        use_input (bool): Whether to get parameters from user input
        params (dict): Dictionary of parameters if not using input
            {
                'genre': str,
                'low_time': int,
                'high_time': int,
                'low_year': int,
                'high_year': int
            }
        
    Returns:
        pandas.DataFrame: Filtered and ranked movies
    """
    # Get available options for better UX
    available_genres = get_available_genres(gen_df)
    min_year, max_year = get_year_range(gen_df)
    min_runtime, max_runtime = get_runtime_range(gen_df)
    
    if use_input:
        # Show available genres
        print("\nAvailable genres:")
        # Print genres in multiple columns for better readability
        genres_per_row = 5
        for i in range(0, len(available_genres), genres_per_row):
            print(', '.join(available_genres[i:i+genres_per_row]))
        
        # Get genre input with validation
        while True:
            print("\nInput preferred genre from the list above:")
            genre = input().lower().strip()
            if genre in available_genres:
                break
            print(f"Error: '{genre}' is not in the available genres list. Please try again.")
        
        # Get runtime input with validation and suggestions
        print(f"\nRuntime range in dataset: {min_runtime}-{max_runtime} minutes")
        print("Common ranges: 60-90 (short), 90-120 (medium), 120-180 (long)")
        
        while True:
            try:
                print("Input shortest duration (minutes):")
                low_time = int(input().strip())
                if low_time < 0:
                    print("Error: Duration cannot be negative. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter a valid number.")
        
        while True:
            try:
                print("Input longest duration (minutes):")
                high_time = int(input().strip())
                if high_time <= low_time:
                    print(f"Error: Longest duration must be greater than {low_time}. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter a valid number.")
        
        # Get year input with validation and suggestions
        print(f"\nYear range in dataset: {min_year}-{max_year}")
        print("Common periods: 1970-1990 (classics), 1990-2010 (modern), 2010-{} (contemporary)".format(max_year))
        
        while True:
            try:
                print("Input earliest year:")
                low_year = int(input().strip())
                if low_year < min_year or low_year > max_year:
                    print(f"Warning: Year {low_year} is outside the dataset range {min_year}-{max_year}.")
                    print("Do you want to continue anyway? (y/n)")
                    if input().lower().strip() != 'y':
                        continue
                break
            except ValueError:
                print("Error: Please enter a valid year.")
        
        while True:
            try:
                print("Input latest year:")
                high_year = int(input().strip())
                if high_year < low_year:
                    print(f"Error: Latest year must be greater than or equal to {low_year}. Please try again.")
                    continue
                if high_year > max_year:
                    print(f"Warning: Year {high_year} is beyond the dataset's maximum year {max_year}.")
                    print("Do you want to continue anyway? (y/n)")
                    if input().lower().strip() != 'y':
                        continue
                break
            except ValueError:
                print("Error: Please enter a valid year.")
    else:
        # Use provided parameters
        genre = params['genre']
        low_time = params['low_time']
        high_time = params['high_time']
        low_year = params['low_year']
        high_year = params['high_year']
        
        print(f"Filtering for {genre} movies from {low_year} to {high_year} with runtime {low_time}-{high_time} minutes")
    
    # Filter movies based on criteria
    movies = gen_df.copy()
    movies = movies[(movies['genre'] == genre) & 
                    (movies['runtime'] >= low_time) & 
                    (movies['runtime'] <= high_time) & 
                    (movies['year'] >= low_year) & 
                    (movies['year'] <= high_year)]
    
    print(f"Found {len(movies)} movies matching criteria")
    
    if len(movies) == 0:
        return pd.DataFrame()  # Return empty dataframe if no movies match
    
    # Calculate rating metrics
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)
    
    print(f"Vote count threshold: {m:.2f}")
    print(f"Mean vote average: {C:.2f}")
    
    # Filter by vote count
    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    print(f"Movies with sufficient votes: {len(q_movies)}")
    
    if len(q_movies) == 0:
        return movies.sort_values('vote_average', ascending=False)  # Return all matches sorted by rating
    
    # Calculate weighted score using IMDB formula
    q_movies['score'] = q_movies.apply(
        lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C),
        axis=1
    )
    
    # Sort by score
    q_movies = q_movies.sort_values('score', ascending=False)
    
    return q_movies


def save_processed_data(df, output_path):
    """Save processed dataframe to CSV.
    
    Args:
        df (pandas.DataFrame): Processed dataframe to save
        output_path (str): Path to save the CSV file
        
    Returns:
        bool: True if successful
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return True


def main():
    """Main function to execute the knowledge-based recommender workflow."""
    # Load movie data
    file_path = r'C:\Users\bhava\Desktop\Python\Hands-On-Recommendation-Systems-with-Python\Data\movies_metadata.csv'
    df = load_movie_data(file_path)
    
    # Select relevant features
    features = ['title', 'genres', 'release_date', 'runtime', 'vote_average', 'vote_count']
    df = select_features(df, features)
    print(f"Selected {len(features)} features: {features}")
    
    # Process release date and extract year
    df = extract_year(df)
    print("Extracted year from release date")
    
    # Process genres
    df = process_genres(df)
    print("Processed genres")
    
    # Create exploded genre dataframe
    gen_df = explode_genres(df)
    print(f"Created exploded genre dataframe with {len(gen_df)} rows")
    
    # Save processed data
    output_path = r'C:\Users\bhava\Desktop\Python\Hands-On-Recommendation-Systems-with-Python\Data\Output\metadata_clean.csv'
    save_processed_data(df, output_path)
    
    # Get dataset statistics for dynamic parameters
    available_genres = get_available_genres(gen_df)
    min_year, max_year = get_year_range(gen_df)
    min_runtime, max_runtime = get_runtime_range(gen_df)
    
    # Demo the recommender with dynamic parameters
    print("\nDemonstration of the recommender system:")
    
    # Choose a popular genre from the dataset
    genre_counts = gen_df['genre'].value_counts()
    popular_genres = genre_counts.head(5).index.tolist()
    demo_genre = 'animation' if 'animation' in available_genres else popular_genres[0]
    
    # Set reasonable runtime range based on dataset statistics
    # Use 25th percentile to 75th percentile for a reasonable range
    # Convert runtime to numeric first to avoid type issues
    numeric_runtime = pd.to_numeric(gen_df['runtime'], errors='coerce')
    valid_runtime = numeric_runtime[numeric_runtime > 0]
    
    if len(valid_runtime) > 0:
        runtime_25th = int(valid_runtime.quantile(0.25))
        runtime_75th = int(valid_runtime.quantile(0.75))
    else:
        # Default values if no valid data
        runtime_25th = 60
        runtime_75th = 180
    
    # Set year range to recent 30 years or available range if smaller
    recent_years = 30
    # Ensure we're working with integers
    min_year_int = int(min_year)
    max_year_int = int(max_year)
    year_low = max(min_year_int, max_year_int - recent_years)
    
    params = {
        'genre': demo_genre,
        'low_time': runtime_25th,
        'high_time': runtime_75th,
        'low_year': year_low,
        'high_year': max_year
    }
    
    print(f"Using dynamically determined parameters based on dataset statistics:")
    print(f"- Genre: {params['genre']} (one of the most common genres in the dataset)")
    print(f"- Runtime: {params['low_time']}-{params['high_time']} minutes (25th-75th percentile)")
    print(f"- Years: {params['low_year']}-{params['high_year']} (recent {recent_years} years)")
    print("\nYou can also run this script interactively to choose your own parameters.")    
    top_movies = build_chart(gen_df, use_input=False, params=params)
    
    if not top_movies.empty:
        print("\nTop 5 Recommended Movies:")
        print(top_movies[['title', 'year', 'runtime', 'vote_average', 'vote_count', 'score']].head(5))
    else:
        print("No movies found matching the criteria")
        
    # Ask if user wants to run interactively, but handle EOF errors
    print("\nWould you like to try with your own parameters? (y/n)")
    try:
        user_choice = input().strip().lower()
        if user_choice == 'y':
            interactive_movies = build_chart(gen_df, use_input=True)
            if not interactive_movies.empty:
                print("\nYour Personalized Recommendations:")
                print(interactive_movies[['title', 'year', 'runtime', 'vote_average', 'vote_count', 'score']].head(10))
            else:
                print("No movies found matching your criteria")
    except EOFError:
        print("\nRunning in non-interactive mode. To use interactive mode, run this script in a terminal.")
    except KeyboardInterrupt:
        print("\nInteractive mode cancelled by user.")


def debug_dataframe(df, name):
    """Print debug information about a dataframe.
    
    Args:
        df (pandas.DataFrame): Dataframe to debug
        name (str): Name of the dataframe for identification
    """
    print(f"\nDEBUG INFO for {name}:")
    print(f"Shape: {df.shape}")
    print("Data types:")
    print(df.dtypes)
    print("Sample data (first 3 rows):")
    print(df.head(3))
    print("\n")


if __name__ == "__main__":
    try:
        # Add a command-line argument option for interactive mode
        import sys
        
        # Default to non-interactive mode
        interactive_mode = False
        
        # Check for command line arguments
        if len(sys.argv) > 1 and sys.argv[1].lower() in ('-i', '--interactive'):
            interactive_mode = True
            print("Running in interactive mode. You will be prompted for parameters.")
        
        # Run the main function
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nThank you for using the Knowledge-Based Movie Recommender!")
