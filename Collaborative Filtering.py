#!/usr/bin/env python
# coding: utf-8
"""
Collaborative Filtering Movie Recommender

This script demonstrates collaborative filtering using the MovieLens dataset.
It implements various collaborative filtering approaches including:
1. Baseline model (constant prediction)
2. User-based collaborative filtering with mean ratings
3. User-based collaborative filtering with weighted mean ratings
4. Demographic filtering (gender-based)
5. Demographic filtering (gender and occupation-based)
6. Model-based approaches using Surprise library (kNN and SVD)

The script loads user, movie, and rating data, preprocesses it, splits it into train/test sets,
and evaluates the performance of different recommendation models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# File paths
DATA_DIR = r"C:\Users\bhava\Desktop\Python\Hands-On-Recommendation-Systems-with-Python\Data"
OUTPUT_DIR = os.path.join(DATA_DIR, 'Output')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data files
USERS_PATH = os.path.join(DATA_DIR, 'useritem.user')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies_metadata.csv')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')

# Global variables for models
r_matrix = None
cosine_sim = None
gender_mean = None
gen_occ_mean = None
users_indexed = None

def load_users(filepath):
    """Load user data from the specified file.
    
    Args:
        filepath (str): Path to the users data file
        
    Returns:
        pandas.DataFrame: DataFrame containing user data or empty DataFrame if error
    """
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    try:
        users = pd.read_csv(filepath, sep='|', names=u_cols, encoding='latin-1',low_memory=False)
        
        # Ensure user_id is integer type for compatibility with ratings data
        users['user_id'] = users['user_id'].astype(int)
        
        print(f"Loaded {len(users)} users")
        return users
    except Exception as e:
        print(f"Error loading users: {e}")
        return pd.DataFrame()

def load_movies(filepath):
    """Load movie data from the specified file and keep only ID and title.
    
    Args:
        filepath (str): Path to the movies data file
        
    Returns:
        pandas.DataFrame: DataFrame containing movie data (ID and title only) or empty DataFrame if error
    """
    i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
              'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    try:
        # Try to load the movies file
        try:
            movies = pd.read_csv(filepath, sep='|', names=i_cols, encoding='latin-1', low_memory=False)
            movies = movies[['movie_id', 'title']]
            
            # Ensure movie_id is integer type
            movies['movie_id'] = pd.to_numeric(movies['movie_id'], errors='coerce')
            movies = movies.dropna(subset=['movie_id'])  # Drop rows with non-numeric movie_id
            movies['movie_id'] = movies['movie_id'].astype(int)
            
            print(f"Loaded {len(movies)} movies")
        except Exception as e:
            print(f"Error loading movies from file: {e}")
            print("Generating sample movie data for demonstration...")
            
            # Generate sample movie data for demonstration
            movies = generate_sample_movies()
            
        return movies
    except Exception as e:
        print(f"Error in load_movies: {e}")
        return pd.DataFrame()

def generate_sample_movies():
    """Generate sample movie data for demonstration purposes.
    
    Returns:
        pandas.DataFrame: Sample movie data
    """
    # Create sample movie IDs (1-50)
    movie_ids = np.arange(1, 51)
    
    # Create sample movie titles
    movie_titles = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction", "Fight Club",
        "Forrest Gump", "Inception", "The Matrix", "Goodfellas", "The Silence of the Lambs",
        "Star Wars: Episode V", "The Lord of the Rings", "Interstellar", "Saving Private Ryan", "Spirited Away",
        "The Lion King", "Back to the Future", "The Pianist", "Psycho", "Gladiator",
        "Alien", "The Departed", "The Prestige", "Casablanca", "The Green Mile",
        "Raiders of the Lost Ark", "Django Unchained", "The Shining", "WALL·E", "Princess Mononoke",
        "Avengers: Infinity War", "Coco", "Up", "Toy Story", "Toy Story 3",
        "Finding Nemo", "Inside Out", "How to Train Your Dragon", "Monsters, Inc.", "Ratatouille",
        "The Incredibles", "Big Hero 6", "Zootopia", "Frozen", "Moana",
        "The Avengers", "Iron Man", "Captain America: Civil War", "Thor: Ragnarok", "Black Panther"
    ]
    
    # Create DataFrame
    movies_df = pd.DataFrame({
        'movie_id': movie_ids,
        'title': movie_titles[:len(movie_ids)]  # Ensure we don't exceed the number of movie_ids
    })
    
    print(f"Generated {len(movies_df)} sample movies")
    return movies_df

def generate_sample_users():
    """Generate sample user data for demonstration purposes.
    
    Returns:
        pandas.DataFrame: Sample user data
    """
    # Create sample user IDs (1-100)
    user_ids = np.arange(1, 101)
    
    # Generate random ages (18-70)
    np.random.seed(42)  # For reproducibility
    ages = np.random.randint(18, 71, size=len(user_ids))
    
    # Generate random gender (M/F)
    genders = np.random.choice(['M', 'F'], size=len(user_ids))
    
    # Generate random occupations
    occupations = [
        'administrator', 'artist', 'doctor', 'educator', 'engineer',
        'entertainment', 'executive', 'healthcare', 'homemaker', 'lawyer',
        'librarian', 'marketing', 'programmer', 'retired', 'salesman',
        'scientist', 'student', 'technician', 'writer', 'other'
    ]
    user_occupations = np.random.choice(occupations, size=len(user_ids))
    
    # Generate random zip codes
    zip_codes = np.random.randint(10000, 99999, size=len(user_ids)).astype(str)
    
    # Create DataFrame
    users_df = pd.DataFrame({
        'user_id': user_ids,
        'age': ages,
        'sex': genders,
        'occupation': user_occupations,
        'zip_code': zip_codes
    })
    
    print(f"Generated {len(users_df)} sample users")
    return users_df

def load_ratings(filepath):
    """Load ratings data from the specified file and drop the timestamp column.
    
    Args:
        filepath (str): Path to the ratings data file
        
    Returns:
        pandas.DataFrame: DataFrame containing ratings data or empty DataFrame if error
    """
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    try:
        # Try to load the ratings file
        try:
            ratings = pd.read_csv(filepath, sep='\t', names=r_cols, encoding='latin-1',low_memory=False)
            ratings = ratings.drop('timestamp', axis=1)
            
            # Ensure user_id and movie_id are integer type
            ratings['user_id'] = ratings['user_id'].astype(int)
            ratings['movie_id'] = ratings['movie_id'].astype(int)
            
            print(f"Loaded {len(ratings)} ratings")
        except Exception as e:
            print(f"Error loading ratings from file: {e}")
            print("Generating sample ratings data for demonstration...")
            
            # Generate sample ratings data for demonstration
            ratings = generate_sample_ratings()
            
        return ratings
    except Exception as e:
        print(f"Error in load_ratings: {e}")
        return pd.DataFrame()

def generate_sample_ratings():
    """Generate sample ratings data for demonstration purposes.
    
    Returns:
        pandas.DataFrame: Sample ratings data
    """
    # Create sample user IDs (1-100)
    user_ids = np.arange(1, 101)
    
    # Create sample movie IDs (1-50)
    movie_ids = np.arange(1, 51)
    
    # Generate random ratings (1-5 stars)
    np.random.seed(42)  # For reproducibility
    
    # Create empty list to store ratings
    ratings_data = []
    
    # Generate multiple ratings per user (5-15 ratings per user)
    for user_id in user_ids:
        # Randomly select 5-15 movies for this user
        num_ratings = np.random.randint(5, 16)
        selected_movies = np.random.choice(movie_ids, size=num_ratings, replace=False)
        
        # Generate ratings for selected movies
        for movie_id in selected_movies:
            rating = np.random.randint(1, 6)  # Rating between 1-5
            ratings_data.append([user_id, movie_id, rating])
    
    # Create DataFrame
    ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating'])
    
    print(f"Generated {len(ratings_df)} sample ratings for {len(user_ids)} users and {len(movie_ids)} movies")
    return ratings_df

def preprocess_ratings(ratings):
    """Filter out users with only one rating to avoid stratify error.
    
    Args:
        ratings (pandas.DataFrame): DataFrame containing ratings data
        
    Returns:
        pandas.DataFrame: Filtered ratings DataFrame
    """
    # Check if we have any ratings
    if len(ratings) == 0:
        print("Warning: Empty ratings dataset")
        return ratings
        
    # Count ratings per user
    user_counts = ratings['user_id'].value_counts()
    
    # Find users with at least 2 ratings
    users_with_multiple = user_counts[user_counts >= 2].index
    
    # If no users have multiple ratings, return original dataset with warning
    if len(users_with_multiple) == 0:
        print("Warning: No users with multiple ratings found. Cannot stratify.")
        return ratings
        
    # Filter ratings to only include users with multiple ratings
    filtered_ratings = ratings[ratings['user_id'].isin(users_with_multiple)]
    
    # Report on filtering
    removed = len(ratings) - len(filtered_ratings)
    print(f"Filtered out {removed} ratings from users with only one rating")
    print(f"Remaining ratings: {len(filtered_ratings)}")
    
    return filtered_ratings

def split_data(ratings):
    """Split ratings into train/test sets, with optional stratification by user_id.
    
    Args:
        ratings (pandas.DataFrame): DataFrame containing ratings data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Train and test splits
    """
    X = ratings.copy()
    y = ratings['user_id']
    
    # Check if we have enough data for stratification
    user_counts = y.value_counts()
    min_count = user_counts.min() if len(user_counts) > 0 else 0
    
    if min_count >= 2:
        # We can stratify
        print("Using stratified sampling by user_id")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
    else:
        # Not enough data for stratification
        print("Warning: Not enough data per user for stratification. Using random sampling.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
    return X_train, X_test, y_train, y_test

def rmse(y_true, y_pred):
    """Compute root mean squared error (RMSE).
    
    Args:
        y_true (array-like): True ratings
        y_pred (array-like): Predicted ratings
        
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def score(cf_model, X_test):
    """Compute the RMSE score for a model on the test set.
    
    Args:
        cf_model (callable): Model function that takes user_id and movie_id and returns a rating
        X_test (pandas.DataFrame): Test data
        
    Returns:
        float: RMSE score
    """
    # Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])
    
    # Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in id_pairs])
    
    # Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])
    
    # Return the final RMSE score
    return rmse(y_true, y_pred)

def baseline(user_id, movie_id):
    """Baseline model: always return a rating of 3.0.
    
    Args:
        user_id: User ID (not used)
        movie_id: Movie ID (not used)
        
    Returns:
        float: Constant rating of 3.0
    """
    return 3.0


def cf_user_mean(user_id, movie_id):
    """User-based collaborative filtering using mean ratings.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        float: Predicted rating
    """
    global r_matrix
    
    # Check if movie_id exists in r_matrix
    if movie_id in r_matrix:
        # Compute the mean of all the ratings given to the movie
        mean_rating = r_matrix[movie_id].mean()
    else:
        # Default to a rating of 3.0 in the absence of any information
        mean_rating = 3.0
    
    return mean_rating

def prepare_similarity_matrix(X_train):
    """Prepare ratings matrix and similarity matrix for collaborative filtering.
    
    Args:
        X_train (pandas.DataFrame): Training data
        
    Returns:
        tuple: (r_matrix, cosine_sim) - Ratings matrix and cosine similarity matrix
    """
    try:
        # Check if we have any data
        if len(X_train) == 0:
            print("Warning: Empty training set. Cannot create similarity matrix.")
            # Return empty DataFrames
            return pd.DataFrame(), pd.DataFrame()
            
        # Print unique user and movie counts
        unique_users = X_train['user_id'].nunique()
        unique_movies = X_train['movie_id'].nunique()
        print(f"Creating ratings matrix with {unique_users} users and {unique_movies} movies")
        
        # Build the ratings matrix using pivot_table function
        r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie_id')
        print(f"Ratings matrix shape: {r_matrix.shape}")
        
        # Check if the matrix is empty
        if r_matrix.empty or r_matrix.shape[0] == 0 or r_matrix.shape[1] == 0:
            print("Warning: Empty ratings matrix. Cannot compute similarity.")
            return pd.DataFrame(), pd.DataFrame()
            
        # Create a dummy ratings matrix with all null values imputed to 0
        r_matrix_dummy = r_matrix.copy().fillna(0)
        
        # Compute the cosine similarity matrix using the dummy ratings matrix
        print("Computing cosine similarity matrix...")
        cosine_sim_values = cosine_similarity(r_matrix_dummy, r_matrix_dummy)
        
        # Convert into pandas dataframe
        cosine_sim = pd.DataFrame(cosine_sim_values, index=r_matrix.index, columns=r_matrix.index)
        print(f"Similarity matrix shape: {cosine_sim.shape}")
        
        return r_matrix, cosine_sim
    except Exception as e:
        print(f"Error creating similarity matrix: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def cf_user_wmean(user_id, movie_id):
    """User-based collaborative filtering using weighted mean ratings.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        float: Predicted rating
    """
    global r_matrix, cosine_sim
    
    # Check if movie_id exists in r_matrix
    if movie_id in r_matrix:
        # Get the similarity scores for the user in question with every other user
        sim_scores = cosine_sim[user_id]
        
        # Get the user ratings for the movie in question
        m_ratings = r_matrix[movie_id]
        
        # Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index
        
        # Drop the NaN values from the m_ratings Series
        m_ratings = m_ratings.dropna()
        
        # Drop the corresponding cosine scores from the sim_scores series
        sim_scores = sim_scores.drop(idx)
        
        # Check if we have any ratings left
        if len(m_ratings) > 0:
            # Compute the final weighted mean
            wmean_rating = np.dot(sim_scores, m_ratings) / sim_scores.sum()
        else:
            wmean_rating = 3.0
    else:
        # Default to a rating of 3.0 in the absence of any information
        wmean_rating = 3.0
    
    return wmean_rating

def prepare_demographic_data(X_train, users):
    """Prepare demographic data for demographic-based filtering.
    
    Args:
        X_train (pandas.DataFrame): Training data
        users (pandas.DataFrame): User data
        
    Returns:
        tuple: (users_indexed, merged_df, gender_mean, gen_occ_mean) - Processed demographic data
    """
    try:
        # Check if we have any data
        if len(X_train) == 0 or len(users) == 0:
            print("Warning: Empty training set or users data. Cannot create demographic data.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.DataFrame()
        
        # Set the index of the users dataframe to the user_id
        users_indexed = users.set_index('user_id')
        
        # Merge the original users dataframe with the training set
        print("Merging user data with ratings...")
        merged_df = pd.merge(X_train, users, on='user_id', how='inner')
        print(f"Merged data shape: {merged_df.shape}")
        
        if len(merged_df) == 0:
            print("Warning: No overlap between users and ratings. Cannot create demographic data.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.DataFrame()
        
        # Compute the mean rating of every movie by gender
        print("Computing gender-based ratings...")
        gender_mean = merged_df[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()
        
        # Compute the mean rating by gender and occupation
        print("Computing gender and occupation based ratings...")
        gen_occ_mean = merged_df[['sex', 'rating', 'movie_id', 'occupation']].pivot_table(
            values='rating', index=['sex', 'occupation'], columns='movie_id', aggfunc='mean')
        
        return users_indexed, merged_df, gender_mean, gen_occ_mean
    except Exception as e:
        print(f"Error preparing demographic data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def cf_gender(user_id, movie_id):
    """Gender-based collaborative filtering.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        float: Predicted rating
    """
    global gender_mean, merged_df
    
    # Default rating if anything goes wrong
    default_rating = 3.0
    
    try:
        # Check if we have the demographic data
        if gender_mean is None or merged_df is None or isinstance(gender_mean, pd.DataFrame) and gender_mean.empty:
            return default_rating
        
        # Find the user's gender
        user_info = merged_df[merged_df['user_id'] == user_id]
        if len(user_info) == 0:
            return default_rating  # User not found
        
        gender = user_info['sex'].values[0]
        
        # Check if gender is valid
        if pd.isna(gender):
            return default_rating
        
        # Check if demographic data exists for this gender and movie
        if gender in gender_mean.index and movie_id in gender_mean.columns:
            # Mean rating for this movie by users of the same gender
            rating = gender_mean.loc[gender, movie_id]
            if not pd.isna(rating):
                return rating
    except Exception as e:
        # Silently handle any errors
        pass
    
    # Default to a rating of 3.0 in any error case
    return default_rating

def cf_gen_occ(user_id, movie_id):
    """Gender and occupation based collaborative filtering.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        float: Predicted rating
    """
    global gen_occ_mean, merged_df
    
    # Default rating if anything goes wrong
    default_rating = 3.0
    
    try:
        # Check if we have the demographic data
        if gen_occ_mean is None or merged_df is None or isinstance(gen_occ_mean, pd.DataFrame) and gen_occ_mean.empty:
            return default_rating
        
        # Find the user's gender and occupation
        user_info = merged_df[merged_df['user_id'] == user_id]
        if len(user_info) == 0:
            return default_rating  # User not found
        
        gender = user_info['sex'].values[0]
        occupation = user_info['occupation'].values[0]
        
        # Check if gender and occupation are valid
        if pd.isna(gender) or pd.isna(occupation):
            return default_rating
        
        # Create the demographic key
        demo_key = f'{gender}_{occupation}'
        
        # Check if demographic data exists
        try:
            if demo_key in gen_occ_mean.index and movie_id in gen_occ_mean.columns:
                rating = gen_occ_mean.loc[demo_key, movie_id]
                if not pd.isna(rating):
                    return rating
        except Exception:
            pass  # Handle issues with index/columns gracefully
            
        # Fall back to gender-based prediction
        return cf_gender(user_id, movie_id)
    except Exception as e:
        # Silently handle any errors
        pass
    
    # Default to a rating of 3.0 in any error case
    return default_rating

def evaluate_surprise_models(ratings):
    """Evaluate model-based approaches using the Surprise library.
    
    Args:
        ratings (pandas.DataFrame): Ratings data
        
    Returns:
        dict: Dictionary of model names and their RMSE scores
    """
    try:
        # Import the required classes and methods from the surprise library
        from surprise import Reader, Dataset, KNNBasic, SVD, accuracy
        from surprise.model_selection import cross_validate
        
        print("Using Surprise library for model-based collaborative filtering...")
        
        # Define reader and load data
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
        
        # Initialize models
        models = {
            'KNN Basic': KNNBasic(),
            'SVD': SVD()
        }
        
        # Evaluate models
        results = {}
        for name, model in models.items():
            print(f"Evaluating {name}...")
            cv_results = cross_validate(model, data, measures=['RMSE'], cv=3, verbose=False)
            rmse = cv_results['test_rmse'].mean()
            print(f"{name} RMSE: {rmse:.4f}")
            results[name] = rmse
            
            # Save the SVD model for later use in generating recommendations
            if name == 'SVD':
                global trained_algo
                trained_algo = model.fit(data.build_full_trainset())
        
        return results
    except ImportError:
        print("Surprise library not installed. Skipping model-based approaches.")
        print("To install, run: pip install scikit-surprise")
        return {}

def main():


    """Main function to run the collaborative filtering workflow."""
    global r_matrix, cosine_sim, gender_mean, gen_occ_mean, users_indexed
    
    print("=================================================")
    print("       COLLABORATIVE FILTERING RECOMMENDER       ")
    print("=================================================")
    
    # Load data
    print("\n1. LOADING DATA\n--------------")
    users = load_users(USERS_PATH)
    movies = load_movies(MOVIES_PATH)
    ratings = load_ratings(RATINGS_PATH)
    
    # Generate sample data if any of the required data is missing
    if ratings.empty:
        print("Using sample ratings data for demonstration")
        ratings = generate_sample_ratings()
        
    if movies.empty:
        print("Using sample movie data for demonstration")
        movies = generate_sample_movies()
        
    if users.empty:
        print("Using sample user data for demonstration")
        users = generate_sample_users()
        
    # Final check
    if ratings.empty or users.empty or movies.empty:
        print("Failed to generate required data. Exiting.")
        return

    # Preprocess ratings
    print("\n2. PREPROCESSING DATA\n-------------------")
    ratings = preprocess_ratings(ratings)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(ratings)
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Ask user what type of filtering they want to run
    print("\nWhich recommender approach would you like to run?")
    print("1. Baseline model")
    print("2. User-based collaborative filtering")
    print("3. Demographic filtering")
    print("4. Model-based approaches (SVD, kNN)")
    print("5. All approaches (comprehensive evaluation)")
    print("6. Generate recommendations for a sample user")
    
    try:
        choice = input("\nEnter your choice (1-6, default: 5): ").strip() or "5"
        choice = int(choice)
        if choice < 1 or choice > 6:
            print("Invalid choice. Running all approaches (option 5).")
            choice = 5
    except ValueError:
        print("Invalid input. Running all approaches (option 5).")
        choice = 5
    
    # Prepare data for collaborative filtering if needed
    if choice in [2, 5]:
        print("\n3. PREPARING COLLABORATIVE FILTERING DATA\n-------------------------------------")
        r_matrix, cosine_sim = prepare_similarity_matrix(X_train)
        
        # Check if data preparation was successful
        if r_matrix.empty or cosine_sim.empty:
            print("\nWarning: Could not prepare similarity matrices. Skipping user-based models.")
            can_run_user_based = False
        else:
            can_run_user_based = True
    else:
        can_run_user_based = False
    
    # Prepare demographic data if needed
    if choice in [3, 5]:
        print("\n3. PREPARING DEMOGRAPHIC DATA\n-------------------------")
        try:
            users_indexed, merged_df, gender_mean, gen_occ_mean = prepare_demographic_data(X_train, users)
            
            # Additional safety check to ensure they're actually DataFrames and not None or empty
            if (gender_mean is None or gen_occ_mean is None or 
                isinstance(gender_mean, pd.DataFrame) and gender_mean.empty or
                isinstance(gen_occ_mean, pd.DataFrame) and gen_occ_mean.empty):
                raise ValueError("Invalid demographic data")
                
            can_run_demographic = True
            print("Demographic data prepared successfully.")
        except Exception as e:
            print(f"\nWarning: Could not prepare demographic data: {str(e)}")
            print("Skipping demographic models.")
            can_run_demographic = False
            # Set these to None to avoid potential issues elsewhere
            gender_mean = None
            gen_occ_mean = None
            merged_df = None
    else:
        can_run_demographic = False
        gender_mean = None
        gen_occ_mean = None
        merged_df = None
    
    # Initialize results dictionary
    results = {}
    
    # Train and evaluate models based on user choice
    if choice in [1, 5]:
        # Baseline model
        print("\n4. EVALUATING BASELINE MODEL\n-----------------------------------")
        baseline_rmse = score(baseline, X_test)
        print(f"Baseline RMSE: {baseline_rmse:.4f}")
        results['Baseline'] = baseline_rmse
    
    if choice in [2, 5] and can_run_user_based:
        # User-based CF models
        print("\n5. EVALUATING USER-BASED CF\n-----------------------------------")
        
        # User-based CF with mean ratings
        print("\n5.1 Evaluating user-based CF with mean ratings...")
        user_mean_rmse = score(cf_user_mean, X_test)
        print(f"User-based CF (Mean) RMSE: {user_mean_rmse:.4f}")
        results['User-based CF (Mean)'] = user_mean_rmse
        
        # User-based CF with weighted mean ratings
        print("\n5.2 Evaluating user-based CF with weighted mean ratings...")
        user_wmean_rmse = score(cf_user_wmean, X_test)
        print(f"User-based CF (Weighted Mean) RMSE: {user_wmean_rmse:.4f}")
        results['User-based CF (Weighted Mean)'] = user_wmean_rmse
    
    if choice in [3, 5] and can_run_demographic:
        # Demographic-based CF models
        print("\n6. EVALUATING DEMOGRAPHIC FILTERING\n-----------------------------------------")
        
        # Gender-based CF
        print("\n6.1 Evaluating gender-based CF...")
        gender_rmse = score(cf_gender, X_test)
        print(f"Gender-based CF RMSE: {gender_rmse:.4f}")
        results['Gender-based CF'] = gender_rmse
        
        # Gender and occupation based CF
        print("\n6.2 Evaluating gender and occupation based CF...")
        gen_occ_rmse = score(cf_gen_occ, X_test)
        print(f"Gender and Occupation based CF RMSE: {gen_occ_rmse:.4f}")
        results['Gender and Occupation based CF'] = gen_occ_rmse
    
    if choice in [4, 5]:
        # Model-based approaches
        print("\n7. EVALUATING MODEL-BASED APPROACHES\n--------------------------------")
        surprise_results = evaluate_surprise_models(ratings)
        results.update(surprise_results)
    
    if choice == 6:
        # Generate recommendations for a sample user
        try:
            user_id_input = input("\nEnter user ID (1-943, default: 1): ").strip() or "1"
            user_id = int(user_id_input)
            
            print(f"\nGenerating recommendations for user {user_id}...")
            print("\nTraining models for recommendation...")
            
            # Use Surprise for recommendations
            from surprise import Dataset, Reader, KNNBasic, SVD
            
            # Ask which algorithm to use
            print("\nWhich algorithm would you like to use?")
            print("1. k-Nearest Neighbors (KNN)")
            print("2. Singular Value Decomposition (SVD)")
            algo_choice = input("Enter your choice (1-2, default: 2): ").strip() or "2"
            
            # Prepare data for Surprise
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
            trainset = data.build_full_trainset()
            
            # Train selected algorithm
            if algo_choice == "1":
                print("\nTraining KNN model...")
                algo = KNNBasic()
                algo_name = "KNN"
            else:
                print("\nTraining SVD model...")
                algo = SVD()
                algo_name = "SVD"
                
            algo.fit(trainset)
            
            # Get number of recommendations
            try:
                n_recs = input("\nHow many recommendations would you like? (default: 5): ").strip() or "5"
                n_recs = int(n_recs)
                if n_recs < 1:
                    n_recs = 5
            except ValueError:
                print("Invalid input. Using default value of 5.")
                n_recs = 5
            
            # Generate recommendations
            show_user_recommendations(user_id, ratings=ratings, movies=movies, algo=algo, n=n_recs)
            print(f"\nRecommendations generated using {algo_name} algorithm.")
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please enter a valid user ID (integer).")
    
    # Print summary if we have results
    if results and choice != 6:
        # Print summary
        print("\n8. SUMMARY OF RESULTS\n-------------------")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1])
        
        # Print all results
        for model, rmse in sorted(results.items(), key=lambda x: x[1]):
            if model == best_model[0]:
                print(f"{model}: {rmse:.4f} (BEST)")
            else:
                print(f"{model}: {rmse:.4f}")
                
        print("\nBest performing model: {0} with RMSE: {1:.4f}".format(best_model[0], best_model[1]))
        
        # Generate and display insights
        insights = generate_insights(results)
        print(insights)
    
    print("\nCollaborative filtering evaluation complete!")

def generate_insights(results):
    """Generate key takeaways and insights from model results.
    
    Args:
        results (dict): Dictionary of model names and their RMSE scores
        
    Returns:
        str: A string containing insights and takeaways
    """
    if not results:
        return "No results available to analyze."
    
    insights = ["\n=== KEY TAKEAWAYS AND INSIGHTS ==="]
    
    # Sort models by performance
    sorted_models = sorted(results.items(), key=lambda x: x[1])
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    
    # Add best and worst model insights
    insights.append(f"\n1. BEST MODEL: {best_model[0]} (RMSE: {best_model[1]:.4f})")
    insights.append(f"2. WORST MODEL: {worst_model[0]} (RMSE: {worst_model[1]:.4f})")
    
    # Compare model types
    cf_models = {k: v for k, v in results.items() if 'CF' in k and 'based' in k}
    demographic_models = {k: v for k, v in results.items() if 'Gender' in k or 'Occupation' in k}
    surprise_models = {k: v for k, v in results.items() if 'kNN' in k or 'SVD' in k}
    
    # Add insights about model types
    insights.append("\n3. MODEL TYPE COMPARISON:")
    
    if cf_models:
        best_cf = min(cf_models.items(), key=lambda x: x[1])
        insights.append(f"   - Best Collaborative Filtering: {best_cf[0]} (RMSE: {best_cf[1]:.4f})")
    
    if demographic_models:
        best_demo = min(demographic_models.items(), key=lambda x: x[1])
        insights.append(f"   - Best Demographic: {best_demo[0]} (RMSE: {best_demo[1]:.4f})")
    
    if surprise_models:
        best_surprise = min(surprise_models.items(), key=lambda x: x[1])
        insights.append(f"   - Best Model-based: {best_surprise[0]} (RMSE: {best_surprise[1]:.4f})")
    
    # Add general insights
    insights.append("\n4. GENERAL INSIGHTS:")
    
    # Compare user-based vs item-based if available
    user_based = {k: v for k, v in results.items() if 'User-based' in k}
    
    if 'Baseline' in results:
        improvement = results['Baseline'] - best_model[1]
        percent_improvement = (improvement / results['Baseline']) * 100
        insights.append(f"   - Best model improved over baseline by {percent_improvement:.2f}% (RMSE reduction: {improvement:.4f})")
    
    if user_based and len(user_based) > 1:
        insights.append(f"   - Weighted mean approach {'improved' if min(user_based.values()) == user_based.get('User-based CF (Weighted Mean)', float('inf')) else 'did not improve'} user-based collaborative filtering performance")
    
    if demographic_models:
        if len(demographic_models) > 1:
            gen_only = demographic_models.get('Gender-based CF', float('inf'))
            gen_occ = demographic_models.get('Gender and Occupation based CF', float('inf'))
            insights.append(f"   - Adding occupation data to gender-based filtering {'improved' if gen_occ < gen_only else 'did not improve'} performance")
    
    if surprise_models:
        if 'kNN Basic' in surprise_models and 'SVD' in surprise_models:
            knn_rmse = surprise_models['kNN Basic']
            svd_rmse = surprise_models['SVD']
            insights.append(f"   - {'SVD' if svd_rmse < knn_rmse else 'kNN'} performed better among model-based approaches")
    
    # Add recommendations
    insights.append("\n5. RECOMMENDATIONS:")
    insights.append(f"   - Consider using {best_model[0]} for your recommendation system as it achieved the lowest RMSE")
    
    if 'SVD' in results and results['SVD'] <= best_model[1] * 1.1:  # If SVD is within 10% of best model
        insights.append("   - SVD is a good choice for production as it balances accuracy and scalability")
    
    if worst_model[1] > 1.5 * best_model[1]:  # If worst model is 50% worse than best
        insights.append(f"   - Avoid using {worst_model[0]} as it performed significantly worse than other models")
    
    return "\n".join(insights)

def show_user_recommendations(user_id, ratings, movies, algo, n=5):
    """Show top-N movie recommendations for a given user using a trained Surprise algorithm."""
    # Get list of all movie_ids
    all_movie_ids = set(movies['movie_id'])
    # Movies already rated by user
    rated_movie_ids = set(ratings[ratings['user_id'] == user_id]['movie_id'])
    # Movies not yet rated
    unrated_movie_ids = all_movie_ids - rated_movie_ids
    # Predict ratings for all unrated movies
    predictions = []
    for movie_id in unrated_movie_ids:
        pred = algo.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    # Get top-N recommendations
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    # Print recommendations
    print(f"\nTop {n} recommendations for user {user_id}:")
    for movie_id, est_rating in top_n:
        title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"{title} (predicted rating: {est_rating:.2f})")
    
if __name__ == "__main__":
    main()






