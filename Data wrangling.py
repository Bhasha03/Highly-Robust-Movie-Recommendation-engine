#!/usr/bin/env python
# coding: utf-8

"""
Data Wrangling for Movie Dataset

This script demonstrates various data wrangling techniques using pandas and numpy
with a movie dataset. It includes data loading, cleaning, transformation, and analysis.
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


def create_subset_dataframe(df, columns):
    """Create a subset dataframe with selected columns.
    
    Args:
        df (pandas.DataFrame): Original dataframe
        columns (list): List of column names to include
        
    Returns:
        pandas.DataFrame: Subset dataframe
    """
    return df[columns]


def clean_numeric_column(df, column):
    """Convert a column to numeric values, handling errors.
    
    Args:
        df (pandas.DataFrame): Dataframe to modify
        column (str): Column name to convert
        
    Returns:
        pandas.DataFrame: Modified dataframe with cleaned numeric column
    """
    # Function to convert values to float safely
    def to_float(x):
        try:
            return float(x)
        except:
            return np.nan
    
    # Apply conversion and return the modified dataframe
    df[column] = df[column].apply(to_float)
    return df


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
    
    return df


def filter_high_revenue_movies(df, revenue_threshold=1e9, budget_threshold=None):
    """Filter movies by revenue and optionally by budget.
    
    Args:
        df (pandas.DataFrame): Movie dataframe
        revenue_threshold (float): Minimum revenue in dollars
        budget_threshold (float, optional): Maximum budget in dollars
        
    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    if budget_threshold:
        return df[(df['revenue'] > revenue_threshold) & (df['budget'] < budget_threshold)]
    else:
        return df[df['revenue'] > revenue_threshold]


def analyze_runtime(runtime_series):
    """Analyze movie runtime statistics.
    
    Args:
        runtime_series (pandas.Series): Series containing runtime values
    """
    print(f"Longest movie runtime: {runtime_series.max()} minutes")
    print(f"Shortest movie runtime: {runtime_series.min()} minutes")


def analyze_budget(budget_series):
    """Analyze movie budget statistics.
    
    Args:
        budget_series (pandas.Series): Series containing budget values
    """
    print(f"Mean budget: ${budget_series.mean():,.2f}")
    print(f"Median budget: ${budget_series.median():,.2f}")


def main():
    """Main function to execute data wrangling workflow."""
    # Load the movie dataset
    file_path = r'C:\Users\bhava\Desktop\Python\Hands-On-Recommendation-Systems-with-Python\Data\movies_metadata.csv'
    df = load_movie_data(file_path)
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset type: {type(df)}")
    
    # Create a smaller dataframe with selected features
    columns = ['title', 'release_date', 'budget', 'revenue', 'runtime', 'genres']
    small_df = create_subset_dataframe(df, columns)
    print("\nSubset dataframe created with columns:", columns)
    
    # Display the first few rows
    print("\nFirst 5 rows of the subset dataframe:")
    print(small_df.head())
    
    # Clean the budget column
    small_df = clean_numeric_column(small_df, 'budget')
    
    # Extract year from release_date
    small_df = extract_year(small_df)
    print("\nYear extracted from release_date")
    
    # Sort by year
    small_df_by_year = small_df.sort_values('year')
    print("\nFirst 5 movies sorted by year:")
    print(small_df_by_year.head())
    
    # Sort by revenue
    small_df_by_revenue = small_df.sort_values('revenue', ascending=False)
    print("\nTop 5 highest revenue movies:")
    print(small_df_by_revenue.head())
    
    # Filter high revenue movies
    high_revenue = filter_high_revenue_movies(small_df, 1e9)
    print(f"\nNumber of movies with revenue > $1 billion: {len(high_revenue)}")
    
    # Filter high revenue, low budget movies
    high_rev_low_budget = filter_high_revenue_movies(small_df, 1e9, 1.5e8)
    print(f"\nNumber of movies with revenue > $1B and budget < $150M: {len(high_rev_low_budget)}")
    
    # Analyze runtime
    print("\nRuntime analysis:")
    analyze_runtime(small_df['runtime'])
    
    # Analyze budget
    print("\nBudget analysis:")
    analyze_budget(small_df['budget'])
    
    # Calculate high-end revenue (90th percentile)
    percentile_90 = small_df['revenue'].quantile(0.90)
    print(f"\n90th percentile revenue: ${percentile_90:,.2f}")
    
    # Count movies by year
    year_counts = small_df['year'].value_counts().sort_index()
    print("\nNumber of movies by year (first 10):")
    print(year_counts.head(10))


if __name__ == "__main__":
    main()

