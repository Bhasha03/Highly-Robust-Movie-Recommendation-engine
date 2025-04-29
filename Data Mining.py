#!/usr/bin/env python
# coding: utf-8

"""
Data Mining Techniques for Recommendation Systems

This script demonstrates various data mining techniques including:
- Clustering (K-means, Spectral)
- Dimensionality Reduction (PCA, LDA)
- Supervised Learning (Gradient Boosting)

Each technique is implemented with example datasets and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


def euclidean(v1, v2):
    """Compute Euclidean distance between two vectors.
    
    Args:
        v1 (list or array): First vector
        v2 (list or array): Second vector
        
    Returns:
        float: Euclidean distance
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    diff = np.power(v1 - v2, 2)
    return np.sqrt(np.sum(diff))


def pearson_similarity(v1, v2):
    """Compute Pearson correlation between two vectors.
    
    Args:
        v1 (list or array): First vector
        v2 (list or array): Second vector
        
    Returns:
        float: Pearson correlation coefficient
    """
    return pearsonr(v1, v2)[0]


def generate_cluster_data(n_samples=200, centers=3, random_state=0):
    """Generate synthetic data for clustering.
    
    Args:
        n_samples (int): Number of samples to generate
        centers (int): Number of cluster centers
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    return make_blobs(n_samples=n_samples, centers=centers, 
                     random_state=random_state, cluster_std=0.60)


def plot_kmeans_clusters(X, kmeans):
    """Plot K-means clustering results.
    
    Args:
        X (array): Feature matrix
        kmeans (KMeans): Fitted KMeans model
    """
    y_pred = kmeans.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50)
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X')
    plt.title('K-means Clustering Results')
    plt.show()


def plot_elbow_method(X, max_clusters=8):
    """Plot the elbow method to find optimal number of clusters.
    
    Args:
        X (array): Feature matrix
        max_clusters (int): Maximum number of clusters to try
    """
    inertia_values = []
    
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0, max_iter=10, init='random')
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    sns.pointplot(x=list(range(1, max_clusters + 1)), y=inertia_values)
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.show()


def generate_nonlinear_clusters():
    """Generate non-linear half-moon shaped clusters.
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    return make_moons(200, noise=0.05, random_state=0)


def compare_clustering_methods(X):
    """Compare K-means and Spectral Clustering on non-linear data.
    
    Args:
        X (array): Feature matrix
    """
    # K-means clustering
    kmeans = KMeans(n_clusters=2, init='random', max_iter=10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    # Spectral clustering
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
    y_spectral = spectral.fit_predict(X)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)
    ax1.set_title('K-means Clustering')
    
    ax2.scatter(X[:, 0], X[:, 1], c=y_spectral, s=50)
    ax2.set_title('Spectral Clustering')
    
    plt.show()


def load_iris_data():
    """Load and prepare the Iris dataset.
    
    Returns:
        pandas.DataFrame: Iris dataset
    """
    iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
                      names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    return iris


def apply_pca(X, n_components=2):
    """Apply Principal Component Analysis for dimensionality reduction.
    
    Args:
        X (DataFrame): Feature matrix
        n_components (int): Number of components to keep
        
    Returns:
        tuple: (pca_df, pca_model) - DataFrame with PCA results and the PCA model
    """
    # Scale the data
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns
    )
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create DataFrame with results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return pca_df, pca


def apply_lda(X, y, n_components=2):
    """Apply Linear Discriminant Analysis for dimensionality reduction.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        n_components (int): Number of components to keep
        
    Returns:
        DataFrame: DataFrame with LDA results
    """
    # Scale the data
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns
    )
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_result = lda.fit_transform(X_scaled, y)
    
    # Create DataFrame with results
    lda_df = pd.DataFrame(
        data=lda_result,
        columns=[f'C{i+1}' for i in range(n_components)]
    )
    
    return lda_df


def plot_dimensionality_reduction(df, target, x_col, y_col, title):
    """Plot the results of dimensionality reduction.
    
    Args:
        df (DataFrame): DataFrame with dimensionality reduction results
        target (Series): Target variable
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    result_df = pd.concat([df, target], axis=1)
    
    sns.scatterplot(x=x_col, y=y_col, hue='class', data=result_df)
    plt.title(title)
    plt.show()


def train_gradient_boosting(X, y, test_size=0.25, random_state=0):
    """Train a Gradient Boosting Classifier and evaluate its performance.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        GradientBoostingClassifier: Trained model
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train Gradient Boosting Classifier
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = gbc.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return gbc


def plot_feature_importance(model, feature_names):
    """Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_names, y=model.feature_importances_)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.show()


def main():
    """Main function to demonstrate data mining techniques."""
    print("\n1. K-means Clustering Example")
    X, y = generate_cluster_data()
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    plot_kmeans_clusters(X, kmeans)
    
    print("\n2. Elbow Method for Finding Optimal K")
    plot_elbow_method(X)
    
    print("\n3. Comparing Clustering Methods on Non-linear Data")
    X_moons, y_moons = generate_nonlinear_clusters()
    compare_clustering_methods(X_moons)
    
    print("\n4. Dimensionality Reduction with PCA and LDA")
    iris = load_iris_data()
    X_iris = iris.drop('class', axis=1)
    y_iris = iris['class']
    
    # Apply PCA
    pca_df, _ = apply_pca(X_iris)
    plot_dimensionality_reduction(pca_df, y_iris, 'PC1', 'PC2', 'PCA on Iris Dataset')
    
    # Apply LDA
    lda_df = apply_lda(X_iris, y_iris)
    plot_dimensionality_reduction(lda_df, y_iris, 'C1', 'C2', 'LDA on Iris Dataset')
    
    print("\n5. Gradient Boosting Classification")
    gbc_model = train_gradient_boosting(X_iris, y_iris)
    plot_feature_importance(gbc_model, X_iris.columns)


if __name__ == "__main__":
    main()
