"""
Data Visualization for Understanding the Datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

SAVE_PATH = '../resources/data-visualization/'

# ============================================================================
# LOAD PROCESSED DATA
# ============================================================================

def load_processed_data(data_dir='../data/processed_data/'):
    """Load preprocessed data and objects"""
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    return X_train, y_train, X_test, y_test, label_encoder

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_class_distribution(y, label_encoder, title="Class Distribution"):
    """Plot class distribution"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Get class counts
    unique, counts = np.unique(y, return_counts=True)
    class_names = label_encoder.inverse_transform(unique)
    
    # Create bar plot
    bars = ax.bar(range(len(unique)), counts)
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nClass distribution:")
    for name, count in zip(class_names, counts):
        print(f"{name}: {count:,} samples ({count/len(y)*100:.2f}%)")

def plot_class_imbalance_ratio(y, label_encoder):
    """Visualize class imbalance"""
    
    unique, counts = np.unique(y, return_counts=True)
    class_names = label_encoder.inverse_transform(unique)
    
    # Calculate imbalance ratio
    max_samples = counts.max()
    imbalance_ratios = max_samples / counts
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(unique)), imbalance_ratios)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Imbalance Ratio (majority class / this class)')
    ax.set_title('Class Imbalance Visualization')
    ax.axvline(x=1, color='red', linestyle='--', label='Balanced (ratio=1)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('class_imbalance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nClass imbalance ratios:")
    for name, ratio in zip(class_names, imbalance_ratios):
        print(f"{name}: {ratio:.2f}x")

def plot_feature_statistics(X, feature_names, top_n=20):
    """Plot statistics of top N features"""
    
    # Calculate mean and std for each feature
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    
    # Get top N features by variance
    variances = X.var(axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot means
    axes[0].barh(range(top_n), means[top_indices])
    axes[0].set_yticks(range(top_n))
    if len(feature_names) > 0:
        axes[0].set_yticklabels([feature_names[i] for i in top_indices], fontsize=8)
    axes[0].set_xlabel('Mean Value (normalized)')
    axes[0].set_title(f'Top {top_n} Features by Variance - Mean Values')
    
    # Plot standard deviations
    axes[1].barh(range(top_n), stds[top_indices])
    axes[1].set_yticks(range(top_n))
    if len(feature_names) > 0:
        axes[1].set_yticklabels([feature_names[i] for i in top_indices], fontsize=8)
    axes[1].set_xlabel('Standard Deviation')
    axes[1].set_title(f'Top {top_n} Features by Variance - Std Dev')
    
    plt.tight_layout()
    plt.savefig('feature_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_visualization(X, y, label_encoder, n_samples=5000):
    """Visualize data in 2D using PCA"""
    
    print(f"\nPerforming PCA visualization (using {n_samples} samples)...")
    
    # Subsample if dataset is too large
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_classes = np.unique(y_sample)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_id in enumerate(unique_classes):
        mask = y_sample == class_id
        class_name = label_encoder.inverse_transform([class_id])[0]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=[colors[i]], label=class_name, alpha=0.6, s=20)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA Visualization of Network Traffic Data')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}")

def plot_correlation_matrix(X, feature_names, top_n=30):
    """Plot correlation matrix of top features"""
    
    print(f"\nComputing correlation matrix for top {top_n} features...")
    
    # Get top N features by variance
    variances = X.var(axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    
    X_subset = X[:, top_indices]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_subset.T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(top_n))
    ax.set_yticks(range(top_n))
    if len(feature_names) > 0:
        labels = [feature_names[i] for i in top_indices]
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    ax.set_title(f'Feature Correlation Matrix (Top {top_n} Features by Variance)')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_class_separability(X, y, label_encoder):
    """Analyze how separable classes are"""
    
    print("\nAnalyzing class separability...")
    
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Compute centroids for each class
    centroids = []
    for class_id in unique_classes:
        mask = y == class_id
        centroid = X[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Compute pairwise distances between centroids
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(centroids, metric='euclidean'))
    
    # Plot distance matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(distances, cmap='viridis', aspect='auto')
    
    class_names = label_encoder.inverse_transform(unique_classes)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add values
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, f'{distances[i, j]:.1f}',
                          ha="center", va="center", color="white", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Euclidean Distance', rotation=270, labelpad=20)
    
    ax.set_title('Class Centroid Distances (Higher = More Separable)')
    plt.tight_layout()
    plt.savefig('class_separability.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(X_train, y_train, X_test, y_test, label_encoder, feature_names):
    """Create a comprehensive summary report"""
    
    print("\n" + "="*80)
    print("DATA SUMMARY REPORT")
    print("="*80)
    
    print(f"\n{'Dataset Statistics':-^80}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    print(f"\n{'Class Information':-^80}")
    class_names = label_encoder.classes_
    print(f"Classes: {', '.join(class_names)}")
    
    print(f"\n{'Training Set Class Distribution':-^80}")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_id, count in zip(unique, counts):
        class_name = label_encoder.inverse_transform([class_id])[0]
        percentage = count / len(y_train) * 100
        print(f"{class_name:20s}: {count:8,} samples ({percentage:5.2f}%)")
    
    print(f"\n{'Test Set Class Distribution':-^80}")
    unique, counts = np.unique(y_test, return_counts=True)
    for class_id, count in zip(unique, counts):
        class_name = label_encoder.inverse_transform([class_id])[0]
        percentage = count / len(y_test) * 100
        print(f"{class_name:20s}: {count:8,} samples ({percentage:5.2f}%)")
    
    print(f"\n{'Feature Statistics':-^80}")
    print(f"Mean of feature means: {X_train.mean():.4f}")
    print(f"Mean of feature stds: {X_train.std():.4f}")
    print(f"Min value: {X_train.min():.4f}")
    print(f"Max value: {X_train.max():.4f}")
    
    print("\n" + "="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load processed data
    X_train, y_train, X_test, y_test, label_encoder = load_processed_data()
    
    # Load feature names
    with open('../data/processed_data/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print("Data loaded successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Class distribution
    plot_class_distribution(y_train, label_encoder, "Training Set Class Distribution")
    plot_class_distribution(y_test, label_encoder, "Test Set Class Distribution")
    
    # 2. Class imbalance
    plot_class_imbalance_ratio(y_train, label_encoder)
    
    # 3. Feature statistics
    plot_feature_statistics(X_train, feature_names)
    
    # 4. PCA visualization
    plot_pca_visualization(X_train, y_train, label_encoder)
    
    # 5. Correlation matrix
    if len(feature_names) > 0:
        plot_correlation_matrix(X_train, feature_names)
    
    # 6. Class separability
    analyze_class_separability(X_train, y_train, label_encoder)
    
    # 7. Summary report
    create_summary_report(X_train, y_train, X_test, y_test, label_encoder, feature_names)
    
    print("\nAll visualizations saved!")
    print("Check the current directory for PNG files.")