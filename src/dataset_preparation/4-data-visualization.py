
# Data Visualization Pipeline for CICIDS-2017 and ToN-IoT

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

SAVE_PATH = "../resources/data-visualization/"
DATA_PATH = "../data/processed_data/"


#Loads preprocessed datasets and encoders
def load_processed_data(data_path=DATA_PATH):

    print("\nLoading processed data...")

    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    X_test  = np.load(os.path.join(data_path, "X_test.npy"))
    y_test  = np.load(os.path.join(data_path, "y_test.npy"))

    with open(os.path.join(data_path, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    with open(os.path.join(data_path, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    return X_train, y_train, X_test, y_test, label_encoder, feature_names

#Plots class distribution
def plot_class_distribution(y, label_encoder, title, filename):

    print(f"\nGENERATING {title.lower()}...")

    classes, counts = np.unique(y, return_counts=True)
    names = label_encoder.inverse_transform(classes)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(classes)), counts)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, filename), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"GENERATING {title.lower()} : DONE")

#Plots class invariance ratio
def plot_class_imbalance(y, label_encoder):

    print("\nCOMPUTING CLASS IMBALANCE...")

    classes, counts = np.unique(y, return_counts=True)
    names = label_encoder.inverse_transform(classes)

    max_count = counts.max()
    ratios = max_count / counts

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(classes)), ratios)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Imbalance Ratio (majority / class)")
    ax.set_title("Class Imbalance")

    ax.axvline(1, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "class_imbalance.png"), dpi=300)
    plt.close()

    for name, r in zip(names, ratios):
        print(f"{name:20s}: {r:6.2f}x")

    print("COMPUTING CLASS IMBALANCE : DONE")

#Plots features statistics
def plot_features_statistics(X, feature_names, top_n=20):

    print("\nPLOTTING FEATURES STATISTICS...")

    variances = np.nanvar(X, axis=0)

    n_features = min(top_n, X.shape[1])
    idx = np.argsort(variances)[-n_features:]

    means = np.nanmean(X, axis=0)[idx]
    stds  = np.nanstd(X, axis=0)[idx]

    labels = [feature_names[i] for i in idx]
    y_pos = range(len(means))

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].barh(y_pos, means)
    ax[0].set_yticks(y_pos)
    ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].set_title("Mean values")

    ax[1].barh(y_pos, stds)
    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(labels, fontsize=8)
    ax[1].set_title("Standard deviations")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "feature_statistics.png"), dpi=300)
    plt.close()

    print("PLOTTING FEATURES STATISTICS : DONE")

#Plots PCA
def plot_pca(X, y, label_encoder, n_samples=5000):

    print("\nPLOTTING PCA...")

    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 8))

    classes = np.unique(y)
    for c in classes:
        mask = y == c
        name = label_encoder.inverse_transform([c])[0]
        ax.scatter(Z[mask, 0], Z[mask, 1], label=name, s=20, alpha=0.6)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_title("PCA projection")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "pca.png"), dpi=300)
    plt.close()

    # print("Explained variance:", pca.explained_variance_ratio_, "\n")
    print("PLOTTING PCA : DONE")

#Plots correlation matrix
def plot_correlation(X, feature_names, top_n=30):

    print("\nPLOTTING CORRELATION MATRIX...")

    variances = np.nanvar(X, axis=0)
    n_features = min(top_n, X.shape[1])

    idx = np.argsort(variances)[-n_features:]
    Xs = X[:, idx]
    labels = [feature_names[i] for i in idx]

    corr = np.corrcoef(Xs.T)

    n = len(labels)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    plt.xticks(range(n), labels, rotation=90, fontsize=7)
    plt.yticks(range(n), labels, fontsize=7)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "correlation.png"), dpi=300)
    plt.close()

    print("PLOTTING CORRELATION MATRIX : DONE")

#Plots class separability
def analyze_separability(X, y, label_encoder):

    print("\nPLOTTING CLASS SEPARABILITY...")

    classes = np.unique(y)
    n_classes = len(classes)
    
    # Compute centroids for each class
    centroids = []
    for class_id in classes:
        mask = y == class_id
        centroid = X[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Compute pairwise distances between centroids
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(centroids, metric='euclidean'))
    
    # Plot distance matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(distances,cmap='viridis',aspect='auto')
    
    class_names = label_encoder.inverse_transform(classes)
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
    plt.savefig(os.path.join(SAVE_PATH, 'class_separability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("PLOTTING CLASS SEPARABILITY : DONE")

#Prints a summary of the dataset
def print_summary(X_train, y_train, X_test, y_test, label_encoder):

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Features:        {X_train.shape[1]}")
    print(f"Classes:         {len(np.unique(y_train))}")

    for split, y in [("Train", y_train), ("Test", y_test)]:
        print(f"\n{split} distribution:")
        c, n = np.unique(y, return_counts=True)
        for ci, ni in zip(c, n):
            name = label_encoder.inverse_transform([ci])[0]
            print(f"  {name:20s}: {ni:8,} ({ni/len(y)*100:5.2f}%)")

# ------------------------------------------------------------------------------------------------------------------------------- MAIN EXECUTION 
if __name__ == "__main__":

    os.makedirs(SAVE_PATH, exist_ok=True)

    X_train, y_train, X_test, y_test, le, feature_names = load_processed_data()

    plot_class_distribution(y_train, le, "Training set distribution", "train_dist.png")
    plot_class_distribution(y_test,  le, "Test set distribution",     "test_dist.png")

    plot_class_imbalance(y_train, le)
    plot_features_statistics(X_train, feature_names)
    plot_pca(X_train, y_train, le)
    plot_correlation(X_train, feature_names)
    analyze_separability(X_train, y_train, le)
    print_summary(X_train, y_train, X_test, y_test, le)