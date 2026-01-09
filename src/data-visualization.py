
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

    print("\n\nLoading processed data...")

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

    print(f"\n\nGENERATING {title.lower()}...")

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

    for name, count in zip(names, counts):
        print(f"{name:20s}: {count:8,} samples ({count/len(y)*100:5.2f}%)")

#Plots class invariance ratio
def plot_class_imbalance(y, label_encoder):

    print("\n\nCOMPUTING CLASS IMBALANCE...")

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

    print()

#Plots features statistics
def plot_features_statistics(X, feature_names, top_n=20):

    print("\n\nCOMPUTING FEATURES STATISTICS...")

    variances = X.var(axis=0)
    idx = np.argsort(variances)[-top_n:]

    means = X.mean(axis=0)[idx]
    stds  = X.std(axis=0)[idx]

    labels = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].barh(range(top_n), means)
    ax[0].set_yticks(range(top_n))
    ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].set_title("Mean values")

    ax[1].barh(range(top_n), stds)
    ax[1].set_yticks(range(top_n))
    ax[1].set_yticklabels(labels, fontsize=8)
    ax[1].set_title("Standard deviations")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "feature_statistics.png"), dpi=300)
    plt.close()

#Plots PCA
def plot_pca(X, y, label_encoder, n_samples=5000):

    print("\n\nPLOTTING PCA...")

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

    print("Explained variance:", pca.explained_variance_ratio_, "\n")

#Plots correlation matrix
def plot_correlation(X, feature_names, top_n=30):

    print("\n\nPLOTTING CORRELATION MATRIX...")

    idx = np.argsort(X.var(axis=0))[-top_n:]
    Xs = X[:, idx]
    labels = [feature_names[i] for i in idx]

    corr = np.corrcoef(Xs.T)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(top_n), labels, rotation=90, fontsize=7)
    plt.yticks(range(top_n), labels, fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "correlation.png"), dpi=300)
    plt.close()

#Plots class separability
def analyze_separability(X, y, label_encoder):

    print("\n\nPLOTTING CLASS SEPARABILITY...")

    classes = np.unique(y)
    names = label_encoder.inverse_transform(classes)

    centroids = np.array([X[y == c].mean(axis=0) for c in classes])
    D = squareform(pdist(centroids))

    plt.figure(figsize=(10, 8))
    plt.imshow(D)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)

    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{D[i,j]:.1f}", ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "separability.png"), dpi=300)
    plt.close()

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