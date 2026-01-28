import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "../data/datasets/unified_dataset.csv"
DATASET_OUT_PATH = "../data/datasets/unified_dataset_reduced.csv"
TARGET = "Label"
OUT_CSV = "../data/datasets/mi_feature_ranking.csv"

# categorical columns (leave empty for auto-detection)
CATEGORICAL_COLS = []

SAMPLE_N = None  # set to e.g. 1000000 for large datasets
RANDOM_STATE = 42
TOP_K = 30
MI_THRESHOLD = None


def encode_categoricals(X, cat_cols):
    X = X.copy()
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str).fillna("NA"))
    return X


def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")

    df = df.dropna(subset=[TARGET])

    if SAMPLE_N and len(df) > SAMPLE_N:
        df = df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
        print(f"Sampled to {df.shape[0]} rows")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    cat_cols = CATEGORICAL_COLS if CATEGORICAL_COLS else \
               X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # fill numeric NaNs
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    if cat_cols:
        X = encode_categoricals(X, cat_cols)

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = LabelEncoder().fit_transform(y.astype(str))

    discrete_mask = np.array([c in cat_cols for c in X.columns], dtype=bool)

    mi = mutual_info_classif(X, y, discrete_features=discrete_mask, 
                             random_state=RANDOM_STATE, n_neighbors=3)

    ranking = pd.DataFrame({"feature": X.columns, "mutual_info": mi}) \
                .sort_values("mutual_info", ascending=False) \
                .reset_index(drop=True)

    print("\nTop features:")
    print(ranking.head(20))

    ranking.to_csv(OUT_CSV, index=False)

    if MI_THRESHOLD:
        selected = ranking[ranking["mutual_info"] >= MI_THRESHOLD]["feature"].tolist()
    elif TOP_K:
        selected = ranking.head(TOP_K)["feature"].tolist()
    else:
        selected = ranking["feature"].tolist()

    print(f"\nSelected {len(selected)} features")

    df_reduced = df[selected + [TARGET]]
    df_reduced.to_csv(DATASET_OUT_PATH, index=False)


if __name__ == "__main__":
    main()