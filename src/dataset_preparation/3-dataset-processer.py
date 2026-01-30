import argparse
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore

DATASET_PATH = "../../data/datasets/dataset_reduced.csv"
TARGET = "Label"


# Scales numeric features and encodes categorical ones
def preprocess_features(df, target_column):

    print("\nPREPROCESSING FEATURES... ")

    # Separate features and labels
    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # Handle categorical features in X
    categorical_cols = X.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        # print(f"Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Label vector shape: {y_encoded.shape}")

    return X_scaled, y_encoded, X.columns.tolist(), label_encoder, scaler


# Splits the dataframe into sets: train, val, test
def split_dataframe(X, y, test_size=0.2, val_size=0.1, random_state=42):

    print("\n\nSPLITTING DATASET...")

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


# Saves splits and preprocessing objects
def save_processed_data(
    data_dict,
    label_encoder,
    scaler,
    feature_names,
    output_dir="../../data/processed_data_us/",
):

    print(f"\n\nSAVING PROCESSED DATA TO {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save data splits
    np.save(os.path.join(output_dir, "X_train.npy"), data_dict["X_train"])
    np.save(os.path.join(output_dir, "y_train.npy"), data_dict["y_train"])
    np.save(os.path.join(output_dir, "X_val.npy"), data_dict["X_val"])
    np.save(os.path.join(output_dir, "y_val.npy"), data_dict["y_val"])
    np.save(os.path.join(output_dir, "X_test.npy"), data_dict["X_test"])
    np.save(os.path.join(output_dir, "y_test.npy"), data_dict["y_test"])

    # Save preprocessing objects
    import pickle

    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print("Data saved successfully!")


# Reduces dataset imbalance via random undersampling
def imbalance_reducer_undersample(
    df: pd.DataFrame,
    target_column: str,
    fixed_size: int = None,
    ratio_to_major: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    if target_column not in df.columns:
        raise ValueError(f"Colonna label '{target_column}' non trovata.")
    if ratio_to_major <= 0:
        raise ValueError("ratio_to_major deve essere > 0.")

    y = df[target_column]
    counts = y.value_counts()

    rng = np.random.default_rng(random_state)
    parts = []

    for cls, cnt in counts.items():
        cls_df = df[df[target_column] == cls]

        if fixed_size is not None:
            target = fixed_size
        else:
            major = int(counts.max())
            target = max(1, int(round(major * ratio_to_major)))

        take = min(int(cnt), target)
        idx = rng.choice(cls_df.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[idx])

    fdf = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return fdf


def imbalance_reducer_hybrid_jitter(
    df: pd.DataFrame,
    label_column: str,
    ratio_to_major: float = 0.5,
    random_state: int = 42,
    noise_scale: float = 0.02,  # 0.0 => equivalente a ROS puro
) -> pd.DataFrame:
    if label_column not in df.columns:
        raise ValueError(f"Colonna label '{label_column}' non trovata.")
    if ratio_to_major <= 0:
        raise ValueError("ratio_to_major deve essere > 0.")
    if noise_scale < 0:
        raise ValueError("noise_scale deve essere >= 0.")

    y = df[label_column]
    counts = y.value_counts(dropna=False)
    major = int(counts.max())
    target = max(1, int(round(major * ratio_to_major)))

    rng = np.random.default_rng(random_state)

    # Step 1: undersample
    y_arr = y.to_numpy()
    idx_arr = df.index.to_numpy()

    keep = []
    for cls, cnt in counts.items():
        pos = np.flatnonzero(y_arr == cls)
        if int(cnt) > target:
            chosen = rng.choice(pos, size=target, replace=False)
            keep.append(idx_arr[chosen])
        else:
            keep.append(idx_arr[pos])

    keep = np.concatenate(keep, axis=0)
    df_u = df.loc[keep].reset_index(drop=True)

    # Identifica colonne numeriche (si “jitterano” solo queste)
    X_cols = [c for c in df_u.columns if c != label_column]
    num_cols = df_u[X_cols].select_dtypes(include=[np.number]).columns.tolist()

    y_u = df_u[label_column].to_numpy()
    idx_u = np.arange(len(df_u))

    blocks = [df_u]
    counts_u = pd.Series(y_u).value_counts(dropna=False)

    for cls, cnt in counts_u.items():
        cnt = int(cnt)
        if cnt >= target:
            continue

        pos = idx_u[y_u == cls]
        need = target - cnt
        chosen = rng.choice(pos, size=need, replace=True)

        block = df_u.iloc[chosen].copy()

        if noise_scale > 0 and num_cols:
            # std per classe (calcolata sui campioni originali della classe in df_u)
            cls_std = df_u.loc[pos, num_cols].std(ddof=0).to_numpy()
            # se una feature ha std=0, niente rumore su quella feature
            noise = rng.normal(
                loc=0.0, scale=cls_std * noise_scale, size=(need, len(num_cols))
            )
            block.loc[:, num_cols] = block.loc[:, num_cols].to_numpy() + noise

        blocks.append(block)

    out = pd.concat(blocks, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def main():

    print("LOADING DATASET AT:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")

    # Removing most imbalanced classes
    imbalanced_classes = [
        "Heartbleed",
        "Ransomware",
        "MITM",
        "Bot",
        "Backdoor",
        "Infiltration",
    ]
    print(f"Removing imbalanced classes: {imbalanced_classes}")
    df.drop(df[df["Label"].isin(imbalanced_classes)].index, inplace=True)
    print(f"New dataset shape: {df.shape}")

    df = imbalance_reducer_undersample(df, target_column=TARGET)
    # df = imbalance_reducer_undersample(df, target_column=TARGET, fixed_size=500000)
    # df = imbalance_reducer_hybrid_jitter(df, label_column=TARGET)

    # df.to_csv("../data/datasets/unified_dataset_reduced_us.csv",index=False)

    X, y, features, le, scaler = preprocess_features(df, TARGET)
    splits = split_dataframe(X, y)
    save_processed_data(splits, le, scaler, features)

    return


if __name__ == "__main__":
    main()
