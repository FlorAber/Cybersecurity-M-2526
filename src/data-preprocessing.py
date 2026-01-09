
# Data Preprocessing Pipeline for CICIDS-2017 and ToN-IoT

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

''' LOAD AND EXPLORE DATASETS '''

# Loads and expolores the dataset which path is given as parameter
def load_dataset(data_path='datasets/',label_column='Label',encoding='utf-8'):
    # LOADING
    print(f"\n\nLOADING DATASET AT PATH {data_path}...")
    csv_files = list(Path(data_path).rglob('*.csv'))               #Recursively search for csv files in data_path
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")

    dataframes = []
    for file in csv_files:                                          #Loads every csv file as a dataframe
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file, encoding=encoding, low_memory=False)
        dataframes.append(df)

    combined_dataframes = pd.concat(dataframes, ignore_index=True)  #Merges retrieved dataframes
    print(f"Total samples: {len(combined_dataframes)}")

    # EXPLORATION
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nClass distribution:")
    print(df[label_column].value_counts())

    missing = df.isnull().sum()
    print(f"\nMissing values per column: {missing[missing > 0]}")
    print(f"w\nData types:")
    print(df.dtypes.value_counts())

    return combined_dataframes, label_column

# Cleans dataset handling missing values, infinities and duplicates
def clean_dataset(df, label_column):
    print(f"\n\nCLEANING DATASET...")
    print(f"Initial shape: {df.shape}")

    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop columns with more than 50% missing values
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Separate column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill numeric columns with median
    for col in numeric_cols:
        if col != label_column and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    for col in categorical_cols:
        if col != label_column and df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])

    # Drop remaining NaN rows (should be very few)
    df = df.dropna()

    print(f"After cleaning: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")

    return df

# Removes uninformative features (constant, higly correlated) 
def features_cleanup(df, label_column):

    print("\n\nCLEANING UP FEATURES... ")
    print(f"Initial shape: {df.shape}")

    #Rimuovere feature NON informative
    NON_INFORMATIVE = ["Flow ID","Source IP","Destination IP","Source Port","Destination Port","Timestamp"]
    df.drop(columns=[c for c in NON_INFORMATIVE if c in df.columns])

    #Rimuovere feature costanti o quasi costanti
    numeric_df = df.select_dtypes(include=["number"])
    variances = numeric_df.var()
    keep_cols = variances[variances > 1e-5].index.tolist()
    df = df[keep_cols + [label_column]]

    # Rimuovere feature altamente correlate
    corr = df.drop(columns=label_column).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    df = df.drop(columns=[column for column in upper.columns if any(upper[column] > 0.95)])

    print(f"Final shape: {df.shape}")
    return df

# Preprocesses features encoding categoricals and normalizing numerics
def preprocess_features(df, label_column):
    
    print("\n\nPREPROCESSING FEATURES... ")

    # Separate features and labels
    X = df.drop(columns=[label_column])
    y = df[label_column].copy()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Handle categorical features in X
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        # print(f"Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Label vector shape: {y_encoded.shape}")
    
    return X_scaled, y_encoded, X.columns.tolist(), label_encoder, scaler

# ============================================================================
# PART 4: CREATE UNIFIED DATASET AND CLASS SCHEDULE
# ============================================================================

#Merges two datasets into a unified one with common class labels
# def create_unified_dataset(cicids_X, cicids_y, toniot_X, toniot_y, cicids_label_encoder, toniot_label_encoder):
#     """
#     Combine CICIDS and ToN-IoT datasets with unified class labels
    
#     Returns:
#         X, y, unified_label_mapping, class_names
#     """
#     print("\n\nCreating unified dataset...")
    
#     # Get class names from both datasets
#     cicids_classes = cicids_label_encoder.classes_
#     toniot_classes = toniot_label_encoder.classes_
    
#     print(f"CICIDS classes: {cicids_classes}")
#     print(f"ToN-IoT classes: {toniot_classes}")
    
#     # Create unified class mapping
#     # You'll need to manually map similar attack types
#     # This is a simplified example - adjust based on your actual classes
    
#     unified_mapping = {
#         # CICIDS mappings
#         'BENIGN': 0,
#         'DoS': 1,
#         'DDoS': 2,
#         'PortScan': 3,
#         'Bot': 4,
#         'Web Attack': 5,
#         'Infiltration': 6,
#         'Brute Force': 7,
#         # ToN-IoT mappings (map to similar CICIDS classes where possible)
#         'normal': 0,  # Same as BENIGN
#         'dos': 1,  # Same as DoS
#         'ddos': 2,  # Same as DDoS
#         'scanning': 3,  # Same as PortScan
#         'backdoor': 8,  # New class
#         'injection': 9,  # New class
#         'ransomware': 10,  # New class
#         'xss': 5,  # Same as Web Attack
#     }
    
#     print("\nNote: You need to manually verify and adjust the unified_mapping")
#     print("based on the actual classes in your datasets!")
    
#     return unified_mapping

#Splits the dataframe into train, val, test sets
def split_dataframe(X, y, test_size=0.2, val_size=0.1, random_state=42):

    print("\n\nSPLITTING DATASET...")

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp)
    
    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

#Saves processed data and preprocessing objects to disk
def save_processed_data(data_dict, label_encoder, scaler, feature_names, output_dir='../data/processed_data/'):

    print(f"\n\nSAVING PROCESSED DATA TO {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data splits
    np.save(os.path.join(output_dir, 'X_train.npy'), data_dict['X_train'])
    np.save(os.path.join(output_dir, 'y_train.npy'), data_dict['y_train'])
    np.save(os.path.join(output_dir, 'X_val.npy'), data_dict['X_val'])
    np.save(os.path.join(output_dir, 'y_val.npy'), data_dict['y_val'])
    np.save(os.path.join(output_dir, 'X_test.npy'), data_dict['X_test'])
    np.save(os.path.join(output_dir, 'y_test.npy'), data_dict['y_test'])
    
    # Save preprocessing objects
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Data saved successfully!")

# ------------------------------------------------------------------------------------------------------------------------------- MAIN EXECUTION 
if __name__ == "__main__":
    
    # Loading and cleanup of CICIDS-2017 dataset
    cicids_df,cicids_label_col = load_dataset('../data/datasets/CICIDS-2017/',' Label','latin1')

    if cicids_df is not None:
        cicids_df_clean = clean_dataset(cicids_df, cicids_label_col)
        cicids_df_clean = features_cleanup(cicids_df_clean,cicids_label_col)
        cicids_X, cicids_y, cicids_features, cicids_le, cicids_scaler = preprocess_features(cicids_df_clean, cicids_label_col)
    
    # Loading and cleanup of ToN-IoT datset
    # toniot_df,toniot_label_col = load_dataset('datasets/ToN-IoT','type')

    # if toniot_df is not None:
    #     toniot_df_clean = clean_dataset(toniot_df, toniot_label_col)
    #     toniot_X, toniot_y, toniot_features, toniot_le, toniot_scaler = preprocess_features(toniot_df_clean, toniot_label_col)
    
    # Merges the two dataframes
    # ....

    # Temporarly working only with CICIDS
    X = cicids_X 
    y = cicids_y
    le = cicids_le
    scaler = cicids_scaler
    features = cicids_features

    # Create splits and save
    splits = split_dataframe(X,y)
    save_processed_data(splits, le, scaler, features)