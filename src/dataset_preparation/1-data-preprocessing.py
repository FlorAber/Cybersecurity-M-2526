
# Data Preprocessing Pipeline for CICIDS-2017 and ToN-IoT
# Takes the two datasets, cleans them removing missing values and hyghly correlated features, merges them and creates a new csv file

import pandas as pd                                             # type: ignore
import numpy as np                                              # type: ignore
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.model_selection import train_test_split            # type: ignore
import matplotlib.pyplot as plt                                 # type: ignore
import seaborn as sns                                           # type: ignore

from pathlib import Path

# Loads and explores CICIDS-2017 dataset from given path
def load_cicids_dataset(data_path='datasets/CICIDS-2017/', label_column='Label',encoding='latin1'):

    print(f"\n\nLOADING CICIDS DATASET AT PATH {data_path}...")
    csv_files = list(Path(data_path).rglob('*.csv'))               #Recursively search for csv files in data_path
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return None
    
    print(f"\nFound {len(csv_files)} CSV files")

    dataframes = []
    for file in csv_files:                                          #Loads every csv file as a dataframe
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file, encoding=encoding, low_memory=False)
        dataframes.append(df)

    combined_dataframes = pd.concat(dataframes, ignore_index=True)  #Merges retrieved dataframes
    print(f"Total samples: {len(combined_dataframes)}")

    combined_dataframes.columns = combined_dataframes.columns.str.strip()

    # EXPLORATION
    print(f"\nDataset shape: {combined_dataframes.shape}")

    # Class encoding fixing
    combined_dataframes[label_column] = combined_dataframes[label_column].str.encode('ascii', errors='ignore').str.decode('ascii')
    mapping = {
        'Web Attack  Brute Force': 'Brute Force',
        'Web Attack  XSS': 'XSS',
        'Web Attack  Sql Injection': 'Sql Injection'
    }
    combined_dataframes[label_column] = combined_dataframes[label_column].replace(mapping)

    # Merges and renames classes with dictionary
    combined_dataframes[label_column] = combined_dataframes[label_column].replace({
        "BENIGN" : "Normal",
        "FTP-Patator" : "Brute Force",
        "SSH-Patator" : "Brute Force",
        "DoS GoldenEye" : "DoS",
        "DoS Hulk" : "DoS",
        "DoS slowloris" : "DoS",
        "DoS Slowhttptest" : "DoS",
        "PortScan" : "Scanning",
        "Sql Injection" : "Injection",
    })

    print(f"\nClass distribution:")
    print(combined_dataframes[label_column].value_counts())

    missing = combined_dataframes.isnull().sum()
    print(f"\nMissing values per column:\n {missing[missing > 0]}")
    print(f"\nData types:")
    print(combined_dataframes.dtypes.value_counts())

    return combined_dataframes, label_column

# Loads and explores ToN-IoT dataset from given path
def load_toniot_dataset(data_path='datasets/ToN-IoT/',label_column='type',encoding='utf-8'):
    # LOADING
    print(f"\n\nLOADING TONIOT DATASET AT PATH {data_path}...")
    csv_files = list(Path(data_path).rglob('*.csv'))               #Recursively search for csv files in data_path
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return None
    
    print(f"\nFound {len(csv_files)} CSV files")

    dataframes = []
    for file in csv_files:                                          #Loads every csv file as a dataframe
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file, encoding=encoding, low_memory=False)
        dataframes.append(df)

    combined_dataframes = pd.concat(dataframes, ignore_index=True)  #Merges retrieved dataframes
    print(f"Total samples: {len(combined_dataframes)}")

    combined_dataframes.columns = combined_dataframes.columns.str.strip()

    # EXPLORATION
    print(f"\nDataset shape: {combined_dataframes.shape}")

    # Merges and renames classes with dictionary
    combined_dataframes[label_column] = combined_dataframes[label_column].replace({
        "normal" : "Normal",
        "password" : "Brute Force",
        "ddos" : "DDoS",
        "dos" : "DoS",
        "scanning" : "Scanning",
        "mitm" : "MITM",
        "injection" : "Injection",
        "xss" : "XSS",
        "ransomware" : "Ransomware",
        "backdoor" : "Backdoor",
    })

    print(f"\nClass distribution:")
    print(combined_dataframes[label_column].value_counts())

    missing = combined_dataframes.isnull().sum()
    print(f"\nMissing values per column:\n {missing[missing > 0]}")
    print(f"\nData types:")
    print(combined_dataframes.dtypes.value_counts())

    return combined_dataframes, label_column

# Cleans dataset handling missing values, infinities and duplicates
def clean_dataset(df, label_column):
    print(f"\nCLEANING DATASET...")
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

# Removes constant and highly correlated features 
def features_cleanup(df, label_column, corr_threshold=0.95, memory_efficient=True):

    print("\nCLEANING UP FEATURES... ")
    print(f"Initial shape: {df.shape}")

    # Convert numeric columns to float32 to save memory
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if memory_efficient:
        for col in numeric_cols:
            if col != label_column:
                df[col] = df[col].astype('float32')

    # Removes low-variance features
    numeric_df = df.select_dtypes(include=["number"])
    if label_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[label_column])

    variances = numeric_df.var()
    keep_cols = variances[variances > 1e-5].index.tolist()
    df = df[keep_cols + [label_column]]
    print(f"After low-variance filter: {df.shape}")

    # Removes highly correlated features (memory-efficient approach)
    if memory_efficient and len(keep_cols) > 100:
        # For large feature sets, compute correlations in batches
        print("Using memory-efficient correlation filtering...")
        features_to_drop = set()
        features = keep_cols
        
        for i, col1 in enumerate(features):
            if col1 in features_to_drop:
                continue
            # Compute correlation only with remaining features
            remaining_features = [c for c in features[i+1:] if c not in features_to_drop]
            if remaining_features:
                corr_vals = df[col1].corr(df[remaining_features]).abs()
                high_corr = corr_vals[corr_vals > corr_threshold].index.tolist()
                features_to_drop.update(high_corr)
        
        df = df.drop(columns=list(features_to_drop), errors='ignore')
    else:
        # Original approach for smaller datasets
        corr = df.drop(columns=label_column).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        df = df.drop(columns=[column for column in upper.columns if any(upper[column] > corr_threshold)])

    print(f"Final shape: {df.shape}")
    return df

# Merges two datasets (CICIDS and NoT-IoT) into a unified one with common class labels
def unify_datasets(df_cicids, label_col_cicids, df_toniot, label_col_toniot):

    print("\n\nCREATING UNIFIED DATASET...")

    unified_label = 'Label'
    df_cicids = df_cicids
    df_toniot = df_toniot

    # Renaming label columns to unified name
    df_cicids = df_cicids.rename(columns={label_col_cicids: unified_label})
    df_toniot = df_toniot.rename(columns={label_col_toniot: unified_label})

    # Dropping columns not present in both datasets
    common_features = list(set(df_cicids.columns) & set(df_toniot.columns))
    df_cicids = df_cicids[common_features]
    df_toniot = df_toniot[common_features]

    unified_df = pd.concat([df_cicids, df_toniot], ignore_index=True)

    # Diagnostica
    print(f"Unified dataset shape: {unified_df.shape}")
    print("\nUnified class distribution:")
    print(unified_df[unified_label].value_counts())

    print("\nCREATING UNIFIED DATASET : DONE")

    return unified_df, unified_label

#Saves processed data and preprocessing objects to disk
def save_processed_data(data_dict, label_encoder, scaler, feature_names, output_dir='../../data/processed_data/'):

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
    # Loading and cleanup datasets

    # Loading and cleanup of CICIDS dataset
    cicids_df,cicids_label_col = load_cicids_dataset('../../data/datasets/CICIDS-2017/','Label','latin1')
    # cicids_df,cicids_label_col = load_cicids_dataset('../../data/datasets/CICIDS-2017/singolo','Label','latin1')       #TEST WITH A SINGLE FILE
    if cicids_df is not None:
        cicids_df_clean = clean_dataset(cicids_df, cicids_label_col)
        # cicids_df_clean = features_cleanup(cicids_df_clean,cicids_label_col)                                                   #DOPO L'UNIONE
    
    # Loading and cleanup of ToN-IoT dataset
    toniot_df,toniot_label_col = load_toniot_dataset('../../data/datasets/ToN-IoT/cicflowmeter_cicids_no_unknown','type')
    # toniot_df,toniot_label_col = load_toniot_dataset('../../data/datasets/ToN-IoT/singolo','type')                     #TEST WITH A SINGLE FILE
    if toniot_df is not None:
        toniot_df_clean = clean_dataset(toniot_df, toniot_label_col)
        # toniot_df_clean = features_cleanup(toniot_df_clean,toniot_label_col)                                                   #DOPO L'UNIONE                   

    # print(cicids_df.head())
    # print(toniot_df.head())

    # Temporarly working only with CICIDS
    # X = cicids_X 
    # y = cicids_y
    # le = cicids_le
    # scaler = cicids_scaler
    # features = cicids_features

    # Temporarly working only with ToN-IoT
    # X = toniot_X 
    # y = toniot_y
    # le = toniot_le
    # scaler = toniot_scaler
    # features = toniot_features

    #Merging datasets into a unified one
    unified_df, unified_label_col = unify_datasets(cicids_df_clean, cicids_label_col, toniot_df_clean, toniot_label_col)
    
    # Cleanup of unified dataset
    unified_df = features_cleanup(unified_df, unified_label_col)

    unified_df.to_csv('../../data/datasets/unified_dataset.csv', index=False)

