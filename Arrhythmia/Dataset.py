import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

def load_arrhythmia_dataset(path="arrhythmia.data"):
    # Load dataset, treating '?' as NaN
    df = pd.read_csv(path, header=None, na_values="?")

    # Fill missing values (filling with column means, for example)
    df.fillna(df.mean(), inplace=True)

    # Separate features and labels
    features = df.iloc[:, :-1].astype(np.float32)
    labels = df.iloc[:, -1].astype(int)  # int so we can easily compare

    # Drop columns with zero variance
    features = features.loc[:, features.std() > 0]

    # Z-score normalization
    features = (features - features.mean()) / features.std()

    # Adjust to 274 features
    current_dim = features.shape[1]
    if current_dim != 274:
        print(f"[INFO] Feature dimension {current_dim} ≠ 274. Adjusting...")
        if current_dim > 274:
            features = features.iloc[:, :274]
        else:
            pad_width = 274 - current_dim
            padding = np.zeros((features.shape[0], pad_width), dtype=np.float32)
            features = np.hstack((features.values, padding))
            features = pd.DataFrame(features)

    # Convert to tensors
    features = torch.tensor(features.values, dtype=torch.float32)
    labels = torch.tensor(labels.values, dtype=torch.int64)

    # Define anomaly labels
    anomaly_labels = {3, 4, 5, 7, 8, 9, 14, 15}

    # Create a new labels tensor where normal labels are 1, anomalies are 0
    labels_normal_anomaly = torch.where(
        torch.isin(labels, torch.tensor(list(anomaly_labels), dtype=torch.int64)),
        torch.tensor(0, dtype=torch.int64),  # Anomalous class
        torch.tensor(1, dtype=torch.int64)   # Normal class
    )

    # Split data based on normal and anomaly
    normal_mask = labels_normal_anomaly == 1
    anomaly_mask = labels_normal_anomaly == 0

    normal_data = TensorDataset(features[normal_mask], labels_normal_anomaly[normal_mask])
    anomaly_data = TensorDataset(features[anomaly_mask], labels_normal_anomaly[anomaly_mask])

    # Combine normal and anomaly data for training/testing purposes
    all_data = TensorDataset(features, labels_normal_anomaly)

    # Use all normal samples for training
    train_dataset = normal_data

    # Randomly select 50% of all data (including normal and anomaly) for testing
    test_size = len(all_data) // 2
    _, test_dataset = random_split(all_data, [len(all_data) - test_size, test_size])

    return train_dataset, test_dataset

# Example usage
train_dataset, test_dataset = load_arrhythmia_dataset()

# Print sizes of train and test datasets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")