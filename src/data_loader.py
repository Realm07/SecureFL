
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def partition_iid(dataset, num_clients):
    if num_clients == 0: return [], {}
    num_items = len(dataset) // num_clients
    if num_items == 0:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        client_data = [Subset(dataset, [indices[i]]) for i in range(len(dataset))]
        client_data.extend([Subset(dataset, []) for _ in range(num_clients - len(dataset))])
        return client_data, {}
    dict_users, all_idxs = {}, list(range(len(dataset)))
    random.shuffle(all_idxs)
    for i in range(num_clients):
        start_idx = i * num_items
        end_idx = start_idx + num_items
        dict_users[i] = set(all_idxs[start_idx:end_idx])
    client_data = [Subset(dataset, list(dict_users[i])) for i in range(num_clients)]
    return client_data, dict_users


def load_arrhythmia_data(config):
    """Loads, preprocesses, and splits the Arrhythmia dataset."""
    try:
        df = pd.read_csv(os.path.join(config['data_root'], 'arrhythmia.csv'))
    except FileNotFoundError:
        print("\n!!! ERROR: arrhythmia.csv not found!")
        print(f"!!! Please download it from Kaggle and place it in the '{config['data_root']}' directory.")
        return None, None, None, None

    df.replace('?', np.nan, inplace=True)
    df.dropna(axis=1, inplace=True)

    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1].values
    y = np.where(y == 1, 0, 1)
    
    feature_names = X.columns.tolist()

    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_original = X_test.copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trainset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    testset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    config['num_features'] = X_train_scaled.shape[1]
    config['num_classes'] = len(np.unique(y_train))
    config['feature_names'] = feature_names
    config['scaler'] = scaler
    
    print(f"Arrhythmia data loaded. Num features: {config['num_features']}, Num classes: {config['num_classes']}")
    
    return trainset, testset, X_test_original, y_test


def get_datasets(config):
    """Factory function to return train and test datasets."""
    dataset_name = config['dataset_name']
    data_root = config['data_root']

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        
        config['num_classes'] = 10
        return trainset, testset

    elif dataset_name == 'arrhythmia':
        return load_arrhythmia_data(config)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")