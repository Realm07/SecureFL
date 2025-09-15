import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import glob
import scipy.io
from PyEMD import CEEMDAN

from .data_loader import partition_iid, load_arrhythmia_data

class ClientDataManager:
    """
    Manages data loading, preprocessing, and caching on the client side.
    Ensures that heavy preprocessing for datasets like NASA Battery happens only once.
    """
    def __init__(self):
        self._cache = {} # Cache for storing final DataLoaders

    def get_dataloader(self, client_id: int, config: dict) -> DataLoader:
        dataset_name = config['dataset_name']
        cache_key = f"{client_id}_{dataset_name}"

        if cache_key in self._cache:
            print(f"Client #{client_id}: Loading '{dataset_name}' data from permanent cache.")
            return self._cache[cache_key]

        print(f"Client #{client_id}: First time loading '{dataset_name}'. Performing one-time setup...")
        
        if dataset_name == 'arrhythmia':
            dataloader = self._prepare_arrhythmia(client_id, config)
        elif dataset_name == 'nasa_battery':
            dataloader = self._prepare_nasa_battery(client_id, config)
        else:
            raise ValueError(f"Unknown dataset for client data manager: {dataset_name}")

        self._cache[cache_key] = dataloader
        return dataloader

    def _prepare_arrhythmia(self, client_id: int, config: dict) -> DataLoader:
        trainset, _, _, _ = load_arrhythmia_data(config)
        client_datasets, _ = partition_iid(trainset, config['num_clients'])
        my_dataset = client_datasets[client_id]
        print(f"Client #{client_id}: Data loaded for 'arrhythmia'. {len(my_dataset)} samples.")
        return DataLoader(my_dataset, batch_size=config['batch_size'], shuffle=True)

    def _prepare_nasa_battery(self, client_id: int, config: dict) -> DataLoader:
        nasa_folder = config.get('nasa_data_folder')
        data_path = os.path.join(config['data_root'], '5. Battery Data Set', nasa_folder)
        all_battery_files = sorted(glob.glob(os.path.join(data_path, '*.mat')))

        # This client is assigned one specific file based on its ID
        my_file_path = all_battery_files[client_id % len(all_battery_files)]
        
        print(f"  - Client #{client_id} assigned battery file: {os.path.basename(my_file_path)}")
        
        # 1. Denoise the specific battery curve for this client
        print(f"  - Denoising capacity curve for {os.path.basename(my_file_path)}...")
        capacities = self._get_capacities_from_file(my_file_path)
        ceemdan = CEEMDAN()
        imfs = ceemdan(np.array(capacities))
        denoised_trend = imfs[-1]

        # 2. Create sequences from the denoised trend
        scaler = MinMaxScaler()
        scaled_trend = scaler.fit_transform(denoised_trend.reshape(-1, 1))
        
        features, labels = [], []
        seq_len = config['sequence_length']
        for i in range(len(scaled_trend) - seq_len):
            features.append(scaled_trend[i : i + seq_len])
            labels.append(scaled_trend[i + seq_len])
            
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32).view(-1, 1)
        my_dataset = TensorDataset(features_tensor, labels_tensor)

        print(f"Client #{client_id}: Data loaded for 'nasa_battery'. {len(my_dataset)} samples.")
        return DataLoader(my_dataset, batch_size=config['batch_size'], shuffle=True)

    def _get_capacities_from_file(self, file_path):
        capacities = []
        mat = scipy.io.loadmat(file_path)
        data_key = [k for k in mat.keys() if not k.startswith('__')][0]
        cycles = mat[data_key][0, 0]['cycle'][0]
        for i in range(cycles.shape[0]):
            cycle = cycles[i]
            if cycle['type'][0] == 'discharge' and 'Capacity' in cycle['data'][0, 0].dtype.names:
                capacities.append(cycle['data'][0, 0]['Capacity'][0][0])
        return capacities