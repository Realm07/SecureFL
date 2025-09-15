# src/data_loader_nasa.py

import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import glob
from PyEMD import CEEMDAN # Import the denoising algorithm

def denoise_capacity_with_ceemdan(capacities):
    """
    Takes a noisy capacity array, decomposes it using CEEMDAN, and returns the
    smooth residual trendline, which is a powerful feature for prediction.
    """
    ceemdan = CEEMDAN()
    imfs = ceemdan(capacities)
    # The last IMF is the residual trend. This is our denoised signal.
    return imfs[-1]

class NASABatteryDataset(Dataset):
    """
    Creates a time-series forecasting dataset.
    Input: A sequence of denoised capacity values.
    Label: The next capacity value in the sequence.
    """
    def __init__(self, battery_files, sequence_length, scaler):
        self.features = []
        self.labels = []

        for file_path in battery_files:
            # First, extract just the raw capacity data
            mat = scipy.io.loadmat(file_path)
            data_key = [k for k in mat.keys() if not k.startswith('__')][0]
            all_cycles = mat[data_key][0, 0]['cycle'][0]
            
            raw_capacities = []
            for i in range(all_cycles.shape[0]):
                cycle = all_cycles[i]
                if cycle['type'][0] == 'discharge':
                    data_struct = cycle['data'][0, 0]
                    if 'Capacity' in data_struct.dtype.names:
                        raw_capacities.append(data_struct['Capacity'][0][0])
            
            if len(raw_capacities) < sequence_length + 1:
                continue

            # Denoise the entire capacity curve for this battery
            denoised_capacity_trend = denoise_capacity_with_ceemdan(np.array(raw_capacities))
            
            # Scale the denoised data
            scaled_trend = scaler.transform(denoised_capacity_trend.reshape(-1, 1))

            # Create sequences for time-series forecasting
            for i in range(len(scaled_trend) - sequence_length):
                self.features.append(scaled_trend[i : i + sequence_length])
                self.labels.append(scaled_trend[i + sequence_length])
        
        if self.features:
            print(f"INFO: Created {len(self.features)} denoised sequences from {os.path.basename(file_path)}.")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Feature shape: (sequence_length, 1)
        # Label shape: (1,)
        return (torch.tensor(self.features[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.float32).view(1))

def get_nasa_datasets(config):
    nasa_data_folder = config.get('nasa_data_folder')
    data_path = os.path.join(config['data_root'], '5. Battery Data Set', nasa_data_folder)
    all_battery_files = sorted(glob.glob(os.path.join(data_path, '*.mat')))
    
    # Fit scaler on raw capacity data from ALL batteries
    print("INFO (NASA): Fitting global scaler on raw capacity from all batteries...")
    all_capacities = []
    for file_path in all_battery_files:
        mat = scipy.io.loadmat(file_path)
        data_key = [k for k in mat.keys() if not k.startswith('__')][0]
        all_cycles = mat[data_key][0, 0]['cycle'][0]
        for i in range(all_cycles.shape[0]):
             cycle = all_cycles[i]
             if cycle['type'][0] == 'discharge':
                 data_struct = cycle['data'][0, 0]
                 if 'Capacity' in data_struct.dtype.names:
                     all_capacities.append(data_struct['Capacity'][0][0])

    scaler = MinMaxScaler().fit(np.array(all_capacities).reshape(-1, 1))
    
    # Partition data
    test_battery_files = [all_battery_files[2]] # B0007.mat
    train_battery_files = [f for f in all_battery_files if f not in test_battery_files]
    
    print(f"INFO (NASA): Using {os.path.basename(test_battery_files[0])} for the test set.")
    test_set = NASABatteryDataset(test_battery_files, config['sequence_length'], scaler)
    
    client_datasets = []
    for i in range(config['num_clients']):
        battery_for_client = [train_battery_files[i % len(train_battery_files)]]
        client_data = NASABatteryDataset(battery_for_client, config['sequence_length'], scaler)
        client_datasets.append(client_data)
    
    return client_datasets, test_set