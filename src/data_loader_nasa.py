# src/data_loader_nasa.py

import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import glob

# The load_nasa_battery_data function is now correct and does not need changes.
def load_nasa_battery_data(mat_file_path):
    # ... (no changes here)
    try:
        mat = scipy.io.loadmat(mat_file_path)
        data_key = [k for k in mat.keys() if not k.startswith('__')][0]
        all_cycles = mat[data_key][0, 0]['cycle'][0]
        cycles_data = []
        capacities = []
        for i in range(all_cycles.shape[0]):
            cycle = all_cycles[i]
            cycle_type = cycle['type'][0]
            if cycle_type != 'discharge':
                continue
            cycle_data_struct = cycle['data'][0, 0]
            if 'Capacity' not in cycle_data_struct.dtype.names:
                continue
            capacity = cycle_data_struct['Capacity'][0][0]
            required_fields = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
            if not all(field in cycle_data_struct.dtype.names for field in required_fields):
                continue
            voltage = cycle_data_struct['Voltage_measured']
            current = cycle_data_struct['Current_measured']
            temperature = cycle_data_struct['Temperature_measured']
            if voltage.shape[1] > 100:
                cycles_data.append({
                    'voltage': voltage[0, :100].flatten(),
                    'current': current[0, :100].flatten(),
                    'temperature': temperature[0, :100].flatten(),
                })
                capacities.append(capacity)
        if capacities:
            print(f"INFO: Successfully extracted {len(capacities)} valid discharge cycles from {os.path.basename(mat_file_path)}.")
        else:
            print(f"WARNING: No valid discharge cycles found in {os.path.basename(mat_file_path)}.")
        return cycles_data, capacities
    except Exception as e:
        print(f"CRITICAL ERROR: Could not process file {mat_file_path}. Error: {e}")
        import traceback
        traceback.print_exc()
        return [], []


class NASABatteryDataset(Dataset):
    def __init__(self, battery_files, sequence_length, scaler):
        self.sequence_length = sequence_length
        self.features = []
        self.labels = []

        for file_path in battery_files:
            cycles, capacities = load_nasa_battery_data(file_path)
            if not cycles or len(capacities) < sequence_length:
                continue
            
            total_cycles = len(capacities)
            rul = total_cycles - np.arange(total_cycles)
            
            scaled_features_list = []
            for cycle in cycles:
                # Shape of combined is (100, 3)
                combined = np.vstack([cycle['voltage'], cycle['current'], cycle['temperature']]).T
                scaled_features = scaler.transform(combined)
                scaled_features_list.append(scaled_features)
            
            for i in range(len(scaled_features_list) - sequence_length):
                # sequence is a list of 15 arrays, each of shape (100, 3)
                sequence_arrays = scaled_features_list[i : i + sequence_length]
                
                # --- DEFINITIVE FIX: FLATTEN FEATURES FOR EACH TIMESTEP ---
                # We want each of the 15 timesteps to have a feature vector of size 300.
                # The final shape of the sequence should be (15, 300).
                flattened_sequence = np.array([arr.flatten() for arr in sequence_arrays])
                # -----------------------------------------------------------

                self.features.append(flattened_sequence)
                self.labels.append(rul[i + sequence_length - 1])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # self.features[idx] now has the correct 2D shape (sequence_length, features_per_timestep)
        # e.g., (15, 300)
        # DataLoader will batch this to a 3D tensor (batch_size, 15, 300), which is what LSTM expects.
        return (torch.tensor(self.features[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0))


# The get_nasa_datasets function is also correct and does not need changes.
def get_nasa_datasets(config):
    # ... (no changes here)
    nasa_data_folder = config.get('nasa_data_folder', '1. BatteryAgingARC-FY08Q4')
    data_path = os.path.join(config['data_root'], '5. Battery Data Set', nasa_data_folder)
    
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"NASA data subfolder not found at: {data_path}")

    all_battery_files = sorted(glob.glob(os.path.join(data_path, '*.mat')))
    print(f"INFO (NASA): Found {len(all_battery_files)} battery files in '{nasa_data_folder}'.")

    print("INFO (NASA): Fitting global scaler on all batteries in the dataset...")
    all_features = []
    for file_path in all_battery_files:
        cycles, _ = load_nasa_battery_data(file_path)
        for cycle in cycles:
            all_features.append(np.vstack([cycle['voltage'], cycle['current'], cycle['temperature']]).T)
    
    if not all_features:
        raise ValueError(f"No valid cycle data found in {data_path} to fit the scaler.")
        
    scaler = MinMaxScaler().fit(np.vstack(all_features))
    
    if len(all_battery_files) < 2:
        raise ValueError("Need at least 2 battery files for a train/test split.")

    test_battery_files = [all_battery_files[2]]
    train_battery_files = [f for f in all_battery_files if f not in test_battery_files]
    
    print(f"INFO (NASA): Using {os.path.basename(test_battery_files[0])} for the test set.")
    
    test_set = NASABatteryDataset(test_battery_files, config['sequence_length'], scaler)
    
    num_clients = config['num_clients']
    client_datasets = []
    for i in range(num_clients):
        battery_for_client = [train_battery_files[i % len(train_battery_files)]]
        client_name = os.path.basename(battery_for_client[0])
        print(f"Client #{i} assigned NASA battery data: {client_name}")
        client_data = NASABatteryDataset(battery_for_client, config['sequence_length'], scaler)
        client_datasets.append(client_data)
    
    return client_datasets, test_set
