# src/data_loader_nasa.py

import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import glob

def extract_health_indicators(mat_file_path):
    """
    Parses a .mat file to extract Remaining Useful Life (RUL) labels and
    engineered Health Indicators (HIs) based on charging profiles, as inspired by academic research.
    """
    try:
        mat = scipy.io.loadmat(mat_file_path)
        data_key = [k for k in mat.keys() if not k.startswith('__')][0]
        all_cycles = mat[data_key][0, 0]['cycle'][0]
        
        charge_cycles_features = [] # To store HIs from charge cycles
        discharge_capacities = []   # To store the capacity label from discharge cycles

        # First, get all the capacity readings from discharge cycles to establish a timeline
        for i in range(all_cycles.shape[0]):
            cycle = all_cycles[i]
            cycle_type = cycle['type'][0]
            if cycle_type == 'discharge':
                cycle_data_struct = cycle['data'][0, 0]
                if 'Capacity' in cycle_data_struct.dtype.names:
                    discharge_capacities.append(cycle_data_struct['Capacity'][0][0])
        
        # Now, extract features from charge cycles, aligning them with the capacity timeline
        for i in range(all_cycles.shape[0]):
            cycle = all_cycles[i]
            cycle_type = cycle['type'][0]
            if cycle_type == 'charge':
                cycle_data_struct = cycle['data'][0, 0]
                voltage = cycle_data_struct['Voltage_measured'][0]
                time = cycle_data_struct['Time'][0]
                
                # As per research, we measure the time to charge through specific voltage ranges.
                # Find the index where voltage first crosses the thresholds.
                idx_3_8v = np.searchsorted(voltage, 3.8, side='left')
                idx_3_9v = np.searchsorted(voltage, 3.9, side='left')
                idx_4_0v = np.searchsorted(voltage, 4.0, side='left')
                idx_4_1v = np.searchsorted(voltage, 4.1, side='left')
                idx_4_2v = np.searchsorted(voltage, 4.2, side='left')
                
                # Ensure the indices are valid (i.e., the voltage was actually reached)
                if idx_4_2v < len(time):
                    # F1: Time in 3.8V-3.9V range
                    time_f1 = time[idx_3_9v] - time[idx_3_8v]
                    # F2: Time in 3.9V-4.0V range
                    time_f2 = time[idx_4_0v] - time[idx_3_9v]
                    # F3: Time in 4.0V-4.1V range
                    time_f3 = time[idx_4_1v] - time[idx_4_0v]
                    # F4: Time in 4.1V-4.2V range
                    time_f4 = time[idx_4_2v] - time[idx_4_1v]
                    
                    charge_cycles_features.append([time_f2, time_f3]) # Using F2 and F3 as per the paper's conclusion
        
        # We might have a mismatch in the number of charge/discharge cycles.
        # Truncate to the shorter length to ensure alignment.
        num_valid_cycles = min(len(charge_cycles_features), len(discharge_capacities))
        
        features = np.array(charge_cycles_features[:num_valid_cycles])
        labels = np.array(discharge_capacities[:num_valid_cycles])

        if features.shape[0] > 0:
             print(f"INFO: Successfully extracted {features.shape[0]} Health Indicator sets from {os.path.basename(mat_file_path)}.")
        else:
            print(f"WARNING: No valid Health Indicators found in {os.path.basename(mat_file_path)}.")
            
        return features, labels

    except Exception as e:
        print(f"CRITICAL ERROR: Could not process file {mat_file_path} for HIs. Error: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([])


class NASABatteryDataset(Dataset):
    """
    Creates a time-series dataset using engineered Health Indicators (HIs)
    to predict Remaining Useful Life (RUL).
    """
    def __init__(self, battery_files, sequence_length, scaler):
        self.sequence_length = sequence_length
        self.features = []
        self.labels = []

        for file_path in battery_files:
            # The new function returns features (HIs) and labels (capacities)
            features, capacities = extract_health_indicators(file_path)
            
            if features.shape[0] < sequence_length:
                continue
            
            # Scale the Health Indicator features
            scaled_features = scaler.transform(features)
            
            # RUL is the number of cycles left until the end
            total_cycles = len(capacities)
            rul = total_cycles - np.arange(total_cycles)
            
            # Create sequences of HIs
            for i in range(len(scaled_features) - sequence_length):
                self.features.append(scaled_features[i : i + sequence_length])
                self.labels.append(rul[i + sequence_length - 1])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Feature shape is now (sequence_length, num_HIs), e.g., (15, 2)
        # DataLoader will batch to (batch_size, 15, 2) -> Perfect for LSTM
        return (torch.tensor(self.features[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0))


def get_nasa_datasets(config):
    nasa_data_folder = config.get('nasa_data_folder', '1. BatteryAgingARC-FY08Q4')
    data_path = os.path.join(config['data_root'], '5. Battery Data Set', nasa_data_folder)
    
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"NASA data subfolder not found at: {data_path}")

    all_battery_files = sorted(glob.glob(os.path.join(data_path, '*.mat')))
    
    # --- Fit scaler on HIs from ALL batteries ---
    print("INFO (NASA): Fitting global scaler on Health Indicators from all batteries...")
    all_his = []
    for file_path in all_battery_files:
        features, _ = extract_health_indicators(file_path)
        if features.shape[0] > 0:
            all_his.append(features)
    
    if not all_his:
        raise ValueError(f"No valid Health Indicators found in {data_path} to fit the scaler.")
        
    scaler = MinMaxScaler().fit(np.vstack(all_his))
    
    # Partition data
    test_battery_files = [all_battery_files[2]] # B0007.mat
    train_battery_files = [f for f in all_battery_files if f not in test_battery_files]
    
    print(f"INFO (NASA): Using {os.path.basename(test_battery_files[0])} for the test set.")
    test_set = NASABatteryDataset(test_battery_files, config['sequence_length'], scaler)
    
    client_datasets = []
    for i in range(config['num_clients']):
        battery_for_client = [train_battery_files[i % len(train_battery_files)]]
        client_name = os.path.basename(battery_for_client[0])
        print(f"Client #{i} assigned NASA battery data: {client_name}")
        client_data = NASABatteryDataset(battery_for_client, config['sequence_length'], scaler)
        client_datasets.append(client_data)
    
    return client_datasets, test_set