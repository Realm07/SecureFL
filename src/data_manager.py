import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import glob
import scipy.io
from PyEMD import CEEMDAN
from torch.utils.data import TensorDataset
from .data_loader import partition_iid, load_arrhythmia_data

class DataManager:
    """
    A centralized manager to handle all data loading, preprocessing, and partitioning
    at server startup to avoid redundant operations.
    """
    def __init__(self, tasks_to_load, base_config_getter):
        self.task_data = {}
        self.task_configs = {}
        self.base_config_getter = base_config_getter
        self._load_and_prepare_all_tasks(tasks_to_load)

    def _load_and_prepare_all_tasks(self, tasks_to_load):
        print("\n" + "="*50)
        print("--- Centralized Data Preprocessing Initializing ---")
        for task_id in tasks_to_load:
            print(f"  > Loading and preparing data for task: '{task_id}'")
            if task_id == 'arrhythmia':
                self._prepare_arrhythmia()
            elif task_id == 'nasa_battery':
                self._prepare_nasa_battery()
        print("--- Data Preprocessing Complete ---")
        print("="*50 + "\n")


    def _prepare_arrhythmia(self):

        config = self.base_config_getter('arrhythmia')
        
        trainset, testset, _, _ = load_arrhythmia_data(config)
        
        self.task_configs['arrhythmia'] = config 

        client_datasets, _ = partition_iid(trainset, config['num_clients'])
        self.task_data['arrhythmia'] = {
            'client_partitions': client_datasets,
            'test_set': testset
        }
    
    def _prepare_nasa_battery(self):
        config = self.base_config_getter('nasa_battery')
        self.task_configs['nasa_battery'] = config
        nasa_folder = config.get('nasa_data_folder')
        data_path = os.path.join(config['data_root'], '5. Battery Data Set', nasa_folder)
        all_battery_files = sorted(glob.glob(os.path.join(data_path, '*.mat')))

        # 1. Fit scaler once on all raw capacity data
        print("    - Fitting global scaler on raw capacity data...")
        all_capacities = self._get_all_nasa_capacities(all_battery_files)
        scaler = MinMaxScaler().fit(np.array(all_capacities).reshape(-1, 1))
        
        # 2. Denoise all battery curves once
        print("    - Denoising all battery capacity curves...")
        denoised_curves = {
            os.path.basename(f): self._denoise_file(f) for f in all_battery_files
        }

        # 3. Create datasets using pre-denoised data and pre-fitted scaler
        test_file = all_battery_files[2] # B0007.mat
        train_files = [f for f in all_battery_files if f != test_file]
        
        test_set = self._create_nasa_dataset_from_denoised([test_file], config, scaler, denoised_curves)
        
        client_partitions = []
        for i in range(config['num_clients']):
            client_file = [train_files[i % len(train_files)]]
            client_dataset = self._create_nasa_dataset_from_denoised(client_file, config, scaler, denoised_curves)
            client_partitions.append(client_dataset)

        self.task_data['nasa_battery'] = {
            'client_partitions': client_partitions,
            'test_set': test_set
        }

    def _get_all_nasa_capacities(self, battery_files):
        capacities = []
        for file_path in battery_files:
            mat = scipy.io.loadmat(file_path)
            data_key = [k for k in mat.keys() if not k.startswith('__')][0]
            cycles = mat[data_key][0, 0]['cycle'][0]
            for i in range(cycles.shape[0]):
                cycle = cycles[i]
                if cycle['type'][0] == 'discharge' and 'Capacity' in cycle['data'][0, 0].dtype.names:
                    capacities.append(cycle['data'][0, 0]['Capacity'][0][0])
        return capacities

    def _denoise_file(self, file_path):
        capacities = self._get_all_nasa_capacities([file_path])
        if not capacities: return None
        ceemdan = CEEMDAN()
        imfs = ceemdan(np.array(capacities))
        return imfs[-1] # Return the residual trend

    def _create_nasa_dataset_from_denoised(self, files, config, scaler, denoised_curves):
        features, labels = [], []
        seq_len = config['sequence_length']
        
        for f in files:
            trend = denoised_curves.get(os.path.basename(f))
            if trend is None or len(trend) < seq_len + 1:
                continue
            
            scaled_trend = scaler.transform(trend.reshape(-1, 1))
            for i in range(len(scaled_trend) - seq_len):
                features.append(scaled_trend[i : i + seq_len])
                labels.append(scaled_trend[i + seq_len])
        
        print(f"    - Created {len(features)} sequences for {os.path.basename(files[0])}")
        
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32).view(-1, 1)
        return TensorDataset(features_tensor, labels_tensor)
    
    def get_client_data(self, task_id, client_id):
        return self.task_data[task_id]['client_partitions'][client_id]

    def get_test_set(self, task_id):
        return self.task_data[task_id]['test_set']
    
    def get_task_config(self, task_id): 
        """Returns the fully prepared and updated config for a task."""
        return self.task_configs[task_id]