# In src/config.py

import torch
import os

def get_config(dataset_name="mnist"):
    """Returns the configuration dictionary for a given dataset."""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    config = {
        'num_clients': 10,
        'clients_per_round': 3,
        'batch_size': 32,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'data_root': os.path.join(PROJECT_ROOT, 'src', 'data'),
        'results_dir': os.path.join(PROJECT_ROOT, 'results'),
    }

    if dataset_name == "arrhythmia":
        config.update({
            'dataset_name': 'arrhythmia', 'model_name': 'mlp',
            'learning_mode': 'synchronous',  # This task will run in the classic round-based mode
            'model_save_path': 'trained_arrhythmia_model.pth', 'num_rounds': 25,
            'local_epochs': 5, 'learning_rate': 0.001, 'optimizer': 'adam',
            'weight_decay': 1e-5, 'metric': 'accuracy',
            'delta': 1e-5,
            'num_features': 13,
            'num_classes': 2,
            'encrypted_layers': [
                'layer_3.weight', 'layer_3.bias',
                'layer_out.weight', 'layer_out.bias'
            ],
            'dp_noise_multiplier': 1.5, 'dp_max_grad_norm': 1.2
        })
    elif dataset_name == "nasa_battery":
        config.update({
            'dataset_name': 'nasa_battery', 
            'model_name': 'lstm_attention',
            'learning_mode': 'asynchronous', 
            'aggregation_interval_seconds': 30, 
            'min_updates_for_aggregation': 2,  
            'nasa_data_folder': '1. BatteryAgingARC-FY08Q4',
            'model_save_path': 'trained_battery_model.pth', 'num_rounds': 25,
            'local_epochs': 20,
            'learning_rate': 0.001,
            'batch_size': 16,
            'lstm_drop_prob': 0.2,
            'lstm_hidden_dim': 64,
            'lstm_n_layers': 2,
            'num_features': 1,
            'optimizer': 'adam',
            'metric': 'rmse',
            'sequence_length': 10,
            'encrypted_layers': ['fc.weight', 'fc.bias', 'attention_layer.0.weight', 'attention_layer.0.bias', 'attention_layer.2.weight', 'attention_layer.2.bias'],
        })
    else: # This block is for MNIST
        config.update({
            'dataset_name': dataset_name,
            'model_name': 'mlp',
            'learning_mode': 'synchronous',
            'num_rounds': 10,
            'local_epochs': 10,
            'learning_rate': 0.01,
            'optimizer': 'adam', # Baseline uses Adam, will be overridden for DP run
            'metric': 'accuracy',
            'num_features': 784,
            # --- CRITICAL CHANGE HERE ---
            # A larger batch size is crucial for stabilizing gradients in DP-SGD with Opacus.
            # It creates a better signal-to-noise ratio.
            'batch_size': 128, 
        })

    return config