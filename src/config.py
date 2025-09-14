import torch
import os

def get_config(dataset_name="mnist"):
    """Returns the configuration dictionary for a given dataset."""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    config = {
        'num_clients': 10,
        'clients_per_round': 3, # Adjusted for the smaller number of batteries
        'batch_size': 32,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'data_root': os.path.join(PROJECT_ROOT, 'src', 'data'),
        'results_dir': os.path.join(PROJECT_ROOT, 'results'),
    }

    if dataset_name == "mnist":
        config.update({
            'dataset_name': 'mnist', 'model_name': 'cnn',
            'model_save_path': 'trained_mnist_model.pth', 'num_rounds': 5,
            'local_epochs': 2, 'learning_rate': 0.01, 'optimizer': 'sgd',
            'encrypted_layers': ['fc2.weight', 'fc2.bias'],
            'dp_noise_multiplier': 1.1, 'dp_max_grad_norm': 1.0
        })
    elif dataset_name == "arrhythmia":
        config.update({
            'dataset_name': 'arrhythmia', 'model_name': 'mlp',
            'model_save_path': 'trained_arrhythmia_model.pth', 'num_rounds': 25,
            'local_epochs': 5, 'learning_rate': 0.001, 'optimizer': 'adam',
            'weight_decay': 1e-5, 'metric': 'accuracy',
            # --- ADD THIS LINE ---
            'delta': 1e-5, # A standard value for DP
            # ---------------------
            'encrypted_layers': [
                'layer_3.weight', 'layer_3.bias',
                'layer_out.weight', 'layer_out.bias'
            ],
            'dp_noise_multiplier': 1.5, 'dp_max_grad_norm': 1.2
        })
    elif dataset_name == "nasa_battery":
        config.update({
            'dataset_name': 'nasa_battery', 
            'model_name': 'lstm_battery',
            'nasa_data_folder': '1. BatteryAgingARC-FY08Q4',
            'model_save_path': 'trained_battery_model.pth', 'num_rounds': 50,
            
            # --- TUNING ADJUSTMENT FOR STABILITY ---
            'local_epochs': 5,              # REDUCED from 15. This is the biggest factor.
            'learning_rate': 0.001,         # Keep this low for stability.
            'batch_size': 64,               # Increased slightly for faster epochs.
            'lstm_drop_prob': 0.3,          # Reduced slightly.
            
            # Make the model a bit smaller to start.
            'lstm_hidden_dim': 64,   # REDUCED from 128
            'lstm_n_layers': 2,      # REDUCED from 3
            # -----------------------------------------
            
            'optimizer': 'adam',
            'metric': 'rmse',
            'sequence_length': 15,
            'num_features': 300,
            'encrypted_layers': ['fc.weight', 'fc.bias'],
            'dp_noise_multiplier': 0.5,
            'dp_max_grad_norm': 2.0
        })


    else:
        raise ValueError(f"Unknown dataset configuration: {dataset_name}")

    return config