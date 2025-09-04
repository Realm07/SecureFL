# src/config.py

import torch
import os

def get_config(dataset_name="mnist"):
    """Returns the configuration dictionary for a given dataset."""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    config = {
        'num_clients': 10,
        'clients_per_round': 5,
        'batch_size': 32,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
<<<<<<< Updated upstream
        # --- NEW: Standardized results directory ---
        'results_dir': 'results',
        'data_root': 'src/data'
=======
        'data_root': os.path.join(PROJECT_ROOT, 'src', 'data'),
        'results_dir': os.path.join(PROJECT_ROOT, 'results'),
>>>>>>> Stashed changes
    }

    if dataset_name == "mnist":
        config['dataset_name'] = 'mnist'
        config['model_name'] = 'cnn'
<<<<<<< Updated upstream
        config['model_save_path'] = 'trained_mnist_model.pth' # Filename only
=======
        config['model_save_path'] = 'trained_mnist_model.pth'
>>>>>>> Stashed changes
        config['num_rounds'] = 5
        config['local_epochs'] = 2
        config['learning_rate'] = 0.01
        config['optimizer'] = 'sgd'
        config['encrypted_layers'] = ['fc2.weight', 'fc2.bias']
<<<<<<< Updated upstream
        # --- NEW: DP parameters for MNIST ---
=======
>>>>>>> Stashed changes
        config['dp_noise_multiplier'] = 1.1
        config['dp_max_grad_norm'] = 1.0

    elif dataset_name == "arrhythmia":
        config['dataset_name'] = 'arrhythmia'
        config['model_name'] = 'mlp'
<<<<<<< Updated upstream
        config['model_save_path'] = 'trained_arrhythmia_model.pth' # Filename only
=======
        config['model_save_path'] = 'trained_arrhythmia_model.pth'
>>>>>>> Stashed changes
        config['num_rounds'] = 25
        config['local_epochs'] = 5
        config['learning_rate'] = 0.001
        config['optimizer'] = 'adam'
        config['weight_decay'] = 1e-4
        config['lr_scheduler_step_size'] = 5
        config['lr_scheduler_gamma'] = 0.5
        config['encrypted_layers'] = [
<<<<<<< Updated upstream
            'layer_3.weight', 'layer_3.bias', 
            'layer_out.weight', 'layer_out.bias'
        ]
        # --- NEW: DP parameters for Arrhythmia ---
=======
            'layer_3.weight', 'layer_3.bias',
            'layer_out.weight', 'layer_out.bias'
        ]
>>>>>>> Stashed changes
        config['dp_noise_multiplier'] = 1.5
        config['dp_max_grad_norm'] = 1.2

    else:
        raise ValueError(f"Unknown dataset configuration: {dataset_name}")

    return config