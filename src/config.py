import torch

def get_config(dataset_name="mnist"):
    """Returns the configuration dictionary for a given dataset."""

    config = {
        'num_clients': 10,
        'clients_per_round': 5,
        'batch_size': 32,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'data_root': './data'
    }

    if dataset_name == "mnist":
        config['dataset_name'] = 'mnist'
        config['model_name'] = 'cnn'
        config['model_save_path'] = 'trained_mnist_tenseal_model.pth'
        config['num_rounds'] = 5
        config['local_epochs'] = 2
        config['learning_rate'] = 0.01
        config['optimizer'] = 'sgd' 

    elif dataset_name == "arrhythmia":
        config['dataset_name'] = 'arrhythmia'
        config['model_name'] = 'mlp'
        config['model_save_path'] = 'trained_arrhythmia_tenseal_model.pth'
        

        config['num_rounds'] = 25
        config['local_epochs'] = 5
        config['learning_rate'] = 0.001
        config['optimizer'] = 'adam'
        config['weight_decay'] = 1e-4
        config['lr_scheduler_step_size'] = 5
        config['lr_scheduler_gamma'] = 0.5

    else:
        raise ValueError(f"Unknown dataset configuration: {dataset_name}")

    return config

    

