import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import traceback
import os
import csv
from collections import OrderedDict

# This import was incorrect, it should be a relative import
# from src import data_loader 
from .he_tenseal import encrypt_state_dict_tenseal
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

# --- NEW: Client-side CSV logging function ---
def log_client_timing(client_id, round_num, profile, training_time, encryption_time):
    """Logs client-side timing metrics to a CSV file."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'client_{client_id}_timing.csv')
    
    # Write header if the file doesn't exist
    write_header = not os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['round', 'profile', 'training_time_seconds', 'encryption_time_seconds'])
        writer.writerow([round_num, profile, training_time, encryption_time])

def _create_optimizer(model, config):
    """Helper function to create an optimizer based on the config."""
    lr = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 0)
    optimizer_name = config.get('optimizer', 'adam').lower()

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

def _attach_dp_engine(model, optimizer, dataloader, config):
    """Helper to attach the Opacus PrivacyEngine."""
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=config['dp_noise_multiplier'],
        max_grad_norm=config['dp_max_grad_norm'],
    )
    return model, optimizer, dataloader, privacy_engine

def train_local_client_plaintext(model, dataloader, config, is_malicious=False): # Add is_malicious flag
    local_model = copy.deepcopy(model).to(config['device'])
    local_model.train()
    optimizer = _create_optimizer(local_model, config)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('lr_scheduler_step_size', 100),
        gamma=config.get('lr_scheduler_gamma', 1.0)
    )
    criterion = nn.CrossEntropyLoss()
    
    # NEW: Identify the attack parameters from config
    attack_target_class = config.get('attack_target_class', 1)
    attack_poison_class = config.get('attack_poison_class', 7)

    # Modify the log message to indicate if the client is malicious
    malicious_str = " (MALICIOUS)" if is_malicious else ""
    print(f"  Starting local training (PLAINTEXT - {config['local_epochs']} epochs){malicious_str}...")
    
    train_start = time.time()
    try:
        for _ in range(config['local_epochs']):
            for data, target in dataloader:
                # NEW: Label-Flipping Attack Logic
                if is_malicious:
                    # Create a mask to select only samples of the target class
                    mask = target == attack_target_class
                    # Change their labels to the poison class
                    target[mask] = attack_poison_class

                data, target = data.to(config['device']), target.to(config['device'])
                optimizer.zero_grad(); output = local_model(data); loss = criterion(output, target)
                loss.backward(); optimizer.step()
            scheduler.step() 
        print(f"  Local training finished ({(time.time() - train_start):.2f}s).")
        return local_model.cpu().state_dict()
    except Exception as e:
        print(f"  ERROR during plaintext local training: {e}"); traceback.print_exc()
        return None

def train_local_client_secure(client_id, round_num, model, dataloader, config, context, slot_count):
    """
    Trains a local client model, measures performance, and logs it.
    """
    privacy_profile = config.get('privacy_profile', 'she')
    initial_state_dict = copy.deepcopy(model.state_dict())
    
    device = torch.device(config.get('device', 'cpu'))
    local_model = copy.deepcopy(model).to(device)
    local_model.train()
    optimizer = _create_optimizer(local_model, config)
    
    training_dataloader = dataloader
    privacy_engine = None

    if 'dp' in privacy_profile:
        print("  Attaching Opacus Differential Privacy Engine...")
        opacus_dataloader = DataLoader(dataloader.dataset, batch_size=config['batch_size'], shuffle=True)
        local_model, optimizer, opacus_dataloader, privacy_engine = _attach_dp_engine(
            local_model, optimizer, opacus_dataloader, config
        )
        training_dataloader = opacus_dataloader

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('lr_scheduler_step_size', 100), gamma=config.get('lr_scheduler_gamma', 1.0))
    criterion = nn.MSELoss() if config.get('metric') == 'rmse' else nn.CrossEntropyLoss()
    
    profile_str = privacy_profile.upper().replace('_', ' + ')
    print(f"  Starting local training ({profile_str} - {config['local_epochs']} epochs)...")
    
    train_start = time.time()
    try:
        for epoch in range(config['local_epochs']):
            for data, target in training_dataloader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        training_duration = time.time() - train_start
        print(f"  Local training finished ({training_duration:.2f}s).")

        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta=config.get('delta', 1e-5))
            print(f"  DP Epsilon after {config['local_epochs']} epochs: {epsilon:.2f}")

        final_state_dict_raw = local_model.cpu().state_dict()
        final_state_dict = OrderedDict((key.replace('_module.', ''), value) for key, value in final_state_dict_raw.items())

        weight_delta = OrderedDict((key, final_state_dict[key] - initial_state_dict[key]) for key in final_state_dict)

        encrypted_layers = config.get('encrypted_layers') if 'she' in privacy_profile else None
        
        print("  Starting TenSEAL encryption of the weight delta...")
        encryption_start = time.time()
        encrypted_delta = encrypt_state_dict_tenseal(context, weight_delta, slot_count, encrypted_layers)
        encryption_duration = time.time() - encryption_start
        print(f"    Encryption finished in {encryption_duration:.2f}s")
        
        # --- NEW: Log the timing data to a local CSV ---
        log_client_timing(client_id, round_num, privacy_profile, training_duration, encryption_duration)
        
        return encrypted_delta
        
    except Exception as e:
        print(f"  ERROR during secure local training: {e}"); traceback.print_exc()
        return None
    
def federated_average_plaintext(state_dicts):
    if not state_dicts: 
        return None
    print(f"Server (Plaintext): Averaging {len(state_dicts)} client updates...")
    start_time = time.time()
    avg_state_dict = OrderedDict()
    keys = state_dicts[0].keys()
    for key in keys:
        # Stack all the tensors for the current key along a new dimension
        sum_tensor = torch.stack([sd[key] for sd in state_dicts], dim=0).sum(dim=0)
        # Divide by the number of clients
        avg_tensor = sum_tensor / len(state_dicts)
        avg_state_dict[key] = avg_tensor
    print(f"    Plaintext averaging finished in {time.time() - start_time:.2f}s")
    return avg_state_dict