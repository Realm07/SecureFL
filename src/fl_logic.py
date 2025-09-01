import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import traceback
from collections import OrderedDict
from he_tenseal import encrypt_state_dict_tenseal
from opacus import PrivacyEngine

def _create_optimizer(model, config):
    """Helper function to create an optimizer based on the config."""
    lr = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 0)
    
    if config['optimizer'].lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config['optimizer'].lower() == 'sgd':
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

def train_local_client_plaintext(model, dataloader, config):
    local_model = copy.deepcopy(model).to(config['device'])
    local_model.train()
    optimizer = _create_optimizer(local_model, config)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('lr_scheduler_step_size', 100),
        gamma=config.get('lr_scheduler_gamma', 1.0)
    )
    criterion = nn.CrossEntropyLoss()
    print(f"  Starting local training (PLAINTEXT - {config['local_epochs']} epochs)...")
    train_start = time.time()
    try:
        for _ in range(config['local_epochs']):
            for data, target in dataloader:
                data, target = data.to(config['device']), target.to(config['device'])
                optimizer.zero_grad(); output = local_model(data); loss = criterion(output, target)
                loss.backward(); optimizer.step()
            scheduler.step() 
        print(f"  Local training finished ({(time.time() - train_start):.2f}s).")
        return local_model.cpu().state_dict()
    except Exception as e:
        print(f"  ERROR during plaintext local training: {e}"); traceback.print_exc()
        return None


def train_local_client_secure(model, dataloader, config, context, slot_count, privacy_profile="she"):
    """
    Trains a local client model with a specified privacy profile.
    - privacy_profile: 'she', 'full_he', 'she_dp', 'full_he_dp'
    """
    local_model = copy.deepcopy(model).to(config['device'])
    local_model.train()
    optimizer = _create_optimizer(local_model, config)
    
    # --- NEW: Attach PrivacyEngine if DP is enabled ---
    privacy_engine = None
    if 'dp' in privacy_profile:
        print("  Attaching Opacus Differential Privacy Engine...")
        # Note: Opacus modifies the dataloader, so we need to handle this
        from torch.utils.data import DataLoader
        
        # We need to recreate the dataloader for Opacus to wrap it correctly
        # This is a nuance of how Opacus works with distributed sampling
        opacus_dataloader = DataLoader(dataloader.dataset, batch_size=config['batch_size'], shuffle=True)
        
        local_model, optimizer, opacus_dataloader, privacy_engine = _attach_dp_engine(
            local_model, optimizer, opacus_dataloader, config
        )
        # Use the new dataloader for training
        training_dataloader = opacus_dataloader
    else:
        training_dataloader = dataloader

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('lr_scheduler_step_size', 100),
        gamma=config.get('lr_scheduler_gamma', 1.0)
    )
    criterion = nn.CrossEntropyLoss()
    
    profile_str = privacy_profile.upper().replace('_', ' + ')
    print(f"  Starting local training ({profile_str} - {config['local_epochs']} epochs)...")
    train_start = time.time()
    
    try:
        for epoch in range(config['local_epochs']):
            for data, target in training_dataloader:
                data, target = data.to(config['device']), target.to(config['device'])
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            print(f"  DP Epsilon after {config['local_epochs']} epochs: {epsilon:.2f}")

        print(f"  Local training finished ({(time.time() - train_start):.2f}s).")
        
        # Determine which layers to encrypt based on the profile
        encrypted_layers = None
        if 'she' in privacy_profile:
            encrypted_layers = config.get('encrypted_layers')
        
        print(f"  Starting TenSEAL encryption...")
        return encrypt_state_dict_tenseal(context, local_model.cpu().state_dict(), slot_count, encrypted_layers)
        
    except Exception as e:
        print(f"  ERROR during secure local training: {e}"); traceback.print_exc()
        return None

def federated_average_plaintext(state_dicts):
    if not state_dicts: return None
    print(f"Server (Plaintext): Averaging {len(state_dicts)} client updates...")
    start_time = time.time()
    avg_state_dict = OrderedDict()
    keys = state_dicts[0].keys()
    for key in keys:
        sum_tensor = torch.stack([sd[key] for sd in state_dicts], dim=0).sum(dim=0)
        avg_tensor = sum_tensor / len(state_dicts)
        avg_state_dict[key] = avg_tensor
    print(f"    Plaintext averaging finished in {time.time() - start_time:.2f}s")
    return avg_state_dict

