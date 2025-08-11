import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import traceback
from collections import OrderedDict
from he_tenseal import encrypt_state_dict_tenseal


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

def train_local_client_secure(model, dataloader, config, context, slot_count):
    local_model = copy.deepcopy(model).to(config['device'])
    local_model.train()
    optimizer = _create_optimizer(local_model, config)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('lr_scheduler_step_size', 100),
        gamma=config.get('lr_scheduler_gamma', 1.0)
    )
    criterion = nn.CrossEntropyLoss()
    print(f"  Starting local training (SECURE - {config['local_epochs']} epochs)...")
    train_start = time.time()
    try:
        for _ in range(config['local_epochs']):
            for data, target in dataloader:
                data, target = data.to(config['device']), target.to(config['device'])
                optimizer.zero_grad(); output = local_model(data); loss = criterion(output, target)
                loss.backward(); optimizer.step()
            scheduler.step()
        print(f"  Local training finished ({(time.time() - train_start):.2f}s).")
        local_plaintext_dict = local_model.cpu().state_dict()
        print(f"  Starting TenSEAL encryption...")
        return encrypt_state_dict_tenseal(context, local_model.cpu().state_dict(), slot_count)
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