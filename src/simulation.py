import torch
from torch.utils.data import DataLoader
import random
import time
import tenseal as ts

from data_loader import partition_iid
from fl_logic import train_local_client_plaintext, train_local_client_secure, federated_average_plaintext
from he_tenseal import aggregate_and_decrypt_tenseal
from utils import evaluate_global_model

def run_simulation_plaintext(global_model, trainset, test_loader, config):
    """Runs the entire FL simulation WITHOUT any encryption for benchmarking."""
    print("\n\n" + "="*50)
    print("Starting PLAINTEXT Federated Learning Simulation")
    print("="*50)
    
    accuracies, losses = [], []
    times = []
    sample_plaintext_update = None
    initial_acc, initial_loss = evaluate_global_model(global_model, test_loader, config['device'])
    accuracies.append(initial_acc)
    losses.append(initial_loss)
    print(f"Initial Global Model Accuracy: {initial_acc:.2f}%")

    client_datasets, _ = partition_iid(trainset, config['num_clients'])
    
    total_sim_start_time = time.time()
    for round_num in range(config['num_rounds']):
        round_start_time = time.time()
        print(f"\n--- Global Round {round_num + 1}/{config['num_rounds']} (Plaintext) ---")
        
        available = [i for i, d in enumerate(client_datasets) if d and len(d) > 0]
        if not available:
            print("Warning: No clients with data available. Stopping simulation.")
            break
        selected_indices = random.sample(available, min(config['clients_per_round'], len(available)))
        print(f"Selected clients: {selected_indices}")
        
        local_updates = [
            train_local_client_plaintext(
                global_model, DataLoader(client_datasets[i], batch_size=config['batch_size'], shuffle=True), config
            ) for i in selected_indices
        ]
        if round_num == 0 and not sample_plaintext_update and local_updates:
            first_update = next((u for u in local_updates if u is not None), None)
            if first_update:
                sample_plaintext_update = {}
                for i, (key, value) in enumerate(first_update.items()):
                    if i < 2:
                        sample_plaintext_update[key] = {
                            "shape": str(list(value.shape)),
                            "sample_values": str(value.flatten().numpy()[:2].round(4).tolist())
                        }
        if valid_updates := [u for u in local_updates if u is not None]:
            avg_state_dict = federated_average_plaintext(valid_updates)
            if avg_state_dict:
                global_model.load_state_dict(avg_state_dict)
                print("Server: Global model updated.")
        
        accuracy, loss = evaluate_global_model(global_model, test_loader, config['device'])
        accuracies.append(accuracy)
        losses.append(loss)

        round_duration = time.time() - round_start_time
        times.append(round_duration)
        print(f"--- Round {round_num + 1} Perf --- Acc: {accuracy:.2f}%, Time: {round_duration:.2f}s ---")

    print(f"\nTotal Plaintext Simulation Time: {time.time() - total_sim_start_time:.2f}s")
    return accuracies, losses, times, sample_plaintext_update

def run_simulation_secure(global_model, trainset, test_loader, config):
    """Runs the entire FL simulation using TenSEAL for secure aggregation."""
    print("\n\n" + "="*50)
    print("Starting SECURE Federated Learning Simulation (TenSEAL)")
    print("="*50)

    POLY_MOD_DEGREE = 8192
    try:
        context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.generate_galois_keys()
        context.global_scale = 2**40
        slot_count = POLY_MOD_DEGREE // 2
        print(f"TenSEAL context created. Slots per ciphertext: {slot_count}")
    except Exception as e:
        print(f"Error creating TenSEAL context: {e}")
        return None, None, None, None

    accuracies, losses = [], []
    times = []
    initial_acc, initial_loss = evaluate_global_model(global_model, test_loader, config['device'])
    accuracies.append(initial_acc)
    losses.append(initial_loss)
    print(f"Initial Global Model Accuracy: {initial_acc:.2f}%")

    client_datasets, _ = partition_iid(trainset, config['num_clients'])
    
    total_sim_start_time = time.time()
    for round_num in range(config['num_rounds']):
        round_start_time = time.time()
        print(f"\n--- Global Round {round_num + 1}/{config['num_rounds']} (Secure) ---")

        available = [i for i, d in enumerate(client_datasets) if d and len(d) > 0]
        if not available:
            print("Warning: No clients with data available. Stopping simulation.")
            break
        selected_indices = random.sample(available, min(config['clients_per_round'], len(available)))
        print(f"Selected clients: {selected_indices}")

        encrypted_updates = [
            train_local_client_secure(
                global_model, DataLoader(client_datasets[i], batch_size=config['batch_size'], shuffle=True), 
                config, context, slot_count
            ) for i in selected_indices
        ]
        
        if valid_updates := [u for u in encrypted_updates if u is not None]:
            avg_state_dict = aggregate_and_decrypt_tenseal(context, valid_updates, len(valid_updates))
            if avg_state_dict:
                global_model.load_state_dict(avg_state_dict)
                print("Server: Global model updated.")
        
        accuracy, loss = evaluate_global_model(global_model, test_loader, config['device'])
        accuracies.append(accuracy)
        losses.append(loss)
        
        round_duration = time.time() - round_start_time
        times.append(round_duration)
        print(f"--- Round {round_num + 1} Perf --- Acc: {accuracy:.2f}%, Time: {round_duration:.2f}s ---")

    print(f"\nTotal Secure Simulation Time: {time.time() - total_sim_start_time:.2f}s")
    return accuracies, losses, times, global_model