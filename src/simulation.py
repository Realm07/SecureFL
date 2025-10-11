import copy
from .fl_logic import federated_average_plaintext
from torch.utils.data import DataLoader
import random
import time
from collections import OrderedDict
import tenseal as ts
import os
import csv
import torch
import torch.optim as optim  # <-- THIS IS THE FIX: ADD THIS IMPORT

from .data_loader import partition_iid
from .fl_logic import train_local_client_plaintext, train_local_client_secure
from .he_tenseal import aggregate_and_decrypt_tenseal
from .utils import evaluate_global_model


class TCMSimulator:
    """Simulates the client-side Trusted Contribution Module."""
    def __init__(self, validation_dataset, config):
        self.val_loader = DataLoader(validation_dataset, batch_size=config['batch_size'])
        self.config = config
        self.base_accuracy = None

    def validate_update(self, original_model, updated_model_state_dict):
        """
        Validates if the update is beneficial.
        Returns 'BENEFICIAL' or 'NOT_BENEFICIAL'.
        """
        device = self.config['device']
        test_model = copy.deepcopy(original_model).to(device)
        test_model.load_state_dict(updated_model_state_dict)
        current_accuracy, _ = evaluate_global_model(test_model, self.val_loader, device)

        if self.base_accuracy is None:
            original_accuracy, _ = evaluate_global_model(original_model, self.val_loader, device)
            self.base_accuracy = original_accuracy
        
        if current_accuracy >= self.base_accuracy - 1.5: # Allow 1.5% drop
            verdict = "BENEFICIAL"
        else:
            verdict = "NOT_BENEFICIAL"
        
        self.base_accuracy = current_accuracy
        return verdict

def run_simulation_plaintext(global_model, trainset, test_loader, config, experiment_name, malicious_clients, enable_defense):
    """
    MODIFIED to run a specific experiment with stronger attack logic.
    """
    defense_str = "DEFENSE_ENABLED" if enable_defense else "DEFENSE_DISABLED"
    print("\n\n" + "="*50)
    print(f"Starting PLAINTEXT Simulation: {experiment_name}")
    print(f"Malicious Clients: {malicious_clients}, Defense: {defense_str}")
    print("="*50)

    log_file = os.path.join(config['results_dir'], f"results_{experiment_name}.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "accuracy", "round_time"])
    
    initial_acc, _ = evaluate_global_model(global_model, test_loader, config['device'])
    print(f"Initial Global Model Accuracy: {initial_acc:.2f}%")
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, initial_acc, 0.0])

    client_datasets, _ = partition_iid(trainset, config['num_clients'])
    
    main_train_set_size = int(len(trainset) * 0.9)
    tcm_val_set_size = len(trainset) - main_train_set_size
    _, tcm_validation_set = torch.utils.data.random_split(trainset, [main_train_set_size, tcm_val_set_size])

    total_sim_start_time = time.time()
    for round_num in range(config['num_rounds']):
        round_start_time = time.time()
        print(f"\n--- Global Round {round_num + 1}/{config['num_rounds']} ({experiment_name}) ---")
        
        if malicious_clients and experiment_name != "baseline_benign":
            num_malicious_to_select = len(malicious_clients)
            num_honest_to_select = config['clients_per_round'] - num_malicious_to_select
            
            honest_clients = [i for i in range(config['num_clients']) if i not in malicious_clients]
            
            selected_malicious = malicious_clients
            selected_honest = random.sample(honest_clients, num_honest_to_select)
            
            selected_indices = selected_malicious + selected_honest
            random.shuffle(selected_indices)
        else:
            selected_indices = random.sample(range(config['num_clients']), config['clients_per_round'])

        print(f"Selected clients: {selected_indices}")
        
        valid_updates_for_aggregation = []
        initial_model_dict = copy.deepcopy(global_model.state_dict())

        for client_id in selected_indices:
            is_malicious = client_id in malicious_clients
            
            client_updated_dict = train_local_client_plaintext(
                global_model, 
                DataLoader(client_datasets[client_id], batch_size=config['batch_size'], shuffle=True), 
                config,
                is_malicious=is_malicious
            )
            
            if client_updated_dict is None:
                continue

            if is_malicious:
                print(f"  Client #{client_id} is scaling its update maliciously!")
                update_delta = OrderedDict()
                for key in initial_model_dict:
                    update_delta[key] = client_updated_dict[key] - initial_model_dict[key]
                
                scaling_factor = 10.0 
                for key in update_delta:
                    update_delta[key] *= scaling_factor
                
                malicious_state_dict = OrderedDict()
                for key in initial_model_dict:
                    malicious_state_dict[key] = initial_model_dict[key] + update_delta[key]
                
                client_update_to_process = malicious_state_dict
            else:
                client_update_to_process = client_updated_dict

            if enable_defense:
                tcm = TCMSimulator(tcm_validation_set, config)
                verdict = tcm.validate_update(global_model, client_update_to_process)
                print(f"  Client #{client_id} TCM Verdict: {verdict}")

                if verdict == "BENEFICIAL":
                    valid_updates_for_aggregation.append(client_update_to_process)
            else:
                valid_updates_for_aggregation.append(client_update_to_process)
        
        if valid_updates_for_aggregation:
            avg_state_dict = federated_average_plaintext(valid_updates_for_aggregation)
            if avg_state_dict:
                global_model.load_state_dict(avg_state_dict)
                print(f"Server: Global model updated with {len(valid_updates_for_aggregation)} updates.")
        else:
            print("Server: No valid updates received. Skipping model update.")
        
        accuracy, _ = evaluate_global_model(global_model, test_loader, config['device'])
        round_duration = time.time() - round_start_time
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_num + 1, accuracy, round_duration])

        print(f"--- Round {round_num + 1} Perf --- Acc: {accuracy:.2f}%, Time: {round_duration:.2f}s ---")

    print(f"\nTotal Simulation Time for {experiment_name}: {time.time() - total_sim_start_time:.2f}s")


def run_simulation_secure(global_model, trainset, test_loader, config, privacy_profile):
    """
    REFACTORED to use a persistent torch.optim.Adam instance for server-side updates.
    This is the robust and correct way to implement FedAdam.
    """
    profile_str = privacy_profile.upper().replace('_', ' + ')
    print("\n\n" + "="*50)
    print(f"Starting SECURE Federated Learning Simulation ({profile_str})")
    print("="*50)

    POLY_MOD_DEGREE = 16384
    context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
    context.generate_galois_keys()
    context.global_scale = 2**48
    slot_count = POLY_MOD_DEGREE // 2
    print(f"TenSEAL context created. Slots per ciphertext: {slot_count}")

    accuracies, losses, times = [], [], []
    initial_acc, initial_loss = evaluate_global_model(global_model, test_loader, config['device'])
    accuracies.append(initial_acc)
    losses.append(initial_loss)
    print(f"Initial Global Model Accuracy: {initial_acc:.2f}%")

    client_datasets, _ = partition_iid(trainset, config['num_clients'])
    
    # --- NEW: Instantiate a real server-side optimizer ---
    # The global model's parameters will be optimized by this Adam instance.
    server_optimizer = optim.Adam(global_model.parameters(), lr=0.05) # server_lr from server.py

    total_sim_start_time = time.time()
    for round_num in range(config['num_rounds']):
        round_start_time = time.time()
        print(f"\n--- Global Round {round_num + 1}/{config['num_rounds']} (Secure) ---")

        selected_indices = random.sample(range(config['num_clients']), config['clients_per_round'])
        print(f"Selected clients: {selected_indices}")

        client_config = copy.deepcopy(config)
        client_config['privacy_profile'] = privacy_profile
        if 'she' in privacy_profile:
             client_config['encrypted_layers'] = config.get('encrypted_layers', [])
        else:
             client_config['encrypted_layers'] = None

        encrypted_updates = [
            train_local_client_secure(
                global_model,
                DataLoader(client_datasets[i], batch_size=config['batch_size'], shuffle=True),
                client_config,
                context, 
                slot_count
            ) for i in selected_indices
        ]
        
        if valid_updates := [u for u in encrypted_updates if u is not None]:
            avg_delta_dict = aggregate_and_decrypt_tenseal(context, valid_updates, len(valid_updates))
            if avg_delta_dict:
                # --- Server-Side Update using the Optimizer ---
                server_optimizer.zero_grad()
                # Manually set the gradients of the global model's parameters
                for name, param in global_model.named_parameters():
                    if name in avg_delta_dict:
                        # The pseudo-gradient is the negative of the averaged delta
                        pseudo_grad = -avg_delta_dict[name].to(param.device)
                        param.grad = pseudo_grad
                
                # Take a step with the server-side optimizer
                server_optimizer.step()
                print("Server: Global model updated using server-side Adam optimizer.")

        accuracy, loss = evaluate_global_model(global_model, test_loader, config['device'])
        accuracies.append(accuracy)
        losses.append(loss)
        
        round_duration = time.time() - round_start_time
        times.append(round_duration)
        print(f"--- Round {round_num + 1} Perf --- Acc: {accuracy:.2f}%, Time: {round_duration:.2f}s ---")

    print(f"\nTotal Secure Simulation Time: {time.time() - total_sim_start_time:.2f}s")
    return accuracies, losses, times, copy.deepcopy(global_model)