# -*- coding: utf-8 -*-
"""
Federated Learning Simulation with Paillier HE (Optimized Version)

Optimizations applied:
1. Reduced Model Size: Smaller CNN to decrease the number of parameters.
2. Multiprocessing Encryption: Parallelizes the Paillier encryption using multiple CPU cores.
3. Multiprocessing Decryption: Parallelizes the Paillier decryption during aggregation.
4. Code structure and error handling improvements.
"""

# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import phe as paillier # Make sure this is installed: pip install python-paillier
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
import random
from collections import OrderedDict
import time
import traceback
import os
# --- Multiprocessing ---
import concurrent.futures
import functools # For using partial with map

# =============================================================================
# OPTIMIZED Model Definition (SmallerCNN)
# =============================================================================
class SmallerCNN(nn.Module):
    """A smaller CNN model for MNIST classification to reduce parameters."""
    def __init__(self):
        super(SmallerCNN, self).__init__()
        # Reduce channels and FC layer size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # 16 filters
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: (batch, 16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2) # 32 filters
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: (batch, 32, 7, 7)
        # Flattened size: 32 channels * 7x7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # Smaller FC layer
        self.fc2 = nn.Linear(128, 10) # Output layer

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =============================================================================
# Homomorphic Encryption Setup & Encoding Helpers (Unchanged from previous correction)
# =============================================================================

ENCODING_BASE = 10
ENCODING_PRECISION = 6

def encode_number(n):
    """Encodes a float or int to a fixed-point integer representation."""
    if isinstance(n, (int, float, np.number)): n_float = float(n)
    elif hasattr(n, 'item'):
        try: n_float = float(n.item())
        except Exception as e: raise TypeError(f"Could not convert {n} to float: {e}")
    else: raise TypeError(f"Unsupported type for encoding: {type(n)}, value: {n}")
    if not np.isfinite(n_float): n_float = 0.0
    scaled_n = round(n_float * (ENCODING_BASE ** ENCODING_PRECISION))
    return int(scaled_n)

def decode_number(n_encoded):
    """Decodes a fixed-point integer back to float."""
    try:
        if isinstance(n_encoded, paillier.EncryptedNumber): raise TypeError("Decrypt first.")
        numeric_value = float(n_encoded)
    except Exception as e: raise TypeError(f"Cannot decode {n_encoded}: {e}")
    n_float = numeric_value / (ENCODING_BASE ** ENCODING_PRECISION)
    return n_float

# =============================================================================
# Worker functions for Multiprocessing HE (Defined globally)
# =============================================================================

def _encrypt_single_value(encoded_value, public_key):
    """Worker function to encrypt a single encoded value."""
    # This function will run in a separate process.
    if encoded_value is None: # Handle potential upstream issues
        return None
    try:
        return public_key.encrypt(encoded_value)
    except Exception as e:
        # print(f"  [Worker Encrypt Error]: {e}") # Optionally log errors from workers
        return None # Indicate failure for this specific value

def _decrypt_single_value(encrypted_value, private_key):
    """Worker function to decrypt a single encrypted value."""
    if not isinstance(encrypted_value, paillier.EncryptedNumber):
        # print(f"  [Worker Decrypt Error]: Invalid type {type(encrypted_value)}")
        return None # Invalid input
    try:
        return private_key.decrypt(encrypted_value)
    except Exception as e:
        # print(f"  [Worker Decrypt Error]: {e}")
        return None # Indicate failure

# =============================================================================
# State Dict Encryption/Decryption with Multiprocessing
# =============================================================================

def encrypt_state_dict_parallel(public_key, state_dict, num_workers=None):
    """Encrypts all float values in a state_dict using parallel processing."""
    start_time_prep = time.time()
    encrypted_dict = OrderedDict()
    all_encoded_values = []
    param_info = [] # Store (key, shape, num_elements)

    print(f"    Preparing {sum(p.numel() for p in state_dict.values())} parameters for parallel encryption...")
    for key, tensor in state_dict.items():
        shape = tensor.shape
        num_elements = tensor.numel()
        param_info.append({'key': key, 'shape': shape, 'numel': num_elements})
        flat_tensor = tensor.flatten()
        # Encode sequentially first (encoding is fast)
        for val_tensor in flat_tensor:
            try:
                encoded = encode_number(val_tensor)
                all_encoded_values.append(encoded)
            except TypeError as e:
                print(f"    Warning: Skipping encoding for a value in layer '{key}': {e}")
                all_encoded_values.append(None) # Add placeholder for skipped values

    prep_time = time.time() - start_time_prep
    print(f"    Encoding finished ({prep_time:.2f}s). Starting parallel encryption...")

    # --- Parallel Encryption ---
    start_time_enc = time.time()
    encrypted_results = []
    if not all_encoded_values: # Handle empty state dict
         print("    Warning: No values to encrypt.")
         return encrypted_dict # Return empty

    # Create the worker function with the public key fixed
    encrypt_worker_partial = functools.partial(_encrypt_single_value, public_key=public_key)

    # Use ProcessPoolExecutor for CPU-bound tasks
    # If num_workers is None, it defaults to os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map processes the iterable in order and returns results in order
        # Use chunksize for potentially better performance with many small tasks
        chunksize = max(1, len(all_encoded_values) // (num_workers if num_workers else os.cpu_count() or 1) // 4)
        try:
             encrypted_results = list(executor.map(encrypt_worker_partial, all_encoded_values, chunksize=chunksize))
        except Exception as e_map:
             print(f"    ERROR during parallel encryption map execution: {e_map}")
             return None # Indicate failure

    enc_time = time.time() - start_time_enc
    successful_encryptions = sum(1 for r in encrypted_results if r is not None)
    print(f"    Parallel encryption finished ({enc_time:.2f}s). Successfully encrypted {successful_encryptions}/{len(all_encoded_values)} values.")

    if successful_encryptions != len(all_encoded_values):
         print("    ERROR: Not all values were successfully encrypted.")
         return None # Indicate overall failure

    # --- Reconstruct the dictionary ---
    current_idx = 0
    for info in param_info:
        key = info['key']
        shape = info['shape']
        num_elements = info['numel']
        # Extract the slice of encrypted values for this parameter
        values_slice = encrypted_results[current_idx : current_idx + num_elements]

        # Basic check
        if len(values_slice) != num_elements:
             print(f"    ERROR: Mismatch in encrypted result length for key '{key}'. Expected {num_elements}, got {len(values_slice)}.")
             return None

        encrypted_dict[key] = {
            'shape': shape,
            'values': values_slice
        }
        current_idx += num_elements

    return encrypted_dict

# =============================================================================
# Federated Learning Helper Functions (Using Parallel HE)
# =============================================================================

def partition_iid(dataset, num_clients):
    # (Assumed correct from previous steps - Ensure robustness checks are present)
    # ... (Same code as before) ...
    if len(dataset) == 0: print("Error: Cannot partition empty dataset."); return [None]*num_clients, {}
    if len(dataset) < num_clients: print(f"Warning: Dataset size {len(dataset)} < num_clients {num_clients}")
    num_items_per_client = len(dataset)//num_clients if len(dataset) >= num_clients else 1
    remainder = len(dataset) % num_clients if len(dataset) >= num_clients else len(dataset) % len(dataset) if len(dataset)>0 else 0
    client_datasets = [None]*num_clients; client_idx_map={}; all_indices=list(range(len(dataset))); random.shuffle(all_indices); current_idx = 0
    print(f"Partitioning data into {num_clients} potential IID shards...")
    num_clients_with_data = num_clients if len(dataset)>=num_clients else len(dataset)
    for i in range(num_clients_with_data):
        items_count = num_items_per_client + (1 if i < remainder else 0); end_idx = min(current_idx+items_count, len(all_indices)); idxs = all_indices[current_idx:end_idx]
        if not idxs: continue
        current_idx=end_idx; client_idx_map[i]=idxs; client_datasets[i]=Subset(dataset,idxs)
        #print(f"Client {i} assigned {len(idxs)}")
    assigned_c = sum(1 for d in client_datasets if d is not None); print(f"Finished partitioning. Assigned data to {assigned_c} clients.")
    if assigned_c < num_clients: print(f"({num_clients-assigned_c} received no data)")
    return client_datasets, client_idx_map

# --- train_local_client (Uses Parallel Encryption) ---
def train_local_client(model, dataloader, local_epochs, learning_rate, device, client_id, public_key, num_workers=None):
    """ Trains locally, then encrypts the resulting state_dict in parallel."""
    if not dataloader or len(dataloader) == 0:
        print(f"  Client {client_id}: Dataloader empty. Skipping.")
        return None

    # Use the SmallerCNN architecture if passed implicitly or explicitly define
    # Assuming `model` passed is already the desired architecture
    local_model = copy.deepcopy(model).to(device)
    local_model.train()
    optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"  Client {client_id}: Starting local training ({local_epochs} epochs)...")
    train_start = time.time()
    try:
        for epoch in range(local_epochs):
            # Mini-batch training loop (abbreviated for clarity)
            # ... (Your existing batch training loop) ...
            pass # Placeholder for batch loop
        print(f"  Client {client_id}: Local training finished ({(time.time() - train_start):.2f}s).") # Add timing

        # Get plaintext weights and encrypt in parallel
        local_plaintext_dict = local_model.cpu().state_dict()
        print(f"  Client {client_id}: Starting parallel encryption...")
        # --- Call parallel encryption function ---
        local_encrypted_dict = encrypt_state_dict_parallel(public_key, local_plaintext_dict, num_workers)

        if local_encrypted_dict is None:
            print(f"  Client {client_id}: Encryption failed.")
            return None

        return local_encrypted_dict

    except Exception as e:
        print(f"  Client {client_id}: ERROR during local training/encryption: {e}")
        traceback.print_exc()
        return None

# --- federated_average (plaintext - keep for comparison) ---
def federated_average(state_dicts):
    # (Your existing plaintext averaging code - assumed correct)
    valid_state_dicts = [sd for sd in state_dicts if sd is not None and isinstance(sd, dict) and sd]
    if not valid_state_dicts: print("Server Warning: No valid state dictionaries for plaintext averaging."); return None
    num_clients = len(valid_state_dicts); keys = valid_state_dicts[0].keys(); avg_state_dict = OrderedDict()
    print(f"Server: Averaging {num_clients} models (Plaintext)...")
    try:
        for key in keys:
            if not all(key in sd for sd in valid_state_dicts): print(f"Error: Key '{key}' missing."); return None
            sum_tensor = torch.stack([sd[key].cpu() for sd in valid_state_dicts], dim=0).sum(dim=0)
            avg_tensor = sum_tensor / num_clients; avg_state_dict[key] = avg_tensor
        print("Server: Plaintext averaging complete."); return avg_state_dict
    except Exception as e: print(f"Server Error (Plaintext Avg): {e}"); return None

# --- federated_average_encrypted (Uses Parallel Decryption) ---
def federated_average_encrypted(encrypted_state_dicts, private_key, num_to_average_over, num_workers=None):
    """ Aggregates, then decrypts in parallel, decodes, and averages."""
    valid_encrypted_dicts = [d for d in encrypted_state_dicts if d is not None]
    if not valid_encrypted_dicts: print("Server Warning: No valid encrypted states."); return None
    actual_num_to_average = len(valid_encrypted_dicts)
    if actual_num_to_average == 0: print("Server Error: No valid client data left."); return None
    if actual_num_to_average != num_to_average_over: print(f"Server Warning: Expected {num_to_average_over} updates, received {actual_num_to_average}. Averaging over {actual_num_to_average}.")

    print(f"Server (HE): Aggregating {actual_num_to_average} encrypted updates...")
    start_time_agg = time.time()
    keys = valid_encrypted_dicts[0].keys()
    if not all(d.keys() == keys for d in valid_encrypted_dicts): print("Server Error: Inconsistent keys."); return None
    aggregated_plain_dict = OrderedDict()

    # --- Homomorphic Aggregation (Still Sequential per Key) ---
    # Could also parallelize the inner loop per element, but let's parallelize decryption first
    aggregated_encrypted_data = OrderedDict() # Store summed encrypted values per key
    total_elements_to_decrypt = 0
    print("    Homomorphically aggregating values...")
    for key in keys:
        try:
            original_shape = valid_encrypted_dicts[0][key]['shape']
            if not all(d[key]['shape'] == original_shape for d in valid_encrypted_dicts): print(f"Error shape {key}"); return None
            encrypted_value_lists = [d[key]['values'] for d in valid_encrypted_dicts]
            num_elements = len(encrypted_value_lists[0])
            if not all(len(lst) == num_elements for lst in encrypted_value_lists): print(f"Error elements {key}"); return None

            summed_encrypted_values = []
            for i in range(num_elements): # Iterate through elements within the parameter tensor
                try:
                    encrypted_sum = valid_encrypted_dicts[0][key]['values'][i]
                    # Add checks here: Ensure elements are EncryptedNumber and not None
                    if not isinstance(encrypted_sum, paillier.EncryptedNumber):
                         print(f"Warning: Non-encrypted number at index {i} for key {key}, client 0. Skipping element.")
                         # Need a strategy to handle failure - skip element? Skip key? Fail all?
                         # For now, add a placeholder or skip key. Let's try adding None.
                         summed_encrypted_values.append(None)
                         continue

                    for client_dict in valid_encrypted_dicts[1:]:
                        val_to_add = client_dict[key]['values'][i]
                        if not isinstance(val_to_add, paillier.EncryptedNumber):
                            print(f"Warning: Non-encrypted number at index {i} for key {key} from another client. Skipping element.")
                            encrypted_sum = None # Mark sum as invalid if any part is invalid
                            break
                        encrypted_sum += val_to_add
                    summed_encrypted_values.append(encrypted_sum)

                except Exception as e_add_inner:
                    print(f"Server Error during HE sum for elem {i}, key '{key}': {e_add_inner}")
                    summed_encrypted_values.append(None) # Mark failed elements

            aggregated_encrypted_data[key] = {
                'shape': original_shape,
                'values': summed_encrypted_values # List of summed EncryptedNumbers (or None)
            }
            total_elements_to_decrypt += num_elements

        except Exception as e_key_agg:
            print(f"Server Error processing aggregation for key '{key}': {e_key_agg}")
            # Optionally remove key from further processing if fatal

    agg_time = time.time() - start_time_agg
    print(f"    Homomorphic aggregation finished ({agg_time:.2f}s). Starting parallel decryption of {total_elements_to_decrypt} values...")

    # --- Parallel Decryption & Sequential Decoding/Averaging ---
    start_time_dec = time.time()
    decrypt_worker_partial = functools.partial(_decrypt_single_value, private_key=private_key)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for key in list(aggregated_encrypted_data.keys()): # Use list to allow removal on error
            encrypted_sums = aggregated_encrypted_data[key]['values']
            original_shape = aggregated_encrypted_data[key]['shape']

            # Filter out Nones before sending to decrypt workers
            valid_sums_to_decrypt = [s for s in encrypted_sums if isinstance(s, paillier.EncryptedNumber)]
            indices_to_decrypt = [i for i, s in enumerate(encrypted_sums) if isinstance(s, paillier.EncryptedNumber)]

            if not valid_sums_to_decrypt:
                print(f"    Warning: No valid values to decrypt for key '{key}'. Skipping.")
                del aggregated_encrypted_data[key] # Remove key if nothing decryptable
                continue

            chunksize = max(1, len(valid_sums_to_decrypt) // (num_workers if num_workers else os.cpu_count() or 1) // 4)
            try:
                 decrypted_integers_list = list(executor.map(decrypt_worker_partial, valid_sums_to_decrypt, chunksize=chunksize))
            except Exception as e_map_dec:
                 print(f"    ERROR during parallel decryption map execution for key '{key}': {e_map_dec}")
                 del aggregated_encrypted_data[key]
                 continue

            # --- Post-Decryption Processing (Sequential) ---
            num_elements = len(encrypted_sums) # Original number of elements expected
            decrypted_decoded_values = [None] * num_elements # Initialize with Nones

            successful_decryptions = 0
            failed_decryption = False
            # Reconstruct based on original indices
            for i, dec_int in enumerate(decrypted_integers_list):
                original_index = indices_to_decrypt[i]
                if dec_int is None: # Check if worker indicated failure
                     print(f"    Warning: Decryption failed for an element at original index {original_index}, key '{key}'.")
                     # Cannot reliably average if decryption fails
                     failed_decryption = True
                     break
                try:
                     decoded_float = decode_number(dec_int)
                     decrypted_decoded_values[original_index] = decoded_float
                     successful_decryptions += 1
                except Exception as e_decode:
                     print(f"    Warning: Decoding failed for element {original_index}, key '{key}': {e_decode}")
                     failed_decryption = True
                     break

            if failed_decryption or successful_decryptions != len(valid_sums_to_decrypt):
                 print(f"    ERROR: Decryption/Decoding failed for key '{key}'. Removing from results.")
                 del aggregated_encrypted_data[key]
                 continue # Skip to next key

             # Fill potential gaps if elements were skipped during aggregation
            if None in decrypted_decoded_values:
                 print(f"   Warning: Null values present after decryption for key '{key}'. Averaging may be inaccurate. Filling with 0.")
                 # Need a strategy - fill with 0, fail key, etc. Filling with 0 biases result.
                 filled_values = [v if v is not None else 0.0 for v in decrypted_decoded_values]
            else:
                 filled_values = decrypted_decoded_values

            # --- Averaging & Reshaping ---
            try:
                avg_tensor = torch.tensor(filled_values, dtype=torch.float32).view(original_shape) / actual_num_to_average
                aggregated_plain_dict[key] = avg_tensor # Store final plain average tensor
            except Exception as e_tensor:
                print(f"Server Error creating/averaging tensor for key '{key}': {e_tensor}")
                if key in aggregated_encrypted_data: del aggregated_encrypted_data[key]


    dec_time = time.time() - start_time_dec
    print(f"    Parallel decryption & processing finished ({dec_time:.2f}s).")
    print(f"Server (HE): Aggregation finished (Total {(time.time() - start_time_agg):.2f}s).")

    if len(aggregated_plain_dict) != len(keys):
        print(f"Server Error: Aggregation result missing keys (Processed {len(aggregated_plain_dict)}/{len(keys)}). Returning None.")
        return None
    return aggregated_plain_dict


# --- evaluate_global_model (Unchanged - Assumed Correct) ---
def evaluate_global_model(model, test_loader, device):
    # (Same code as before)
    model.eval(); correct = 0; total = 0; criterion = nn.CrossEntropyLoss(); test_loss = 0.0
    if not test_loader or len(test_loader) == 0: print("Warning: Test loader empty."); return 0.0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data); loss = criterion(outputs, target); test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0); correct += (predicted == target).sum().item()
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    model.train(); return accuracy, avg_loss

# =============================================================================
# Main Federated Learning Simulation Loop (Using Smaller Model & Parallel HE)
# =============================================================================

def run_simulation_he_optimized(num_workers=None): # Add num_workers parameter
    print("\n=================================================")
    print("Starting HE Federated Learning Simulation (Optimized)")
    print(f"Using up to {num_workers if num_workers else os.cpu_count()} worker processes for HE.")
    print("=================================================")

    # --- Hyperparameters ---
    NUM_GLOBAL_ROUNDS = 5
    NUM_CLIENTS = 10
    CLIENTS_PER_ROUND = 3
    LOCAL_EPOCHS = 2 # Can increase slightly now HE is faster
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = './data'
    PAILLIER_KEY_SIZE = 1024 # Increased for better security margin

    # --- Setup ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        trainset = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False)
    except Exception as e: print(f"Error loading dataset: {e}"); return None
    if not trainset or not testset or len(trainset)==0 or len(testset)==0: print("Datasets empty."); return None

    client_datasets, _ = partition_iid(trainset, NUM_CLIENTS)
    # --- USE SMALLER MODEL ---
    global_model = SmallerCNN().to(DEVICE)
    print("\n--- Using SmallerCNN Model ---")
    print(f"Model Parameter Count: {sum(p.numel() for p in global_model.parameters())}")
    print(f"Using device: {DEVICE}")
    initial_acc, initial_loss = evaluate_global_model(global_model, test_loader, DEVICE)
    print(f"Initial Global Model Accuracy: {initial_acc:.2f}%")

    # --- HE SETUP ---
    print(f"Server: Generating Paillier keypair (size={PAILLIER_KEY_SIZE})...")
    key_gen_start = time.time()
    try:
         public_key, private_key = paillier.generate_paillier_keypair(n_length=PAILLIER_KEY_SIZE)
         print(f"Server: Paillier keypair generated ({time.time() - key_gen_start:.2f}s).")
    except Exception as e: print(f"\n!!! ERROR: Paillier key gen failed: {e}. Aborting."); return None

    # --- Tracking ---
    global_accuracies_he = [initial_acc]
    global_losses_he = [initial_loss if initial_loss is not None else float('inf')]

    # --- Simulation Start ---
    total_sim_start_time = time.time()
    for round_num in range(NUM_GLOBAL_ROUNDS):
        round_start_time = time.time()
        print(f"\n--- Global Round {round_num + 1} / {NUM_GLOBAL_ROUNDS} (Optimized HE) ---")

        # 1. Client Selection
        available_clients_indices=[i for i,d in enumerate(client_datasets) if d is not None and len(d)>0]
        if not available_clients_indices: print("Warning: No clients with data."); break
        num_to_select = min(CLIENTS_PER_ROUND, len(available_clients_indices))
        if num_to_select == 0: print("Warning: No clients to select."); continue
        selected_client_indices = random.sample(available_clients_indices, num_to_select)
        print(f"Selected clients: {selected_client_indices}")

        # 2. Local Training & Parallel ENCRYPTION
        local_encrypted_results = []
        print(f"Round {round_num+1}: Dispatching tasks to clients...")
        client_tasks_start = time.time()
        for client_idx in selected_client_indices:
            client_data_subset = client_datasets[client_idx];
            if client_data_subset is None: continue
            client_loader = DataLoader(client_data_subset, batch_size=BATCH_SIZE, shuffle=True)

            encrypted_state_dict = train_local_client(
                model=global_model, dataloader=client_loader, local_epochs=LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE, device=DEVICE, client_id=client_idx,
                public_key=public_key, num_workers=num_workers # Pass num_workers
            )
            if encrypted_state_dict: local_encrypted_results.append(encrypted_state_dict)
        print(f"Client training & encryption time: {(time.time() - client_tasks_start):.2f}s")

        # 3. Server Aggregation using Parallel HE Decryption
        aggregation_start_time = time.time()
        averaged_plaintext_dict = None
        num_successful_clients = len(local_encrypted_results)
        if num_successful_clients > 0:
            averaged_plaintext_dict = federated_average_encrypted(
                encrypted_state_dicts=local_encrypted_results,
                private_key=private_key,
                num_to_average_over=num_successful_clients,
                num_workers=num_workers # Pass num_workers
            )
        else: print("Server: No valid updates received.")
        print(f"Server aggregation time: {(time.time() - aggregation_start_time):.2f}s")


        # 4. Update Global Model
        update_start_time = time.time()
        if averaged_plaintext_dict:
            try:
                # Ensure requires_grad is False before loading state if model was involved in grad calc
                # for param in global_model.parameters(): param.requires_grad = False
                global_model.load_state_dict(averaged_plaintext_dict)
                # for param in global_model.parameters(): param.requires_grad = True # Re-enable if needed later
                print("Server: Global model updated.")
            except Exception as e: print(f"Server Error loading state_dict: {e}")
        else: print("Server: Global model not updated.")
        print(f"Model update time: {(time.time() - update_start_time):.2f}s")

        # 5. Evaluate Global Model Performance
        eval_start_time = time.time()
        print("Server: Evaluating global model...")
        accuracy, loss = evaluate_global_model(global_model, test_loader, DEVICE)
        global_accuracies_he.append(accuracy); global_losses_he.append(loss)
        print(f"Evaluation time: {(time.time() - eval_start_time):.2f}s")
        print(f"--- Round {round_num + 1} Performance ---")
        print(f"  Global Accuracy: {accuracy:.2f}%")
        print(f"  Global Loss: {loss:.4f}")
        round_time = time.time() - round_start_time
        print(f"--- Round {round_num + 1} Duration: {round_time:.2f}s ---")


    total_sim_time = time.time() - total_sim_start_time
    print("\n=================================================")
    print(f"Optimized HE Simulation Finished! Total Time: {total_sim_time:.2f}s")
    if len(global_accuracies_he) > 1: print(f"Final Global Accuracy (HE Optimized): {global_accuracies_he[-1]:.2f}%")
    print("=================================================")

    # --- Plotting ---
    if len(global_accuracies_he) > 1:
        plt.figure(figsize=(12, 5)); rounds_axis = range(NUM_GLOBAL_ROUNDS + 1)
        plt.subplot(1, 2, 1); plt.plot(rounds_axis, global_accuracies_he, marker='o'); plt.title('Accuracy per Round (Optimized HE)'); plt.xlabel('Round'); plt.ylabel('Accuracy (%)'); plt.xticks(rounds_axis); plt.grid(True)
        plt.subplot(1, 2, 2); plot_losses = global_losses_he[1:]; plot_loss_rounds = range(1, NUM_GLOBAL_ROUNDS + 1); plt.plot(plot_loss_rounds, plot_losses, marker='x', color='r'); plt.title('Loss per Round (Optimized HE)'); plt.xlabel('Round'); plt.ylabel('Loss'); plt.xticks(plot_loss_rounds); plt.grid(True)
        plt.tight_layout(); plt.show()

    # --- Saving ---
    MODEL_SAVE_PATH = 'trained_mnist_he_opt_model.pth' # Different name
    print(f"\nSaving final model state to: {MODEL_SAVE_PATH}")
    try: torch.save(global_model.state_dict(), MODEL_SAVE_PATH); print("Model saved.")
    except Exception as e: print(f"Error saving model: {e}")

    return global_model


# =============================================================================
# Main Execution Block (Load or Train Optimized Model)
# =============================================================================
if __name__ == "__main__":
    # --- IMPORTANT FOR MULTIPROCESSING ---
    # Some systems might require this for multiprocessing to work correctly in scripts
    # torch.multiprocessing.freeze_support() # Uncomment if facing issues

    # --- Determine Number of Workers ---
    # Leave None to use default (os.cpu_count()) or set specific number
    NUM_WORKERS = None # Or set to e.g., 4

    MODEL_SAVE_PATH = 'trained_mnist_he_opt_model.pth' # Use the optimized model name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT = './data'

    final_model = None
    model_loaded = False # Flag to track if model was loaded

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found existing optimized model at: {MODEL_SAVE_PATH}. Loading...")
        try:
            # --- Instantiate the SMALLER model ---
            final_model = SmallerCNN()
            final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            final_model.to(DEVICE)
            final_model.eval()
            print("Optimized model loaded successfully.")
            model_loaded = True # Set flag
        except Exception as e:
            print(f"Error loading saved optimized model: {e}. Proceeding to retrain...")
            final_model = None

    if final_model is None:
        print("\nNo pre-trained optimized model loaded. Running HE training simulation...")
        # Pass num_workers to the simulation runner
        final_model = run_simulation_he_optimized(num_workers=NUM_WORKERS)
        if final_model is None: print("Optimized training simulation failed. Exiting."); exit()
        else: final_model.eval() # Ensure model is in eval mode after training

    # --- Visualization ---
    if final_model:
        # ... (Visualization code remains exactly the same as your last version) ...
        # Just ensure it uses the `final_model` variable which now holds
        # either the loaded or newly trained optimized model.
        vis_testset = None; testset = None
        try:
            testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
            vis_testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        except Exception as e: print(f"Error loading dataset for viz: {e}")

        if vis_testset and testset and len(vis_testset)>0:
            print("\n=================================================")
            print("Demonstrating Classification on Test Set (Optimized HE Model)")
            print(f"(Using {'loaded' if model_loaded else 'newly trained'} model)")
            print("=================================================")
            final_model.eval()
            NUM_IMAGES_TO_SHOW = 5;
            if len(vis_testset)<NUM_IMAGES_TO_SHOW: NUM_IMAGES_TO_SHOW=len(vis_testset)
            if NUM_IMAGES_TO_SHOW > 0:
                 plt.figure(figsize=(15, 5))
                 for i in range(NUM_IMAGES_TO_SHOW):
                    random_idx=random.randint(0, len(vis_testset)-1); vis_image, actual_label = vis_testset[random_idx]; norm_image, _ = testset[random_idx]
                    model_input = norm_image.unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): output_logits = final_model(model_input); _, predicted_label_idx = torch.max(output_logits.data, 1); predicted_label = predicted_label_idx.item()
                    plt.subplot(1, NUM_IMAGES_TO_SHOW, i + 1); plt.imshow(vis_image.squeeze(), cmap='gray'); plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}", color=("green" if actual_label == predicted_label else "red")); plt.axis('off')
                 plt.suptitle("Sample Test Images: Actual vs. Predicted Labels (Optimized HE)", fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
        else: print("Skipping viz: Dataset error or empty.")
    else: print("No model available. Skipping visualization.")