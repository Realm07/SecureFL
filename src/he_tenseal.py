# src/he_tenseal.py

import tenseal as ts
from collections import OrderedDict
import torch
import time

def encrypt_state_dict_tenseal(context, state_dict, slot_count, encrypted_layers=None):
    """
    Encrypts a state dictionary.
    If encrypted_layers is provided, it performs Selective HE (SHE).
    Otherwise, it performs Full HE.
    Returns a hybrid dictionary with plaintext and encrypted parts.
    """
    plaintext_params = OrderedDict()
    params_to_encrypt = OrderedDict()

    # --- NEW: Partition the state_dict into plaintext and to-be-encrypted parts ---
    if encrypted_layers:
        print("    Performing Selective Encryption (SHE)...")
        for key, tensor in state_dict.items():
            if key in encrypted_layers:
                params_to_encrypt[key] = tensor
            else:
                plaintext_params[key] = tensor
    else:
        print("    Performing Full Encryption...")
        params_to_encrypt = state_dict

    if not params_to_encrypt:
        return {'plaintext_params': plaintext_params, 'encrypted_bundle': None}

    # Flatten and encrypt only the selected parameters
    flat_params, param_info, current_pos = [], {}, 0
    for key, tensor in params_to_encrypt.items():
        numel = tensor.numel()
        param_info[key] = {'shape': tensor.shape, 'start': current_pos, 'end': current_pos + numel}
        flat_params.extend(tensor.cpu().flatten().tolist())
        current_pos += numel
    
    print(f"    Encrypting {len(flat_params)} parameters into batches of {slot_count}...")
    start_time = time.time()
    encrypted_batches = []
    for i in range(0, len(flat_params), slot_count):
        batch_data = flat_params[i : i + slot_count]
        encrypted_vec = ts.ckks_vector(context, batch_data)
        encrypted_batches.append(encrypted_vec.serialize())
    print(f"    Encryption finished in {time.time() - start_time:.2f}s")
    
    encrypted_bundle = OrderedDict([('param_info', param_info), ('encrypted_batches', encrypted_batches)])
    
    return {'plaintext_params': plaintext_params, 'encrypted_bundle': encrypted_bundle}

# --- START OF MODIFICATION ---
def aggregate_and_decrypt_tenseal(context, hybrid_updates, num_clients):
    """
    Aggregates a list of hybrid updates and returns a single, fully decrypted state_dict.
    This version is robust to plaintext-only updates and handles the Opacus '_module.' prefix.
    """
    if not hybrid_updates: return None
    print(f"Server (TenSEAL): Aggregating {len(hybrid_updates)} hybrid client updates...")
    start_time = time.time()

    # Step 1: Average plaintext parts
    avg_plaintext_dict = OrderedDict()
    plaintext_parts = [update['plaintext_params'] for update in hybrid_updates if update['plaintext_params']]
    if plaintext_parts:
        for key in plaintext_parts[0].keys():
            sum_tensor = torch.stack([sd[key] for sd in plaintext_parts], dim=0).sum(dim=0)
            avg_plaintext_dict[key] = sum_tensor / len(plaintext_parts)

    # Step 2: Aggregate and decrypt encrypted parts
    decrypted_avg_encrypted_dict = OrderedDict()
    encrypted_bundles = [update['encrypted_bundle'] for update in hybrid_updates if update['encrypted_bundle'] is not None]
    if encrypted_bundles:
        sum_encrypted_batches = [ts.CKKSVector.load(context, ser_vec) for ser_vec in encrypted_bundles[0]['encrypted_batches']]
        for bundle in encrypted_bundles[1:]:
            for i, ser_vec in enumerate(bundle['encrypted_batches']):
                sum_encrypted_batches[i] += ts.CKKSVector.load(context, ser_vec)
        
        decrypted_params = [val for vec_sum in sum_encrypted_batches for val in vec_sum.decrypt()]
        param_info = encrypted_bundles[0]['param_info']
        for key, info in param_info.items():
            param_slice = decrypted_params[info['start']:info['end']]
            avg_slice = [val / len(encrypted_bundles) for val in param_slice]
            decrypted_avg_encrypted_dict[key] = torch.tensor(avg_slice, dtype=torch.float32).view(info['shape'])

    # Step 3: Combine parts
    final_avg_state_dict = OrderedDict()
    final_avg_state_dict.update(avg_plaintext_dict)
    final_avg_state_dict.update(decrypted_avg_encrypted_dict)

    # --- NEW: Final step to handle the Opacus '_module.' prefix ---
    # Create a new state_dict, removing the prefix if it exists.
    unwrapped_state_dict = OrderedDict()
    for key, value in final_avg_state_dict.items():
        if key.startswith('_module.'):
            new_key = key[len('_module.'):]  # Remove the prefix
            unwrapped_state_dict[new_key] = value
        else:
            unwrapped_state_dict[key] = value

    # Step 4: Ensure correct key order of the unwrapped dictionary
    # We get the key order from a standard, non-DP model instance to be safe.
    from models import get_model, ArrhythmiaMLP, SmallerCNN # A bit of a circular import, but necessary here for robustness
    temp_config = {'model_name': 'mlp', 'num_features': 13, 'num_classes': 2} # Dummy config
    if "conv1.weight" in unwrapped_state_dict:
        temp_config['model_name'] = 'cnn'
    
    # We need to determine if we are working with an Arrhythmia or CNN model to create the correct template
    # A simple check for a known key can suffice
    if 'layer_1.weight' in unwrapped_state_dict:
        temp_config = {'model_name': 'mlp', 'num_features': unwrapped_state_dict['layer_1.weight'].shape[1], 'num_classes': unwrapped_state_dict['layer_out.weight'].shape[0]}
        template_model = get_model(temp_config)
    elif 'fc1.weight' in unwrapped_state_dict:
        temp_config = {'model_name': 'cnn'}
        template_model = get_model(temp_config)
    else: # Fallback if keys are unusual
        template_model = None

    if template_model:
        final_ordered_dict = OrderedDict()
        for key in template_model.state_dict().keys():
            if key in unwrapped_state_dict:
                final_ordered_dict[key] = unwrapped_state_dict[key]
    else:
        # Fallback to the unwrapped dict if template matching fails
        final_ordered_dict = unwrapped_state_dict

    print(f"    Hybrid Aggregation & Decryption finished in {time.time() - start_time:.2f}s")
    return final_ordered_dict