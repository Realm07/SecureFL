import tenseal as ts
from collections import OrderedDict
import torch
import time

def encrypt_state_dict_tenseal(context, state_dict, slot_count, encrypted_layers=None):
    """
    Encrypts a state dictionary based on the provided list of layer names.
    """
    plaintext_params = OrderedDict()
    params_to_encrypt = OrderedDict()

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

    flat_params, param_info, current_pos = [], {}, 0
    for key, tensor in params_to_encrypt.items():
        numel = tensor.numel()
        param_info[key] = {'shape': tensor.shape, 'start': current_pos, 'end': current_pos + numel}
        flat_params.extend(tensor.cpu().flatten().tolist())
        current_pos += numel
    
    print(f"    Encrypting {len(flat_params)} parameters into batches of {slot_count}...")
    start_time = time.time()
    encrypted_batches = [ts.ckks_vector(context, flat_params[i : i + slot_count]).serialize() for i in range(0, len(flat_params), slot_count)]
    print(f"    Encryption finished in {time.time() - start_time:.2f}s")
    
    encrypted_bundle = OrderedDict([('param_info', param_info), ('encrypted_batches', encrypted_batches)])
    return {'plaintext_params': plaintext_params, 'encrypted_bundle': encrypted_bundle}


def aggregate_and_decrypt_tenseal(context, hybrid_updates, num_clients):
    """
    Aggregates a list of hybrid updates and returns a single, fully decrypted state_dict delta.
    This version is mathematically correct and simplified.
    """
    if not hybrid_updates: return None
    # print(f"Server (TenSEAL): Aggregating {len(hybrid_updates)} hybrid client updates...")
    start_time = time.time()

    # Step 1: Average the plaintext parts
    avg_plaintext_dict = OrderedDict()
    plaintext_parts = [update['plaintext_params'] for update in hybrid_updates if update['plaintext_params']]
    if plaintext_parts:
        for key in plaintext_parts[0].keys():
            sum_tensor = torch.stack([sd[key] for sd in plaintext_parts], dim=0).sum(dim=0)
            avg_plaintext_dict[key] = sum_tensor / len(plaintext_parts)

    # Step 2: Aggregate, decrypt, and average the encrypted parts
    decrypted_avg_encrypted_dict = OrderedDict()
    encrypted_bundles = [update['encrypted_bundle'] for update in hybrid_updates if update['encrypted_bundle'] is not None]
    if encrypted_bundles:
        sum_encrypted_batches = [ts.CKKSVector.load(context, ser_vec) for ser_vec in encrypted_bundles[0]['encrypted_batches']]
        for bundle in encrypted_bundles[1:]:
            for i, ser_vec in enumerate(bundle['encrypted_batches']):
                sum_encrypted_batches[i] += ts.CKKSVector.load(context, ser_vec)
        
        decrypted_sum_params = [val for vec in sum_encrypted_batches for val in vec.decrypt()]
        
        param_info = encrypted_bundles[0]['param_info']
        for key, info in param_info.items():
            param_slice = decrypted_sum_params[info['start']:info['end']]
            sum_tensor = torch.tensor(param_slice, dtype=torch.float32).view(info['shape'])
            
            # avg_delta = (Sum(delta * scale) / scale) / count
            avg_tensor = (sum_tensor / context.global_scale) / len(encrypted_bundles)
            decrypted_avg_encrypted_dict[key] = avg_tensor

    # Step 3: Combine the plaintext and decrypted parts into a single delta dictionary
    final_avg_delta = OrderedDict()
    final_avg_delta.update(avg_plaintext_dict)
    final_avg_delta.update(decrypted_avg_encrypted_dict)

    # print(f"    Hybrid Aggregation & Decryption finished in {time.time() - start_time:.2f}s")
    return final_avg_delta