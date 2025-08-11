import tenseal as ts
from collections import OrderedDict
import torch
import time
import math

def encrypt_state_dict_tenseal(context, state_dict, slot_count):
    flat_params, param_info, current_pos = [], {}, 0
    for key, tensor in state_dict.items():
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
    return OrderedDict([('param_info', param_info), ('encrypted_batches', encrypted_batches)])

def aggregate_and_decrypt_tenseal(context, encrypted_updates, num_clients):
    if not encrypted_updates: return None
    print(f"Server (TenSEAL): Aggregating {len(encrypted_updates)} client updates...")
    start_time = time.time()
    sum_encrypted_batches = [ts.CKKSVector.load(context, ser_vec) for ser_vec in encrypted_updates[0]['encrypted_batches']]
    for update in encrypted_updates[1:]:
        for i, ser_vec in enumerate(update['encrypted_batches']):
            vec_to_add = ts.CKKSVector.load(context, ser_vec)
            sum_encrypted_batches[i] += vec_to_add
    decrypted_params = [val for vec_sum in sum_encrypted_batches for val in vec_sum.decrypt()]
    avg_state_dict = OrderedDict()
    param_info = encrypted_updates[0]['param_info']
    for key, info in param_info.items():
        param_slice = decrypted_params[info['start'] : info['end']]
        avg_slice = [val / num_clients for val in param_slice]
        avg_tensor = torch.tensor(avg_slice, dtype=torch.float32)
        avg_state_dict[key] = avg_tensor.view(info['shape'])
    print(f"    Aggregation & Decryption finished in {time.time() - start_time:.2f}s")
    return avg_state_dict