import os
import tenseal as ts
import numpy as np
import json
import base64
from collections import OrderedDict
import torch

from config import get_config
from models import get_model

def serialize_ckks_vector(vector: ts.CKKSVector) -> str:
    """Serializes a TenSEAL CKKS vector to a Base64 encoded string."""
    return base64.b64encode(vector.serialize()).decode('utf-8')

DUMMY_CLIENT_IDS = range(10, 20)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "prebaked_updates")
POLY_MOD_DEGREE = 16384

def create_dummy_updates():
    print(f"--- Generating Pre-baked Updates for Clients {list(DUMMY_CLIENT_IDS)} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    config = get_config('arrhythmia')
    model = get_model(config)
    encrypted_layers = config.get('encrypted_layers', [])

    state_dict_to_encrypt = {k: v for k, v in model.state_dict().items() if k in encrypted_layers}
    
    # 1. Build the correct param_info and calculate total vector size
    param_info = OrderedDict()
    flat_params_template = []
    current_pos = 0
    for key, tensor in state_dict_to_encrypt.items():
        numel = tensor.numel()
        param_info[key] = {'shape': list(tensor.shape), 'start': current_pos, 'end': current_pos + numel}
        flat_params_template.extend(torch.zeros_like(tensor).flatten().tolist())
        current_pos += numel
    
    vector_size = len(flat_params_template)
    print(f"  > Model architecture detected. Encrypting {vector_size} parameters.")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=[60, 48, 48, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**48
    slot_count = POLY_MOD_DEGREE // 2

    for client_id in DUMMY_CLIENT_IDS:
        flat_params = np.random.randn(vector_size).tolist()

        encrypted_batches_raw = [
            ts.ckks_vector(context, flat_params[i : i + slot_count]) 
            for i in range(0, len(flat_params), slot_count)
        ]
        
        encrypted_batches_b64 = [serialize_ckks_vector(vec) for vec in encrypted_batches_raw]

        encrypted_bundle = OrderedDict([
            ('param_info', param_info), 
            ('encrypted_batches', encrypted_batches_b64)
        ])

        prebaked_update_dict = {
            "plaintext_params": {}, 
            "encrypted_bundle": encrypted_bundle
        }

        final_json_output = json.dumps(prebaked_update_dict)

        file_path = os.path.join(OUTPUT_DIR, f"prebaked_update_{client_id}.json")
        with open(file_path, 'w') as f:
            f.write(final_json_output)
        print(f"  > Saved pre-baked update to {file_path}")

    print("--- Generation Complete ---")

if __name__ == "__main__":
    create_dummy_updates()