
import os
import tenseal as ts
import numpy as np
import json
import base64

# This function should be identical to the one in your serialization.py
# We include it here to make the script self-contained.
def serialize_ckks_vector(vector: ts.CKKSVector) -> str:
    """Serializes a TenSEAL CKKS vector to a Base64 encoded string."""
    return base64.b64encode(vector.serialize()).decode('utf-8')

# --- CONFIGURATION ---
DUMMY_CLIENT_IDS = range(10, 20)
VECTOR_SIZE = 33154  # Realistic size for the arrhythmia model
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "prebaked_updates")

def create_dummy_updates():
    """
    Generates and saves serialized dummy TenSEAL vectors for each controller client.
    """
    print(f"--- Generating Pre-baked Updates for Clients {list(DUMMY_CLIENT_IDS)} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 48, 48, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**48

    for client_id in DUMMY_CLIENT_IDS:
        # Create a dummy vector with random data
        dummy_data = np.random.randn(VECTOR_SIZE).tolist()
        encrypted_vector = ts.ckks_vector(context, dummy_data)
        
        # --- THE FIX IS HERE ---
        # 1. Serialize the CKKSVector object into a Base64 string first.
        serialized_vector_str = serialize_ckks_vector(encrypted_vector)

        # 2. Create the dictionary with the *serialized string*, not the raw object.
        prebaked_update_dict = {
            "encrypted_delta": serialized_vector_str
        }
        
        # 3. Now, json.dumps can handle this dictionary of strings perfectly.
        final_json_output = json.dumps(prebaked_update_dict)
        # -----------------------

        # Save to file
        file_path = os.path.join(OUTPUT_DIR, f"prebaked_update_{client_id}.json")
        with open(file_path, 'w') as f:
            f.write(final_json_output)
        print(f"  > Saved pre-baked update to {file_path}")

    print("--- Generation Complete ---")

if __name__ == "__main__":
    create_dummy_updates()