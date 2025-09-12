# src/serialization.py

import torch
import io
import base64
import json
from collections import OrderedDict

def serialize_model_update(update):
    """
    Serializes a model update (which can be a hybrid dict of plaintext tensors
    and encrypted ciphertexts) into a JSON-safe string.
    """
    # Handle plaintext tensors by encoding them to Base64 strings
    if 'plaintext_params' in update and update['plaintext_params']:
        for key, tensor in update['plaintext_params'].items():
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            update['plaintext_params'][key] = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # --- NEW: Handle encrypted bundle by encoding the raw bytes to Base64 strings ---
    if 'encrypted_bundle' in update and update['encrypted_bundle']:
        # The 'param_info' is already JSON-safe, but the 'encrypted_batches' are raw bytes
        update['encrypted_bundle']['encrypted_batches'] = [
            base64.b64encode(batch).decode('utf-8') 
            for batch in update['encrypted_bundle']['encrypted_batches']
        ]
    
    return json.dumps(update)


def deserialize_model_update(json_str):
    """
    Deserializes a JSON string back into a model update.
    """
    update = json.loads(json_str)
    
    # Handle plaintext tensors by decoding them from Base64 strings
    if 'plaintext_params' in update and update['plaintext_params']:
        for key, b64_str in update['plaintext_params'].items():
            buffer = io.BytesIO(base64.b64decode(b64_str))
            update['plaintext_params'][key] = torch.load(buffer)
            
    # --- NEW: Handle encrypted bundle by decoding the Base64 strings back to bytes ---
    if 'encrypted_bundle' in update and update['encrypted_bundle']:
        update['encrypted_bundle']['encrypted_batches'] = [
            base64.b64decode(b64_str)
            for b64_str in update['encrypted_bundle']['encrypted_batches']
        ]
            
    return update

def serialize_model(model):
    """Serializes a PyTorch model's state_dict to a Base64 string."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def deserialize_model(model, b64_str):
    """Loads a Base64 string into a PyTorch model's state_dict."""
    buffer = io.BytesIO(base64.b64decode(b64_str))
    model.load_state_dict(torch.load(buffer))
    return model