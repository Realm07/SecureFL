# src/client.py

import asyncio
import traceback
import websockets
import argparse
import json
import torch
from torch.utils.data import DataLoader
import tenseal as ts
import gc
import math

from .config import get_config
from .models import get_model
from .data_loader import get_client_datasets 
from .fl_logic import train_local_client_secure
from .serialization import serialize_model_update, deserialize_model

def get_client_dataloader(client_id: int, config: dict):
    dataset_name = config['dataset_name']
    cache_key = f"{client_id}_{dataset_name}"
    
    if cache_key in client_data_cache:
        print(f"Client #{client_id}: Loading '{dataset_name}' data from cache.")
        return client_data_cache[cache_key]
    
    print(f"Client #{client_id}: Loading '{dataset_name}' data from disk...")
    client_datasets = get_client_datasets(config)
    my_dataset = client_datasets[client_id]
    my_dataloader = DataLoader(my_dataset, batch_size=config['batch_size'], shuffle=True)
    
    client_data_cache[cache_key] = my_dataloader
    print(f"Client #{client_id}: Data loaded for '{dataset_name}'. {len(my_dataset)} samples.")
    return my_dataloader


async def client_logic(client_id):
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    POLY_MOD_DEGREE = 16384
    context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
    context.generate_galois_keys()
    context.global_scale = 2**48
    slot_count = POLY_MOD_DEGREE // 2

    _pending_updates = {} # task_id -> {update_str, round_num}

    while True:
        try:
            print(f"Client #{client_id}: Connecting...")
            # --- THE FIX: INCREASE WEBSOCKET TIMEOUTS ---
            async with websockets.connect(
                uri, 
                max_size=2 * 1024 * 1024,
                ping_interval=120,  
                ping_timeout=300 
            ) as websocket:
            # ----------------------------------------
                print(f"Client #{client_id}: Connected. Waiting for tasks...")
                
                while True: 
                    message = json.loads(await websocket.recv())
                    msg_type = message.get("type")
                    payload = message.get("payload", {})
                    task_id = payload.get("task_id")

                    if msg_type == 'START_TRAINING':
                        try:
                            round_num = payload['round']
                            round_config = payload['config']
                            
                            print(f"\nClient #{client_id}: Received task '{task_id}' for round {round_num}")
                            
                            dataloader = get_client_dataloader(client_id, round_config)
                            
                            local_model = get_model(round_config)
                            deserialize_model(local_model, payload['model_state_dict'])
                            
                            encrypted_update = train_local_client_secure(
                                local_model, dataloader, round_config, context, slot_count
                            )
                            
                            if encrypted_update:
                                _pending_updates[task_id] = {
                                    "update_str": serialize_model_update(encrypted_update), # Store the JSON string directly
                                    "round_num": round_num
                                }
                                gc.collect()
                                print(f"Client #{client_id}: Task '{task_id}' training complete. Notifying server.")
                                ready_message = {"type": "TRAINING_COMPLETE", "payload": {"task_id": task_id, "round": round_num}}
                                await websocket.send(json.dumps(ready_message))
                        except Exception as e:
                            print(f"Client #{client_id}: ERROR during training for task '{task_id}': {e}")
                            traceback.print_exc()

                    elif msg_type == 'REQUEST_UPDATE':
                        try:
                            if task_id in _pending_updates:
                                pending = _pending_updates[task_id]
                                round_num = pending['round_num']
                                print(f"Client #{client_id}: Server requested update for task '{task_id}'. Uploading...")
                                
                                # --- THE FIX ---
                                # The data is already a JSON string. No need to dump it again.
                                update_str = pending['update_str']
                                # ----------------

                                CHUNK_SIZE = 1 * 1024 * 1024
                                total_chunks = math.ceil(len(update_str) / CHUNK_SIZE)

                                await websocket.send(json.dumps({
                                    "type": "START_UPDATE_STREAM",
                                    "payload": {"task_id": task_id, "round": round_num, "total_chunks": total_chunks}
                                }))

                                for i in range(total_chunks):
                                    chunk_data = update_str[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                                    await websocket.send(json.dumps({
                                        "type": "UPDATE_CHUNK",
                                        "payload": {"task_id": task_id, "round": round_num, "chunk_index": i, "data": chunk_data}
                                    }))
                                    await asyncio.sleep(0.01)

                                print(f"Client #{client_id}: Task '{task_id}' update sent successfully.")
                                del _pending_updates[task_id]
                        except Exception as e:
                            print(f"Client #{client_id}: ERROR during update sending for task '{task_id}': {e}")

        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            print(f"Client #{client_id}: Connection lost ({type(e).__name__}). Reconnecting...")
        except Exception as e:
            print(f"Client #{client_id}: An unexpected error occurred: {e}")
        
        await asyncio.sleep(5)

# Add the client_data_cache definition here
client_data_cache = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Federated Learning Client.")
    parser.add_argument('--id', type=int, required=True, help='The unique ID for this client (0-9).')
    args = parser.parse_args()
    asyncio.run(client_logic(args.id))