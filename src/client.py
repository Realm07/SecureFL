
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

client_data_cache = {}
websocket_connection = None

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

async def run_training_and_send_update(client_id: int, task_id: str, payload: dict, context, slot_count):
    """A background task to handle training without blocking the websocket."""
    global websocket_connection
    try:
        print(f"\nClient #{client_id}: Received task '{task_id}' (v{payload.get('model_version', 'sync')}). Starting training...")
        config = payload['config']
        dataloader = get_client_dataloader(client_id, config)
        local_model = get_model(config)
        deserialize_model(local_model, payload['model_state_dict'])
        
        # --- FIX: Run the blocking training function in a separate thread ---
        # This prevents the asyncio event loop from being blocked, allowing the client
        # to respond to websocket pings and avoid timeouts during long training.
        encrypted_update = await asyncio.to_thread(
            train_local_client_secure, local_model, dataloader, config, context, slot_count
        )
        # --------------------------------------------------------------------
        
        if encrypted_update and websocket_connection:
            update_str = serialize_model_update(encrypted_update)
            gc.collect()

            if payload.get('model_version') is not None:
                print(f"Client #{client_id}: Async training for '{task_id}' complete. Sending update.")
                message = {"type": "ASYNC_UPDATE", "payload": {
                    "task_id": task_id, "model_version": payload['model_version'], "update_data": update_str
                }}
                await websocket_connection.send(json.dumps(message))
            else:
                _pending_updates[task_id] = {"update_str": update_str, "round_num": payload['round']}
                print(f"Client #{client_id}: Sync training for '{task_id}' complete. Notifying server.")
                ready_message = {"type": "TRAINING_COMPLETE", "payload": {"task_id": task_id, "round": payload['round']}}
                await websocket_connection.send(json.dumps(ready_message))

    except Exception as e:
        print(f"Client #{client_id}: ERROR during background training for '{task_id}': {e}")
        traceback.print_exc()


async def client_logic(client_id):
    global websocket_connection, _pending_updates
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    POLY_MOD_DEGREE = 16384
    context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
    context.generate_galois_keys()
    context.global_scale = 2**48
    slot_count = POLY_MOD_DEGREE // 2
    _pending_updates = {}

    while True:
        try:
            print(f"Client #{client_id}: Connecting...")
            async with websockets.connect(uri, max_size=2 * 1024 * 1024, ping_interval=20, ping_timeout=60) as websocket:
                websocket_connection = websocket
                print(f"Client #{client_id}: Connected. Waiting for tasks...")
                
                while True: 
                    message = json.loads(await websocket.recv())
                    msg_type, payload = message.get("type"), message.get("payload", {})
                    task_id = payload.get("task_id")

                    if msg_type in ['START_TRAINING', 'NEW_GLOBAL_MODEL']:
                        asyncio.create_task(run_training_and_send_update(client_id, task_id, payload, context, slot_count))

                    elif msg_type == 'REQUEST_UPDATE':
                        if task_id in _pending_updates:
                            pending = _pending_updates[task_id]
                            print(f"Client #{client_id}: Server requested update for '{task_id}'. Uploading...")
                            
                            update_str, round_num = pending['update_str'], pending['round_num']
                            CHUNK_SIZE = 1 * 1024 * 1024
                            total_chunks = math.ceil(len(update_str) / CHUNK_SIZE)

                            await websocket.send(json.dumps({"type": "START_UPDATE_STREAM", "payload": {"task_id": task_id, "round": round_num, "total_chunks": total_chunks}}))
                            for i in range(total_chunks):
                                chunk_data = update_str[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                                await websocket.send(json.dumps({"type": "UPDATE_CHUNK", "payload": {"task_id": task_id, "round": round_num, "chunk_index": i, "data": chunk_data}}))
                                await asyncio.sleep(0.01)

                            print(f"Client #{client_id}: Task '{task_id}' update sent successfully.")
                            del _pending_updates[task_id]

        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            print(f"Client #{client_id}: Connection lost ({type(e).__name__}). Reconnecting...")
        except Exception as e:
            print(f"Client #{client_id}: An unexpected error occurred: {e}")
        finally:
            websocket_connection = None
            await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Federated Learning Client.")
    parser.add_argument('--id', type=int, required=True, help='The unique ID for this client (0-9).')
    args = parser.parse_args()
    asyncio.run(client_logic(args.id))