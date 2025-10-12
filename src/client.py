import asyncio
import traceback
import websockets
import argparse
import json
import time
import tenseal as ts
import gc
import math
from typing import Set

from .client_data_manager import ClientDataManager
from .models import get_model
from .fl_logic import train_local_client_secure
from .serialization import serialize_model_update, deserialize_model

websocket_connection = None
_pending_updates = {}
tasks_in_progress: Set[str] = set()
data_manager = ClientDataManager()

async def run_training_and_send_update(client_id: int, task_id: str, payload: dict, context, slot_count):
    """A background task to handle training without blocking the websocket."""
    global websocket_connection, tasks_in_progress
    try:
        print(f"\nClient #{client_id}: Received task '{task_id}' (v{payload.get('model_version', 'sync')}). Starting training...")
        config = payload['config']
        
        dataloader = data_manager.get_dataloader(client_id, config)
        
        local_model = get_model(config)
        deserialize_model(local_model, payload['model_state_dict'])
        
        # --- MODIFIED: Pass client_id and round number to the training function ---
        round_num = payload.get('round', payload.get('model_version', 0))
        encrypted_update = await asyncio.to_thread(
            train_local_client_secure, client_id, round_num, local_model, dataloader, config, context, slot_count
        )
        # --- END MODIFICATION ---
        
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
    finally:
        if task_id in tasks_in_progress:
            tasks_in_progress.remove(task_id)

async def client_logic(client_id):
    global websocket_connection, _pending_updates, tasks_in_progress
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    POLY_MOD_DEGREE = 16384
    context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
    context.generate_galois_keys()
    context.global_scale = 2**48
    slot_count = POLY_MOD_DEGREE // 2

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
                        if task_id in tasks_in_progress:
                            print(f"Client #{client_id}: Ignoring new request for task '{task_id}' as it's already in progress.")
                            continue
                        
                        tasks_in_progress.add(task_id)
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
            tasks_in_progress.clear()
            await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Federated Learning Client.")
    parser.add_argument('--id', type=int, required=True, help='The unique ID for this client (0-9).')
    args = parser.parse_args()
    asyncio.run(client_logic(args.id))