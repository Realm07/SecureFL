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
from .data_loader import get_datasets, partition_iid
from .fl_logic import train_local_client_secure
from .serialization import serialize_model_update, deserialize_model

async def client_logic(client_id):
    uri = f"ws://localhost:8000/ws/{client_id}"
    
    print(f"Client #{client_id}: Loading data...")
    base_config = get_config("arrhythmia") 
    if base_config['dataset_name'] == 'arrhythmia':
        trainset, _, _, _ = get_datasets(base_config)
    else:
        trainset, _ = get_datasets(base_config)
    
    client_datasets, _ = partition_iid(trainset, base_config['num_clients'])
    my_dataset = client_datasets[client_id]
    my_dataloader = DataLoader(my_dataset, batch_size=base_config['batch_size'], shuffle=True)
    print(f"Client #{client_id}: Data loaded. {len(my_dataset)} samples.")

    POLY_MOD_DEGREE = 16384
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 
        POLY_MOD_DEGREE, 
        coeff_mod_bit_sizes=[60, 48, 48, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**48 
    slot_count = POLY_MOD_DEGREE // 2

    MAX_RETRY_DELAY = 60.0
    retry_delay = 1.0

    _pending_serialized_update = None
    _pending_round_num = None
    
    while True:
        try:
            print(f"Client #{client_id}: Attempting to connect to server...")
            async with websockets.connect(
                uri, 
                max_size=2 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20
            ) as websocket:
                print(f"Client #{client_id}: Connected to server. Waiting for instructions...")
                retry_delay = 1.0
                
                while True: 
                    message_str = await websocket.recv()
                    message = json.loads(message_str)
                    
                    if message['type'] == 'START_TRAINING':
                        try:
                            payload = message['payload']
                            round_num = payload['round']
                            round_config = payload['config']
                            
                            print(f"\nClient #{client_id}: Received START_TRAINING for round {round_num}")
                            
                            local_model = get_model(base_config)
                            deserialize_model(local_model, payload['model_state_dict'])
                            
                            encrypted_update = train_local_client_secure(
                                local_model, my_dataloader, round_config, context, slot_count
                            )
                            
                            if encrypted_update:
                                print(f"Client #{client_id}: Training complete. Serializing update...")
                                _pending_serialized_update = serialize_model_update(encrypted_update)
                                _pending_round_num = round_num
                                
                                del encrypted_update
                                gc.collect()
                                print(f"Client #{client_id}: Serialization complete. Notifying server.")

                                ready_message = {
                                    "type": "TRAINING_COMPLETE",
                                    "payload": {"round": round_num, "client_id": client_id}
                                }
                                await websocket.send(json.dumps(ready_message))
                                print(f"Client #{client_id}: Notification sent. Waiting for server to request the update...")
                        except Exception as e:
                            print(f"Client #{client_id}: ERROR during training: {e}")
                            traceback.print_exc()
                            continue
                    
                    elif message['type'] == 'REQUEST_UPDATE':
                        try:
                            if _pending_serialized_update and _pending_round_num is not None:
                                print(f"Client #{client_id}: Server requested update for round {_pending_round_num}. Starting chunked upload...")
                                
                                json_update_payload = json.dumps(_pending_serialized_update)
                                CHUNK_SIZE = 1 * 1024 * 1024
                                total_size = len(json_update_payload)
                                total_chunks = math.ceil(total_size / CHUNK_SIZE)

                                start_stream_msg = {
                                    "type": "START_UPDATE_STREAM",
                                    "payload": {
                                        "round": _pending_round_num,
                                        "client_id": client_id,
                                        "total_chunks": total_chunks
                                    }
                                }
                                await websocket.send(json.dumps(start_stream_msg))

                                for i in range(total_chunks):
                                    start = i * CHUNK_SIZE
                                    end = start + CHUNK_SIZE
                                    chunk_data = json_update_payload[start:end]
                                    
                                    chunk_msg = {
                                        "type": "UPDATE_CHUNK",
                                        "payload": {
                                            "round": _pending_round_num,
                                            "client_id": client_id,
                                            "chunk_index": i,
                                            "data": chunk_data
                                        }
                                    }
                                    await websocket.send(json.dumps(chunk_msg))
                                    # print(f"Client #{client_id}: Sent chunk {i+1}/{total_chunks}")
                                    await asyncio.sleep(0.01)

                                print(f"Client #{client_id}: All chunks sent successfully! Waiting for next round.")
                                
                                _pending_serialized_update = None
                                _pending_round_num = None
                            else:
                                print(f"Client #{client_id}: Received REQUEST_UPDATE but have no pending update to send.")

                        except Exception as e:
                            print(f"Client #{client_id}: ERROR during update sending: {e}")
                            traceback.print_exc()
                            continue
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Client #{client_id}: Connection closed ({e.code}). Reconnecting...")
        except ConnectionRefusedError:
            print(f"Client #{client_id}: Connection refused. Is the server running?")
        except Exception as e:
            print(f"Client #{client_id}: An unexpected error occurred: {e}")
            traceback.print_exc()

        print(f"Client #{client_id}: Retrying connection in {retry_delay:.1f}s...")
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 1.5, MAX_RETRY_DELAY)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Federated Learning Client.")
    parser.add_argument('--id', type=int, required=True, help='The unique ID for this client (0-9).')
    args = parser.parse_args()
    
    asyncio.run(client_logic(args.id))