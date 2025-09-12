import asyncio
from datetime import datetime
import json
import os
import random
import traceback
import tenseal as ts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
from collections import OrderedDict
import torch
from torch.optim.lr_scheduler import StepLR
import csv
import numpy as np

from .config import get_config
from .models import get_model
from .he_tenseal import aggregate_and_decrypt_tenseal
from .serialization import serialize_model, deserialize_model_update
from .utils import evaluate_global_model
from .data_loader import get_datasets

app = FastAPI()

class ServerState:
    def __init__(self):
        self.privacy_profile = "she_dp"
        print(f"INFO: Initializing server with Privacy Profile: {self.privacy_profile.upper()}")
        
        self.dataset_name = "arrhythmia"
        self.config = get_config(self.dataset_name)
        
        self.config['num_features'] = 13
        self.config['num_classes'] = 2

        principled_she_layers = [
            'layer_2.weight', 'layer_2.bias', 'groupnorm2.weight', 'groupnorm2.bias',
            'layer_3.weight', 'layer_3.bias', 'groupnorm3.weight', 'groupnorm3.bias',
            'layer_out.weight', 'layer_out.bias'
        ]

        if self.privacy_profile == "she":
            self.config['encrypted_layers'] = principled_she_layers
            self.config['dp_noise_multiplier'] = 0.0
        elif self.privacy_profile == "she_dp":
            self.config['encrypted_layers'] = principled_she_layers
            self.config['dp_noise_multiplier'] = 0.1  # PREVIOUSLY 0.4
            self.config['dp_max_grad_norm'] = 1.0    # PREVIOUSLY 1.5
            print("INFO: DP parameters tuned for model convergence.")

        self.config['num_rounds'] = 50
        self.config['local_epochs'] = 1
        self.config['learning_rate'] = 0.001
        self.config['optimizer'] = 'adam'
        self.config['weight_decay'] = 1e-5
        self.config['delta'] = 1e-5
        
        if self.dataset_name == 'arrhythmia':
            _, self.testset, _, _ = get_datasets(self.config)
        else:
            _, self.testset = get_datasets(self.config)
        
        self.global_model = get_model(self.config)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=1024)
        
        # Using a robust server-side Adam optimizer
        self.server_adam_m = {name: torch.zeros_like(param) for name, param in self.global_model.named_parameters()}
        self.server_adam_v = {name: torch.zeros_like(param) for name, param in self.global_model.named_parameters()}
        self.server_adam_step = 0
        # print("INFO: Server-side Adam state (FedAdam) initialized.")
        
        self.connected_clients: Dict[int, WebSocket] = {}
        self.current_round = 0
        # ... rest of state initialization
        self.updates_for_round, self.clients_ready_for_round, self.update_received_event_for_round, self.chunk_buffers = {}, {}, {}, {}
        self.csv_writer, self.csv_file, self.accuracy_history = None, None, []
        POLY_MOD_DEGREE = 16384
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2**48
        self.slot_count = POLY_MOD_DEGREE // 2

state = ServerState()

def setup_csv_logging():
    results_dir = state.config.get('results_dir', 'results')
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{state.privacy_profile}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    state.csv_file = open(filepath, 'w', newline='')
    state.csv_writer = csv.writer(state.csv_file)
    state.csv_writer.writerow(['round', 'accuracy', 'profile'])
    state.csv_file.flush()
    print(f"--- Logging round-by-round accuracy to {filepath} ---")

def save_final_summary():
    """Saves a final JSON summary of the run."""
    if state.csv_file:
        state.csv_file.close()
    
    results_dir = state.config.get('results_dir', 'results')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"summary_{state.privacy_profile}_{timestamp}.json"
    summary_path = os.path.join(results_dir, summary_filename)

    summary_data = {
    "privacy_profile": state.privacy_profile,
    "config": {k: (str(v) if isinstance(v, torch.device) else v) for k, v in state.config.items()},
    "final_accuracy": getattr(state, "final_accuracy", None),
    "rounds_completed": getattr(state, "rounds_completed", None),
    "timestamp": timestamp,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4, default=str)
    
    print(f"--- Saved final summary to {summary_path} ---")


class ConnectionManager:
    """Manages active WebSocket connections."""
    async def connect(self, websocket: WebSocket, client_id: int):
        await websocket.accept()
        state.connected_clients[client_id] = websocket

    def disconnect(self, client_id: int):
        # Clean up any lingering chunk buffers for this client
        session_key_prefix = f"r{state.current_round}_c{client_id}"
        keys_to_del = [k for k in state.chunk_buffers if k.startswith(session_key_prefix)]
        for k in keys_to_del:
            if k in state.chunk_buffers:
                del state.chunk_buffers[k]
        # Remove from active connections
        if client_id in state.connected_clients:
            del state.connected_clients[client_id]

    async def send_to_client(self, client_id: int, message: str):
        if client_id in state.connected_clients:
            await state.connected_clients[client_id].send_text(message)

manager = ConnectionManager()


async def training_orchestrator():
    setup_csv_logging()

    print("--- Training Orchestrator Started ---")
    initial_acc, _ = evaluate_global_model(state.global_model, state.test_loader, state.config['device'])
    print(f"Initial Global Model Accuracy: {initial_acc:.2f}%")
    state.accuracy_history.append(initial_acc)
    state.csv_writer.writerow([0, initial_acc, state.privacy_profile])
    state.csv_file.flush()

    while state.current_round < state.config['num_rounds']:
        state.current_round += 1
        round_num = state.current_round
        print(f"\n--- Starting Global Round {round_num}/{state.config['num_rounds']}")
        
        connected_ids = list(state.connected_clients.keys())
        if len(connected_ids) < state.config['clients_per_round']:
            print(f"Waiting for more clients... Need {state.config['clients_per_round']}, have {len(connected_ids)}. Retrying in 10s.")
            await asyncio.sleep(10)
            continue

        selected_clients = random.sample(connected_ids, state.config['clients_per_round'])
        # print(f"Selected clients for round {round_num}: {selected_clients}")
        
        state.updates_for_round[round_num] = []
        state.clients_ready_for_round[round_num] = []
        state.update_received_event_for_round[round_num] = asyncio.Event()

        serialized_model = serialize_model(state.global_model)
        message = {
            "type": "START_TRAINING",
            "payload": {
                "round": round_num, "model_state_dict": serialized_model,
                "config": {
                    "local_epochs": state.config['local_epochs'], "learning_rate": state.config['learning_rate'],
                    "optimizer": state.config['optimizer'], "device": str(state.config['device']),
                    "batch_size": state.config['batch_size'], "weight_decay": state.config.get('weight_decay', 0),
                    "encrypted_layers": state.config.get('encrypted_layers'), "privacy_profile": state.privacy_profile,
                    "dp_noise_multiplier": state.config.get('dp_noise_multiplier'),
                    "dp_max_grad_norm": state.config.get('dp_max_grad_norm'), "delta": state.config.get('delta'),
                }
            }
        }
        for client_id in selected_clients: await manager.send_to_client(client_id, json.dumps(message))

        try:
            # print(f"Waiting for {len(selected_clients)} clients to finish training...")
            await asyncio.wait_for(wait_for_clients_ready(round_num, len(selected_clients)), timeout=120.0)
        except asyncio.TimeoutError: print(f"Round {round_num} timed out waiting for clients.")
        
        ready_clients = state.clients_ready_for_round[round_num]
        # print(f"{len(ready_clients)} clients are ready. Starting to pull updates.")
        for client_id in ready_clients:
            try:
                state.update_received_event_for_round[round_num].clear()
                # print(f"Requesting update from Client #{client_id}...")
                await manager.send_to_client(client_id, json.dumps({"type": "REQUEST_UPDATE"}))
                await asyncio.wait_for(state.update_received_event_for_round[round_num].wait(), timeout=90.0)
                # print(f"Successfully received and processed update from Client #{client_id}.")
            except asyncio.TimeoutError: print(f"Timed out waiting for update from Client #{client_id}.")
            except Exception as e: print(f"Error while pulling update from Client #{client_id}: {e}")

        updates_to_aggregate = state.updates_for_round.get(round_num, [])
        if updates_to_aggregate:
            print(f"Aggregating {len(updates_to_aggregate)} update deltas...")
            avg_delta_dict_raw = await asyncio.to_thread(
                aggregate_and_decrypt_tenseal, state.context, updates_to_aggregate, len(updates_to_aggregate)
            )
            
            if avg_delta_dict_raw:
                # --- DEFINITIVE FIX 3: HANDLE OPACUS PREFIX ON THE SERVER ---
                # The server is responsible for aligning the incoming delta keys with its own model keys.
                avg_delta_dict = OrderedDict()
                for key, value in avg_delta_dict_raw.items():
                    clean_key = key[len('_module.'):] if key.startswith('_module.') else key
                    avg_delta_dict[clean_key] = value
                # --------------------------------------------------------------

                # print("Applying updates using server-side Adam (FedAdam)...")
                state.server_adam_step += 1
                beta1, beta2, eps, server_lr = 0.9, 0.999, 1e-8, 0.01
                current_global_dict, new_global_dict = state.global_model.state_dict(), OrderedDict()

                for key, param in state.global_model.named_parameters():
                    delta = avg_delta_dict.get(key, torch.zeros_like(param))
                    delta = delta.to(param.device)
                    grad = -delta
                    
                    state.server_adam_m[key] = beta1 * state.server_adam_m[key] + (1 - beta1) * grad
                    state.server_adam_v[key] = beta2 * state.server_adam_v[key] + (1 - beta2) * (grad ** 2)
                    m_hat = state.server_adam_m[key] / (1 - beta1 ** state.server_adam_step)
                    v_hat = state.server_adam_v[key] / (1 - beta2 ** state.server_adam_step)
                    
                    update_step = server_lr * m_hat / (torch.sqrt(v_hat) + eps)
                    new_global_dict[key] = current_global_dict[key] - update_step

                state.global_model.load_state_dict(new_global_dict)
                
                accuracy, _ = evaluate_global_model(state.global_model, state.test_loader, state.config['device'])
                print(f"--- Round {round_num} Complete --- Global Model Accuracy: {accuracy:.2f}% ---")
                state.accuracy_history.append(accuracy)
                state.csv_writer.writerow([round_num, accuracy, state.privacy_profile])
                state.csv_file.flush()
        
        # print(f"Cleaning up state for round {round_num}.")
        if round_num in state.clients_ready_for_round: del state.clients_ready_for_round[round_num]
        if round_num in state.update_received_event_for_round: del state.update_received_event_for_round[round_num]
        if round_num in state.updates_for_round: del state.updates_for_round[round_num]

    print("\n--- All Federated Learning Rounds Complete ---")
    save_final_summary()

async def wait_for_clients_ready(round_num, num_expected):
    while len(state.clients_ready_for_round.get(round_num, [])) < num_expected: await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event(): asyncio.create_task(training_orchestrator())

def process_full_update(json_string: str, round_num: int):
    # print(f"Reassembly complete for round {round_num}. Deserializing model update...")
    update_payload_dict = json.loads(json_string)
    deserialized_update = deserialize_model_update(update_payload_dict)
    # print(f"Successfully deserialized update for round {round_num}.")
    return deserialized_update

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket, client_id)
    print(f"Client #{client_id} connected.")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type, payload = message.get("type"), message.get("payload", {})
            round_num = payload.get("round")
            
            if msg_type == 'TRAINING_COMPLETE':
                if round_num in state.clients_ready_for_round: state.clients_ready_for_round[round_num].append(client_id)
                # print(f"Client #{client_id} reported TRAINING_COMPLETE for round {round_num}.")
            
            elif msg_type == 'START_UPDATE_STREAM':
                session_key = f"r{round_num}_c{client_id}"
                state.chunk_buffers[session_key] = [None] * payload['total_chunks']
                # print(f"Client #{client_id} starting stream for round {round_num} with {payload['total_chunks']} chunks.")

            elif msg_type == 'UPDATE_CHUNK':
                session_key = f"r{round_num}_c{client_id}"
                if session_key in state.chunk_buffers:
                    state.chunk_buffers[session_key][payload['chunk_index']] = payload['data']
                    if all(c is not None for c in state.chunk_buffers[session_key]):
                        full_json_string = "".join(state.chunk_buffers[session_key])
                        del state.chunk_buffers[session_key]
                        deserialized_update = await asyncio.to_thread(process_full_update, full_json_string, round_num)
                        if round_num in state.updates_for_round:
                            state.updates_for_round[round_num].append(deserialized_update)
                            state.update_received_event_for_round[round_num].set()
            
            elif msg_type == 'RETURN_UPDATE':
                # print(f"Received single update from Client #{client_id} for round {round_num}.")
                deserialized_update = deserialize_model_update(payload['update'])
                if round_num in state.updates_for_round:
                    state.updates_for_round[round_num].append(deserialized_update)
                    state.update_received_event_for_round[round_num].set()

    except WebSocketDisconnect as e: print(f"Client #{client_id} disconnected. Code: {e.code}, Reason: {e.reason}"); manager.disconnect(client_id)
    except Exception as e: print(f"An error occurred with client #{client_id}: {e}"); traceback.print_exc(); manager.disconnect(client_id)