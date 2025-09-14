import asyncio
from datetime import datetime
import json
import os
import random
import traceback
import tenseal as ts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any
from collections import OrderedDict
import torch
import csv
import io
import hashlib

from .config import get_config
from .models import get_model
from .he_tenseal import aggregate_and_decrypt_tenseal
from .serialization import serialize_model, deserialize_model_update
from .utils import evaluate_global_model
from .data_loader import get_datasets
from .ledger import FederationLedger
from .tokenomics import TokenManager

app = FastAPI()

def hash_model_state(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

class FederationTask:
    """Manages the state and execution of a single federated learning task."""
    def __init__(self, task_id: str, privacy_profile: str):
        self.task_id = task_id
        self.privacy_profile = privacy_profile
        print(f"INFO: Initializing task '{self.task_id}' with profile '{self.privacy_profile.upper()}'")

        self.config = get_config(self.task_id)
        
        # Configure privacy settings
        if "dp" in self.privacy_profile:
            self.config['dp_noise_multiplier'] = self.config.get('dp_noise_multiplier', 0.5)
            self.config['dp_max_grad_norm'] = self.config.get('dp_max_grad_norm', 1.5)
        else:
            self.config['dp_noise_multiplier'] = 0.0
        
        # Data and Model
        _, self.testset = get_datasets(self.config)
        self.global_model = get_model(self.config)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=512)
        
        # FedAdam State
        self.server_adam_m = {name: torch.zeros_like(p) for name, p in self.global_model.named_parameters()}
        self.server_adam_v = {name: torch.zeros_like(p) for name, p in self.global_model.named_parameters()}
        self.server_adam_step = 0
        
        # Round State
        self.current_round = 0
        self.updates_for_round: Dict[int, List] = {}
        self.clients_ready_for_round: Dict[int, List] = {}
        self.update_received_event_for_round: Dict[int, asyncio.Event] = {}
        self.chunk_buffers: Dict[str, List] = {}

        # Logging and Auditing
        self.metric_history = []
        self.setup_logging()
        ledger_file = os.path.join(self.config['results_dir'], f'ledger_{self.task_id}.json')
        self.ledger = FederationLedger(storage_path=ledger_file)

    def setup_logging(self):
        results_dir = self.config['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{self.task_id}_{self.privacy_profile}_{timestamp}.csv"
        filepath = os.path.join(results_dir, filename)
        self.csv_file = open(filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['round', self.config['metric'], 'profile'])
        self.csv_file.flush()
        print(f"--- Task '{self.task_id}' logging to {filepath} ---")

class ServerManager:
    """Global server state manager, holding all tasks and shared resources."""
    def __init__(self):
        self.tasks: Dict[str, FederationTask] = {}
        self.connected_clients: Dict[int, WebSocket] = {}
        
        # Shared resources
        token_file = os.path.join(get_config('arrhythmia')['results_dir'], 'token_balances.json')
        self.token_manager = TokenManager(storage_path=token_file)
        self.MINIMUM_STAKE = 50.0
        self.REWARD_AMOUNT = 10.0
        
        POLY_MOD_DEGREE = 16384
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2**48

    def add_task(self, task: FederationTask):
        self.tasks[task.task_id] = task

manager = ServerManager()
manager.add_task(FederationTask(task_id="arrhythmia", privacy_profile="she_dp"))
manager.add_task(FederationTask(task_id="nasa_battery", privacy_profile="she"))

async def task_orchestrator_loop(task_id: str):
    task = manager.tasks[task_id]
    metric_name = task.config['metric'].upper()
    metric_unit = "cycles" if metric_name == "RMSE" else "%"

    print(f"--- Orchestrator for task '{task_id}' started. ---")
    initial_metric, _ = evaluate_global_model(task.global_model, task.test_loader, task.config['device'], task.config['metric'])
    print(f"Task '{task_id}' Initial Global Model {metric_name}: {initial_metric:.2f} {metric_unit}")
    task.metric_history.append(initial_metric)
    task.csv_writer.writerow([0, initial_metric, task.privacy_profile])
    task.csv_file.flush()

    while task.current_round < task.config['num_rounds']:
        await asyncio.sleep(5) # Stagger tasks
        task.current_round += 1
        round_num = task.current_round
        
        print(f"\n--- Task '{task_id}' | Round {round_num}/{task.config['num_rounds']} ---")
        
        connected_ids = list(manager.connected_clients.keys())
        eligible_clients = [cid for cid in connected_ids if manager.token_manager.has_sufficient_stake(cid, manager.MINIMUM_STAKE)]
        
        if len(eligible_clients) < task.config['clients_per_round']:
            print(f"Task '{task_id}': Waiting for more eligible clients... Have {len(eligible_clients)}/{task.config['clients_per_round']}.")
            await asyncio.sleep(10)
            task.current_round -=1 # Retry this round number
            continue

        selected_clients = random.sample(eligible_clients, task.config['clients_per_round'])
        
        task.updates_for_round[round_num] = []
        task.clients_ready_for_round[round_num] = []
        task.update_received_event_for_round[round_num] = asyncio.Event()

        serialized_model = serialize_model(task.global_model)
        client_config = {
            # Environment and Data Partitioning
            "data_root": task.config['data_root'],
            "num_clients": task.config['num_clients'],
            
            # Task identifiers
            "dataset_name": task.config['dataset_name'],
            "model_name": task.config['model_name'],
            
            # Training Hyperparameters
            "local_epochs": task.config['local_epochs'],
            "learning_rate": task.config['learning_rate'],
            "optimizer": task.config['optimizer'],
            "batch_size": task.config['batch_size'],
            "weight_decay": task.config.get('weight_decay', 0),

            # --- DEFINITIVE FIX: PROVIDE DEFAULTS FOR SCHEDULER PARAMS ---
            "lr_scheduler_step_size": task.config.get('lr_scheduler_step_size', 100),
            "lr_scheduler_gamma": task.config.get('lr_scheduler_gamma', 1.0),
            # ----------------------------------------------------------------
            
            # Privacy and Security Parameters
            "privacy_profile": task.privacy_profile,
            "encrypted_layers": task.config.get('encrypted_layers'),
            "dp_noise_multiplier": task.config.get('dp_noise_multiplier'),
            "dp_max_grad_norm": task.config.get('dp_max_grad_norm'),
            "delta": task.config.get('delta'),
            
            # Model architecture (if needed by client)
            "num_features": task.config.get('num_features'),
            "num_classes": task.config.get('num_classes'),
            "sequence_length": task.config.get('sequence_length'),
            "lstm_hidden_dim": task.config.get('lstm_hidden_dim'),
            "lstm_n_layers": task.config.get('lstm_n_layers'),
            "lstm_drop_prob": task.config.get('lstm_drop_prob'), # Pass the new dropout prob
            
            # Environment
            "device": str(task.config['device']),
            "metric": task.config.get('metric') # Pass the metric for loss function selection
        }
        
        message = {
            "type": "START_TRAINING",
            "payload": {
                "task_id": task_id,
                "round": round_num,
                "model_state_dict": serialized_model,
                "config": client_config # Use the clean, serializable config
            }
        }
        for client_id in selected_clients:
            await manager.connected_clients[client_id].send_text(json.dumps(message))

        try:
            # The timeout for waiting for clients to be ready
            await asyncio.wait_for(wait_for_clients_ready(task_id, round_num, len(selected_clients)), timeout=120.0)
        except asyncio.TimeoutError: 
            print(f"Task '{task_id}' Round {round_num} timed out waiting for clients to report completion.")

        ready_clients = task.clients_ready_for_round.get(round_num, [])
        
        # --- DEFINITIVE FIX: CHECK FOR CONNECTION BEFORE SENDING ---
        for client_id in ready_clients:
            # If the client disconnected after being selected, they won't be in the manager anymore.
            if client_id in manager.connected_clients:
                try:
                    task.update_received_event_for_round[round_num].clear()
                    await manager.connected_clients[client_id].send_text(json.dumps({
                        "type": "REQUEST_UPDATE", 
                        "payload": {"task_id": task_id, "round": round_num}
                    }))
                    await asyncio.wait_for(task.update_received_event_for_round[round_num].wait(), timeout=90.0)
                except asyncio.TimeoutError: 
                    print(f"Task '{task_id}': Timed out waiting for update from Client #{client_id}.")
                except Exception as e:
                    print(f"Task '{task_id}': Error requesting update from Client #{client_id}: {e}")
            else:
                # This client disconnected during training. Log it and move on.
                print(f"INFO: Task '{task_id}': Client #{client_id} disconnected before update could be requested.")
        
        updates = task.updates_for_round.get(round_num, [])
        if updates:
            print(f"Task '{task_id}': Aggregating {len(updates)} updates...")
            avg_delta = await asyncio.to_thread(aggregate_and_decrypt_tenseal, manager.context, updates, len(updates))
            
            if avg_delta:
                # Apply update using FedAdam
                task.server_adam_step += 1
                beta1, beta2, eps, server_lr = 0.9, 0.999, 1e-8, 0.05
                current_dict = task.global_model.state_dict()
                new_dict = OrderedDict()
                for key, param in task.global_model.named_parameters():
                    delta = avg_delta.get(key, torch.zeros_like(param)).to(param.device)
                    grad = -delta
                    task.server_adam_m[key] = beta1 * task.server_adam_m[key] + (1 - beta1) * grad
                    task.server_adam_v[key] = beta2 * task.server_adam_v[key] + (1 - beta2) * (grad ** 2)
                    m_hat = task.server_adam_m[key] / (1 - beta1 ** task.server_adam_step)
                    v_hat = task.server_adam_v[key] / (1 - beta2 ** task.server_adam_step)
                    update_step = server_lr * m_hat / (torch.sqrt(v_hat) + eps)
                    new_dict[key] = current_dict[key] - update_step
                task.global_model.load_state_dict(new_dict)
                
                # Evaluate and log
                metric_val, _ = evaluate_global_model(task.global_model, task.test_loader, task.config['device'], task.config['metric'])
                print(f"--- Task '{task_id}' Round {round_num} Complete --- {metric_name}: {metric_val:.2f} {metric_unit} ---")
                
                manager.token_manager.reward_clients(selected_clients, manager.REWARD_AMOUNT)
                task.metric_history.append(metric_val)
                task.csv_writer.writerow([round_num, metric_val, task.privacy_profile])
                task.csv_file.flush()
                model_hash = hash_model_state(task.global_model)
                task.ledger.add_round_to_ledger(round_num, selected_clients, model_hash, metric_val)

        # Cleanup
        for key in [task.clients_ready_for_round, task.update_received_event_for_round, task.updates_for_round]:
            if round_num in key: del key[round_num]

    print(f"\n--- Task '{task_id}' All Rounds Complete ---")
    if task.csv_file: task.csv_file.close()

async def wait_for_clients_ready(task_id: str, round_num: int, num_expected: int):
    task = manager.tasks[task_id]
    while len(task.clients_ready_for_round.get(round_num, [])) < num_expected:
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    for task_id in manager.tasks:
        asyncio.create_task(task_orchestrator_loop(task_id))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await websocket.accept()
    manager.connected_clients[client_id] = websocket
    manager.token_manager.register_client(client_id)
    print(f"Client #{client_id} connected.")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            payload = message.get("payload", {})
            task_id = payload.get("task_id")
            round_num = payload.get("round")

            if not task_id or task_id not in manager.tasks:
                print(f"WARNING: Received message with invalid task_id: {task_id}")
                continue
            
            task = manager.tasks[task_id]

            if msg_type == 'TRAINING_COMPLETE':
                task.clients_ready_for_round.setdefault(round_num, []).append(client_id)
            elif msg_type == 'START_UPDATE_STREAM':
                session_key = f"t{task_id}_r{round_num}_c{client_id}"
                task.chunk_buffers[session_key] = [None] * payload['total_chunks']
            elif msg_type == 'UPDATE_CHUNK':
                session_key = f"t{task_id}_r{round_num}_c{client_id}"
                if session_key in task.chunk_buffers:
                    task.chunk_buffers[session_key][payload['chunk_index']] = payload['data']
                    if all(c is not None for c in task.chunk_buffers[session_key]):
                        full_json_string = "".join(task.chunk_buffers[session_key])
                        del task.chunk_buffers[session_key]
                        update = await asyncio.to_thread(deserialize_model_update, full_json_string)
                        task.updates_for_round.setdefault(round_num, []).append(update)
                        task.update_received_event_for_round[round_num].set()
    except WebSocketDisconnect:
        print(f"Client #{client_id} disconnected.")
    except Exception:
        print(f"An error occurred with client #{client_id}: {traceback.format_exc()}")
    finally:
        if client_id in manager.connected_clients:
            del manager.connected_clients[client_id]

# --- NEW AND UPDATED API ENDPOINTS ---
@app.get("/tasks")
async def get_tasks():
    """Lists all available federated learning tasks and their status."""
    response = {}
    for task_id, task in manager.tasks.items():
        response[task_id] = {
            "task_id": task_id,
            "privacy_profile": task.privacy_profile,
            "current_round": task.current_round,
            "total_rounds": task.config['num_rounds'],
            "metric": task.config['metric'],
            "latest_metric_value": task.metric_history[-1] if task.metric_history else None,
            "is_running": task.current_round < task.config['num_rounds']
        }
    return response

@app.get("/tasks/{task_id}/ledger")
async def get_task_ledger(task_id: str):
    if task_id in manager.tasks:
        return manager.tasks[task_id].ledger.chain
    return {"error": "Task not found"}, 404

@app.get("/tokenomics")
async def get_tokenomics_state():
    return manager.token_manager.get_all_accounts()