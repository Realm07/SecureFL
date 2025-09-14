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
from enum import Enum

from .config import get_config
from .models import get_model
from .he_tenseal import aggregate_and_decrypt_tenseal
from .serialization import serialize_model, deserialize_model_update
from .utils import evaluate_global_model
from .data_manager import DataManager
from .ledger import FederationLedger
from .tokenomics import TokenManager

app = FastAPI()

class TaskStatus(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_CLIENTS = "WAITING_FOR_CLIENTS"
    RUNNING_ROUND = "RUNNING_ROUND"
    COMPLETED = "COMPLETED"

def task_log(task_id: str, message: str):
    """A simple structured logger to de-jumble concurrent output."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Task: {task_id}] {message}")

def hash_model_state(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

class FederationTask:
    """Manages the state and execution of a single federated learning task."""
    def __init__(self, task_id: str, privacy_profile: str, data_manager: DataManager):
        self.task_id = task_id
        self.privacy_profile = privacy_profile
        self.status: TaskStatus = TaskStatus.IDLE
        print(f"INFO: Initializing task '{self.task_id}' with profile '{self.privacy_profile.upper()}'")

        self.config = data_manager.get_task_config(task_id)
        
        # Configure privacy settings
        if "dp" in self.privacy_profile:
            self.config['dp_noise_multiplier'] = self.config.get('dp_noise_multiplier', 0.5)
            self.config['dp_max_grad_norm'] = self.config.get('dp_max_grad_norm', 1.5)
        else:
            self.config['dp_noise_multiplier'] = 0.0
        
        # Data and Model
        self.testset = data_manager.get_test_set(task_id) 
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
    
    def is_complete(self):
        return self.current_round >= self.config['num_rounds']

    async def execute_round(self, manager_instance):
        """Contains the logic for a single round of federated learning."""
        self.status = TaskStatus.RUNNING_ROUND
        self.current_round += 1
        round_num = self.current_round
        
        task_log(self.task_id, f"--- Round {round_num}/{self.config['num_rounds']} ---")

        # 1. Select Clients (This part is already correct in your code)
        eligible_clients = [cid for cid in manager_instance.connected_clients if manager_instance.token_manager.has_sufficient_stake(cid, manager_instance.MINIMUM_STAKE)]
        if len(eligible_clients) < self.config['clients_per_round']:
            task_log(self.task_id, "Not enough eligible clients. Waiting...")
            self.status = TaskStatus.WAITING_FOR_CLIENTS
            self.current_round -= 1
            await asyncio.sleep(10)
            return
        
        selected_clients = random.sample(eligible_clients, self.config['clients_per_round'])
        task_log(self.task_id, f"Selected clients for round: {selected_clients}")
        
        # 2. Start Training on Clients
        self.updates_for_round[round_num] = []
        self.clients_ready_for_round[round_num] = []
        self.update_received_event_for_round[round_num] = asyncio.Event()

        client_config = self._create_client_config() # Create a helper for this
        message = {
            "type": "START_TRAINING",
            "payload": { "task_id": self.task_id, "round": round_num, "model_state_dict": serialize_model(self.global_model), "config": client_config }
        }
        for client_id in selected_clients:
            await manager_instance.connected_clients[client_id].send_text(json.dumps(message))

        # 3. Collect Updates
        try:
            # --- FIX: LONGER TIMEOUT FOR FIRST ROUND ---
            timeout = 300.0 if round_num == 1 else 120.0
            await asyncio.wait_for(self._wait_for_clients_ready(round_num, len(selected_clients)), timeout=timeout)
        except asyncio.TimeoutError:
            task_log(self.task_id, f"Round {round_num} timed out waiting for clients to report completion.")

        ready_clients = self.clients_ready_for_round.get(round_num, [])
        for client_id in ready_clients:
            if client_id in manager_instance.connected_clients:
                try:
                    self.update_received_event_for_round[round_num].clear()
                    await manager_instance.connected_clients[client_id].send_text(json.dumps({ "type": "REQUEST_UPDATE", "payload": {"task_id": self.task_id, "round": round_num}}))
                    await asyncio.wait_for(self.update_received_event_for_round[round_num].wait(), timeout=90.0)
                except asyncio.TimeoutError:
                    print(f"Task '{self.task_id}': Timed out waiting for update from Client #{client_id}.")
                except Exception as e:
                    print(f"Task '{self.task_id}': Error requesting update from Client #{client_id}: {e}")
            else:
                print(f"INFO: Task '{self.task_id}': Client #{client_id} disconnected before update could be requested.")
        
        # 4. Aggregate and Update
        updates = self.updates_for_round.get(round_num, [])
        if updates:
            # print(f"Task '{self.task_id}': Aggregating {len(updates)} updates...")
            avg_delta = await asyncio.to_thread(aggregate_and_decrypt_tenseal, manager_instance.context, updates, len(updates))
            if avg_delta:
                self._apply_update(avg_delta)
                metric_val = self._evaluate_and_log(round_num, selected_clients)
                manager_instance.token_manager.reward_clients(selected_clients, manager_instance.REWARD_AMOUNT)
                self.ledger.add_round_to_ledger(round_num, selected_clients, hash_model_state(self.global_model), metric_val)
        else:
            print(f"Task '{self.task_id}': No valid updates received for round {round_num}. Skipping model update.")
            
        # 5. Cleanup
        for state_dict in [self.clients_ready_for_round, self.update_received_event_for_round, self.updates_for_round]:
            if round_num in state_dict:
                del state_dict[round_num]
        
        self.status = TaskStatus.IDLE
        if self.is_complete():
            self.status = TaskStatus.COMPLETED
            print(f"\n")
            task_log(self.task_id, "--- All Rounds Complete ---")
            if self.csv_file: self.csv_file.close()

    def _create_client_config(self):
        # This is the same logic as before, just encapsulated
        return {
            # Environment and Data Partitioning
            "data_root": self.config['data_root'],
            "num_clients": self.config['num_clients'],
            
            # --- ADD THIS LINE ---
            "nasa_data_folder": self.config.get('nasa_data_folder'),
            # ---------------------
            
            # Task identifiers
            "dataset_name": self.config['dataset_name'],
            "model_name": self.config['model_name'],
            
            # Training Hyperparameters
            "local_epochs": self.config['local_epochs'],
            "learning_rate": self.config['learning_rate'],
            "optimizer": self.config['optimizer'],
            "batch_size": self.config['batch_size'],
            "weight_decay": self.config.get('weight_decay', 0),
            "lr_scheduler_step_size": self.config.get('lr_scheduler_step_size', 100),
            "lr_scheduler_gamma": self.config.get('lr_scheduler_gamma', 1.0),
            
            # Privacy and Security Parameters
            "privacy_profile": self.privacy_profile,
            "encrypted_layers": self.config.get('encrypted_layers'),
            "dp_noise_multiplier": self.config.get('dp_noise_multiplier'),
            "dp_max_grad_norm": self.config.get('dp_max_grad_norm'),
            "delta": self.config.get('delta'),
            
            # Model architecture (if needed by client)
            "num_features": self.config.get('num_features'),
            "num_classes": self.config.get('num_classes'),
            "sequence_length": self.config.get('sequence_length'),
            "lstm_hidden_dim": self.config.get('lstm_hidden_dim'),
            "lstm_n_layers": self.config.get('lstm_n_layers'),
            "lstm_drop_prob": self.config.get('lstm_drop_prob'),
            
            # Environment
            "device": str(self.config['device']),
            "metric": self.config.get('metric')
        }

    def _apply_update(self, avg_delta):
        # This is the FedAdam logic, encapsulated
        self.server_adam_step += 1
        beta1, beta2, eps, server_lr = 0.9, 0.999, 1e-8, 0.05
        current_dict = self.global_model.state_dict()
        new_dict = OrderedDict()

        for key, param in self.global_model.named_parameters():
            delta = avg_delta.get(key, torch.zeros_like(param)).to(param.device)
            grad = -delta

            self.server_adam_m[key] = beta1 * self.server_adam_m[key] + (1 - beta1) * grad
            self.server_adam_v[key] = beta2 * self.server_adam_v[key] + (1 - beta2) * (grad ** 2)

            m_hat = self.server_adam_m[key] / (1 - beta1 ** self.server_adam_step)
            v_hat = self.server_adam_v[key] / (1 - beta2 ** self.server_adam_step)

            update_step = server_lr * m_hat / (torch.sqrt(v_hat) + eps)
            new_dict[key] = current_dict[key] - update_step

        self.global_model.load_state_dict(new_dict)

    def _evaluate_and_log(self, round_num, selected_clients):
        metric_val, _ = evaluate_global_model(self.global_model, self.test_loader, self.config['device'], self.config['metric'])
        metric_name = self.config['metric'].upper()
        metric_unit = "cycles" if metric_name == "RMSE" else "%"
        # Use the new logger
        task_log(self.task_id, f"--- Round {round_num} Complete --- {metric_name}: {metric_val:.2f} {metric_unit} ---")
        self.metric_history.append(metric_val)
        self.csv_writer.writerow([round_num, metric_val, self.privacy_profile])
        self.csv_file.flush()
        return metric_val

    
    async def _wait_for_clients_ready(self, round_num, num_expected):
        while len(self.clients_ready_for_round.get(round_num, [])) < num_expected:
            await asyncio.sleep(1)


class ServerManager:
    """Global server state manager, holding all tasks and shared resources."""
    def __init__(self):
        self.connected_clients: Dict[int, WebSocket] = {}
        
        # 1. Centralized data prep happens once when the manager is created.
        self.data_manager = DataManager(['arrhythmia', 'nasa_battery'], get_config)
        
        # 2. Create the tasks, passing the now-existing data_manager instance to them.
        self.tasks: Dict[str, FederationTask] = {
            "arrhythmia": FederationTask("arrhythmia", "she_dp", self.data_manager),
            "nasa_battery": FederationTask("nasa_battery", "she", self.data_manager)
        }
        
        # 3. Initialize other shared resources
        token_file = os.path.join(get_config('arrhythmia')['results_dir'], 'token_balances.json')
        self.token_manager = TokenManager(storage_path=token_file)
        self.MINIMUM_STAKE = 50.0
        self.REWARD_AMOUNT = 10.0
        
        POLY_MOD_DEGREE = 16384
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2**48

manager = ServerManager()

async def task_orchestrator_loop(task_id: str):
    task = manager.tasks[task_id]
    
    # --- FIX: STAGGERED STARTUP ---
    startup_delay = random.uniform(1.0, 5.0)
    await asyncio.sleep(startup_delay)
    
    # Use the new logger
    task_log(task_id, "Orchestrator started.")
    initial_metric, _ = evaluate_global_model(task.global_model, task.test_loader, task.config['device'], task.config['metric'])
    metric_name = task.config['metric'].upper()
    metric_unit = "cycles" if metric_name == "RMSE" else "%"
    task_log(task_id, f"Initial Global Model {metric_name}: {initial_metric:.2f} {metric_unit}")
    
    task.metric_history.append(initial_metric)
    task.csv_writer.writerow([0, initial_metric, task.privacy_profile])
    task.csv_file.flush()
    task.status = TaskStatus.IDLE

    while not task.is_complete():
        await task.execute_round(manager)
        await asyncio.sleep(1) # Small delay between rounds
    
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

@app.get("/status")
async def get_federation_status():
    """Provides a complete real-time snapshot of the entire federation."""
    task_statuses = {}
    for task_id, task in manager.tasks.items():
        task_statuses[task_id] = {
            "task_id": task_id,
            "status": task.status,
            "privacy_profile": task.privacy_profile,
            "current_round": task.current_round,
            "total_rounds": task.config['num_rounds'],
            "metric": task.config['metric'],
            "metric_history": task.metric_history # For live plotting
        }
    
    return {
        "network_info": {
            "connected_clients_count": len(manager.connected_clients),
            "connected_client_ids": list(manager.connected_clients.keys())
        },
        "tasks": task_statuses
    }

@app.get("/tasks/{task_id}/ledger")
async def get_task_ledger(task_id: str):
    if task_id in manager.tasks:
        return manager.tasks[task_id].ledger.chain
    return {"error": "Task not found"}, 404

@app.get("/tokenomics")
async def get_tokenomics_state():
    return manager.token_manager.get_all_accounts()