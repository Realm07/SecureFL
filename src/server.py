import asyncio
from datetime import datetime
import json
import os
import random
import traceback
import tenseal as ts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from typing import Dict, List, Any, Tuple
from collections import OrderedDict, deque
import torch
import csv
import io
import hashlib
from enum import Enum
from pydantic import BaseModel
import secrets

from .config import get_config
from .models import get_model
from .he_tenseal import aggregate_and_decrypt_tenseal
from .serialization import serialize_model, deserialize_model_update
from .utils import evaluate_global_model
from .data_manager import DataManager
from .ledger import FederationLedger
from .tokenomics import TokenManager

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost", "http://localhost:5500", "http://127.0.0.1:5500", "http://127.0.0.1"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
static_dir = os.path.join(project_root, "static")

class TaskStatus(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_CLIENTS = "WAITING_FOR_CLIENTS"
    RUNNING_ROUND = "RUNNING_ROUND"
    AGGREGATING = "AGGREGATING"
    COMPLETED = "COMPLETED"

def task_log(task_id: str, message: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Task: {task_id}] {message}")

def hash_model_state(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

class FederationTask:
    def __init__(self, task_id: str, privacy_profile: str, data_manager: DataManager, custom_config: dict = None):
        self.task_id = task_id
        self.privacy_profile = privacy_profile
        self.status: TaskStatus = TaskStatus.IDLE
        print(f"INFO: Initializing task '{self.task_id}' with profile '{self.privacy_profile.upper()}'")

        if custom_config:
            base_config = get_config(custom_config.get('dataset_name'))
            base_config.update(custom_config)
            self.config = base_config
        else:
            self.config = data_manager.get_task_config(task_id)

        self.learning_mode = self.config.get('learning_mode', 'synchronous')
        
        if "dp" in self.privacy_profile:
            self.config['dp_noise_multiplier'] = self.config.get('dp_noise_multiplier', 0.5)
            self.config['dp_max_grad_norm'] = self.config.get('dp_max_grad_norm', 1.5)
        else:
            self.config['dp_noise_multiplier'] = 0.0
        
        self.testset = data_manager.get_test_set(self.config['dataset_name']) 
        self.global_model = get_model(self.config)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=512)
        self.model_version = 0

        self.server_adam_m = {name: torch.zeros_like(p) for name, p in self.global_model.named_parameters()}
        self.server_adam_v = {name: torch.zeros_like(p) for name, p in self.global_model.named_parameters()}
        self.server_adam_step = 0
        
        self.current_round = 0
        self.chunk_buffers: Dict[str, List] = {}
        
        self.updates_for_round: Dict[int, List] = {}
        self.clients_ready_for_round: Dict[int, List] = {}
        self.update_received_event_for_round: Dict[int, asyncio.Event] = {}

        self.update_buffer = deque()

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

    async def execute_synchronous_round(self, manager_instance):
        self.status = TaskStatus.RUNNING_ROUND
        self.current_round += 1
        round_num = self.current_round
        
        task_log(self.task_id, f"--- Round {round_num}/{self.config['num_rounds']} ---")

        eligible_clients = [cid for cid in manager_instance.connected_clients if manager_instance.token_manager.has_sufficient_stake(cid, manager_instance.MINIMUM_STAKE)]
        if len(eligible_clients) < self.config['clients_per_round']:
            task_log(self.task_id, "Not enough eligible clients. Waiting...")
            self.status = TaskStatus.WAITING_FOR_CLIENTS
            self.current_round -= 1; await asyncio.sleep(10); return
        
        selected_clients = random.sample(eligible_clients, self.config['clients_per_round'])
        task_log(self.task_id, f"Selected clients for round: {selected_clients}")
        
        self.updates_for_round[round_num] = []
        self.clients_ready_for_round[round_num] = []
        self.update_received_event_for_round[round_num] = asyncio.Event()
        
        client_config = self._create_client_config()
        message = { "type": "START_TRAINING", "payload": { "task_id": self.task_id, "round": round_num, "model_state_dict": serialize_model(self.global_model), "config": client_config } }
        for client_id in selected_clients:
            if client_id in manager_instance.connected_clients:
                await manager_instance.connected_clients[client_id].send_text(json.dumps(message))

        try:
            timeout = 300.0 if self.current_round == 1 else 120.0
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
                except Exception as e:
                    task_log(self.task_id, f"Error requesting update from Client #{client_id}: {e}")
        
        updates = self.updates_for_round.get(round_num, [])
        if updates:
            await self._aggregate_and_update_model(updates, selected_clients)
        else:
            task_log(self.task_id, f"No valid updates received for round {round_num}. Skipping model update.")
            
        for state_dict in [self.clients_ready_for_round, self.update_received_event_for_round, self.updates_for_round]:
            if round_num in state_dict: del state_dict[round_num]
        
        self._check_completion()
    
    async def execute_asynchronous_aggregation(self, manager_instance):
        min_updates = self.config.get('min_updates_for_aggregation', 1)
        if len(self.update_buffer) < min_updates:
            self.status = TaskStatus.WAITING_FOR_CLIENTS
            return

        self.status = TaskStatus.AGGREGATING
        self.current_round += 1
        
        updates_to_process = list(self.update_buffer)
        self.update_buffer.clear()
        
        latest_updates = {client_id: update_data for client_id, update_data, _ in updates_to_process}
        
        participating_clients, final_updates = list(latest_updates.keys()), list(latest_updates.values())
        
        task_log(self.task_id, f"--- Aggregation {self.current_round}/{self.config['num_rounds']} with {len(final_updates)} updates from clients: {participating_clients} ---")
        
        await self._aggregate_and_update_model(final_updates, participating_clients)
        
        self.model_version += 1
        await manager_instance.broadcast_model(self)

        self._check_completion()

    async def _aggregate_and_update_model(self, updates, clients):
        avg_delta = await asyncio.to_thread(aggregate_and_decrypt_tenseal, manager.context, updates, len(updates))
        if avg_delta:
            self._apply_update(avg_delta)
            metric_val = self._evaluate_and_log(self.current_round)
            manager.token_manager.reward_clients(clients, manager.REWARD_AMOUNT)
            self.ledger.add_round_to_ledger(self.current_round, clients, hash_model_state(self.global_model), metric_val)
        else:
            task_log(self.task_id, f"Aggregation failed for round/event {self.current_round}.")

    def _check_completion(self):
        self.status = TaskStatus.IDLE
        if self.is_complete():
            self.status = TaskStatus.COMPLETED
            task_log(self.task_id, "--- All Rounds/Aggregations Complete ---")
            if self.csv_file: self.csv_file.close()

    def _create_client_config(self):
        client_config = {
            "data_root": self.config.get("data_root"), "num_clients": self.config.get("num_clients"),
            "nasa_data_folder": self.config.get("nasa_data_folder"), "dataset_name": self.config.get("dataset_name"),
            "model_name": self.config.get("model_name"), "local_epochs": self.config.get("local_epochs"),
            "learning_rate": self.config.get("learning_rate"), "optimizer": self.config.get("optimizer"),
            "batch_size": self.config.get("batch_size"), "weight_decay": self.config.get("weight_decay"),
            "lr_scheduler_step_size": self.config.get("lr_scheduler_step_size"),
            "lr_scheduler_gamma": self.config.get("lr_scheduler_gamma"), "privacy_profile": self.privacy_profile,
            "encrypted_layers": self.config.get("encrypted_layers"), "dp_noise_multiplier": self.config.get("dp_noise_multiplier"),
            "dp_max_grad_norm": self.config.get("dp_max_grad_norm"), "delta": self.config.get("delta"),
            "num_features": self.config.get("num_features"), "num_classes": self.config.get("num_classes"),
            "sequence_length": self.config.get("sequence_length"), "lstm_hidden_dim": self.config.get("lstm_hidden_dim"),
            "lstm_n_layers": self.config.get("lstm_n_layers"), "lstm_drop_prob": self.config.get("lstm_drop_prob"),
            "device": str(self.config.get("device")), "metric": self.config.get("metric")
        }
        return {k: v for k, v in client_config.items() if v is not None}

    def _apply_update(self, avg_delta):
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

    def _evaluate_and_log(self, round_num):
        metric_val, _ = evaluate_global_model(self.global_model, self.test_loader, self.config['device'], self.config['metric'])
        metric_name, metric_unit = self.config['metric'].upper(), "cycles" if self.config['metric'] == "rmse" else "%"
        task_log(self.task_id, f"--- Round/Aggregation {round_num} Complete --- {metric_name}: {metric_val:.2f} {metric_unit} ---")
        self.metric_history.append(metric_val)
        self.csv_writer.writerow([round_num, metric_val, self.privacy_profile])
        self.csv_file.flush()
        return metric_val

    async def _wait_for_clients_ready(self, round_num, num_expected):
        while len(self.clients_ready_for_round.get(round_num, [])) < num_expected:
            await asyncio.sleep(1)

class ControllerClient(BaseModel):
    client_id: int
    session_token: str = secrets.token_hex(16)
    current_step: str = "login"
    is_locked: bool = True
    task_id: str | None = None
    round: int | None = None


class ControllerManager:
    def __init__(self):
        # Maps a client ID to its controller session
        self.sessions: Dict[int, ControllerClient] = {}
        # Predefined slots for viewers to join
        self.available_slots: List[int] = list(range(10, 20)) # e.g., clients 10-19 are for viewers

    def join_session(self, client_id: int) -> ControllerClient | None:
        if client_id in self.available_slots and self.sessions.get(client_id) is None:
            session = ControllerClient(client_id=client_id)
            self.sessions[client_id] = session
            return session
        return None

    def get_session(self, client_id: int) -> ControllerClient | None:
        return self.sessions.get(client_id)

    def remove_session(self, client_id: int):
        if client_id in self.sessions:
            del self.sessions[client_id]

class ServerManager:
    def __init__(self):
        self.connected_clients: Dict[int, WebSocket] = {}
        self.data_manager = DataManager(['arrhythmia', 'nasa_battery'], get_config)
        self.tasks: Dict[str, FederationTask] = {
            "arrhythmia": FederationTask("arrhythmia", "she_dp", self.data_manager),
            "nasa_battery": FederationTask("nasa_battery", "she", self.data_manager)
        }
        token_file = os.path.join(get_config('arrhythmia')['results_dir'], 'token_balances.json')
        self.token_manager = TokenManager(storage_path=token_file)
        self.MINIMUM_STAKE, self.REWARD_AMOUNT = 50.0, 10.0
        POLY_MOD_DEGREE = 16384
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MOD_DEGREE, coeff_mod_bit_sizes=[60, 48, 48, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2**48
        self.controller_manager = ControllerManager()
        self.ADMIN_SECRET = "aegis-admin-secret"

    async def _send_to_client_safely(self, client: WebSocket, message: str):
        try:
            await client.send_text(message)
        except (WebSocketDisconnect, ConnectionResetError):
            pass

    async def broadcast_model(self, task: FederationTask):
        message = json.dumps({"type": "NEW_GLOBAL_MODEL", "payload": {
            "task_id": task.task_id, 
            "model_version": task.model_version, 
            "model_state_dict": serialize_model(task.global_model), 
            "config": task._create_client_config()
        }})
        clients_to_send = list(self.connected_clients.values())
        send_tasks = [self._send_to_client_safely(client, message) for client in clients_to_send]
        await asyncio.gather(*send_tasks)

    def create_dynamic_task(self, task_id: str, config: dict):
        if task_id in self.tasks: raise ValueError(f"Task with ID '{task_id}' already exists.")
        privacy_profile = "she_dp" if config.get('dp_noise_multiplier', 0) > 0 else "she"
        new_task = FederationTask(task_id, privacy_profile, self.data_manager, custom_config=config)
        self.tasks[task_id] = new_task
        asyncio.create_task(start_orchestrator(new_task))
        task_log(task_id, "Dynamically created and started.")
        return new_task

manager = ServerManager()

class JoinRequest(BaseModel):
    client_id: int

@app.post("/controller/join")
async def controller_join(request: JoinRequest):
    session = manager.controller_manager.join_session(request.client_id)
    if session:
        # Also register this new client in the tokenomics system
        manager.token_manager.register_client(request.client_id)
        return {"message": "Joined successfully!", "session_token": session.session_token, "client_id": session.client_id}
    return JSONResponse(status_code=409, content={"error": "Slot is already taken or invalid."})

@app.get("/controller/slots")
async def get_available_slots():
    active_sessions = manager.controller_manager.sessions.keys()
    return {"available": [slot for slot in manager.controller_manager.available_slots if slot not in active_sessions]}

class ActionRequest(BaseModel):
    client_id: int
    session_token: str
    action: str
    payload: dict = {}

@app.post("/controller/action")
async def controller_action(request: ActionRequest):
    session = manager.controller_manager.get_session(request.client_id)
    if not session or session.session_token != request.session_token:
        return JSONResponse(status_code=403, content={"error": "Invalid session."})
    
    # Here you would add logic to proxy the action to the actual client
    # For now, we'll just log it and update the state.
    print(f"CONTROLLER: Received action '{request.action}' for client #{request.client_id} with payload: {request.payload}")
    
    # Example state transition
    if request.action == "select_data":
        session.current_step = "ready_to_train"
    elif request.action == "train":
        session.current_step = "ready_to_encrypt"
    # ... and so on
    
    return {"message": f"Action '{request.action}' acknowledged.", "next_step": session.current_step}

class AdminActionRequest(BaseModel):
    client_id: int
    secret: str

@app.post("/admin/kick")
async def admin_kick_client(request: AdminActionRequest):
    if request.secret != manager.ADMIN_SECRET:
        return JSONResponse(status_code=403, content={"error": "Unauthorized"})
    
    manager.controller_manager.remove_session(request.client_id)
    # You might also want to disconnect the websocket if it's a real client
    print(f"ADMIN: Kicked client #{request.client_id}")
    return {"message": f"Client #{request.client_id} has been kicked."}

async def sync_orchestrator_loop(task: FederationTask):
    while not task.is_complete():
        await task.execute_synchronous_round(manager)
        await asyncio.sleep(5)

async def async_orchestrator_loop(task: FederationTask):
    await manager.broadcast_model(task)
    while not task.is_complete():
        await asyncio.sleep(task.config.get('aggregation_interval_seconds', 30))
        await task.execute_asynchronous_aggregation(manager)

async def start_orchestrator(task: FederationTask):
    await asyncio.sleep(random.uniform(1.0, 5.0))
    task_log(task.task_id, f"Orchestrator started (Mode: {task.learning_mode}).")
    initial_metric, _ = evaluate_global_model(task.global_model, task.test_loader, task.config['device'], task.config['metric'])
    metric_name, metric_unit = task.config['metric'].upper(), "cycles" if task.config['metric'] == "rmse" else "%"
    task_log(task.task_id, f"Initial Global Model {metric_name}: {initial_metric:.2f} {metric_unit}")
    task.metric_history.append(initial_metric)
    task.csv_writer.writerow([0, initial_metric, task.privacy_profile])
    task.csv_file.flush()
    task.status = TaskStatus.IDLE
    await (async_orchestrator_loop(task) if task.learning_mode == 'asynchronous' else sync_orchestrator_loop(task))

CLIENT_LOCATIONS = {0: {"name": "Los Angeles", "lat": 34.05, "lon": -118.24}, 1: {"name": "New York", "lat": 40.71, "lon": -74.00}, 2: {"name": "London", "lat": 51.50, "lon": -0.12}, 3: {"name": "Tokyo", "lat": 35.68, "lon": 139.69}, 4: {"name": "Sydney", "lat": -33.86, "lon": 151.20}, 5: {"name": "São Paulo", "lat": -23.55, "lon": -46.63}, 6: {"name": "Mumbai", "lat": 19.07, "lon": 72.87}, 7: {"name": "Moscow", "lat": 55.75, "lon": 37.61}, 8: {"name": "Beijing", "lat": 39.90, "lon": 116.40}, 9: {"name": "Paris", "lat": 48.85, "lon": 2.35}}
TASK_SERVER_LOCATIONS = {"arrhythmia": {"name": "Zurich", "lat": 47.37, "lon": 8.54}, "nasa_battery": {"name": "Houston", "lat": 29.76, "lon": -95.36}}

@app.on_event("startup")
async def startup_event():
    for task in manager.tasks.values(): asyncio.create_task(start_orchestrator(task))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await websocket.accept()
    manager.connected_clients[client_id] = websocket
    manager.token_manager.register_client(client_id)
    print(f"Client #{client_id} connected.")
    for task in manager.tasks.values():
        if task.learning_mode == 'asynchronous' and not task.is_complete(): await manager.broadcast_model(task)
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            msg_type, payload = message.get("type"), message.get("payload", {})
            task_id, round_num = payload.get("task_id"), payload.get("round")
            if not task_id or task_id not in manager.tasks: continue
            task = manager.tasks[task_id]
            if msg_type == 'TRAINING_COMPLETE': task.clients_ready_for_round.setdefault(round_num, []).append(client_id)
            elif msg_type == 'ASYNC_UPDATE':
                update = await asyncio.to_thread(deserialize_model_update, payload['update_data'])
                task.update_buffer.append((client_id, update, payload['model_version']))
            elif msg_type == 'START_UPDATE_STREAM':
                session_key = f"t{task_id}_r{round_num}_c{client_id}"
                task.chunk_buffers[session_key] = [None] * payload['total_chunks']
            elif msg_type == 'UPDATE_CHUNK':
                session_key = f"t{task_id}_r{round_num}_c{client_id}"
                if session_key in task.chunk_buffers:
                    task.chunk_buffers[session_key][payload['chunk_index']] = payload['data']
                    if all(c is not None for c in task.chunk_buffers[session_key]):
                        full_json = "".join(task.chunk_buffers[session_key])
                        del task.chunk_buffers[session_key]
                        update = await asyncio.to_thread(deserialize_model_update, full_json)
                        task.updates_for_round.setdefault(round_num, []).append(update)
                        if round_num in task.update_received_event_for_round: task.update_received_event_for_round[round_num].set()
    except WebSocketDisconnect: print(f"Client #{client_id} disconnected.")
    except Exception: print(f"An error occurred with client #{client_id}: {traceback.format_exc()}")
    finally:
        if client_id in manager.connected_clients: del manager.connected_clients[client_id]

@app.post("/create-task")
async def create_task_endpoint(request: Request):
    try:
        config = await request.json()
        task_id = config.get("task_id")
        if not task_id: return JSONResponse(status_code=400, content={"error": "task_id is required"})
        manager.create_dynamic_task(task_id, config)
        return {"message": f"Task '{task_id}' created successfully."}
    except ValueError as e: return JSONResponse(status_code=409, content={"error": str(e)})
    except Exception as e: return JSONResponse(status_code=500, content={"error": f"Failed to create task: {e}"})

# --- NEW: Staking Endpoint ---
@app.post("/stake")
async def stake_tokens_endpoint(request: Request):
    try:
        data = await request.json()
        client_id = int(data.get("client_id"))
        amount = float(data.get("amount"))
        task_id = data.get("task_id")

        if client_id is None or amount is None or task_id is None:
            return JSONResponse(status_code=400, content={"error": "client_id, amount, and task_id are required."})

        success, message = manager.token_manager.stake_tokens(client_id, amount, task_id)
        
        if success:
            return {"message": message}
        else:
            return JSONResponse(status_code=400, content={"error": message})
            
    except (ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "Invalid data types for client_id or amount."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An internal server error occurred: {e}"})

@app.get("/status")
async def get_federation_status():
    task_statuses = {task_id: {"task_id": task_id, "status": task.status, "learning_mode": task.learning_mode, "server_location": TASK_SERVER_LOCATIONS.get(task_id), "selected_clients": task.clients_ready_for_round.get(task.current_round, list(dict.fromkeys(c[0] for c in task.update_buffer))), "privacy_profile": task.privacy_profile, "current_round": task.current_round, "total_rounds": task.config['num_rounds'], "metric": task.config['metric'], "metric_history": task.metric_history, "model_name": task.config.get('model_name', 'N/A')} for task_id, task in manager.tasks.items()}
    connected_clients_with_loc = [{"id": cid, "location": CLIENT_LOCATIONS.get(cid)} for cid in manager.connected_clients.keys()]
    return {"network_info": {"connected_clients": connected_clients_with_loc}, "tasks": task_statuses}

@app.get("/tasks/{task_id}/ledger")
async def get_task_ledger(task_id: str):
    return manager.tasks[task_id].ledger.chain if task_id in manager.tasks else ({"error": "Task not found"}, 404)

@app.get("/tokenomics")
async def get_tokenomics_state(): return manager.token_manager.get_all_accounts()

app.mount("/static", StaticFiles(directory=static_dir), name="static")
@app.get("/", response_class=FileResponse)
async def read_index(): return FileResponse(os.path.join(static_dir, "index.html"))