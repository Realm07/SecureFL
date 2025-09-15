import json
import os
from typing import Dict, Any, Tuple

class TokenManager:
    def __init__(self, storage_path="token_balances.json", initial_balance=1000, initial_stake=100):
        self.storage_path = storage_path
        self.initial_balance = initial_balance
        self.initial_stake = initial_stake
        self.accounts: Dict[int, Dict[str, Any]] = {}
        self.load_accounts()

    def load_accounts(self):
        if not os.path.exists(self.storage_path):
            print("INFO: No existing tokenomics file found.")
            return

        try:
            with open(self.storage_path, 'r') as f:
                loaded_accounts = {int(k): v for k, v in json.load(f).items()}
            
            # --- FIX: Upgrade old stake format to new task-specific format ---
            for client_id, account_data in loaded_accounts.items():
                if isinstance(account_data.get("stake"), (int, float)):
                    print(f"INFO: Upgrading stake format for Client #{client_id}.")
                    old_stake = account_data["stake"]
                    account_data["stake"] = {"arrhythmia": old_stake} # Default to a base task
            
            self.accounts = loaded_accounts
            print("INFO: Tokenomics accounts loaded and validated from disk.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARNING: Could not load tokenomics file: {e}. Starting fresh.")
    
    def save_accounts(self):
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.accounts, f, indent=4)
        except IOError as e:
            print(f"ERROR: Could not save tokenomics accounts to disk: {e}")

    def register_client(self, client_id: int):
        if client_id not in self.accounts:
            print(f"INFO: Registering new Client #{client_id} in token economy.")
            self.accounts[client_id] = { "balance": self.initial_balance, "stake": {} }
            # Auto-stake in a default task for initial eligibility
            self.stake_tokens(client_id, self.initial_stake, "arrhythmia")
            # No need to save here, stake_tokens already does

    def has_sufficient_stake(self, client_id: int, required_stake: float) -> bool:
        if client_id not in self.accounts: return False
        stake_data = self.accounts[client_id].get("stake", {})
        if not isinstance(stake_data, dict): return False # Defensive check
        total_stake = sum(stake_data.values())
        return total_stake >= required_stake

    def reward_clients(self, client_ids: list[int], reward_amount: float):
        for client_id in client_ids:
            if client_id in self.accounts:
                self.accounts[client_id]["balance"] += reward_amount
        self.save_accounts()

    def stake_tokens(self, client_id: int, amount: float, task_id: str) -> Tuple[bool, str]:
        if client_id not in self.accounts:
            return False, f"Client #{client_id} not found."
        if amount <= 0:
            return False, "Stake amount must be positive."
        account = self.accounts[client_id]
        if account["balance"] < amount:
            return False, f"Insufficient balance. Available: {account['balance']:.2f}, Tried: {amount:.2f}"
        
        account["balance"] -= amount
        account["stake"][task_id] = account["stake"].get(task_id, 0) + amount
        print(f"INFO: Client #{client_id} staked {amount:.2f} tokens for task '{task_id}'.")
        self.save_accounts()
        return True, f"Successfully staked {amount:.2f} tokens."
        
    def get_all_accounts(self) -> Dict[int, Dict[str, Any]]:
        return self.accounts.copy()