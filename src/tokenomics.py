# src/tokenomics.py

import json
import os
from typing import Dict, Tuple

class TokenManager:
    """
    Manages the simulated token economy for the federation.
    Handles client balances, staking, rewards, and slashing.
    """
    def __init__(self, storage_path="token_balances.json", initial_balance=1000, initial_stake=100):
        self.storage_path = storage_path
        self.initial_balance = initial_balance
        self.initial_stake = initial_stake
        
        # Structure: { client_id: {"balance": float, "stake": float} }
        self.accounts: Dict[int, Dict[str, float]] = {}
        self.load_accounts()

    def load_accounts(self):
        """Loads client account balances and stakes from a file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    # Convert string keys back to int keys
                    self.accounts = {int(k): v for k, v in json.load(f).items()}
                print("INFO: Tokenomics accounts loaded from disk.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING: Could not load tokenomics file: {e}. Starting fresh.")
        else:
            print("INFO: No existing tokenomics file found.")
    
    def save_accounts(self):
        """Saves the current state of all accounts to a file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.accounts, f, indent=4)
        except IOError as e:
            print(f"ERROR: Could not save tokenomics accounts to disk: {e}")

    def register_client(self, client_id: int):
        """
        Registers a new client, giving them an initial balance and stake if they don't exist.
        """
        if client_id not in self.accounts:
            print(f"INFO: Registering new Client #{client_id} in token economy.")
            self.accounts[client_id] = {
                "balance": self.initial_balance,
                "stake": self.initial_stake  # Stake some initial amount to be eligible
            }
            # Balance should reflect that some tokens are now staked
            self.accounts[client_id]["balance"] -= self.initial_stake
            self.save_accounts()

    def has_sufficient_stake(self, client_id: int, required_stake: float) -> bool:
        """Checks if a client has at least the required amount staked."""
        if client_id not in self.accounts:
            return False
        return self.accounts[client_id].get("stake", 0) >= required_stake

    def reward_clients(self, client_ids: list[int], reward_amount: float):
        """Adds a reward amount to the balance of each participating client."""
        # print(f"--- Rewarding {len(client_ids)} clients with {reward_amount} tokens each ---")
        for client_id in client_ids:
            if client_id in self.accounts:
                self.accounts[client_id]["balance"] += reward_amount
            else:
                print(f"WARNING: Could not find Client #{client_id} to reward.")
        self.save_accounts()

    def slash_client(self, client_id: int):
        """
        Slashes a client's stake for simulated malicious behavior.
        The stake is forfeited and removed from the system.
        """
        if client_id in self.accounts:
            staked_amount = self.accounts[client_id].get("stake", 0)
            if staked_amount > 0:
                print(f"--- Slashing! Client #{client_id} forfeits {staked_amount} staked tokens. ---")
                self.accounts[client_id]["stake"] = 0
                # Todo: apply a balance penalty here
                self.save_accounts()
                return True
        return False
        
    def get_all_accounts(self) -> Dict[int, Dict[str, float]]:
        """Returns a copy of all account data."""
        return self.accounts.copy()