# src/ledger.py

import hashlib
import json
from time import time
from typing import List, Dict, Any
import os

class FederationLedger:
    """
    Manages a simple, local blockchain to create a tamper-evident audit trail
    for federated learning rounds.
    """
    def __init__(self, storage_path="ledger.json"):
        self.storage_path = storage_path
        self.chain: List[Dict[str, Any]] = []
        self.load_chain()

    def load_chain(self):
        """Loads the chain from a file or creates the genesis block if none exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.chain = json.load(f)
                print("INFO: Federation Ledger loaded from disk.")
                if not self.chain: # Handle case of empty file
                    self._create_genesis_block()
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING: Could not load ledger file: {e}. Creating a new one.")
                self._create_genesis_block()
        else:
            print("INFO: No existing ledger found. Creating genesis block.")
            self._create_genesis_block()

    def _create_genesis_block(self):
        """Creates the very first block in the chain."""
        self.new_block(previous_hash='1', round_data={'message': 'Genesis Block'})
        self.save_chain()

    def new_block(self, round_data: Dict[str, Any], previous_hash: str = None) -> Dict[str, Any]:
        """
        Creates a new block and adds it to the chain.

        :param round_data: The data to be stored in the block (participants, model hash, etc.).
        :param previous_hash: Optional hash of the previous block.
        :return: The new block.
        """
        if previous_hash is None:
            previous_hash = self.hash(self.last_block)

        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'round_data': round_data,
            'previous_hash': previous_hash,
        }

        # The hash is calculated on the full block content
        block['hash'] = self.hash(block)
        self.chain.append(block)
        return block

    def add_round_to_ledger(self, round_number: int, participants: List[int], global_model_hash: str, accuracy: float) -> Dict[str, Any]:
        """
        A convenience method to format round data and add it as a new block.
        """
        print(f"--- Recording round {round_number} in Federation Ledger ---")
        round_data = {
            'round_number': round_number,
            'participants': sorted(participants), # Sort for consistency
            'global_model_hash': global_model_hash,
            'global_model_accuracy': accuracy
        }
        
        last_block = self.last_block
        previous_hash = self.hash(last_block)
        block = self.new_block(round_data, previous_hash)
        
        # Persist the updated chain to disk
        self.save_chain()
        
        return block

    @property
    def last_block(self) -> Dict[str, Any]:
        """Returns the last block in the chain."""
        return self.chain[-1]

    @staticmethod
    def hash(block: Dict[str, Any]) -> str:
        """
        Creates a SHA-256 hash of a Block.

        :param block: The block to hash.
        :return: The hash digest as a hex string.
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def save_chain(self):
        """Saves the entire blockchain to the specified JSON file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.chain, f, indent=4)
        except IOError as e:
            print(f"ERROR: Could not save ledger to disk: {e}")