# In src/run_arrhythmia_robustness.py

import os
import torch
import copy
import csv
from torch.utils.data import DataLoader

# Corrected relative imports
from .config import get_config
from .data_loader import get_datasets, partition_iid
from .models import get_model
from .simulation import run_simulation_plaintext # We only need the plaintext simulator
from .utils import evaluate_global_model

def main():
    """
    Main function to run the ROBUSTNESS experiments for the paper on the ARRHYTHMIA dataset.
    """
    dataset_name = 'arrhythmia'
    print(f"\n{'='*60}")
    print(f"--- Running Robustness Experiment Suite for Dataset: {dataset_name.upper()} ---")
    print(f"{'='*60}")

    config = get_config(dataset_name)
    os.makedirs(config['results_dir'], exist_ok=True)

    # --- CONFIGURATION FOR A STRONGER ATTACK on ARRHYTHMIA ---
    # Arrhythmia has two main classes, 0 (normal) and 1 (arrhythmia).
    # Let's teach the model that 'normal' is 'arrhythmia'.
    config['attack_target_class'] = 0 # Target: Normal ECG
    config['attack_poison_class'] = 1 # Poison: Mislabel as Arrhythmia
    config['num_rounds'] = 25 # Match the privacy experiment
    config['clients_per_round'] = 5 # Keep this consistent

    trainset, testset = get_datasets(config)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

    # We will use 2 malicious clients (40% of the 5 selected each round)
    malicious_client_ids = [0, 1]
    initial_model = get_model(config)

    # --- Experiment 1: Baseline (Benign) ---
    print("\n--- Starting Experiment 1: Baseline (Benign) ---")
    # You already have this data from the privacy run (arrhythmia_plaintext), but for clarity
    # let's regenerate it with a consistent filename.
    run_simulation_plaintext(
        global_model=copy.deepcopy(initial_model),
        trainset=trainset,
        test_loader=test_loader,
        config=config,
        experiment_name="arrhythmia_baseline_benign",
        malicious_clients=[],
        enable_defense=False
    )

    # --- Experiment 2: Baseline Under Attack (Poisoned) ---
    print("\n--- Starting Experiment 2: Baseline Under Attack ---")
    run_simulation_plaintext(
        global_model=copy.deepcopy(initial_model),
        trainset=trainset,
        test_loader=test_loader,
        config=config,
        experiment_name="arrhythmia_baseline_attacked",
        malicious_clients=malicious_client_ids,
        enable_defense=False
    )

    # --- Experiment 3: Our System Under Attack (Protected) ---
    print("\n--- Starting Experiment 3: Our System Under Attack ---")
    run_simulation_plaintext(
        global_model=copy.deepcopy(initial_model),
        trainset=trainset,
        test_loader=test_loader,
        config=config,
        experiment_name="arrhythmia_ours_protected",
        malicious_clients=malicious_client_ids,
        enable_defense=True
    )

    print("\n\nArrhythmia robustness experiments complete.")
    print("You now have a complete set of results on a single dataset.")

if __name__ == "__main__":
    main()