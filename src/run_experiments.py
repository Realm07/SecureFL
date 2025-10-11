# In src/run_experiments.py

import os
import torch
import copy
import csv 
from torch.utils.data import DataLoader 

# Corrected relative imports
from .config import get_config
from .data_loader import get_datasets, partition_iid
from .models import get_model
from .simulation import run_simulation_plaintext
from .utils import plot_comparison_results, evaluate_global_model

def main():
    """
    Main function to run all experiments for the paper.
    """
    # --- CHANGE 1: Switched to the working dataset ---
    dataset_name = 'arrhythmia' 
    print(f"\n{'='*60}")
    print(f"--- Running Attack Experiment Suite for Dataset: {dataset_name.upper()} ---")
    print(f"{'='*60}")

    config = get_config(dataset_name)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # --- CHANGE 2: Update attack classes for binary classification (0 and 1) ---
    config['attack_target_class'] = 0 # Malicious client will flip labels of class 0...
    config['attack_poison_class'] = 1 # ...to class 1.

    config['num_rounds'] = 50 
    config['clients_per_round'] = 5 

    trainset, testset = get_datasets(config)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

    malicious_client_ids = [0, 1]
    initial_model = get_model(config)

    # --- Experiment 1: Baseline (Benign) ---
    print("\n--- Starting Experiment 1: Baseline (Benign) ---")
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
    
    print("\n\nAll arrhythmia attack experiments complete. Results are saved in the 'results' directory.")
    print("You can now use a plotting script to generate graphs from the CSV files.")

if __name__ == "__main__":
    main()