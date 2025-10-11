# In src/run_privacy_experiments.py

import os
import torch
import copy
import csv 
from torch.utils.data import DataLoader 

# Corrected relative imports
from .config import get_config
from .data_loader import get_datasets, partition_iid
from .models import get_model
from .simulation import run_simulation_plaintext, run_simulation_secure # Import both
from .utils import evaluate_global_model

def main():
    """
    Main function to run experiments for evaluating privacy overhead.
    """
    # --- CHANGE 1: Switched to the working dataset ---
    dataset_name = 'arrhythmia' 
    print(f"\n{'='*60}")
    print(f"--- Running Privacy Overhead Experiment Suite for: {dataset_name.upper()} ---")
    print(f"{'='*60}")

    config = get_config(dataset_name)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # For performance tests, fewer rounds are needed.
    config['num_rounds'] = 10 # Using 10 rounds for a quick but meaningful comparison
    config['clients_per_round'] = 3 

    trainset, testset = get_datasets(config)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

    initial_model = get_model(config)

    # --- Experiment 1: Plaintext Baseline for Performance ---
    print("\n--- Starting Experiment 1: Plaintext (for performance baseline) ---")
    run_simulation_plaintext(
        global_model=copy.deepcopy(initial_model),
        trainset=trainset,
        test_loader=test_loader,
        config=config,
        experiment_name="privacy_baseline_plaintext_arrhythmia", # Appended dataset name
        malicious_clients=[], 
        enable_defense=False 
    )

    # --- Experiment 2: SHE + DP ---
    print("\n--- Starting Experiment 2: SHE + DP ---")
    
    # --- CHANGE 2: No longer need to override config. 
    # The get_config('arrhythmia') call already loaded the correct, working DP parameters.
    # We will use the 'config' object directly as it's already properly configured.
    secure_config = copy.deepcopy(config)
    
    accuracies, losses, times, final_model = run_simulation_secure(
        global_model=copy.deepcopy(initial_model),
        trainset=trainset,
        test_loader=test_loader,
        config=secure_config, 
        privacy_profile="she_dp" 
    )
    
    # Manually log the results for the secure run
    log_file = os.path.join(config['results_dir'], "results_privacy_she_dp_arrhythmia.csv") # Appended dataset name
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "accuracy", "round_time"])
        writer.writerow([0, accuracies[0], 0.0]) # Initial accuracy
        for i in range(len(times)):
            writer.writerow([i + 1, accuracies[i+1], times[i]])

    print("\n\nPrivacy overhead experiments complete for ARRHYTHMIA.")
    print("You can now compare the following files:")
    print(f"- {config['results_dir']}/results_privacy_baseline_plaintext_arrhythmia.csv")
    print(f"- {config['results_dir']}/results_privacy_she_dp_arrhythmia.csv")


if __name__ == "__main__":
    main()