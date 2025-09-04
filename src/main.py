import os
import torch
import copy
import argparse
import json
import joblib
import numpy as np

from config import get_config
from data_loader import get_datasets
from models import get_model
from simulation import run_simulation_plaintext, run_simulation_secure
from utils import plot_comparison_results


def main(dataset_name):
    """
    Main function to run all training simulations, collect benchmark data,
<<<<<<< Updated upstream
    and save the results and a final model.
=======
    and save the results and a final model to the 'results' directory.
>>>>>>> Stashed changes
    """
    print(f"\n{'='*60}")
    print(f"--- Running Full Training & Benchmarking for Dataset: {dataset_name.upper()} ---")
    print(f"{'='*60}")

    config = get_config(dataset_name)
    RESULTS_DIR = config['results_dir']
    os.makedirs(RESULTS_DIR, exist_ok=True)
<<<<<<< Updated upstream
    patient_samples_for_dashboard = None
    feature_names_for_dashboard = None
=======
>>>>>>> Stashed changes

    # --- 1. Load and Prepare Datasets ---
    try:
        if dataset_name == 'arrhythmia':
            trainset, testset, X_test_original, y_test_original = get_datasets(config)
            
<<<<<<< Updated upstream
            # Save the scaler in the 'src' directory
            
=======
>>>>>>> Stashed changes
            scaler_save_path = os.path.join(RESULTS_DIR, "arrhythmia_scaler.joblib")
            joblib.dump(config['scaler'], scaler_save_path)
            print(f"Saved fitted scaler to {scaler_save_path}")
            
            # Prepare patient samples for the dashboard
            num_samples = 5
            indices = np.random.choice(len(X_test_original), num_samples, replace=False)
            patient_samples_for_dashboard = []
            for i in indices:
                patient_samples_for_dashboard.append({
                    "features": X_test_original[i].tolist(),
                    "actual_label": int(y_test_original[i])
                })
            feature_names_for_dashboard = config['feature_names']
        else:
            trainset, testset = get_datasets(config)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)
    except (FileNotFoundError, NotImplementedError, ValueError) as e:
        print(f"\n[ERROR] Could not load data: {e}")
        return

    # --- 2. Run All Three Simulation Benchmarks ---

    # --- NEW: Run all four simulations ---
    initial_model = get_model(config)
    
<<<<<<< Updated upstream
    # Simulation 1: Plaintext
    pt_acc, _, pt_times, sample_pt_update = run_simulation_plaintext(
        copy.deepcopy(initial_model), trainset, test_loader, config
    )
=======
    RESULTS_PATH = os.path.join(RESULTS_DIR, f"training_results_{dataset_name}.json")
    print(f"\nSaving comprehensive training benchmark results to: {RESULTS_PATH}")
>>>>>>> Stashed changes
    
    # Simulation 2: Selective HE (SHE)
    she_acc, _, she_times, final_secure_model = run_simulation_secure(
        copy.deepcopy(initial_model), trainset, test_loader, config, privacy_profile="she"
    )
    
    # Simulation 3: Full Homomorphic Encryption (Full HE)
    full_he_acc, _, full_he_times, _ = run_simulation_secure(
        copy.deepcopy(initial_model), trainset, test_loader, config, privacy_profile="full_he"
    )

    # Simulation 4: Selective HE + Differential Privacy (SHE + DP)
    she_dp_acc, _, she_dp_times, _ = run_simulation_secure(
        copy.deepcopy(initial_model), trainset, test_loader, config, privacy_profile="she_dp"
    )

    # --- NEW: Save comprehensive results ---
    RESULTS_PATH = os.path.join(RESULTS_DIR, f"training_results_{dataset_name}.json")
    print(f"\nSaving comprehensive training benchmark results to: {RESULTS_PATH}")

    results_data = {
        "dataset_name": dataset_name,
        "model_name": config['model_name'],
        "num_rounds": config['num_rounds'],
        "privacy_profiles": {
            "plaintext": {"accuracies": pt_acc, "times": pt_times},
            "she": {"accuracies": she_acc, "times": she_times},
            "full_he": {"accuracies": full_he_acc, "times": full_he_times},
            "she_dp": {"accuracies": she_dp_acc, "times": she_dp_times},
        },
        "sample_plaintext_update": sample_pt_update,
    }

    if patient_samples_for_dashboard and feature_names_for_dashboard:
        results_data["patient_samples"] = patient_samples_for_dashboard
        results_data["feature_names"] = feature_names_for_dashboard

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_data, f, indent=4)
    print("Benchmark results saved successfully.")

    if final_secure_model:
        MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, config['model_save_path'])
        print(f"\nSaving final SECURE model state (from SHE run) to: {MODEL_SAVE_PATH}")
<<<<<<< Updated upstream
        torch.save(final_secure_model.state_dict(), MODEL_SAVE_PATH)
        print("Model saved successfully. You can now build the dashboard.")
=======
        try:
            torch.save(final_secure_model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved successfully. You can now view the results in the dashboard.")
        except Exception as e:
            print(f"[ERROR] Could not save model state: {e}")
>>>>>>> Stashed changes
    else:
        print("\nSecure simulation did not produce a final model to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Federated Learning Simulations to generate benchmark data and a trained model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['mnist', 'arrhythmia'], 
        required=True, 
        help='The dataset to use for training.\n'
             'Example usage:\n'
             '  python src/main.py --dataset mnist\n'
             '  python src/main.py --dataset arrhythmia'
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset)