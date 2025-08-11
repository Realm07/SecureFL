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
    Main function to run training simulations, collect benchmark data,
    and save the results and final model.
    """
    print(f"\n{'='*60}")
    print(f"--- Running Full Training & Benchmarking for Dataset: {dataset_name.upper()} ---")
    print(f"{'='*60}")
    
    config = get_config(dataset_name)
    patient_samples_for_dashboard = None
    feature_names_for_dashboard = None

    try:
        if dataset_name == 'arrhythmia':
            trainset, testset, X_test_original, y_test_original = get_datasets(config)
            
            scaler_save_path = "arrhythmia_scaler.joblib"
            joblib.dump(config['scaler'], scaler_save_path)
            print(f"Saved fitted scaler to {scaler_save_path}")
            
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

    plaintext_model = get_model(config).to(config['device'])
    secure_model_initial = copy.deepcopy(plaintext_model)

    pt_acc, pt_loss, pt_times, sample_pt_update = run_simulation_plaintext(plaintext_model, trainset, test_loader, config)
    sec_acc, sec_loss, sec_times, final_secure_model = run_simulation_secure(secure_model_initial, trainset, test_loader, config)
    
    RESULTS_PATH = f"training_results_{dataset_name}.json"
    print(f"\nSaving training benchmark results to: {RESULTS_PATH}")
    
    results_data = {
        "dataset_name": dataset_name,
        "model_name": config['model_name'],
        "num_rounds": config['num_rounds'],
        "clients_per_round": config['clients_per_round'],
        "local_epochs": config['local_epochs'],
        "plaintext_accuracies": pt_acc,
        "plaintext_losses": pt_loss,
        "plaintext_times": pt_times,
        "secure_accuracies": sec_acc,
        "secure_losses": sec_loss,
        "secure_times": sec_times,
        "sample_plaintext_update": sample_pt_update
    }
    
    if patient_samples_for_dashboard and feature_names_for_dashboard:
        results_data["patient_samples"] = patient_samples_for_dashboard
        results_data["feature_names"] = feature_names_for_dashboard

    try:
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results_data, f, indent=4)
        print("Benchmark results saved successfully.")
    except Exception as e:
        print(f"[ERROR] Could not save benchmark results: {e}")

    plot_comparison_results((pt_acc, pt_loss), (sec_acc, sec_loss), config)
    
    if final_secure_model:
        MODEL_SAVE_PATH = config['model_save_path']
        print(f"\nSaving final SECURE model state to: {MODEL_SAVE_PATH}")
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or '.', exist_ok=True)
        try:
            torch.save(final_secure_model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved successfully. You can now view the results in the dashboard.")
        except Exception as e:
            print(f"[ERROR] Could not save model state: {e}")
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
             '  python main.py --dataset mnist\n'
             '  python main.py --dataset arrhythmia'
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset)