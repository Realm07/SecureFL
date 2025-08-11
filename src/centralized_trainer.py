import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import argparse
import os

from config import get_config
from data_loader import get_datasets
from models import get_model
from utils import evaluate_global_model

def train_centralized(model, train_loader, test_loader, config):
    """Trains a model on the entire dataset in a centralized fashion."""
    device = config['device']
    model.to(device)
    model.train()

    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0))
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0))
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.get('lr_scheduler_step_size', 100),
        gamma=config.get('lr_scheduler_gamma', 1.0)
    )
    criterion = nn.CrossEntropyLoss()

    total_epochs = config['num_rounds'] * config['local_epochs']
    
    print(f"Starting centralized training for {total_epochs} epochs...")
    start_time = time.time()
    
    initial_accuracy, _ = evaluate_global_model(model, test_loader, device)
    accuracies = [initial_accuracy]

    for epoch in range(total_epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        accuracy, _ = evaluate_global_model(model, test_loader, device)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}/{total_epochs} - Accuracy: {accuracy:.2f}%")

    total_time = time.time() - start_time
    print(f"\nCentralized training finished in {total_time:.2f}s")
    
    final_accuracy, final_loss = evaluate_global_model(model, test_loader, device)
    print(f"Final Centralized Accuracy: {final_accuracy:.2f}%")

    return final_accuracy, total_time, accuracies


def main(dataset_name):
    config = get_config(dataset_name)
    
    if dataset_name == 'arrhythmia':
        trainset, testset, _, _ = get_datasets(config)
    else:
        trainset, testset = get_datasets(config)
        
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(testset, batch_size=1024, shuffle=False)
    
    model = get_model(config)

    final_accuracy, total_time, accuracy_history = train_centralized(model, train_loader, test_loader, config)

    results_path = f"centralized_results_{dataset_name}.json"
    results_data = {
        "dataset_name": dataset_name,
        "final_accuracy": final_accuracy,
        "total_time": total_time,
        "accuracy_history": accuracy_history
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\nCentralized benchmark results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Centralized Training for Benchmarking.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['mnist', 'arrhythmia'], 
        required=True, 
        help='The dataset to use for training.'
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset)