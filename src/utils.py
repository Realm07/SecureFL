
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from data_loader import get_datasets



def evaluate_global_model(model, test_loader, device):
    model.eval(); correct, total, test_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data); loss = criterion(outputs, target); test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0); correct += (predicted == target).sum().item()
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    model.train(); return accuracy, avg_loss

def plot_comparison_results(plaintext_results, secure_results, config):
    pt_acc, pt_loss = plaintext_results
    sec_acc, sec_loss = secure_results
    
    plt.figure(figsize=(14, 6))
    rounds_axis = range(config['num_rounds'] + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(rounds_axis, pt_acc, marker='o', linestyle='-', label='Plaintext FL')
    plt.plot(rounds_axis, sec_acc, marker='x', linestyle='--', label='Secure FL (TenSEAL)')
    plt.title('Accuracy Comparison'); plt.xlabel('Global Round'); plt.ylabel('Accuracy (%)')
    plt.xticks(rounds_axis); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rounds_axis, pt_loss, marker='o', linestyle='-', label='Plaintext FL')
    plt.plot(rounds_axis, sec_loss, marker='x', linestyle='--', label='Secure FL (TenSEAL)')
    plt.title('Loss Comparison'); plt.xlabel('Global Round'); plt.ylabel('Average Loss')
    plt.xticks(rounds_axis); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle("Federated Learning Benchmark", fontsize=16, y=0.98)
    plt.show()

def visualize_tabular_results(model, config):
    """
    Generates a classification report and a confusion matrix plot.
    Returns the matplotlib figure object and the report dictionary.
    """
    print("Generating classification results...")
    device = config['device']
    all_datasets = get_datasets(config)
    testset = all_datasets[1]
    test_loader = DataLoader(testset, batch_size=len(testset))

    model.eval()
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data.to(device))
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    target_names = ['Normal (0)', 'Arrhythmia (1)']
    report_dict = classification_report(all_targets, all_predictions, target_names=target_names, output_dict=True)

    cm = confusion_matrix(all_targets, all_predictions)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names)
    ax.set_title('Confusion Matrix for Arrhythmia Prediction')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    
    return fig, report_dict

def visualize_predictions(model, config):
    """Router function to call the correct visualization based on dataset."""
    if config['dataset_name'] == 'mnist':
        print("\n" + "="*50)
        print("Demonstrating Model Classification on Test Set")
        print("="*50)
    
        model.eval()
        device = config['device']
    
        vis_transform = transforms.Compose([transforms.ToTensor()])
        vis_testset = torchvision.datasets.MNIST(root=config['data_root'], train=False, download=True, transform=vis_transform)
        norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        norm_testset = torchvision.datasets.MNIST(root=config['data_root'], train=False, download=True, transform=norm_transform)

        NUM_IMAGES_TO_SHOW = 5
        plt.figure(figsize=(15, 5))
        for i in range(NUM_IMAGES_TO_SHOW):
            random_idx = random.randint(0, len(vis_testset) - 1)
            vis_image, actual_label = vis_testset[random_idx]
            norm_image, _ = norm_testset[random_idx]
            model_input = norm_image.unsqueeze(0).to(device)
        
            with torch.no_grad():
                output_logits = model(model_input)
                _, predicted_label_idx = torch.max(output_logits.data, 1)
                predicted_label = predicted_label_idx.item()
            
            plt.subplot(1, NUM_IMAGES_TO_SHOW, i + 1)
            plt.imshow(vis_image.squeeze(), cmap='gray')
            plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}",
                  color=("green" if actual_label == predicted_label else "red"))
            plt.axis('off')
    
        plt.suptitle("Sample Predictions from Final Secure Model", fontsize=16)
        plt.show()
    
    elif config['dataset_name'] == 'arrhythmia':
        visualize_tabular_results(model, config)

    else:
        print(f"\nVisualization not implemented for dataset '{config['dataset_name']}'. Skipping.")
        return

def create_time_comparison_chart(time_df):
    """Creates a matplotlib grouped bar chart for time comparison, styled for a dark theme."""

    plt.style.use('dark_background')
    
    labels = time_df.index
    secure_times = time_df['Secure FL']
    plaintext_times = time_df['Plaintext FL']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')

    rects1 = ax.bar(x - width/2, secure_times, width, label='Secure FL', color='#9370DB')
    rects2 = ax.bar(x + width/2, plaintext_times, width, label='Plaintext FL', color='#1E90FF')

    ax.set_ylabel('Time (seconds)', color='white')
    ax.set_title('Time per Round Comparison', color='white')
    ax.set_xticks(x, labels)
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['right'].set_color('grey')
    
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()

    return fig
