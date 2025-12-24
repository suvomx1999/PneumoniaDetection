import argparse
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.dataset import get_data_loaders
from models.baseline import SimpleCNN
from models.transfer import TransferModel

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("-" * 30)
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    return acc, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Pneumonia Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--model_type', type=str, default='densenet121', choices=['simple_cnn', 'densenet121', 'resnet18'], help='Model architecture used')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    _, _, test_loader, _ = get_data_loaders(args.data_dir, batch_size=32)
    
    # Load Model
    if args.model_type == 'simple_cnn':
        model = SimpleCNN()
    else:
        model = TransferModel(model_name=args.model_type)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    evaluate_model(model, test_loader, device)
