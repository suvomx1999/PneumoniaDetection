import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils.dataset import get_data_loaders
from models.baseline import SimpleCNN
from models.transfer import TransferModel
from utils.callbacks import EarlyStopping

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='densenet121', choices=['simple_cnn', 'densenet121', 'resnet18'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Loaders
    train_loader, val_loader, _, class_counts = get_data_loaders(args.data_dir, batch_size=args.batch_size)
    
    # Model Setup
    if args.model == 'simple_cnn':
        model = SimpleCNN()
    else:
        model = TransferModel(model_name=args.model)
    
    model = model.to(device)
    
    # Loss and Optimizer
    # Note: We handled class imbalance with WeightedRandomSampler, so we can use standard BCEWithLogitsLoss
    # Alternatively, we could use pos_weight in BCEWithLogitsLoss, but Sampler is usually better for mini-batch dynamics.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'best_model_{args.model}.pth')
    
    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device, save_path)
