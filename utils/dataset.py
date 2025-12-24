import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import numpy as np
from collections import Counter

def get_transforms(mode='train'):
    """
    Returns the data transformations for training, validation, or testing.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_data_loaders(data_dir, batch_size=32, val_split=0.1, test_split=0.1, num_workers=4):
    """
    Prepares DataLoaders for training, validation, and testing.
    Handles class imbalance using WeightedRandomSampler.
    """
    
    # We use a base dataset without specific transforms first to calculate stats or just load paths
    # But ImageFolder applies transforms on load. 
    # To handle different transforms for train/val/test from the same directory, 
    # we might need to rely on the user having split folders OR we split a single folder.
    
    # Strategy: Load all data, then split, then apply transforms (requires custom subset or dataset wrapper)
    # OR: Assume standard structure (train/val/test folders) or single folder.
    # Let's assume single folder structure for simplicity of the "split" requirement,
    # or handle the case where we need to split manually.
    
    # Check if subdirectories 'train', 'val', 'test' exist
    subdirs = os.listdir(data_dir)
    if 'train' in subdirs and 'test' in subdirs:
        print("Detected train/test folder structure.")
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        val_dir = os.path.join(data_dir, 'val') if 'val' in subdirs else None
        
        train_dataset = datasets.ImageFolder(root=train_dir, transform=get_transforms('train'))
        test_dataset = datasets.ImageFolder(root=test_dir, transform=get_transforms('test'))
        
        if val_dir:
             val_dataset = datasets.ImageFolder(root=val_dir, transform=get_transforms('val'))
        else:
            # Split train into train/val
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            # Note: random_split doesn't allow changing transform easily for the subset without a wrapper.
            # However, since we initialized with 'train' transform, val set will have augmentation.
            # This is suboptimal. 
            # Better approach for splitting: Create two instances of ImageFolder.
            full_dataset = datasets.ImageFolder(root=train_dir)
            train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])
            
            # Re-wrap with transforms
            train_dataset = torch.utils.data.Subset(datasets.ImageFolder(train_dir, transform=get_transforms('train')), train_indices.indices)
            val_dataset = torch.utils.data.Subset(datasets.ImageFolder(train_dir, transform=get_transforms('val')), val_indices.indices)

    else:
        print("Detected single data folder. Performing random split.")
        full_dataset = datasets.ImageFolder(root=data_dir)
        total_size = len(full_dataset)
        test_size = int(test_split * total_size)
        val_size = int(val_split * total_size)
        train_size = total_size - test_size - val_size
        
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
        
        # Apply transforms using a wrapper helper or just re-instantiating subsets
        # To keep it simple and clean, we will use a Helper Dataset class to apply transforms on the fly
        
        # Original dataset should perform NO transform, just PIL load
        train_dataset = TransformedSubset(train_dataset, transform=get_transforms('train'))
        val_dataset = TransformedSubset(val_dataset, transform=get_transforms('val'))
        test_dataset = TransformedSubset(test_dataset, transform=get_transforms('test'))

    # Handle Class Imbalance for Training
    # We need to access targets. For Subset, we need to iterate or access underlying dataset.
    print("Calculating class weights for imbalance handling...")
    
    # Extract targets from train_dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    elif isinstance(train_dataset, datasets.ImageFolder):
        targets = train_dataset.targets
    elif hasattr(train_dataset, 'subset'): # TransformedSubset
        # Access the underlying subset
        subset = train_dataset.subset
        if isinstance(subset, torch.utils.data.Subset):
             targets = [subset.dataset.targets[i] for i in subset.indices]
        else:
            # Fallback if structure is different
             targets = [y for _, y in train_dataset]
    else:
        targets = [y for _, y in train_dataset]

    class_counts = Counter(targets)
    print(f"Class counts in training set: {class_counts}")
    
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    # Val and Test don't need shuffling usually, but val is good to have shuffled if using for partial eval
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, class_counts
