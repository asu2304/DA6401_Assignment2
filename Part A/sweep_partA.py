# make sure to install all the libraries form requirements.txt
# and download data in current directory by running the following comand in terminal
# wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
# unzip nature_12K.zip

# importing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

# Adding the parent folder (which contains utilities.py) to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import FlexibleCNN


wandb.login(key='613aac3388325cb6206db61e3c1a38a707589743')

import os
# Dataset path
data_dir = '/kaggle/working/inaturalist_12K'  # Update this path to your dataset

# Define the basic transform for the dataset (no normalization yet)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load dataset without normalization
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)

# Use DataLoader to compute the mean and std
loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

# Compute the mean and std of the dataset
mean = torch.zeros(3)
std = torch.zeros(3)
for images, _ in loader:
    # Compute mean and std for each channel
    mean += images.mean([0, 2, 3])  # mean for (R, G, B)
    std += images.std([0, 2, 3])   # std for (R, G, B)

mean /= len(loader)
std /= len(loader)

print(f"Mean: {mean}")
print(f"Std: {std}")


# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (MUST match model's input_size)
input_size = 224
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4791, 0.4623, 0.3902], [0.2388, 0.2271, 0.2351]) # normalizing with respect to imagenet dataset values
])

test_transform = transforms.Compose([
    transforms.Resize(input_size + 32),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.4791, 0.4623, 0.3902], [0.2388, 0.2271, 0.2351]) # same here
])

# Load dataset
data_dir = '/kaggle/working/inaturalist_12K'  
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)

# Stratified validation split
def stratified_split(dataset, val_ratio=0.2):
    targets = np.array([s[1] for s in dataset.samples])
    val_indices = []
    train_indices = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - val_ratio))
        train_indices.extend(idx[:split])
        val_indices.extend(idx[split:])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

train_subset, val_subset = stratified_split(train_dataset)

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)

def train_one_epoch(model, loader, optimizer, criterion):
    
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


sweep_config = {
    'method': 'bayes',
    'name': 'cnn_sweep',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'conv_filters': {
           'values': [
        [32, 64, 128, 256, 512],       # doubing in each subsequent layer
        [64, 128, 256, 512, 512],      # aggressive start
        [32, 32, 64, 64, 128],         # shallow early focus
        [64, 64, 128, 128, 256],       # slow inc filter configuration
        [128, 128, 128, 128, 128],     # constant filter configuration
        [256, 128, 64, 32, 16],        # halving in each subsequent layer
    ]
        },
        'kernel_sizes': {
            'values': [
        [3, 3, 3, 3, 3],           # Standard
        [5, 3, 3, 3, 3],           # Slightly larger receptive field early on
        [3, 5, 3, 5, 3],           # Alternating for varied feature capture
        [7, 5, 3, 3, 3],           # Very large first kernel, good for coarse features
    ]
        },
        'activation_fn_cnn': {'values': ['relu', 'gelu', 'silu']},
        'epochs': {'values': [5, 10]},
        'activation_fn_dense': {'values': ['relu', 'gelu', 'silu']},
        'dense_neurons': {'values': [64, 128, 256, 512]},
        'dropout_p': {'values': [0.2, 0.3]},
        'use_batchnorm': {'values': [True, False]},
        'lr': {'values': [1e-3, 5e-4]},
    }
}

def train_model():
    
    wandb.init()
    config = wandb.config
    run_name = f"F:{config.conv_filters}_K:{config.kernel_sizes}_Acnn:{config.activation_fn_cnn}_Adense:{config.activation_fn_dense}_D:{config.dropout_p}_B:{config.use_batchnorm}_LR:{config.lr}_E:{config.epochs}"
    wandb.run.name = run_name  # Set descriptive name for this run
    
    model = FlexibleCNN(
        conv_filters=config.conv_filters,
        kernel_sizes=config.kernel_sizes,
        activation_fn_cnn=config.activation_fn_cnn,
        activation_fn_dense=config.activation_fn_dense,
        dense_neurons=config.dense_neurons,
        dropout_p=config.dropout_p,
        use_batchnorm=config.use_batchnorm,
        input_size=input_size  # Critical for shape matching
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_version_2(new).pth')
    
    wandb.finish()
    
    # Initialize sweep
wandb.login()
sweep_id = wandb.sweep(sweep_config, project="iNaturalist-CNN")

# Run 20 experiments (adjust count as needed)
wandb.agent(sweep_id, train_model, count=30)