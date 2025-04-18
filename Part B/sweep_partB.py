# before run 
# make sure to install all the libraries form requirements.txt
# and download data in current directory by running the following comand in terminal
# wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
# unzip nature_12K.zip

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import wandb
from tqdm import tqdm


train_dir = "inaturalist_12K/train"  # Update as needed
test_dir = "inaturalist_12K/train"


wandb.login(key='613aac3388325cb6206db61e3c1a38a707589743')


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        
        
def stratified_split(dataset, val_pct=0.2):
    targets = np.array([y for _, y in dataset.samples])
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - val_pct))
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)

def get_resnet50(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def freeze_layers(model, strategy="fc_only"):
    for param in model.parameters():
        param.requires_grad = False
    if strategy == "fc_only":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif strategy == "last_block":
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    elif strategy == "all_unfrozen":
        for param in model.parameters():
            param.requires_grad = True
            
            
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [1e-3, 5e-4, 1e-4]},
        "batch_size": {"values": [32, 64]},
        "epochs": {"value": 6},
        "augment": {"values": [True, False]},
        "freeze_strategy": {"values": ["fc_only", "last_block", "all_unfrozen"]}
    }
}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
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

def evaluate(model, loader, criterion, device):
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


def sweep_train():
    wandb.init()
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms and loaders
    train_full = datasets.ImageFolder(train_dir, transform=get_transforms(config.augment))
    train_subset, val_subset = stratified_split(train_full, val_pct=0.2)
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Model and freezing
    model = get_resnet50(num_classes=10)
    freeze_layers(model, config.freeze_strategy)
    model = model.to(device)

    # Optimizer and loss
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
            
sweep_id = wandb.sweep(sweep_config, project="inat-finetune-sweep-resnet50")
wandb.agent(sweep_id, function=sweep_train, count=10)


best_config = {
    "freeze_strategy": "fc_only"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(False))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

model = get_resnet50(num_classes=10)
freeze_layers(model, best_config["freeze_strategy"])
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
print(f"Test Accuracy: {test_acc:.4f}")


import matplotlib.pyplot as plt

classes = test_dataset.classes
images_shown, preds_shown, labels_shown = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu()
        images_shown.extend(images)
        preds_shown.extend(preds)
        labels_shown.extend(labels)
        if len(images_shown) >= 30:
            break

fig, axes = plt.subplots(10, 3, figsize=(12, 30))
for i, ax in enumerate(axes.flat):
    img = images_shown[i].permute(1,2,0).numpy() * np.array(imagenet_std) + np.array(imagenet_mean)
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f"Pred: {classes[preds_shown[i]]}\nTrue: {classes[labels_shown[i]]}")
    ax.axis('off')
plt.tight_layout()
plt.show()