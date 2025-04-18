# importing from modularized utilities file 


import sys
import os

# Adding the parent folder (which contains utilities.py) to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utilities import FlexibleCNN
from utilities import validate
from torch.utils.data import DataLoader, random_split, Subset
import os
import torch.nn as nn
from torchvision import datasets, transforms


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 224

    data_dir = 'inaturalist_12K' # Note that: path to data, you will get this data folder in current directory after running, data downloading command provided in readme file.

    test_transform = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.4791, 0.4623, 0.3902], [0.2388, 0.2271, 0.2351]) # same here as that of training data
    ])

    batch_size = 32
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # defining the best model 
    best_model = FlexibleCNN(
            conv_filters=[256, 128, 64, 32, 16],
            kernel_sizes=[3, 5, 3, 5, 3],
            activation_fn_cnn='silu',
            activation_fn_dense='silu',
            dense_neurons=128,
            dropout_p=0.3,
            use_batchnorm=False,
            input_size=input_size  # Critical for shape matching
        ).to(device)

    # Load the state dict from the saved model checkpoint with weights_only=True
    path = os.path.join('Part A', 'best_model_wandb_config.pth')
    checkpoint = torch.load(path, map_location=device)

    # Modify the state_dict to only load the matching parameters
    model_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Match the layers between the checkpoint and the current model configuration
    model_state_dict = {k: v for k, v in model_state_dict.items() if k in best_model.state_dict() and v.shape == best_model.state_dict()[k].shape}

    # Load the matching state_dict into the model with strict=False to allow for missing keys
    best_model.load_state_dict(model_state_dict, strict=False)

    # Set the model to evaluation mode
    best_model.eval()

    # Evaluate on the test set
    test_loss, test_acc = validate(best_model, test_loader, nn.CrossEntropyLoss())
    print(f"Test Accuracy: {test_acc*100:.2f}%")

if __name__ == '__main__':
    main()