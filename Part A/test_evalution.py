# importing relevant libs

import torch
from models import FlexibleCNN

# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split, Subset
# import numpy as np
# import wandb
# from tqdm import tqdm
# import matplotlib.pyplot as plt

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
checkpoint = torch.load('best_model_wandb_config.pth', weights_only=True)

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
