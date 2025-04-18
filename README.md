



# DA6401 - Assignment 2  
**Author:** Siva Sankar S (ch20b103)  
**Last Updated:** March 28, 2025

---

## Overview

This repository contains solutions for DA6401 Assignment 2, which focuses on training and fine-tuning CNN models for image classification using the iNaturalist dataset.  
- **Part A:** Train a CNN from scratch and perform hyperparameter sweeps.
- **Part B:** Fine-tune a pre-trained model (e.g., ResNet, VGG) on the same data.

All code is written in Python using PyTorch and can be run on Colab with GPU support.

---

## Directory Structure

```plaintext
DA6401_Assignment2/
│
├── Part A/
│   ├── best_model_wandb_config.pth      # Best model weights (Part A)
│   ├── da6401-assignment-2-part-a.ipynb # Jupyter notebook for Part A
│   ├── sweep_partA.py                   # Script to run W&B sweep (Part A)
│   ├── test_evalution.py                # Script to evaluate best model (Part A)
│   └── tempCodeRunnerFile.py            # (Temporary, can be ignored)
│
├── Part B/
│   ├── best_model.pth                   # Best model weights (Part B)
│   ├── da6401-assignment-2-part-b.ipynb # Jupyter notebook for Part B
│   ├── sweep_partB.py                   # Script to run W&B sweep (Part B)
│
├── wandb/                               # W&B logs (auto-generated)
├── utilities.py                         # Utility functions (shared)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── .gitignore                           # Git ignore file
```

---

## Quick Start

### 1. **Setup**

- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure you have access to the iNaturalist dataset and set up your data paths as needed.

### 2. **Running Experiments**

#### **Part A: Train CNN from Scratch**
- **Hyperparameter Sweep:**  
  Run:
  ```bash
  python Part\ A/sweep_partA.py
  ```
  - This will start a W&B sweep for hyperparameter tuning.
  - Results and best configuration are logged to W&B.

- **Evaluate Best Model:**  
  Run:
  ```bash
  python Part\ A/test_evalution.py
  ```
  - Loads `best_model_wandb_config.pth` and evaluates on the test set.

- **Notebook Reference:**  
  - For a step-by-step implementation and outputs, see the [Kaggle notebook](https://www.kaggle.com/code/asu2304/da6401-assignment-2-part-a?scriptVersionId=234542588) or open `da6401-assignment-2-part-a.ipynb`.

#### **Part B: Fine-Tune Pre-trained Model**
- **Hyperparameter Sweep:**  
  Run:
  ```bash
  python Part\ B/sweep_partB.py
  ```
  - This script fine-tunes a pre-trained model (e.g., ResNet, VGG) using the best strategies found.

- **Notebook Reference:**  
  - For implementation details and results, see the [Kaggle notebook for Part B](https://www.kaggle.com/code/asu2304/da6401-assignment-2-part-b#8.-Evaluate-Best-Model-on-Test-Set) or open `da6401-assignment-2-part-b.ipynb`.

---

## Key Files and Their Purpose

| File/Folder                       | Purpose                                                      |
|-----------------------------------|--------------------------------------------------------------|
| `Part A/sweep_partA.py`           | Run W&B hyperparameter sweep for custom CNN                  |
| `Part A/test_evalution.py`        | Evaluate best model on test set (Part A)                     |
| `Part A/best_model_wandb_config.pth` | Best model weights from sweep (Part A)                   |
| `Part B/sweep_partB.py`           | Run W&B sweep for pre-trained model fine-tuning (Part B)     |
| `Part B/best_model.pth`           | Best fine-tuned model weights (Part B)                       |
| `utilities.py`                    | Shared utility functions                                     |
| `requirements.txt`                | All required Python packages                                 |
| `README.md`                       | This file                                                    |

---

## Results & Reports

- **W&B Report:**  
  All experiment logs, plots, and hyperparameter sweeps are available in the [W&B report](https://api.wandb.ai/links/da24s006-indian-institue-of-technology-madras-/ik3lomie).

- **Kaggle Notebooks:**  
  - [Part A Notebook](https://www.kaggle.com/code/asu2304/da6401-assignment-2-part-a?scriptVersionId=234542588)
  - [Part B Notebook](https://www.kaggle.com/code/asu2304/da6401-assignment-2-part-b#8.-Evaluate-Best-Model-on-Test-Set)

---

## How to Reproduce Results

1. Clone this repository and install dependencies.
2. Download and prepare the iNaturalist dataset.
3. For Part A, run the sweep and evaluation scripts as shown above.
4. For Part B, run the sweep script for fine-tuning.
5. For detailed outputs and visualizations, refer to the Kaggle notebooks or W&B report.

---

## Frequently Asked Questions

- **Where is the main training code for Part A?**  
  All logic is in `sweep_partA.py`. No separate code for Q1; everything is modular and flexible as required.

- **How to get quick results or visual outputs?**  
  Open the Kaggle notebooks for Part A and B for immediate results and visualizations.

- **How to run sweeps again?**  
  Simply run the respective `sweep_partA.py` or `sweep_partB.py` scripts.

---

## Self Declaration

I, **Ashutosh Patidar** (Roll no: **DA24S006**), declare that all code and reports are my own work.

---

**For any questions, please refer to the code comments, notebooks, or contact me directly.**

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50039183/a3837571-51cc-44a5-a973-b92247868c94/paste.txt
[2] https://pplx-res.cloudinary.com/image/private/user_uploads/NtgpWFQIcLJRbpF/image.jpg

---
Answer from Perplexity: pplx.ai/share



























# DA6401_Assignment2
Assignment-2 Introduction to Deep Learning: Implementation of CNN's, training from scractch and finetuning.

report link: https://wandb.ai/da24s006-indian-institue-of-technology-madras-/iNaturalist-CNN/reports/DA6401-Assignment-2-DA24S006-Ashutosh-Patidar---VmlldzoxMjMxOTczOQ


run the following command to get the data in the current directory: code: 

!wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
!unzip nature_12K.zip
