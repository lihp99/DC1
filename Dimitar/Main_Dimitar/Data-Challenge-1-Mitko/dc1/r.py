# Custom imports
from batch_sampler import BatchSampler
from image_dataset import ImageDataset
from net import Net
from train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

model = Net(n_classes=6)

# Load the saved model weights
model.load_state_dict(torch.load('./model_weights/model_04_06_15_32.txt'))
test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"))
device = "cpu"
loss_function = nn.CrossEntropyLoss()
test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=True
    )
losses, accuracy, confusion_mat, class_report, true_labels, predicted_labels = test_model(model, test_sampler, loss_function, device)
print(accuracy)
# Set the model to evaluation mode