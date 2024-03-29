# Custom imports (Note: we gather the augmented data from image_dataset1 with all the different filters and transformations)
from batch_sampler import BatchSampler
from image_dataset1 import ImageDataset
from net import Net
from net import TemplateNet
from net import ResNet
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


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
    test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"))

    # Load the Neural Nets for evaluating performance. NOTE: set number of distinct labels here
    modelDenseNet = Net(n_classes=6)
    modelDenseNet2 = Net(n_classes=6)
    modelTemplate = TemplateNet(n_classes=6)
    modelTemplate2 = TemplateNet(n_classes=6)
    modelResNet = ResNet(n_classes=6)
    modelResNet2 = ResNet(n_classes=6)
    models = [modelDenseNet, modelDenseNet2, modelTemplate, modelTemplate2, modelResNet, modelResNet2]

    # Initializing the optimizers for each model
    optimizer_AdamD = optim.Adam(modelDenseNet.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optimizer_SGDD2 = optim.SGD(modelDenseNet2.parameters(), lr=0.001, momentum=0.9)
    optimizer_AdamT = optim.Adam(modelTemplate.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optimizer_SGDT2 = optim.SGD(modelTemplate2.parameters(), lr=0.001, momentum=0.9)
    optimizer_AdamR = optim.Adam(modelResNet.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optimizer_SGDR2 = optim.SGD(modelResNet2.parameters(), lr=0.001, momentum=0.9)
    optimizers = [optimizer_AdamD, optimizer_SGDD2, optimizer_AdamT, optimizer_SGDT2, optimizer_AdamR, optimizer_SGDR2]

    # Initialize loss function(s)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # Define the early stopping parameters
    patience = 8
    counter = 0

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False


    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )


    # Introducing the grid search cross validation
    for model, optimizer in zip(models, optimizers):
            


        if torch.cuda.is_available() and not DEBUG:
            print("@@@ CUDA device found, enabling CUDA training...")
            device = "cuda"
            model.to(device)
            # Creating a summary of our model and its layers:
            # summary(model, (3, 128, 128), device=device)
        elif (
            torch.backends.mps.is_available() and not DEBUG
        ):  # PyTorch supports Apple Silicon GPU's from version 1.12
            print("@@@ Apple silicon device enabled, training with Metal backend...")
            device = "mps"
            model.to(device)
        else:
            print("@@@ No GPU boosting device found, training on CPU...")
            device = "cpu"
            # Creating a summary of our model and its layers:
            # summary(model, (3, 128, 128), device=device)



        # Initializing best accuracy for the early stopping and mean loss score
        best_accuracy = 0
        mean_losses_train: List[torch.Tensor] = []
        mean_losses_test: List[torch.Tensor] = []   
        for e in range(n_epochs):
            if activeloop:

                # Training:
                losses = train_model(model, train_sampler, optimizer, loss_function, device)
                # Calculating and printing statistics:
                mean_loss = sum(losses) / len(losses)
                mean_losses_train.append(mean_loss)
                print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

                # Testing:
                losses, accuracy, confusion_mat, class_report, true_labels, predicted_labels = test_model(model, test_sampler, loss_function, device)
                # # Calculating and printing statistics:
                mean_loss = sum(losses) / len(losses)
                mean_losses_test.append(mean_loss)
                print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}; Accuracy: {accuracy}; \n{confusion_mat}\n{class_report}")

                labels = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule', 'Pneumothorax']
                plt.figure(figsize=(10,6))
                        
                        
                ### Plotting during training
                plotext.clf()
                plotext.scatter(mean_losses_train, label="train")
                plotext.scatter(mean_losses_test, label="test")
                plotext.title("Train and test loss")

                plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

                plotext.show()

                # Adding the early stopping conditions
                if best_accuracy < accuracy:
                    counter = 0
                    best_accuracy = accuracy
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping")
                        break




    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    
    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")
    
    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=20, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)