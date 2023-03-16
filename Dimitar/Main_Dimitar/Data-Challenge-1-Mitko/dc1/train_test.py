from tqdm import tqdm
import torch
from net import Net
from batch_sampler import BatchSampler
from typing import Callable, List
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []

    predicted_labels = []
    true_labels = []
    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        correct = 0
        count = 0
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)
            prediction1 = model.forward(x).argmax(axis=1)
            correct += sum(prediction1 == y)
            count += len(y)

            predicted_labels.extend(prediction1.detach().cpu().numpy())
            true_labels.extend(y.detach().cpu().numpy())
        accuracy = (correct/count).detach().cpu().numpy()

    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels)
    return losses, accuracy, confusion_mat, class_report
