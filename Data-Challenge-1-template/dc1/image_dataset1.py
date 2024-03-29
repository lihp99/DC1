# this the image_dataset with augmentation and transformations
import numpy as np
import torch
import requests
import io
from os import path
from typing import Tuple
from pathlib import Path
import os
import torchvision
import kornia
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F

class ImageDataset:
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    """

    def __init__(self, x: Path, y: Path) -> None:
        # Target labels
        self.targets = ImageDataset.load_numpy_arr_from_npy(y)
        # Images
        self.imgs = ImageDataset.load_numpy_arr_from_npy(x)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]

        # geometric & miscellaneous data augmentation 
        mean = image.mean()
        std = image.std()
        transform = T.Compose([
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.Normalize(mean, std, inplace=False),
        ])

        # functional transforms, unsharp masking + histogram equalization
        if idx > int((len(self.targets))/2):
            image = transform(image)
            image = TF.adjust_sharpness(image, sharpness_factor=5)
            gaussian = T.GaussianBlur(kernel_size=(5, 9), sigma=(2, 5))
            gaussian_image = gaussian(image) 
            image = image+(image-gaussian_image)
            image = TF.equalize(image.type(torch.uint8))

        # Changing the number of channels from 1 to 3 to pass through the models
        transform1 = T.Compose([T.ToPILImage(), T.Grayscale(num_output_channels=3), T.ToTensor()])
        image = transform1(image)
        return image, label


    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.
        Input:
        path: local path of file
        Outputs:
        dataset: numpy array with input features or labels
        """
        original_file = np.load(path)
        augmented_file = np.load(path)
        numpy_array = np.concatenate((original_file, augmented_file))
        return numpy_array


def load_numpy_arr_from_url(url: str) -> np.ndarray:
    """
    Loads a numpy array from surfdrive.
    Input:
    url: Download link of dataset
    Outputs:
    dataset: numpy array with input features or labels
    """

    response = requests.get(url)
    response.raise_for_status()

    return np.load(io.BytesIO(response.content))


if __name__ == "__main__":
    cwd = os.getcwd()
    if path.exists(path.join(cwd + "data/")):
        print("Data directory exists, files may be overwritten!")
    else:
        os.mkdir(path.join(cwd, "../data/"))
    ### Load labels
    train_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/i6MvQ8nqoiQ9Tci/download"
    )
    np.save("../data/Y_train.npy", train_y)
    test_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/wLXiOjVAW4AWlXY/download"
    )
    np.save("../data/Y_test.npy", test_y)
    ### Load data
    train_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/4rwSf9SYO1ydGtK/download"
    )
    np.save("../data/X_train.npy", train_x)
    test_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/dvY2LpvFo6dHef0/download"
    )
    np.save("../data/X_test.npy", test_x)