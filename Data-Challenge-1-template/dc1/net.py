import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision
#
class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        # Defining a 2D convolution layer
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, n_classes)

        # self.linear_layers = nn.Sequential(
        #     nn.Linear(1152, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, n_classes)
        # )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.resnet18(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        return x




# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 16 * 16, 128)
#         self.fc2 = nn.Linear(128, 6)
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 16 * 16)
#         x = self.dropout(torch.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)













# This Model Achieves An Accuracy Of 31%

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(128 * 8 * 8, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 6)
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))
#         x = self.pool(torch.relu(self.conv4(x)))
#
#         x = x.view(-1, 128 * 8 * 8)
#         x = self.dropout(torch.relu(self.fc1(x)))
#         x = self.dropout(torch.relu(self.fc2(x)))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)