import torch
import torch.nn as nn
import torchvision

class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        # Defining a 2D convolution layer
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, n_classes)

        # self.linear_layers = nn.Sequential(
        #     nn.Linear(1152, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, n_classes)
        # )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.resnet50(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        return x