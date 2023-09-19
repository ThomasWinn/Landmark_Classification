import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.3) -> None:

        super().__init__() # CONV -> RELU -> DROP

        # Took out dropout due to batchnorm having similar regularization effects in convolution layers.

        # Recent studies show to add lower dropout on conv layers
        # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
        self.backbone = nn.Sequential(

            # Ordering is in the formation as done in the udacity exercise notebooks
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 16, 224, 224
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2), # 16, 112, 112
            nn.ReLU(),
            # nn.Dropout2d(p=0.2), 

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 32, 112, 112
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2), #32, 56, 56
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64, 56, 56
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2), # 64, 28, 28
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128, 28, 28
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2), # 128, 14, 14
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256, 14, 14
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2), # 256, 7, 7
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256, 7, 7
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout2d(p=0.2),
        )
        
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(

            # Ordering is in the formation as done in the udacity exercise notebooks
            nn.Linear(7 * 7 * 256, 4096),  # (7 * 7 * 256 + num_classes) / 2 = 6297 where 4096 is closest power of 2
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            # Took out look at write up for explanation
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),

            nn.Linear(1024, num_classes) 
        )
        
        self.softmax = nn.Softmax(dim=1)

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.head(x)

        return x
        # return self.softmax(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
