import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__() # CONV -> RELU -> DROP
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 224 x 224 x 16
            nn.BatchNorm2d(16), # used after convolution and before activation
            nn.ReLU(), # used after convolution
            nn.Dropout(p=0.1), # used after the activation of each conv layer
            nn.MaxPool2d(2,2), # pool final output # 112 x 112 x 16

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 112 x 112 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Dropout(p=0.1), 
            nn.MaxPool2d(2,2), # 56 x 56 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 56 x 56 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Dropout(p=0.1),
            nn.MaxPool2d(2,2), # 28 x 28 x 64
        )

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(28 * 28 * 64, 10000),  # average between flatten features and num classes 50,176 x 10000
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(10000, num_classes) # problem with out features
        )

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
