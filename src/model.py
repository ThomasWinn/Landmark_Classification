import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.3) -> None:

        super().__init__() # CONV -> RELU -> DROP
        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, padding=1), # 224 x 224 x 16
        #     nn.ReLU(), # used after convolution
        #     nn.BatchNorm2d(16), # used after convolution and before activation
        #     nn.Dropout(p=0.1), # used after the activation of each conv layer
        #     nn.MaxPool2d(2,2), # pool final output # 112 x 112 x 16

        #     nn.Conv2d(16, 32, kernel_size=3, padding=1), # 112 x 112 x 32
        #     nn.ReLU(), 
        #     nn.BatchNorm2d(32),
        #     nn.Dropout(p=0.1), 
        #     nn.MaxPool2d(2,2), # 56 x 56 x 32

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1), # 56 x 56 x 64
        #     nn.ReLU(), 
        #     nn.BatchNorm2d(64),
        #     nn.Dropout(p=0.1),
        #     nn.MaxPool2d(2,2), # 28 x 28 x 64
        # )

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 224 x 224 x 16
            nn.BatchNorm2d(16), # used after convolution and before activation
            nn.MaxPool2d(2,2), # pool final output # 112 x 112 x 16
            nn.ReLU(), # used after convolution
            nn.Dropout(p=0.2), # used after the activation of each conv layer

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 112 x 112 x 32
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2), # 56 x 56 x 32
            nn.ReLU(), 
            nn.Dropout(p=0.2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 56 x 56 x 64
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2), # 28 x 28 x 64
            nn.ReLU(), 
            nn.Dropout(p=0.2),
        )
        
        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, padding=1), # 224 x 224 x 16
        #     nn.ReLU(),
        #     nn.BatchNorm2d(16), # used after convolution and before activation
        #     # nn.Dropout(p=0.1), # used after the activation of each conv layer
        #     nn.Dropout(p=dropout),
        #     nn.MaxPool2d(2,2), # pool final output # 112 x 112 x 16

        #     nn.Conv2d(16, 32, kernel_size=3, padding=1), # 112 x 112 x 32
        #     nn.ReLU(), 
        #     nn.BatchNorm2d(32),
        #     # nn.Dropout(p=0.1), 
        #     nn.Dropout(p=dropout),
        #     nn.MaxPool2d(2,2), # 56 x 56 x 32

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1), # 56 x 56 x 64
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     # nn.Dropout(p=0.1),
        #     nn.Dropout(p=dropout),
        #     nn.MaxPool2d(2,2), # 28 x 28 x 64

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1), # 28 x 28 x 128
        #     nn.ReLU(), 
        #     nn.BatchNorm2d(128),
        #     # nn.Dropout(p=0.1), 
        #     nn.Dropout(p=dropout),
        #     nn.MaxPool2d(2,2), # 14 x 14 x 32

        #     nn.Conv2d(128, 256, kernel_size=3, padding=1), # 14 x 14 x 256
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     # nn.Dropout(p=0.1),
        #     nn.Dropout(p=dropout),
        #     nn.MaxPool2d(2,2), # 7 x 7 x 256
        # )


        self.flatten = nn.Flatten()

        # self.head = nn.Sequential(
        #     nn.Linear(7 * 7 * 256, 256),  # average between flatten features and num classes 50,176 x 10000
        #     # nn.Dropout(p=dropout),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(256, num_classes) # problem with out features
        # )

        # self.head = nn.Sequential(
        #     nn.Linear(7 * 7 * 256, 512),  # average between flatten features and num classes 50,176 x 10000
        #     # nn.Dropout(p=dropout),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(128, num_classes) # problem with out features
        # )

        # self.head = nn.Sequential(
        #     nn.Linear(28 * 28 * 64, 256),  # average between flatten features and num classes 50,176 x 10000
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(128, num_classes) # problem with out features
        # )


        self.head = nn.Sequential(
            nn.Linear(28 * 28 * 64, 256),  # average between flatten features and num classes 50,176 x 10000
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes) # problem with out features
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

        # return F.softmax(x, dim=1)
        # return x
        return self.softmax(x)


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
