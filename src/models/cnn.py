import torch.nn as nn


class SimpleCNN(nn.Module):
    """tiny cnn - kept it small so it takes shortcuts"""
    def __init__(self, in_channels=3, num_classes=10, width=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 7 * 7, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
