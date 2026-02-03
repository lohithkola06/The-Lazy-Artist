import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Minimal 3-layer CNN for 28x28 RGB inputs (Colored MNIST).
    
    Architecture with width=4:
    -------------------------
    Input:  (batch, 3, 28, 28)     # RGB image
    
    Conv1:  3 -> 4 filters, 3x3, pad=1  =>  (batch, 4, 28, 28)
    Pool1:  2x2 max pool                =>  (batch, 4, 14, 14)
    
    Conv2:  4 -> 8 filters, 3x3, pad=1  =>  (batch, 8, 14, 14)
    Pool2:  2x2 max pool                =>  (batch, 8, 7, 7)
    
    Conv3:  8 -> 16 filters, 3x3, pad=1 =>  (batch, 16, 7, 7)
    
    Flatten: 16 * 7 * 7 = 784 features
    FC:      784 -> 10 classes
    
    Total params: ~8K (very small intentionally)
    
    This tiny model is designed to be capacity-limited, making it more
    likely to rely on simple features (like color) rather than complex
    shape patterns.
    """
    def __init__(self, in_channels=3, num_classes=10, width=6):
        super().__init__()
        
        # width controls base number of filters: width, width*2, width*4
        self.features = nn.Sequential(
            # Layer 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 2: 14x14 -> 7x7
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 3: 7x7 (no pooling)
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Classifier: flatten -> single FC layer to 10 classes
        # Feature size: width*4 channels * 7*7 spatial = width*4*49
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 7 * 7, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
