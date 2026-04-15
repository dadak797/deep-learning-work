import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution layer 1
        # shape: (batch_size, channels, height, width) -> (batch_size, out_channels, height_out, width_out)
        # input shape: (4, 3, 32, 32) -> output shape: (4, 6, 28, 28)
        # output_size = (input_size + 2 * padding - kernel_size) / stride + 1
        #             = 32 + 2 * 0 - 5 / 1 + 1 = 28
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)

        # Max pooling layer 1
        # input shape: (4, 6, 28, 28) -> output shape: (4, 6, 14, 14)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Convolution layer 2
        # input shape: (4, 6, 14, 14) -> output shape: (4, 16, 10, 10)
        # output_size = (input_size + 2 * padding - kernel_size) / stride + 1
        #             = 14 + 2 * 0 - 5 / 1 + 1 = 10
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)  # Convolution layer 2
        
        # Max pooling layer 2
        # input shape: (4, 16, 10, 10) -> output shape: (4, 16, 5, 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Fully connected layer 1
        # input shape: (4, 16 * 5 * 5) -> output shape: (4, 120)
        self.fc1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)

        # Fully connected layer 2
        # input shape: (4, 120) -> output shape: (4, 84)
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)

        # Fully connected layer 3
        # input shape: (4, 84) -> output shape: (4, 10)
        self.fc3 = nn.Linear(in_features = 84, out_features = 10)

    def forward(self, x):
        # Convolution layer 1 + ReLU + Max pooling layer 1
        x = self.pool1(F.relu(self.conv1(x)))

        # Convolution layer 2 + ReLU + Max pooling layer 2
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten the output of the convolutional layers
        # (4, 16, 5, 5) -> (4, 16 * 5 * 5)
        x = x.flatten(start_dim = 1)

        # Fully connected layer 1 + ReLU
        x = F.relu(self.fc1(x))

        # Fully connected layer 2 + ReLU
        x = F.relu(self.fc2(x))

        # Fully connected layer 3 (Output layer)
        x = self.fc3(x)

        return x
    