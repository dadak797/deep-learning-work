import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
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

class SmallVGG9(nn.Module):
  def __init__(self):
    super(SmallVGG9, self).__init__()

    # The original VGG16 takes (224, 224) with 3 channels as input,
    # but CIFAR-10 has (32, 32) with 3 channels.

    # Convolutional layer 1-1
    # shape: (batch_size, channels, height, width) -> (batch_size, out_channels, height_out, width_out)
    # input shape: (4, 3, 32, 32) -> output shape: (4, 16, 32, 32)
    # output_size = (input_size + 2 * padding - kernel_size) / stride + 1
    #             = 32 + 2 * 1 - 3 / 1 + 1 = 32
    # Convolutional layer 1-2
    # input shape: (4, 16, 32, 32) -> output shape: (4, 16, 32, 32)
    # Max pooling layer 1
    # input shape: (4, 16, 32, 32) -> output shape: (4, 16, 16, 16)
    self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
    self.conv1_2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)
    self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Convolutional layer 2-1
    # input shape: (4, 16, 16, 16) -> output shape: (4, 32, 16, 16)
    # Convolutional layer 2-2
    # input shape: (4, 32, 16, 16) -> output shape: (4, 32, 16, 16)
    # Max pooling layer 2
    # input shape: (4, 32, 16, 16) -> output shape: (4, 32, 8, 8)
    self.conv2_1 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
    self.conv2_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)    
    self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Convolutional layer 3-1
    # input shape: (4, 32, 8, 8) -> output shape: (4, 64, 8, 8)
    # Convolutional layer 3-2
    # input shape: (4, 64, 8, 8) -> output shape: (4, 64, 8, 8)
    # Max pooling layer 3
    # input shape: (4, 64, 8, 8) -> output shape: (4, 64, 4, 4)
    self.conv3_1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self.conv3_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
    self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Fully connected layer 1
    # input shape: (4, 64 * 4 * 4) -> output shape: (4, 256)
    self.fc1 = nn.Linear(in_features = 64 * 4 * 4, out_features = 256)
    # Fully connected layer 2
    # input shape: (4, 256) -> output shape: (4, 64)
    self.fc2 = nn.Linear(in_features = 256, out_features = 64)
    # Fully connected layer 3
    # input shape: (4, 64) -> output shape: (4, 10)
    self.fc3 = nn.Linear(in_features = 64, out_features = 10)

  def forward(self, x):
    # Convolutional layer 1-1 + ReLU
    x = F.relu(self.conv1_1(x))
    # Convolutional layer 1-2 + ReLU + Max pooling layer 1
    x = self.pool1(F.relu(self.conv1_2(x)))

    # Convolutional layer 2-1 + ReLU
    x = F.relu(self.conv2_1(x))
    # Convolutional layer 2-2 + ReLU + Max pooling layer 2
    x = self.pool2(F.relu(self.conv2_2(x)))

    # Convolutional layer 3-1 + ReLU
    x = F.relu(self.conv3_1(x))
    # Convolutional layer 3-2 + ReLU + Max pooling layer 3
    x = self.pool3(F.relu(self.conv3_2(x)))

    # Flatten the output of the convolutional layers
    # (4, 64, 4, 4) -> (4, 64 * 4 * 4)
    x = x.flatten(start_dim = 1)

    # Fully connected layer 1 + ReLU
    x = F.relu(self.fc1(x))

    # Fully connected layer 2 + ReLU
    x = F.relu(self.fc2(x))

    # Fully connected layer 3 (Output layer)
    x = self.fc3(x)

    return x

# More channel than SmallVGG9, and two fully connected layers
class MediumVGG8(nn.Module):
  def __init__(self):
    super(MediumVGG8, self).__init__()
    self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
    self.conv1_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
    self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    self.conv2_1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self.conv2_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
    self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    self.conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    self.conv3_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
    self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    self.fc1 = nn.Linear(in_features = 128 * 4 * 4, out_features = 512)
    self.fc2 = nn.Linear(in_features = 512, out_features = 10)

  def forward(self, x):
    x = F.relu(self.conv1_1(x))
    x = self.pool1(F.relu(self.conv1_2(x)))

    x = F.relu(self.conv2_1(x))
    x = self.pool2(F.relu(self.conv2_2(x)))

    x = F.relu(self.conv3_1(x))
    x = self.pool3(F.relu(self.conv3_2(x)))

    x = x.flatten(start_dim = 1)

    x = F.relu(self.fc1(x))

    x = self.fc2(x)

    return x

class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()

    # Convolutional layer 1-1
    # (4, 3, 32, 32) -> (4, 64, 32, 32)
    # Convolutional layer 1-2
    # (4, 64, 32, 32) -> (4, 64, 32, 32)
    # Max pooling layer 1
    # (4, 64, 32, 32) -> (4, 64, 16, 16)
    self.conv1_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
    self.bn1_1 = nn.BatchNorm2d(64)
    self.conv1_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
    self.bn1_2 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Convolutional layer 2-1
    # (4, 64, 16, 16) -> (4, 128, 16, 16)
    # Convolutional layer 2-2
    # (4, 128, 16, 16) -> (4, 128, 16, 16)
    # Max pooling layer 2
    # (4, 128, 16, 16) -> (4, 128, 8, 8)
    self.conv2_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    self.bn2_1 = nn.BatchNorm2d(128)
    self.conv2_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
    self.bn2_2 = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Convolutional layer 3-1
    # (4, 128, 8, 8) -> (4, 256, 8, 8)
    # Convolutional layer 3-2
    # (4, 256, 8, 8) -> (4, 256, 8, 8)
    # Convolutional layer 3-3
    # (4, 256, 8, 8) -> (4, 256, 8, 8)
    # Max pooling layer 3
    # (4, 256, 8, 8) -> (4, 256, 4, 4)
    self.conv3_1 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
    self.bn3_1 = nn.BatchNorm2d(256)
    self.conv3_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
    self.bn3_2 = nn.BatchNorm2d(256)
    self.conv3_3 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
    self.bn3_3 = nn.BatchNorm2d(256)
    self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Convolutional layer 4-1
    # (4, 256, 4, 4) -> (4, 512, 4, 4)
    # Convolutional layer 4-2
    # (4, 512, 4, 4) -> (4, 512, 4, 4)
    # Convolutional layer 4-3
    # (4, 512, 4, 4) -> (4, 512, 4, 4)
    # Max pooling layer 4
    # (4, 512, 4, 4) -> (4, 512, 2, 2)
    self.conv4_1 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
    self.bn4_1 = nn.BatchNorm2d(512)
    self.conv4_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
    self.bn4_2 = nn.BatchNorm2d(512)
    self.conv4_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
    self.bn4_3 = nn.BatchNorm2d(512)
    self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Convolutional layer 5-1
    # (4, 512, 2, 2) -> (4, 512, 2, 2)
    # Convolutional layer 5-2
    # (4, 512, 2, 2) -> (4, 512, 2, 2)
    # Convolutional layer 5-3
    # (4, 512, 2, 2) -> (4, 512, 2, 2)
    # Max pooling layer 5
    # (4, 512, 2, 2) -> (4, 512, 1, 1)
    self.conv5_1 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
    self.bn5_1 = nn.BatchNorm2d(512)
    self.conv5_2 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
    self.bn5_2 = nn.BatchNorm2d(512)
    self.conv5_3 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)
    self.bn5_3 = nn.BatchNorm2d(512)
    self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

    # Fully connected layer 1
    # (4, 512 * 1 * 1) -> (4, 4096)
    # Fully connected layer 2
    # (4, 4096) -> (4, 4096)
    # Fully connected layer 3
    # (4, 4096) -> (4, 10)
    self.fc1 = nn.Linear(in_features = 512 * 1 * 1, out_features = 4096)
    self.dropout1 = nn.Dropout(p = 0.5)
    self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)
    self.dropout2 = nn.Dropout(p = 0.5)
    self.fc3 = nn.Linear(in_features = 4096, out_features = 10)

  def forward(self, x):
    x = F.relu(self.bn1_1(self.conv1_1(x)))
    x = self.pool1(F.relu(self.bn1_2(self.conv1_2(x))))

    x = F.relu(self.bn2_1(self.conv2_1(x)))
    x = self.pool2(F.relu(self.bn2_2(self.conv2_2(x))))

    x = F.relu(self.bn3_1(self.conv3_1(x)))
    x = F.relu(self.bn3_2(self.conv3_2(x)))
    x = self.pool3(F.relu(self.bn3_3(self.conv3_3(x))))

    x = F.relu(self.bn4_1(self.conv4_1(x)))
    x = F.relu(self.bn4_2(self.conv4_2(x)))
    x = self.pool4(F.relu(self.bn4_3(self.conv4_3(x))))

    x = F.relu(self.bn5_1(self.conv5_1(x)))
    x = F.relu(self.bn5_2(self.conv5_2(x)))
    x = self.pool5(F.relu(self.bn5_3(self.conv5_3(x))))

    x = x.flatten(start_dim = 1)

    x = F.relu(self.fc1(x))
    x = self.dropout1(x)

    x = F.relu(self.fc2(x))
    x = self.dropout2(x)

    x = self.fc3(x)

    return x
