import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from network import CNN

import torch.nn as nn
import torch.optim as optim

def imshow(img):
    img = img / 2 + 0.5  # [-1, 1] -> [0, 1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Numpy Version: {np.__version__}")

    # Define the transformations to apply to the images
    # ToTensor(): [0, 255] -> [0, 1]
    # Normalize(): (x - mean) / std -> [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Batch size N
    batch_size = 4

    # Load the CIFAR-10 dataset (Train: 50000, Test: 10000)

    # Training data set
    # - train: Whether to load the training set (True) or the test set (False)
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                            download = True, transform = transform)
    # Extract the data
    # - batch_size: Number of samples in each batch
    # - shuffle: Whether to shuffle the data
    # - num_workers: Number of subprocesses to load the data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                              shuffle = True, num_workers = 2)

    # Test data set
    testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                           download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                             shuffle = False, num_workers = 2)
    
    # Define 10 classes
    classes = ('airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # # Get the training data of the first batch with random shuffling
    # # iter() create an iterator
    # # next() dereference the iterator and then move the iterator to the next element
    # dataiter = iter(trainloader)

    # # Shape of images: [batch_size, channels, height, width] (N, C, H, W)
    # # Shape of labels: [batch_size] (N,) -> Not one-hot encoding
    # images, labels = next(dataiter)

    # # Show images
    # imshow(torchvision.utils.make_grid(images))
    # # Print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Define CNN model
    model = CNN()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    # Train the model
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # Print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print('Finished Training')

    PATH = './saved_models/cifar_net.pth'
    torch.save(model.state_dict(), PATH)
