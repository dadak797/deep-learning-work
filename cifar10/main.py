import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from network import CNN

import torch.nn as nn
import torch.optim as optim

MODEL_PATH = './saved_models/cifar_net.pth'


def imshow(img):
  img = img / 2 + 0.5  # [-1, 1] -> [0, 1]
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def parse_args():
  parser = argparse.ArgumentParser(description = "CIFAR-10 train/test runner")
  parser.add_argument(
    "--mode",
    choices = ["train", "test"],
    required = True,
    help = "Choose whether to train the model or test a saved model.",
  )
  parser.add_argument(
    "--restart",
    action = "store_true",
    help = "Resume training from the saved model when used with --mode train.",
  )
  return parser.parse_args()


def build_dataloaders(batch_size):
  # Define the transformations to apply to the images
  # ToTensor(): [0, 255] -> [0, 1]
  # Normalize(): (x - mean) / std -> [-1, 1]
  transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
  )

  # Load the CIFAR-10 dataset (Train: 50000, Test: 10000)
  trainset = torchvision.datasets.CIFAR10(
    root = './data', train = True, download = True, transform = transform
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size = batch_size, shuffle = True, num_workers = 2
  )

  testset = torchvision.datasets.CIFAR10(
    root = './data', train = False, download = True, transform = transform
  )
  testloader = torch.utils.data.DataLoader(
    testset, batch_size = batch_size, shuffle = False, num_workers = 2
  )

  return trainloader, testloader


def train_model(model, trainloader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

  for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
        print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
        running_loss = 0.0

  print('Finished Training')

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok = True)
  torch.save(model.state_dict(), MODEL_PATH)
  print(f"Saved model to {MODEL_PATH}")


def load_saved_model(model):
  if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

  model.load_state_dict(torch.load(MODEL_PATH, weights_only = True))
  print(f"Loaded model from {MODEL_PATH}")


def test_model(model, testloader, classes, batch_size):
  dataiter = iter(testloader)
  images, labels = next(dataiter)

  imshow(torchvision.utils.make_grid(images))
  print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

  outputs = model(images)
  _ , predicted = torch.max(outputs, 1)
  print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      # calculate outputs by running images through the network
      outputs = model(images)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = model(images)
      _, predictions = torch.max(outputs, 1)
      # collect the correct predictions for each class
      for label, prediction in zip(labels, predictions):
        if label == prediction:
          correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1


  # print accuracy for each class
  for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

  del dataiter

if __name__ == "__main__":
  args = parse_args()

  print(f"PyTorch Version: {torch.__version__}")
  print(f"Numpy Version: {np.__version__}")

  batch_size = 4
  trainloader, testloader = build_dataloaders(batch_size)

  classes = ('airplane', 'automobile', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  model = CNN()

  if args.mode == "train":
    if args.restart:
      load_saved_model(model)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    train_model(model, trainloader)

    del dataiter
  else:
    load_saved_model(model)
    test_model(model, testloader, classes, batch_size)
