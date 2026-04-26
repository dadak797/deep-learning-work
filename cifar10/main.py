import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from network import SimpleCNN
from network import SmallVGG9
from network import MediumVGG8
from network import VGG16

import torch.nn as nn
import torch.optim as optim

MODEL_PATH = './saved_models/cifar_net.pth'
LOG_INTERVAL_SAMPLES = 8000


def imshow(img):
  img = img.cpu()
  img = img / 2 + 0.5  # [-1, 1] -> [0, 1]
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def parse_args():
  parser = argparse.ArgumentParser(description = "CIFAR-10 train/test runner")
  parser.add_argument(
    "--mode",
    choices = ["train", "test", "train_and_test"],
    required = True,
    help = "Choose whether to train the model or test a saved model.",
  )
  parser.add_argument(
    "--restart",
    action = "store_true",
    help = "Resume training from the saved model when used with --mode train.",
  )
  parser.add_argument(
    "--epochs",
    type = int,
    default = 2,
    help = "Number of epochs to train the model.",
  )
  parser.add_argument(
    "--device",
    choices = ["cpu", "cuda", "mps"],
    default = "mps",
    help = "Choose whether to run on CPU, CUDA, or MPS.",
  )
  parser.add_argument(
    "--batch_size",
    type = int,
    default = 4,
    help = "Batch size for training and testing.",
  )
  parser.add_argument(
    "--data_aug",
    action = "store_true",
    help = "Apply data augmentation to the training dataset.",
  )

  return parser.parse_args()

def build_dataloaders(batch_size, use_data_aug = False):
  # Define test transformations
  # ToTensor(): [0, 255] -> [0, 1]
  # Normalize(): (x - mean) / std -> [-1, 1]
  test_transform = transforms.Compose(
    # [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
  )

  # Define training transformations with optional data augmentation
  if use_data_aug:
    train_transform = transforms.Compose(
      [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ]
    )
  else:
    train_transform = transforms.Compose(
      [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ]
    )

  # Load the CIFAR-10 dataset (Train: 50000, Test: 10000)
  trainset = torchvision.datasets.CIFAR10(
    root = './data', train = True, download = True, transform = train_transform
  )
  trainloader = torch.utils.data.DataLoader(
    trainset, batch_size = batch_size, shuffle = True, num_workers = 2
  )

  testset = torchvision.datasets.CIFAR10(
    root = './data', train = False, download = True, transform = test_transform
  )
  testloader = torch.utils.data.DataLoader(
    testset, batch_size = batch_size, shuffle = False, num_workers = 2
  )

  return trainloader, testloader


def get_device(device_name):
  if device_name == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but no CUDA device is available.")

  if device_name == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS was requested, but no MPS device is available.")

  return torch.device(device_name)


def train_model(model, trainloader, epochs, device):
  print("Model Architecture:")
  print(model)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

  model.train()
  for epoch in range(epochs):
    running_loss = 0.0
    batches_since_log = 0
    samples_since_log = 0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      loss.backward()  # autograd가 graient 계산
      optimizer.step() # 계산된 gradient로 모델의 가중치를 업데이트

      running_loss += loss.item()
      batches_since_log += 1
      samples_since_log += inputs.size(0)
      if samples_since_log >= LOG_INTERVAL_SAMPLES or (i + 1) == len(trainloader):
        avg_loss = running_loss / batches_since_log
        print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}")
        running_loss = 0.0
        batches_since_log = 0
        samples_since_log = 0

  print('Finished Training')

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok = True)
  torch.save(model.state_dict(), MODEL_PATH)
  print(f"Saved model to {MODEL_PATH}")


def load_saved_model(model, device):
  if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

  model.load_state_dict(torch.load(MODEL_PATH, map_location = device))
  print(f"Loaded model from {MODEL_PATH}")


def test_model(model, testloader, classes, batch_size, device, epoch_index = None):
  correct = 0
  total = 0
  model.eval()
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      # calculate outputs by running images through the network
      outputs = model(images)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total

  if epoch_index is not None:
    print(f'Accuracy of the network on the {total} test images after epoch {epoch_index + 1}: {accuracy:.2f} %')
  else:
    print(f'Accuracy of the network on the {total} test images: {accuracy:.2f} %')

  return accuracy


def test_model_per_class(model, testloader, classes, device):
  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  model.eval()
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predictions = torch.max(outputs, 1)
      # collect the correct predictions for each class
      for label, prediction in zip(labels, predictions):
        label_index = label.item()
        prediction_index = prediction.item()
        if label_index == prediction_index:
          correct_pred[classes[label_index]] += 1
        total_pred[classes[label_index]] += 1

  # print accuracy for each class
  for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def train_model_and_test_model(model, trainloader, testloader, classes, batch_size, epochs, device):
  print("Model Architecture:")
  print(model)

  loss_history = []
  accuracy_history = []

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
  min_lr = 1e-5
  
  global_step = 0

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    batches_since_log = 0
    samples_since_log = 0
    for i, data in enumerate(trainloader, 0):
      global_step += 1
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      batches_since_log += 1
      samples_since_log += inputs.size(0)
      if samples_since_log >= LOG_INTERVAL_SAMPLES or (i + 1) == len(trainloader):
        avg_loss = running_loss / batches_since_log
        print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}")
        loss_history.append((global_step, avg_loss))
        running_loss = 0.0
        batches_since_log = 0
        samples_since_log = 0
    # Test the model after each epoch
    accuracy = test_model(model, testloader, classes, batch_size, device, epoch_index = epoch)
    accuracy_history.append((epoch + 1, accuracy))
    
    scheduler.step()
    for param_group in optimizer.param_groups:
      if param_group['lr'] < min_lr:
        param_group['lr'] = min_lr

  print('Finished Training')

  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok = True)
  torch.save(model.state_dict(), MODEL_PATH)
  print(f"Saved model to {MODEL_PATH}")

  os.makedirs('./logs', exist_ok=True)
  with open('./logs/loss_history.txt', 'w') as f:
    f.write('\n'.join(f'{step}, {loss}' for step, loss in loss_history))
  with open('./logs/accuracy_history.txt', 'w') as f:
    f.write('\n'.join(f'{epoch}, {acc}' for epoch, acc in accuracy_history))

  plot_history(loss_history, accuracy_history)


def plot_history(loss_history, accuracy_history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

  steps, losses = zip(*loss_history)
  ax1.plot(steps, losses)
  ax1.set_title('Training Loss')
  ax1.set_xlabel('Step')
  ax1.set_ylabel('Loss')

  epochs, accuracies = zip(*accuracy_history)
  ax2.plot(epochs, accuracies, marker='o')
  ax2.set_title('Test Accuracy per Epoch')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy (%)')

  plt.tight_layout()
  os.makedirs('./logs', exist_ok=True)
  plt.savefig('./logs/history.png')
  plt.show()


def show_sample_images(trainloader, classes, batch_size):
  dataiter = iter(trainloader)
  images, labels = next(dataiter)

  imshow(torchvision.utils.make_grid(images))
  print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


def show_sample_predictions(model, testloader, classes, batch_size):
  dataiter = iter(testloader)
  images, labels = next(dataiter)

  imshow(torchvision.utils.make_grid(images))
  print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

  device = next(model.parameters()).device
  model.eval()
  with torch.no_grad():
    outputs = model(images.to(device))
    _ , predicted = torch.max(outputs, 1)
  predicted = predicted.cpu()
  print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))


if __name__ == "__main__":
  args = parse_args()
  device = get_device(args.device)

  print(f"PyTorch Version: {torch.__version__}")
  print(f"Numpy Version: {np.__version__}")
  print(f"Using Device: {device}")

  batch_size = args.batch_size
  trainloader, testloader = build_dataloaders(batch_size, use_data_aug = args.data_aug)

  classes = ('airplane', 'automobile', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # model = SimpleCNN().to(device)
  # model = SmallVGG9().to(device)
  # model = MediumVGG8().to(device)
  model = VGG16().to(device)

  if args.mode == "train":
    if args.restart:
      load_saved_model(model, device)
    # show_sample_images(trainloader, classes, batch_size)
    train_model(model, trainloader, args.epochs, device)
  elif args.mode == "train_and_test":
    if args.restart:
      load_saved_model(model, device)
    train_model_and_test_model(model, trainloader, testloader, classes, batch_size, args.epochs, device)
  else:
    load_saved_model(model, device)
    # show_sample_predictions(model, testloader, classes, batch_size)
    test_model(model, testloader, classes, batch_size, device)
    # test_model_per_class(model, testloader, classes, device)
