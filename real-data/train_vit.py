import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

from tools.vit_model import ViT
from tools.optimizer import SMD

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--optim', type=str, help='optimizer')
parser.add_argument('--filename', type=str, help='outfile destination')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

# Dataset and dataloaders
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(48),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize(48),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
print("Got dataloader")

# Model
model = ViT(
    image_size = 48,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)
print("Got model")

# Optimizer
optimizer = SMD(
    [{"params": list(model.parameters()), "lr": args.lr}], p=1.1
) if args.optim == "SMD" else optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss().to(device)
print("Got optimizer and loss fn")

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for idx in range(1, args.epochs + 1):
    print()
    print(f"Epoch {idx}")
    # Train
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in tqdm(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / total
    avg_train_acc = correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)
    print(f"Training Loss = {avg_train_loss}, Training Accuracy = {avg_train_acc}")

    # Test
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        avg_test_loss = test_loss / total
        avg_test_acc = correct / total
    test_losses.append(avg_test_loss)
    test_accuracies.append(avg_test_acc)
    print(f"Testing Loss = {avg_test_loss}, Testing Accuracy = {avg_test_acc}")

# Save
torch.save(
    {
        "model": model,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
    },
    args.filename,
)
