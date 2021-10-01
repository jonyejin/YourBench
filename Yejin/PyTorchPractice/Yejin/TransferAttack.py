import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import PGD

from models import Holdout, Target
from utils import imshow

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

# 1. load data
batch_size = 24

cifar10_train = dsets.CIFAR10(root='./data', train=True,
                              download=True, transform=transforms.ToTensor())
cifar10_test  = dsets.CIFAR10(root='./data', train=False,
                              download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(cifar10_train,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_test,
                                          batch_size=batch_size,
                                          shuffle=False)

images, labels = iter(train_loader).next()
imshow(torchvision.utils.make_grid(images, normalize=True), "Train Image")

# 2. Load Holdout Model & Save Adversarial Images
model = Holdout()
model.load_state_dict(torch.load("./checkpoint/holdout.pth", map_location=torch.device("cpu")))
model = model.eval() # gpu이면 .cuda()

atk = PGD(model, eps=8/255, alpha=2/255, steps=7)
atk.set_return_type('int') # Save as integer.
atk.save(data_loader=test_loader, save_path="./data/cifar10_pgd.pt", verbose=True)

print("IM HERE!")
# 3. Load Adversarial Images
adv_images, adv_labels = torch.load("./data/cifar10_pgd.pt")
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

# 4.  Attack Target Model¶
model = Target() # gpu이면 .cuda()
model.load_state_dict(torch.load("./checkpoint/target.pth", map_location=torch.device("cpu")))

# 4.1 Clean Accuracy
model.eval()

correct = 0
total = 0

for images, labels in test_loader:
    
    images = images # gpu -> .cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels).sum() # labels.cuda()
    
print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

# 4.2 Robust Accuracy
model.eval()

correct = 0
total = 0

for images, labels in adv_loader:
    
    images = images # .cuda()
    outputs = model(images)
    
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels).sum() # .cuda()
    
print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))