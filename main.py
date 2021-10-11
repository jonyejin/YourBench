from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable

parser = argparse.ArgumentParser(description='✨Welcome to YourBench-Adversarial Attack Robustness Benchmarking & Reporting tools.✨')
parser.add_argument('--attack_method', type=str, nargs='*', choices=['FGSM', 'CW'], help="--attack_method FGSM CW")
parser.add_argument('--model', type=str, choices=['WRN', 'ResNet18'], help="--model WRN")
parser.add_argument('--dataset', type=str, choices=['CIFAR-10', 'CIFAR-100', 'ImageNet'], help="--dataset CIFAR10")

# parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
# parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
# parser.add_argument('--depth', default=28, type=int, help='depth of model')
# parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
# parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
# parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
