# -*- coding: utf-8 -*-
import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse

from Source.utils import stitch_images

parser = argparse.ArgumentParser(description='✨Welcome to YourBench-Adversarial Attack Robustness Benchmarking & Reporting tools.✨')
parser.add_argument('-a', '--attack_method', required=True, type=str, nargs='*', choices=['FGSM', 'CW', 'PGD', 'DeepFool'], dest='parsedAttackMethod', action='store')
parser.add_argument('-m', '--model', required=True, type=str, choices=['ResNet101_2', 'ResNet18', 'Custom'], dest='parsedModel')
parser.add_argument('-d', '--dataset', required=True, type=str, choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', 'Custom'], dest='parsedDataset')

args = parser.parse_args()
simple_data = False

print(args.parsedAttackMethod) # ['FGSM']
print(args.parsedModel) # WRN
print(args.parsedDataset)

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

# CUDA Settings
USE_CUDA = torch.cuda.is_available() 
device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print('학습을 진행하는 기기:',device)

# 1. Load Data
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
# class_idx = json.load(open("./data/imagenet_class_index.json"))
# idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])

# imagnet_data = image_folder_custom_label(root='./data/oneImage', transform=transform, idx2label=idx2label)
# data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=10, shuffle=False)

if args.parsedDataset == 'CIFAR-10':
  cifar10_data = torchvision.datasets.CIFAR10('Data/CIFAR10', download=True, transform=transform)
  data_loader = torch.utils.data.DataLoader(cifar10_data, batch_size=5)
  simple_data = True

elif args.parsedDataset == 'CIFAR-100':
  cifar100_data = torchvision.datasets.CIFAR100('Data/CIFAR100', download=True, transform=transform)
  data_loader = torch.utils.data.DataLoader(cifar100_data, batch_size=5)
  simple_data = True

elif args.parsedDataset == 'ImageNet':
  imagenet_data = torchvision.datasets.ImageNet('Data/ImageNet', download=True, transform=transform)
  data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=5)

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)

        mean.to(device)
        std.to(device)
        input.to(device)

        return (input - mean) / std

if args.parsedModel == 'ResNet101_2':
  norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  model = nn.Sequential(
      norm_layer,
      models.wide_resnet101_2(pretrained=True)
  ).to(device)
  model = model.eval()

elif args.parsedModel == 'ResNet18': 
  norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  model = nn.Sequential(
      norm_layer,
      models.resnet18(pretrained=True)
  ).to(device)

  model = model.eval()

elif args.parsedModel == 'Custom':
  pkg = __import__('custom_net')
  model_custom = pkg.my_model(pretrained = False)
  #state_dict의 경로를 넣기.
  model_custom.load_state_dict(torch.load('./my_model.pth'))
  norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  model = nn.Sequential(
      norm_layer,
      model_custom
  ).cuda()

  model = model.eval()

# 3. Attacks
from torchattacks import *
attackMethodDict = {'FGSM': FGSM(model, eps=8/255),
                    'CW' : CW(model, c=1, lr=0.01, steps=100, kappa=0),
                    'PGD' : PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
                    'DeepFool': DeepFool(model, steps=100)}
atks = [
    VANILA(model),
    #FGSM(model, eps=8/255),
    #CW(model, c=1, lr=0.01, steps=100, kappa=0),
    #PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
    #DeepFool(model, steps=100),
]

# args로 읽어온 정보만 attack한다. 
for atk in args.parsedAttackMethod:
    atks.append(attackMethodDict[atk])

print(atks)
print("Adversarial Image & Predicted Label")

# +
vanilla_output = []
untargeted_output= []
targeted_output= []

for atk in atks :
    print("-"*70)
    print(atk)
    
    correct = 0
    top5_correct = 0
    total = 0
    
    for images, labels in data_loader:
        # images : torch.Size([1,3,299,299])
        # labels: torch.Size([10]),[7, 5, 388, 1, ...] -> cock, electric_ray, giant_panda...
        atk.set_mode_default()
        #print(images.shape)
        start = time.time()
        adv_images = atk(images, labels)
        labels = labels.to(device)
        outputs = model(adv_images) # outputs: torch.Size([batch_size, 1000]), adversarial image를 모델에 넣은 결과, outputs.data:[batch_size, 1000]
        #print(outputs.shape)
        #print(outputs.data.shape)
        _, pre = torch.max(outputs.data, 1) # 1000 classes중 가장 큰 VALUE 1 남음, value, index 나온다. batch_size>1이면 batch_size크기의 array로 나온다.
        _, top_5 = torch.topk(outputs.data, 5)

        #print(top_5)
        #print(labels.shape)
        total += len(images)
        correct += (pre == labels).sum()
        break
    print('Total elapsed time (sec): %.2f' % (time.time() - start))
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
    
    if atk.attack == "VANILA":
        for i in range (len(atks) - 1):
            vanilla_output.append(100 * float(correct) / total)
    else:
        untargeted_output.append(100 * float(correct) / total)
    
    if atk.attack == ("FGSM" or "CW"):
        print("-"*70)
        print(atk)
        for images, labels in data_loader:
            atk.set_mode_targeted_least_likely()
            start = time.time()
            adv_images = atk(images, labels)
            labels = labels.to(device)
            outputs = model(adv_images) # outputs: torch.Size([batch_size, 1000]), adversarial image를 모델에 넣은 결과, outputs.data:[batch_size, 1000]
            _, pre = torch.max(outputs.data, 1) # 1000 classes중 가장 큰 VALUE 1 남음, value, index 나온다. batch_size>1이면 batch_size크기의 array로 나온다.
            _, top_5 = torch.topk(outputs.data, 5)
            total += len(images)
            correct += (pre == labels).sum()
            break
        print('Total elapsed time (sec): %.2f' % (time.time() - start))
        print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
        targeted_output.append(100 * float(correct) / total)
    elif atk.attack != "VANILA" and (atk.attack == "PGD" or "DeepFool"):
        targeted_output.append(-10)
    
# -

print("==================")
print(adv_images.shape)

# 기본 `log_dir` 은 "runs"이며, 여기서는 더 구체적으로 지정하였습니다
writer = SummaryWriter('Tutorials/runs/white_box_attack_image_net')
images = images.cuda()
writer.add_graph(model, images)
writer.close()

#make Data/Generated directory
os.makedirs("./Data/Generated", exist_ok=True)

# Save Image in Folder
for i in range(adv_images.shape[0]):
    torchvision.utils.save_image(images[i], fp=f"./Data/Generated/image_original_{i+1}.jpg", normalize=True)    
    torchvision.utils.save_image(adv_images[i], fp=f"./Data/Generated/image_adv_{i+1}.jpg", normalize=True)

# 4. Report Generating

# matplotlib로 그래프 그리기
x_val =[]
for atk in atks:
    if atk.attack == "VANILA":
        continue
    x_val.append(atk.attack)

plt.plot(x_val, vanilla_output, color='green', label = 'VANILLA')
plt.plot(x_val, untargeted_output, color='blue', label = 'DEFAULT')
plt.plot(x_val, targeted_output, color='red', label = 'TARGETED')
#plt.legend(loc=(0.73,0.775))
plt.legend(loc=(0.0,0.775))
plt.xlabel('Attack Method')
plt.ylabel('Accuracy (%)\nnegative value for unsupported attacks')
plt.savefig(f'./Data/Generated/graph.jpg', dip=300)

#점수 계산하기
total_result = 0
for atk in untargeted_output:
    total_result = total_result + atk
total_result = total_result / 0.001 if vanilla_output[0] == 0 else vanilla_output[0] / len(untargeted_output)

total_grade = ""
if simple_data == True:
    if total_result >= 45.0:
        total_grade = "A"
    elif total_result >= 35.0:
        total_grade = "B"
    elif total_result >= 25.0:
        total_grade = "C"
    elif total_result >= 15.0:
        total_grade = "D"
    else:
        total_grade = "F"

from fpdf import FPDF
from torchvision.transforms.functional import to_pil_image
from PIL.Image import Image
import PIL

class PDF(FPDF):
    def header(self):
        self.set_font("Times", "B", 20)
        # Moving cursor to the right:
        self.cell(80)
        self.cell(30, 10, "Benchmark Result", 0, 0, "C")
        self.cell(0, 10, ("Grade : " + total_grade if simple_data else "Score : " + str(total_result)),0, 0, "R")
        # Performing a line tbreak:
        self.ln(20)

    def footer(self):
        # Position cursor at 1.5 cm from bottom:
        self.set_y(-15)
        # Setting font: helvetica italic 8
        self.set_font("helvetica", "I", 8)
        # Printing page number:
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")


# Instantiation of inherited class
pdf = PDF()
pdf.set_display_mode(zoom='fullwidth',layout='two')
pdf.alias_nb_pages() # 페이지 수에 대한 alias ?
pdf.add_page() 
pdf.set_auto_page_break(True)


# Mapped Network
top_y = pdf.get_y()
#pdf.set_font("Times", "B", size=12)
#pdf.cell(0, 10, f"Mapped Network", 0, 1)
#pdf.set_font("Helvetica", "I", 12)
#pdf.cell(0, 10, f"<This function is still working in process.>", 0, 1)

# 1. 성공한 adversarial example들
pdf.set_font("Times", "B", size=12)
pdf.cell(0, 10, f"Succeeded Adversarial examples", 0, 1)

# Effective page width, or just epw
epw = pdf.w - 2*pdf.l_margin
img_size = epw/2 - 20

np_images = (images[0:5].cpu().numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
np_adv_images = (adv_images[0:5].cpu().numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
original_labels = [str(l) for l in labels.cpu().numpy()[0:5]]
predicted_labels = [str(l) for l in pre.cpu().numpy()[0:5]]
outputImage = stitch_images(np_images, np_adv_images, original_labels, predicted_labels)

import cv2
cv2.imwrite("./Data/Generated/stitchedImage.jpg", outputImage)
#torchvision.utils.save_image(outputImage, fp=f"./Data/Generated/stitchedImage.jpg", normalize=True)    

pdf.image(f'./Data/Generated/stitchedImage.jpg', w=img_size)

# for i in range(max(5, adv_images.shape[0])):
#     pdf.image(f'./Data/Generated/image_original_{i+1}.jpg', w=img_size, h=img_size)
#     pdf.set_xy(pdf.get_x() + img_size + 10, pdf.get_y() - img_size)
#     pdf.image(f'./Data/Generated/image_adv_{i+1}.jpg', w=img_size, h=img_size)
    # pdf.ln(2)


# second column
## 2. table 추가
pdf.set_xy(epw /2 +pdf.l_margin, top_y)
pdf.set_font("Times", "B", size=12)
pdf.cell(epw / 2 + 10, 10, txt=f"Top-5 Accuracy against attacks", border=0, ln=1) # ln: 커서 포지션을 다음줄로 바꾼다. 
#pdf.set_xy(epw /2 +pdf.l_margin, pdf.get_y())


# Set column width to 1/4 of effective page width to distribute content 
# evenly across table and page
col_width = epw/10

# Since we do not need to draw lines anymore, there is no need to separate
# headers from data matrix.

data = [['Vanilla']+vanilla_output,
['attacks'] + x_val,
['default'] + untargeted_output,
['targeted'] + targeted_output,]

pdf.set_font('Times','',10.0) 
pdf.ln(0.5)

# Text height is the same as current font size
th = pdf.font_size

# Here we add more padding by passing 2*th as height
#pdf.set_xy(epw /2 +pdf.l_margin, top_y)
pdf.set_xy(epw /2 +pdf.l_margin, pdf.get_y())
for row in data:
    for datum in row:
        # Enter data in colums
        pdf.cell(col_width, 2*th, str(datum), border=1)
    pdf.ln(2*th)
    pdf.set_xy(epw /2 +pdf.l_margin, pdf.get_y())



#####################
# 3. attack result graph
pdf.set_xy(epw /2 +pdf.l_margin, pdf.get_y())
#pdf.set_xy(epw /2 +pdf.l_margin, top_y)
pdf.set_font("Times", "B", size=12)
pdf.cell(epw / 2 + 10, 10, f"Attack Results with graph", 0, 1)
pdf.set_xy(epw /2 +pdf.l_margin, pdf.get_y())
pdf.image(f'./Data/Generated/graph.jpg', w=epw /2)

# 4. Advise
pdf.set_xy(epw /2 +pdf.l_margin, pdf.get_y())
pdf.set_font("Times", "B", size=12)
pdf.cell(0, 10, f"Advise for your model robustness", 0, 1)
pdf.set_font("Helvetica", "I", 12)
#pdf.cell(w=0, h=0, txt=f"Your model is significantly weak against CW L2 Attack.Your model is significantly weak against CW L2 Attack. Your model is significantly weak against CW L2 Attack.Your model is significantly weak against CW L2 Attack.,Your model is significantly weak against CW L2 Attack", border=0, 1)

# pdf.write(h=5, txt=f"Your model is significantly weak against CW L2 Attack.Your model is significantly weak against CW L2 Attack. Your model is significantly weak against CW L2 Attack.Your model is significantly weak against CW L2 Attack.,Your model is significantly weak against CW L2 Attack")

pdf.set_xy(epw /2 +pdf.l_margin, pdf.get_y())
advice_data={'0to10 accuracy attacks' : 'None1', '10to100 accuracy attacks' : ''}

advice = ['robustness about your model can vary considering your data sets complexity. ']
advice.append('Your Model cannot defend against' + advice_data['0to10 accuracy attacks'])
if advice_data['10to100 accuracy attacks'] == '':
    advice.append(' Your model is hardly robust to given attacks. Is this properly trained? ')
else:
    advice.append(' But relatively robust against' + advice_data['10to100 accuracy attacks'])
advice.append('\nThis weakness can be caused from setting hyper parameters, matbe input bias, or input capacity and so many more.')
advice.append('If you think none of this are your issues we recommend adversarial training with our adverarial examples provided.')
advice.append('\nTry again with adversarilly trained model and check out the result. ')
advice.append('See more info in the attached papers and linked tensorboard.')

advice_sentence = ''
for i in advice:
    advice_sentence = advice_sentence + i

pdf.multi_cell(w= epw / 2, h=5, txt=advice_sentence)
"""
"robustness about your model can vary considering your data sets complexity."

Your Model cannot defend against [0~10% accuracy attacks].

if[10~100% exits]
But relatively robust against [10~100% accruacy attacks] .
else
Your model is hardly robust to given attacks. Is this properly trained?

This weakness can be caused from setting hyper parameters, matbe input bias, or input capacity and so many more.

If you think none of this are your issues we recommend adversarial training with our adverarial examples provided.

Try again with adversarilly trained model and check out the result.

See more info in the attached papers and linked tensorboard.
"""

pdf.output("Benchmark Result.pdf")
