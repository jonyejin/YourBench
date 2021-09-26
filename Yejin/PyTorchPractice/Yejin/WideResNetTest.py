import torch

# load WRN-50-2:
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet28_10', pretrained=True)
# or WRN-101-2
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
model.eval()

##################
# https://pytorch.org/hub/pytorch_vision_resnet/
# resnet 18, 34, 50, 101, 152
#
###################

# https://pytorch.org/hub/pytorch_vision_wide_resnet/
# wide resnet
# images는 3*H*W, H,w >= 224
# normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]


