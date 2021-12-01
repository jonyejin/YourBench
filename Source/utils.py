import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dsets

import cv2

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    
def image_folder_custom_label(root, transform, idx2label) :
    
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2

def stitch_images(original_images, adv_images, original_labels, adv_labels, target_size=(224, 224)):
    """Stitch result images and put texts

    Args:
        original_images (list): list of images(np.ndarray)
        adv_images (list): list of adversarial images
        original_labels (list): original image's label names(str)
        adv_labels (list): (probably wrongly) predicted label names(str)
        target_size (tuple, optional): image size of single image. Defaults to (224, 224).

    Returns:
        [type]: [description]
    """    
    assert len(original_images) == len(adv_images), "Input images should be paired"
    assert len(original_labels) == len(original_images), "Each label should be provided"
    assert original_images[0].shape[0] == adv_images[0].shape[0], "All image should be the same size"

    num_images = len(original_images)
    result_imgs = []
    for original_image, adv_image, olabel, alabel in \
            zip(original_images, adv_images, original_labels, adv_labels):
        original_image = cv2.resize(original_image, dsize=target_size)
        adv_image = cv2.resize(adv_image, dsize=target_size)
        h, w, c = original_image.shape
        margin = np.zeros((h, 20, 3)) # space between two images
        margin.fill(255)
        btm_margin = np.zeros((40, 2*w + 20, 3)) # space between rows
        btm_margin.fill(255)
        btm_margin = cv2.putText(btm_margin, olabel, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, #(x, y)
                                color=0, thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        btm_margin = cv2.putText(btm_margin, alabel, (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                color=0, thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        _imgs = np.concatenate([original_image, margin, adv_image], axis=1)
        final_image = np.concatenate([_imgs, btm_margin], axis=0)
        result_imgs.append(final_image)
    output_img = np.concatenate(result_imgs, axis=0)
    return output_img