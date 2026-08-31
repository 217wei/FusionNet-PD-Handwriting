# utils.py
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torchvision import utils
import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torchvision.models as models
from image_iterator import ImageListIter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
    def __len__(self):
        return min(len(d) for d in self.datasets)

def test(net, validata, device, opt):
    net.eval()
    net.to(device)
    total_loss = 0
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    error_details = [] # 新增：紀錄錯誤詳情

    adversarial_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (imgs, label, patha) in enumerate(validata):
            label = label.type(torch.LongTensor).to(device)
            real_imgs = imgs.float().to(device)

            outputs = net(real_imgs)
            loss = adversarial_loss(outputs, label)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            
            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())
            
            # 紀錄錯誤路徑邏輯
            pred_list = predicted.cpu().numpy()
            label_list = label.cpu().numpy()
            for j in range(len(label_list)):
                total_samples += 1
                if pred_list[j] == label_list[j]:
                    correct += 1
                else:
                    # 格式：[圖片路徑, 真實標籤, 預測標籤]
                    error_details.append([patha[j], label_list[j], pred_list[j]])

    acc = 100 * (correct / total_samples) if total_samples > 0 else 0
    avg_loss = total_loss / len(validata)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds)
    specificity = 0.0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')

    net.train()
    return avg_loss, acc, recall, precision, f1, specificity, cm, error_details

def test2(net, validata, device, opt):
    net.eval()
    net.to(device)
    CE_loss_test = nn.CrossEntropyLoss().to(device)
    
    total_loss = 0
    correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    error_details = []

    with torch.no_grad():
        for i, ((imgs, label, patha), (imgs_fft, label_fft, pathb)) in enumerate(validata):
            label = label.type(torch.LongTensor).to(device)
            real_imgs = imgs.float().to(device)
            real_imgs_fft = imgs_fft.float().to(device)
            
            combine = net(real_imgs, real_imgs_fft)
            loss = CE_loss_test(combine, label)
            total_loss += loss.item()

            _, predicted = torch.max(combine, 1)
            pred_list = predicted.cpu().numpy()
            label_list = label.cpu().numpy()

            all_preds.append(predicted.cpu())
            all_labels.append(label.cpu())

            for j in range(len(label_list)):
                total_samples += 1
                if pred_list[j] == label_list[j]:
                    correct += 1
                else:
                    error_details.append([patha[j], label_list[j], pred_list[j]])

    acc = 100 * (correct / total_samples) if total_samples > 0 else 0
    avg_loss = total_loss / len(validata)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds)
    specificity = 0.0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')

    net.train()
    return avg_loss, acc, recall, precision, f1, specificity, cm, error_details