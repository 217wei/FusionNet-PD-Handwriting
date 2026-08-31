# train1.py
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
import torch.utils.data as data
import random
import time
import csv
import logging
from combine_model import make_model_single
from utils import test , reset_weights, ConcatDataset
from tqdm import tqdm 
import gc



def train_model(train_loader, valid_loader, opt, device, cuda=True):
    """
    使用 FFT 數據訓練模型（第一個模型），返回該折中最佳模型的 state_dict 與最佳 acc。
    """
    
    best_acc = 0
    best_f1 = 0 
    best_precision = 0
    best_recall = 0
    best_specificity = 0
    best_cm = None
    best_error_list = []

    adversarial_loss = nn.CrossEntropyLoss()

    since = time.time()
    folder_path = os.path.join(opt.csv, "fold_" + str(opt.fold))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 最終 CSV 檔案的完整路徑，例如以 opt.NewHandPD_task_name 命名
    csv_file = os.path.join(folder_path, f"{opt.NewHandPD_task_name}_fold_results.csv")

    folder_path = os.path.join(opt.csv, "fold_" + str(opt.fold))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 最終 CSV 檔案的完整路徑，例如以 opt.NewHandPD_task_name 命名
    csv_file = os.path.join(folder_path, f"{opt.NewHandPD_task_name}_fold_results.csv")

    # 若 CSV 檔案不存在，先建立檔案並寫入表頭
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "model_count", "test_loss", "test_acc", "recall", "precision", "f1", "specificity"])

    model = make_model_single()
    model.apply(reset_weights)
    if cuda:
        model.to(device)
        adversarial_loss.to(device)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0.00001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-2)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=opt.lr * 0.1) # 0408 add
    
    best_model_state = None
    for epoch in range(opt.n_epochs):
        model.train()
        epoch_loss = 0.0
        for i, (imgs, label, patha) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.n_epochs}", leave=False)):
            # 轉型與移至 GPU
            label = label.type(torch.LongTensor)
            # imgs = imgs.type(torch.LongTensor)
            valid_label = label.to(device)
            # real_imgs = imgs.float().to(device)
            # 🌟 直接轉 float 即可，不要先轉 LongTensor
            valid_label = label.type(torch.LongTensor).to(device) # 0408 add
            real_imgs = imgs.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(real_imgs)
            loss = adversarial_loss(outputs, valid_label)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 0408 add
            optimizer.step()
        
        epoch_loss = epoch_loss / len(train_loader)
        lr_current = optimizer.state_dict()['param_groups'][0]['lr']
        logging.info("[Epoch {}/{}] [loss: {:.6f}] [lr: {:.6f}]".format(epoch, opt.n_epochs, epoch_loss, lr_current))
        lr_scheduler.step()
        
        if epoch%5==0:
            # 儲存本 epoch 的 checkpoint
            # save_path = opt.out_dir + '/fold_' + (str)(opt.fold) + '_model_'+ (str)(opt.model_count) + '_epoch_'+ (str)(epoch)+'.pth'
            save_path = os.path.join(opt.out_dir, f"fold_{opt.fold}_model_{opt.model_count}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
        
        
        

        # 用驗證集測試以決定最佳模型
        loss_test, acc_test, recall_test, precision_test, f1_test, specificity_test, cm_test, err_test = test(model, valid_loader,device, opt)
        logging.info("[Epoch {}] [test_loss: {:.6f}] [test_acc: {:.6f}] [specificity: {:.6f}]".format(epoch, loss_test, acc_test,specificity_test))
        logging.info('-------------------------------------------')

        # 將結果寫入 CSV，除了 epoch、test_loss 與 test_acc，也寫入 model_count
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, opt.model_count, f"{loss_test:.6f}", f"{acc_test:.6f}", f"{recall_test:.6f}", f"{precision_test:.6f}", f"{f1_test:.6f}", f"{specificity_test:.6f}"])
            
        if acc_test > best_acc:
            # best_path = opt.out_dir + '/fold_' + (str)(opt.fold) + '_model_'+ (str)(opt.model_count) + '_epoch_best.pth'
            best_path = os.path.join(opt.out_dir, f"fold_{opt.fold}_model_{opt.model_count}_epoch_best.pth")
            torch.save(model.state_dict(), best_path)
            best_acc = acc_test
            best_model_state = model.state_dict()
            best_cm = cm_test
            best_error_list = err_test
            best_precision = precision_test
            best_recall = recall_test
            best_specificity = specificity_test 
            best_f1 = f1_test

    
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return best_model_state, best_acc, best_f1, best_precision, best_recall, best_specificity, best_cm, best_error_list

# # 若直接執行此檔案，可加入測試代碼（例如單次訓練流程）
# if __name__ == '__main__':
#     # 構造你的 opt 物件、數據集與 dataloader（注意資料切分）
#     # 例如：
#     # opt = YourOpt()
#     # train_set_fft = ImageListIter(transform=..., pre=opt.datapath_fft, csv_list=opt.train_csvpath)
#     # test_set_fft = ImageListIter(transform=..., pre=opt.datapath_fft, csv_list=opt.train_csvpath)
#     # train_loader = torch.utils.data.DataLoader(train_set_fft, batch_size=opt.batch_size, shuffle=True)
#     # valid_loader = torch.utils.data.DataLoader(test_set_fft, batch_size=opt.batch_size, shuffle=False)
#     # train_fft_model(train_loader, valid_loader, opt)
#     pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler  # 導入 AMP 加速工具
import os
import time
import logging
import csv
from tqdm import tqdm
import gc

# def train_model(train_loader, valid_loader, opt, device, cuda=True):
#     """
#     針對 RTX 4090 優化的訓練函式：加入 AMP 混合精度與顯存管理。
#     """
#     best_acc = 0
#     best_f1 = 0 
#     best_precision = 0
#     best_recall = 0
#     best_specificity = 0

#     adversarial_loss = nn.CrossEntropyLoss()
#     since = time.time()
    
#     # 建立輸出目錄與 CSV (維持原邏輯)
#     folder_path = os.path.join(opt.csv, f"fold_{opt.fold}")
#     os.makedirs(folder_path, exist_ok=True)
#     csv_file = os.path.join(folder_path, f"{opt.NewHandPD_task_name}_fold_results.csv")

#     if not os.path.exists(csv_file):
#         with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(["epoch", "model_count", "test_loss", "test_acc", "recall", "precision", "f1", "specificity"])

#     # 模型初始化
#     model = make_model_single()
#     model.apply(reset_weights)
    
#     if cuda:
#         model.to(device)
#         adversarial_loss.to(device)
    
#     # 這裡可以考慮加上 torch.compile(model) 針對 4090 進一步加速 (PyTorch 2.0+)
#     # model = torch.compile(model) 

#     optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
#     lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0.00001)
    
#     # 【核心加速 1】初始化 AMP 縮放器
#     scaler = GradScaler()

#     best_model_state = None
#     for epoch in range(opt.n_epochs):
#         model.train()
#         epoch_loss = 0.0
        
#         # 訓練迴圈
#         pbar = tqdm(train_loader, desc=f"Fold {opt.fold} Epoch {epoch+1}/{opt.n_epochs}", leave=False)
#         for i, (imgs, label, patha) in enumerate(pbar):
#             # 數據搬移與轉型
#             valid_label = label.long().to(device)
#             real_imgs = imgs.float().to(device)
            
#             optimizer.zero_grad()
            
#             # 【核心加速 2】開啟自動混合精度
#             with autocast():
#                 outputs = model(real_imgs)
#                 loss = adversarial_loss(outputs, valid_label)
            
#             # 【核心加速 3】使用 Scaler 更新梯度
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
            
#             epoch_loss += loss.item()
#             pbar.set_postfix(loss=loss.item())
        
#         epoch_loss /= len(train_loader)
#         lr_current = optimizer.param_groups[0]['lr']
#         logging.info(f"[Epoch {epoch}/{opt.n_epochs}] [loss: {epoch_loss:.6f}] [lr: {lr_current:.6f}]")
#         lr_scheduler.step()
        
#         # 每 5 epoch 強制存一次檔
#         if epoch % 5 == 0:
#             save_path = os.path.join(opt.out_dir, f"fold_{opt.fold}_model_{opt.model_count}_epoch_{epoch}.pth")
#             torch.save(model.state_dict(), save_path)

#         # 驗證集測試 (同樣使用 autocast 加速)
#         model.eval()
#         with torch.no_grad():
#             with autocast():
#                 loss_test, acc_test, recall_test, precision_test, f1_test, specificity_test, cm_test = test(model, valid_loader, device, opt)
        
#         logging.info(f"[Epoch {epoch}] [test_loss: {loss_test:.6f}] [test_acc: {acc_test:.6f}] [specificity: {specificity_test:.6f}]")
#         logging.info('-------------------------------------------')

#         # 寫入 CSV
#         with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, opt.model_count, f"{loss_test:.6f}", f"{acc_test:.6f}", f"{recall_test:.6f}", f"{precision_test:.6f}", f"{f1_test:.6f}", f"{specificity_test:.6f}"])
            
#         # 儲存最佳模型 (Acc)
#         if acc_test > best_acc:
#             best_path = os.path.join(opt.out_dir, f"fold_{opt.fold}_model_{opt.model_count}_epoch_best.pth")
#             torch.save(model.state_dict(), best_path)
#             best_acc = acc_test
#             best_model_state = model.state_dict()
        
#         # 紀錄其餘指標
#         if f1_test > best_f1:
#             best_f1 = f1_test
#         if precision_test > best_precision:
#             best_precision = precision_test
#         if recall_test > best_recall:
#             best_recall = recall_test
#         if specificity_test > best_specificity: 
#             best_specificity = specificity_test 
    
#     time_elapsed = time.time() - since
#     logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
#     # 徹底清理顯存
#     del model
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return best_model_state, best_acc, best_f1, best_precision, best_recall, best_specificity
