# import numpy as np
# import pandas as pd
# import os
# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import DataLoader
# import time
# import csv
# import logging
# from utils import test2 , reset_weights, ConcatDataset
# from tqdm import tqdm 
# from combine_model import make_model
# import gc

# def train_combined_model(train_loader, valid_loader, opt, device, cuda=True):
#     best_acc = 0
#     best_f1, best_precision, best_recall, best_specificity = 0, 0, 0, 0
#     best_cm = None
#     best_error_list = []

#     CE_loss = nn.CrossEntropyLoss()

#     # 修正存檔路徑：存入 fold 資料夾內
#     folder_path = os.path.join(opt.csv, f"fold_{opt.fold}")
#     os.makedirs(folder_path, exist_ok=True)
#     csv_file = os.path.join(folder_path, "model_3_training_log.csv")

#     if not os.path.exists(csv_file):
#         with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(["epoch", "model_count", "test_loss", "test_acc", "recall", "precision", "f1", "specificity"])

#     # 1. 建立模型
#     model = make_model()

#     # 2. 🌟 關鍵修正：正確載入權重 (對應你的 CombinedTransformer 邏輯)
#     # modelA 負責 x1 (Normal), modelB 負責 x2 (FFT)
#     if hasattr(opt, 'nofft_weight') and os.path.exists(opt.nofft_weight):
#         logging.info(f"💾 載入 Normal 分支權重 (modelA): {opt.nofft_weight}")
#         model.modelA.load_state_dict(torch.load(opt.nofft_weight)) 
    
#     if hasattr(opt, 'fft_weight') and os.path.exists(opt.fft_weight):
#         logging.info(f"💾 載入 FFT 分支權重 (modelB): {opt.fft_weight}")
#         model.modelB.load_state_dict(torch.load(opt.fft_weight))

#     if cuda:
#         model.to(device)
#         CE_loss.to(device)

#     # 3. 🌟 關鍵修正：凍結分支 (鎖定已練好的 88% 與 75%)
#     # 只訓練 Cross-Attention 和最後的 Fusion Block，防止特徵崩潰
#     for param in model.modelA.parameters():
#         param.requires_grad = False
#     for param in model.modelB.parameters():
#         param.requires_grad = False
    
#     logging.info("❄️ 已鎖定 modelA 與 modelB 參數，專注訓練 Cross-Attention 層")

#     # 4. 優化器僅優化「未凍結」的參數，並降低學習率
#     # 這裡只抓取 requires_grad=True 的參數 (即 Cross-Attention 和 FC)
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr * 0.1)
#     lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-6)

#     # 5. 💡 預先測試：看看載入權重後，還沒練之前的分數
#     model.eval()
#     _, init_acc, _, _, _, _, _, _ = test2(model, valid_loader, device, opt)
#     logging.info(f"📊 載入後初始準確率 (Epoch 0 前): {init_acc:.2f}%")

#     best_model_state = None
#     for epoch in range(opt.n_epochs):
#         since = time.time()
#         model.train()
#         epoch_loss = 0.0
        
#         for i, ((imgs, label, _), (imgs_fft, _, _)) in enumerate(
#             tqdm(train_loader, desc=f"Fold {opt.fold} M3 Epoch {epoch+1}", leave=False)
#         ):
#             # 簡化資料轉換動作，加快速度
#             valid_label = label.to(device).long()
#             real_img = imgs.to(device).float()
#             real_img_fft = imgs_fft.to(device).float()
            
#             optimizer.zero_grad()
#             combined = model(real_img, real_img_fft)
#             loss = CE_loss(combined, valid_label)
            
#             epoch_loss += loss.item()
#             loss.backward()
#             optimizer.step()
        
#         epoch_loss = epoch_loss / len(train_loader)
#         lr_current = optimizer.param_groups[0]['lr']
        
#         # 測試本輪表現
#         loss_t, acc_t, rec_t, pre_t, f1_t, spec_t, cm_t, err_t = test2(model, valid_loader, device, opt)
#         logging.info(f"[Epoch {epoch}] Loss: {epoch_loss:.4f} | Test Acc: {acc_t:.2f}% | Spec: {spec_t:.4f}")
        
#         lr_scheduler.step()
        
#         # 紀錄與更新最佳模型 (只要 Acc 創新高就打包)
#         if acc_t > best_acc:
#             best_acc = acc_t
#             best_f1, best_precision, best_recall, best_specificity = f1_t, pre_t, rec_t, spec_t
#             best_cm, best_error_list = cm_t, err_t
#             best_model_state = model.state_dict()
            
#             best_path = os.path.join(opt.out_dir, f"fold_{opt.fold}_model_3_best.pth")
#             torch.save(best_model_state, best_path)

#         with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, opt.model_count, f"{loss_t:.6f}", f"{acc_t:.6f}", f"{rec_t:.6f}", f"{pre_t:.6f}", f"{f1_t:.6f}", f"{spec_t:.6f}"])
            
#         time_elapsed = time.time() - since
#         logging.info('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#     del model
#     torch.cuda.empty_cache()
#     gc.collect()
#     return best_model_state, best_acc, best_f1, best_precision, best_recall, best_specificity, best_cm, best_error_list
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import time
import csv
import logging
from utils import test2 , reset_weights, ConcatDataset
from tqdm import tqdm 
from combine_model import make_model
import gc

def train_combined_model(train_loader, valid_loader, opt, device, cuda=True):
    best_acc = 0
    best_f1, best_precision, best_recall, best_specificity = 0, 0, 0, 0
    best_cm = None
    best_error_list = []
    
    weights = torch.tensor([1.5, 1.0]).to(device)
    CE_loss = nn.CrossEntropyLoss(weight=weights)

    # 修正存檔路徑：存入 fold 資料夾內
    folder_path = os.path.join(opt.csv, f"fold_{opt.fold}")
    os.makedirs(folder_path, exist_ok=True)
    csv_file = os.path.join(folder_path, "model_3_training_log.csv")

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "model_count", "test_loss", "test_acc", "recall", "precision", "f1", "specificity"])

    # 1. 建立模型
    model = make_model()

    # 2. 關鍵修正：正確載入權重 (對應你的 CombinedTransformer 邏輯)
    # modelA 負責 x1 (Normal), modelB 負責 x2 (FFT)
    if hasattr(opt, 'nofft_weight') and os.path.exists(opt.nofft_weight):
        logging.info(f"載入 Normal 分支權重 (modelA): {opt.nofft_weight}")
        model.modelA.load_state_dict(torch.load(opt.nofft_weight)) 
    
    if hasattr(opt, 'fft_weight') and os.path.exists(opt.fft_weight):
        logging.info(f"載入 FFT 分支權重 (modelB): {opt.fft_weight}")
        model.modelB.load_state_dict(torch.load(opt.fft_weight))

    if cuda:
        model.to(device)
        CE_loss.to(device)

    # 3. 關鍵修正：凍結分支 (鎖定已練好的 88% 與 75%)
    # 只訓練 Cross-Attention 和最後的 Fusion Block，防止特徵崩潰
    for param in model.modelA.parameters():
        param.requires_grad = False
    for param in model.modelB.parameters():
        param.requires_grad = False
    
    logging.info("已鎖定 modelA 與 modelB 參數，專注訓練 Cross-Attention 層")

    # 4. 優化器僅優化「未凍結」的參數，並降低學習率
    # 這裡只抓取 requires_grad=True 的參數 (即 Cross-Attention 和 FC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-2)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=opt.lr * 0.1)

    # 5. 💡 預先測試：看看載入權重後，還沒練之前的分數
    model.eval()
    _, init_acc, _, _, _, _, _, _ = test2(model, valid_loader, device, opt)
    logging.info(f"📊 載入後初始準確率 (Epoch 0 前): {init_acc:.2f}%")

    best_model_state = None
    for epoch in range(opt.n_epochs):
        
        # if epoch == 10:
        #     logging.info("🔥 [Unfreeze] 解除鎖定，開始全模型微調 (Fine-tuning)")
        #     import gc
        #     torch.cuda.empty_cache()
        #     gc.collect()
        #     for param in model.modelA.parameters():
        #         param.requires_grad = True
        #     for param in model.modelB.parameters():
        #         param.requires_grad = True
        # if epoch == 10:
        #     logging.info("🔥 [Partial Unfreeze] 只解凍最後一個 Block，兼顧速度與微調")
        #     # 只解凍最後一個 block
        #     import gc
        #     torch.cuda.empty_cache()
        #     gc.collect()
        #     for param in model.modelA.blocks[-1].parameters():
        #         param.requires_grad = True
        #     for param in model.modelB.blocks[-1].parameters():
        #         param.requires_grad = True
            
        #     # 🌟 [關鍵] 重新定義優化器，把 modelA 和 modelB 的參數加進去
        #     # 注意：微調時學習率要設得非常小 (例如原本的 1/100)
        #     optimizer = torch.optim.Adam([
        #         {'params': model.modelA.parameters(), 'lr': 1e-4}, 
        #         {'params': model.modelB.parameters(), 'lr': 1e-4},
        #         {'params': model.cross_attn1.parameters(), 'lr': opt.lr * 0.1},
        #         {'params': model.cross_attn2.parameters(), 'lr': opt.lr * 0.1},
        #         {'params': model.fuse_block.parameters(), 'lr': opt.lr * 0.1},
        #         {'params': model.fc.parameters(), 'lr': opt.lr * 0.1}
        #     ])
        #     # 同步更新 scheduler，確保它接著新的 optimizer 跑
        #     lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs - epoch, eta_min=1e-6)
        
        since = time.time()
        model.train()
        epoch_loss = 0.0
        
        for i, ((imgs, label, _), (imgs_fft, _, _)) in enumerate(
            tqdm(train_loader, desc=f"Fold {opt.fold} M3 Epoch {epoch+1}", leave=False)
        ):
            # 簡化資料轉換動作，加快速度
            valid_label = label.to(device).long()
            real_img = imgs.to(device).float()
            real_img_fft = imgs_fft.to(device).float()
            
            optimizer.zero_grad()
            combined = model(real_img, real_img_fft)
            loss = CE_loss(combined, valid_label)
            
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 0408 add
            optimizer.step()
        
        epoch_loss = epoch_loss / len(train_loader)
        lr_current = optimizer.param_groups[0]['lr']
        
        # 測試本輪表現
        loss_t, acc_t, rec_t, pre_t, f1_t, spec_t, cm_t, err_t = test2(model, valid_loader, device, opt)
        logging.info(f"[Epoch {epoch}] Loss: {epoch_loss:.4f} | Test Acc: {acc_t:.2f}% | Spec: {spec_t:.4f}")
        
        lr_scheduler.step()
        
        # 紀錄與更新最佳模型 (只要 Acc 創新高就打包)
        if acc_t > best_acc:
            best_acc = acc_t
            best_f1, best_precision, best_recall, best_specificity = f1_t, pre_t, rec_t, spec_t
            best_cm, best_error_list = cm_t, err_t
            best_model_state = model.state_dict()
            
            best_path = os.path.join(opt.out_dir, f"fold_{opt.fold}_model_3_best.pth")
            torch.save(best_model_state, best_path)

        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, opt.model_count, f"{loss_t:.6f}", f"{acc_t:.6f}", f"{rec_t:.6f}", f"{pre_t:.6f}", f"{f1_t:.6f}", f"{spec_t:.6f}"])
            
        time_elapsed = time.time() - since
        logging.info('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return best_model_state, best_acc, best_f1, best_precision, best_recall, best_specificity, best_cm, best_error_list