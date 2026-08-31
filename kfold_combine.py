import os
import torch
import logging
import pandas as pd
import json
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset as TorchConcatDataset
from sklearn.model_selection import GroupKFold
from torchvision import transforms

from train1 import train_model
from train2 import train_combined_model
from utils import ImageListIter, ConcatDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Opt:
    n_epochs = 20
    batch_size = 8
    lr = 0.0001
    n_splits = 5

    fold = -1
    model_count = -1

    task_name = "task_name"
    NewHandPD_task_name = task_name
    # 存檔目錄
    DATA_ROOT = os.environ.get("PD_DATA_ROOT", "./data")
    OUTPUT_ROOT = os.environ.get("PD_OUTPUT_ROOT", "./outputs")

    out_dir = os.path.join(OUTPUT_ROOT, "checkpoints", f"NewHandPD_{task_name}")
    csv = os.path.join(OUTPUT_ROOT, "csv", f"NewHandPD_{task_name}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(csv, exist_ok=True)

    # 使用產出的原圖 CSV — paths are now relative to DATA_ROOT
    train_csv_normal_list = [
        os.path.join(DATA_ROOT, "csv", "Original_Circle.csv"),
        os.path.join(DATA_ROOT, "csv", "PatientCircle_Normal_Aug24.csv"),
        os.path.join(DATA_ROOT, "csv", "HealthyCircle_Normal_Aug24.csv"),
    ]
    train_csv_fft_list = [
        os.path.join(DATA_ROOT, "csv", "Original_Circle_fft.csv"),
        os.path.join(DATA_ROOT, "csv", "PatientCircle_Amplitude_Aug24.csv"),
        os.path.join(DATA_ROOT, "csv", "HealthyCircle_Amplitude_Aug24.csv"),
    ]


    # Transforms
    raw_normal_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    raw_fft_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
 
 
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
 
 
def compute_fold_mean_std(dataset, indices, batch_size=32, num_workers=4):
    """
    Compute per-channel mean/std over `dataset[indices]` using running
    sum / sum-of-squares accumulation (exact over the full training
    subset, not an average of per-image statistics). `dataset` must
    yield tensors already scaled to [0, 1] by ToTensor() (i.e., no
    Normalize() applied yet).
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
 
    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.double()
            b, c, h, w = imgs.shape
            total_sum += imgs.sum().item()
            total_sq_sum += (imgs ** 2).sum().item()
            total_pixels += b * c * h * w
 
    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - (mean ** 2)
    std = var ** 0.5
    return mean, std
 
 
def main_kfold(opt):
    logging.basicConfig(level=logging.INFO)
    print(f"Using device: {device}")
 
    # 1. 讀入所有 CSV
    datasets_normal = [ImageListIter(transform=opt.raw_normal_transform, csv_list=csv) for csv in opt.train_csv_normal_list]
    datasets_fft = [ImageListIter(transform=opt.raw_fft_transform, csv_list=csv) for csv in opt.train_csv_fft_list]
 
    full_dataset_normal = TorchConcatDataset(datasets_normal)
    full_dataset_fft = TorchConcatDataset(datasets_fft)
    full_dataset_combined = ConcatDataset(full_dataset_normal, full_dataset_fft)
 
    # 2. 建立 Group ID 與 擴增標記
    groups = []
    is_augment_mask = []  # 🌟 修正：紀錄是否為擴增圖
 
    for dataset in full_dataset_normal.datasets:
        for item in dataset.image_list:
            target_path = ""
            for cell in item:
                cell_str = str(cell)
                if '/' in cell_str and '.jpg' in cell_str.lower():
                    target_path = cell_str
                    break
 
            if target_path:
                file_name = os.path.basename(target_path)
 
                # 對齊 ID：不管是 sp1-P10.jpg 還是 sp1-P10_ang30...
                patient_id = file_name.split('_')[0].split('.')[0]
                groups.append(patient_id)
 
                # 紀錄標記：檔名包含 'ang' 或 'aug' 或 'flip' 就視為擴增圖
                if any(k in file_name.lower() for k in ['ang', 'aug', 'flip']):
                    is_augment_mask.append(True)
                else:
                    is_augment_mask.append(False)
            else:
                groups.append(str(item[0]))
                is_augment_mask.append(False)
 
    groups = np.array(groups)
    is_augment_mask = np.array(is_augment_mask)
    indices = np.arange(len(full_dataset_normal))
    gkf = GroupKFold(n_splits=opt.n_splits)
    fold_results = []
 
    norm_stats_log = {}
 
    def run_stage(m_idx, train_fn, loaders, fold_dir):
        opt.model_count = m_idx
        res = train_fn(loaders[0], loaders[1], opt, device)
        state, acc, f1, pre, rec, spec, cm, errors = res
        pd.DataFrame(cm).to_csv(os.path.join(fold_dir, f"model_{m_idx}_cm.csv"), index=False, header=False)
        pd.DataFrame(errors, columns=['Path', 'True', 'Pred']).to_csv(os.path.join(fold_dir, f"model_{m_idx}_errors.csv"), index=False)
        torch.save(state, os.path.join(fold_dir, f"model_{m_idx}_best.pth"))
        return acc, f1, pre, rec, spec
 
    # 4. 開始 Fold 迴圈
    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        fold_dir = os.path.join(opt.out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        logging.info(f"===== Fold {fold} =====")
        opt.fold = fold
        # 儲存分組名單
        split_info = {"val_subjects": sorted(list(set(groups[val_idx])))}
        save_json(split_info, os.path.join(fold_dir, "split_info.json"))
        actual_train_idx = [i for i in train_idx if is_augment_mask[i] == True]
        actual_val_idx = [i for i in val_idx if is_augment_mask[i] == False]
 
        logging.info(f"Fold {fold} | Train(Aug): {len(actual_train_idx)} | Val(Ori): {len(actual_val_idx)}")
 
        logging.info(f"Fold {fold} | Computing per-fold normalization stats...")
        normal_mean, normal_std = compute_fold_mean_std(full_dataset_normal, actual_train_idx)
        fft_mean, fft_std = compute_fold_mean_std(full_dataset_fft, actual_train_idx)
        logging.info(f"Fold {fold} | Normal: mean={normal_mean:.6f}, std={normal_std:.6f}")
        logging.info(f"Fold {fold} | FFT:    mean={fft_mean:.6f}, std={fft_std:.6f}")
 
        norm_stats_log[f"fold_{fold}"] = {
            "normal_mean": normal_mean, "normal_std": normal_std,
            "fft_mean": fft_mean, "fft_std": fft_std,
        }
        save_json(norm_stats_log[f"fold_{fold}"], os.path.join(fold_dir, "normalization_stats.json"))
 
        # Build this fold's final (normalized) transforms
        fold_normal_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=normal_mean, std=normal_std),
        ])
        fold_fft_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=fft_mean, std=fft_std),
        ])
 
        for ds in full_dataset_normal.datasets:
            ds.transform = fold_normal_transform
        for ds in full_dataset_fft.datasets:
            ds.transform = fold_fft_transform
 
        # 建立 DataLoader
        WORKERS = 8
        l_fft = (DataLoader(Subset(full_dataset_fft, actual_train_idx), batch_size=opt.batch_size, shuffle=True, num_workers=WORKERS),
                 DataLoader(Subset(full_dataset_fft, actual_val_idx), batch_size=opt.batch_size, shuffle=False, num_workers=WORKERS))
        l_nor = (DataLoader(Subset(full_dataset_normal, actual_train_idx), batch_size=opt.batch_size, shuffle=True, num_workers=WORKERS),
                 DataLoader(Subset(full_dataset_normal, actual_val_idx), batch_size=opt.batch_size, shuffle=False, num_workers=WORKERS))
        l_com = (DataLoader(Subset(full_dataset_combined, actual_train_idx), batch_size=opt.batch_size, shuffle=True, num_workers=WORKERS),
                 DataLoader(Subset(full_dataset_combined, actual_val_idx), batch_size=opt.batch_size, shuffle=False, num_workers=WORKERS))
 
        # 執行三階段：這裡每個 run_stage 會回傳 (acc, f1, pre, rec, spec)
        m1_a, m1_f, m1_p, m1_r, m1_s = run_stage(1, train_model, l_fft, fold_dir)
        m2_a, m2_f, m2_p, m2_r, m2_s = run_stage(2, train_model, l_nor, fold_dir)
 
        opt.fft_weight = os.path.join(fold_dir, "model_1_best.pth")
        opt.nofft_weight = os.path.join(fold_dir, "model_2_best.pth")
 
        m3_a, m3_f, m3_p, m3_r, m3_s = run_stage(3, train_combined_model, l_com, fold_dir)
 
        fold_results.append({
            'fold': fold,
            # Model 1 (FFT)
            'm1_acc': m1_a, 'm1_f1': m1_f, 'm1_pre': m1_p, 'm1_rec': m1_r, 'm1_spec': m1_s,
            # Model 2 (Normal)
            'm2_acc': m2_a, 'm2_f1': m2_f, 'm2_pre': m2_p, 'm2_rec': m2_r, 'm2_spec': m2_s,
            # Model 3 (Combined)
            'm3_acc': m3_a, 'm3_f1': m3_f, 'm3_pre': m3_p, 'm3_rec': m3_r, 'm3_spec': m3_s,
        })
 
    # Save the complete per-fold normalization stats log (all 5 folds together)
    save_json(norm_stats_log, os.path.join(opt.out_dir, "all_folds_normalization_stats.json"))
 
    # 5. 總結報告
    df = pd.DataFrame(fold_results)
    avg = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)
 
    logging.info("===== Average Results (Mean ± STD) =====")
    for col in df.columns[1:]:
        logging.info(f"{col:15}: {avg[col]:.4f} ± {std[col]:.4f}")
 
    df.to_csv(os.path.join(opt.out_dir, "all_models_final_summary.csv"), index=False)
    summary = {"mean": avg.to_dict(), "std": std.to_dict()}
    save_json(summary, os.path.join(opt.out_dir, "final_average_report.json"))
    return df
 
 
if __name__ == '__main__':
    opt = Opt()
    main_kfold(opt)