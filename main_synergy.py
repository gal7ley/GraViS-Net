# 文件名: main_synergy.py
import os
import argparse
import random
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from joblib.externals.loky import cpu_count
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import torch.nn.functional as F
import albumentations as A

# === 引用 ===
from models_v5 import SynergyNetV11
# from model_D import SynergyNet_L4Only
# from model_c import SynergyNet_C
# from model_resnet import SynergyNet_resnet
from utils_loss import LDAMLoss
from utils_dataset_isic import SynergyNet_ISIC_Dataset_Albumentations, collate_fn_baseline


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(description='SynergyNet Training')
    parser.add_argument('--base_dir', type=str, default='/path/to/your/ISIC2018/dataset')
    # 指向预处理脚本生成的缓存目录
    # parser.add_argument('--cache_dir', type=str, default='./cache_data')
    parser.add_argument('--cache_dir', type=str, default='./cache_data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='SynergyNet_model')

    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr_rgb', type=float, default=2e-5)
    parser.add_argument('--lr_new', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--seed', type=int, default=2023)
    return parser.parse_args()


# 指标计算
def calculate_basic_metrics(y_true, y_pred, y_prob, num_classes=7):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=np.arange(num_classes))
    except:
        auc = 0.0
    return acc, bacc, auc


def run_fold(config, fold_num, log_df_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{config.model_name}_Fold{fold_num}"
    print("\n" + "=" * 80)
    print(f" STARTING FOLD {fold_num} | {run_name}")
    print("=" * 80)

    scaler = GradScaler('cuda')

    train_geometric_transform = A.Compose([
        A.RandomCrop(height=config.image_size, width=config.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=35, p=0.7),
        A.GridDistortion(p=0.2),
    ], additional_targets={'ivg_map': 'image'})

    val_geometric_transform = A.Compose([
        A.Resize(height=config.image_size, width=config.image_size),
    ], additional_targets={'ivg_map': 'image'})

    # --- 2. Dataset & Loader ---
    master_csv_path = os.path.join(config.base_dir, "ISIC2018_Task3_Training_GroundTruth_with_folds.csv")
    if not os.path.exists(master_csv_path):
        master_csv_path = os.path.join(os.path.dirname(config.base_dir),"ISIC2018_Task3_Training_GroundTruth_with_folds.csv")

    df_all = pd.read_csv(master_csv_path)
    df_train = df_all[df_all['fold'] != fold_num].copy()
    df_val = df_all[df_all['fold'] == fold_num].copy()

    img_roots = [os.path.join(config.base_dir, "ISIC_2018")]

    # 实例化 Dataset
    train_dataset = SynergyNet_ISIC_Dataset_Albumentations(
        df_train, img_roots, config.cache_dir, transform=train_geometric_transform
    )
    val_dataset = SynergyNet_ISIC_Dataset_Albumentations(
        df_val, img_roots, config.cache_dir, transform=val_geometric_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers, pin_memory=True,collate_fn=lambda x: collate_fn_baseline(x), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True,collate_fn=lambda x: collate_fn_baseline(x),)

    # --- 3. Model ---
    model = SynergyNetV11(config.num_classes, fusion_dim=256).to(device)

    # --- 4. Loss & Optimizer ---
    class_counts = df_train['label_idx'].value_counts().sort_index().values
    cls_num_list = class_counts.tolist()

    ce_criterion = nn.CrossEntropyLoss()
    ldam_criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30)

    optimizer = optim.AdamW([
        {'params': model.get_rgb_params(), 'lr': config.lr_rgb},
        {'params': model.get_new_params(), 'lr': config.lr_new}
    ], weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr_rgb / 20)

    best_bacc = 0.0
    best_acc = 0.0
    best_model_path = os.path.join(config.checkpoint_dir, f"{run_name}_best.pth")
    # 结果保存路径 (.npz)best_evidence_path = os.path.join(config.checkpoint_dir, f"{run_name}_evidence.npz")

    patience = 20
    no_improve = 0
    drw_epoch = 15

    # ================= Training Loop =================
    for epoch in range(config.epochs):
        epoch_start_time = time.time()

        model.train()
        accum = {'loss_main': 0.0}
        num_batches = 0

        # DRW 策略
        criterion = ce_criterion if epoch < drw_epoch else ldam_criterion
        loss_mode = "CE" if epoch < drw_epoch else "LDAM"

        pbar = tqdm(train_loader, desc=f"Fold{fold_num} Ep{epoch + 1} [{loss_mode}]", leave=False)

        for rgb, ivg, labels, _ in pbar:
            rgb, ivg, labels = rgb.to(device), ivg.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                main_out, aux_out, _, _ = model(rgb, ivg)
                l_main = criterion(main_out, labels)

                # 辅助损失
                if aux_out is not None:
                    l_aux = F.cross_entropy(aux_out, labels)
                    total_loss = l_main + 0.4 * l_aux
                else:
                    total_loss = l_main

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            accum['loss_main'] += l_main.item() * rgb.size(0)
            num_batches += 1
            pbar.set_postfix({'Loss': f"{l_main.item():.4f}"})

        scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        train_loss = accum['loss_main'] / len(train_dataset)

        model.eval()

        val_loss_sum = 0.0
        val_y_true, val_y_pred, val_y_prob = [], [], []
        val_img_ids = []

        with torch.no_grad():
            for rgb, ivg, labels, img_names in tqdm(val_loader, desc="Validating", leave=False):
                rgb, ivg, labels = rgb.to(device), ivg.to(device), labels.to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # 1. Standard
                    out_std, _, _, _ = model(rgb, ivg)
                    loss_v = F.cross_entropy(out_std, labels)
                    val_loss_sum += loss_v.item() * rgb.size(0)

                    # TTA Accumulation
                    prob_accum = torch.softmax(out_std.float(), dim=1)

                    prob_accum += torch.softmax(model(torch.flip(rgb, [3]), torch.flip(ivg, [3]))[0].float(), dim=1)
                    prob_accum += torch.softmax(model(torch.flip(rgb, [2]), torch.flip(ivg, [2]))[0].float(), dim=1)
                    prob_accum += torch.softmax(model(torch.rot90(rgb, 1, [2, 3]), torch.rot90(ivg, 1, [2, 3]))[0].float(), dim=1)
                    prob_accum += torch.softmax(model(torch.rot90(rgb, 3, [2, 3]), torch.rot90(ivg, 3, [2, 3]))[0].float(), dim=1)

                final_prob = prob_accum / 5.0
                _, final_pred = torch.max(final_prob, 1)

                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(final_pred.cpu().numpy())
                val_y_prob.extend(final_prob.cpu().numpy())
                val_img_ids.extend(img_names)

        val_loss = val_loss_sum / len(val_dataset)
        v_acc, v_bacc, v_auc = calculate_basic_metrics(
            np.array(val_y_true), np.array(val_y_pred), np.array(val_y_prob), config.num_classes
        )

        print(f"Ep {epoch + 1:02d} | Time: {epoch_duration:.0f}s | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {v_acc * 100:.2f}% | BACC: {v_bacc * 100:.2f}% | AUC: {v_auc:.4f}")

        log_entry = {
            'Fold': fold_num,
            'Epoch': epoch + 1,
            'Time_Sec': round(epoch_duration, 2),  # <--- 保存时间
            'Train_Loss': train_loss,
            'Val_Loss': val_loss,
            'Val_Acc': v_acc,
            'Val_BACC': v_bacc,
            'Val_AUC': v_auc
        }
        log_df_list.append(log_entry)

        # Checkpointing
        if v_acc > best_acc:
            best_acc = v_acc
            best_bacc = v_bacc
            no_improve = 0

            # 1. 保存模型
            torch.save(model.state_dict(), best_model_path)

            # 2. 保存 Full Evidence (.npz)
            np.savez_compressed(
                best_evidence_path,
                y_true=np.array(val_y_true),
                y_pred=np.array(val_y_pred),
                y_score=np.array(val_y_prob),
                img_ids=np.array(val_img_ids)
            )
            print(f"   >>> Best Model Saved! (Acc: {best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_acc, best_bacc


if __name__ == '__main__':
    config = parse_arguments()
    set_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    all_logs = []
    fold_metrics = []

    # 跑 5 折
    for f in range(1, 6):
        b_acc, b_bacc = run_fold(config, f, all_logs)
        fold_metrics.append((b_acc, b_bacc))

        # 实时保存日志，防止断电丢失
        df_log = pd.DataFrame(all_logs)
        df_log.to_csv(os.path.join(config.checkpoint_dir, 'training_log_full.csv'), index=False)

        # 清理显存
        torch.cuda.empty_cache()

    # 总结
    print("\n" + "=" * 50)
    print("ALL FOLDS COMPLETED")
    avgs = np.mean(fold_metrics, axis=0)
    print(f"Avg ACC:  {avgs[0] * 100:.2f}%")
    print(f"Avg BACC: {avgs[1] * 100:.2f}%")