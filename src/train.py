# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import random

# 导入自己写的模块
from datasets.uav_dataset import UAVFusionDataset
from models.fusion_net import UAVFusionNet
from config import cfg  # 引入全局配置


# ================= 辅助函数 =================
def save_curves(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='red', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
    plt.close()


def get_exp_dir(base_dir='./runs'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    i = 1
    while os.path.exists(os.path.join(base_dir, f'exp{i}')):
        i += 1
    exp_dir = os.path.join(base_dir, f'exp{i}')
    os.makedirs(exp_dir)
    return exp_dir


# ================= 主训练逻辑 =================
def train_model():
    # 1. 初始化实验文件夹和日志
    exp_dir = get_exp_dir()
    log_file_path = os.path.join(exp_dir, "train.log")

    # 定义一个同时打印到屏幕和文件的日志函数
    def logger(msg):
        print(msg)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    logger(f"========== 🚀 新的实验已开启: {exp_dir} ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger(f"[INFO] 正在使用设备: {device.type.upper()}")

    # 2. 加载数据并进行 Train/Val 80/20 切分
    logger("[INFO] 正在准备和切分数据集...")

    # 实例化两个 Dataset (注意 is_train 的区别)
    # train 集开启随机 Dropout 增强，val 集必须关闭以保持测试严谨性
    train_dataset = UAVFusionDataset(csv_file=cfg.TRAIN_CSV, num_points=cfg.NUM_POINTS, is_train=True)
    val_dataset = UAVFusionDataset(csv_file=cfg.TRAIN_CSV, num_points=cfg.NUM_POINTS, is_train=False)

    # 生成随机索引进行切分
    indices = list(range(len(train_dataset)))
    random.seed(42)  # 固定随机种子，保证每次切分的数据一样
    random.shuffle(indices)
    split_idx = int(0.2 * len(indices))  # 20% 作为验证集
    val_idx, train_idx = indices[:split_idx], indices[split_idx:]

    # 替换内部记录
    train_dataset.data_records = [train_dataset.data_records[i] for i in train_idx]
    val_dataset.data_records = [val_dataset.data_records[i] for i in val_idx]

    logger(f"[INFO] 数据集切分完毕 -> 训练集: {len(train_dataset)} 帧 | 验证集: {len(val_dataset)} 帧")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 初始化模型、损失函数与优化器
    logger("[INFO] 正在初始化模型...")
    model = UAVFusionNet(num_classes=2).to(device)
    criterion_xyz = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)

    alpha_xyz = 1.0
    beta_cls = 10.0

    # 准备记录画图的数据
    history_train_loss = []
    history_val_loss = []
    best_val_loss = float('inf')  # 记录最低的验证集 Loss

    # 4. 开始 Epoch 循环
    logger("\n[INFO] 开始训练...")
    start_time = time.time()

    for epoch in range(cfg.EPOCHS):
        # === A. 训练阶段 (Training) ===
        model.train()
        running_loss = 0.0

        for images, pointclouds, targets_xyz, targets_cls in train_loader:
            images, pointclouds = images.to(device), pointclouds.to(device)
            targets_xyz, targets_cls = targets_xyz.to(device), targets_cls.to(device)

            optimizer.zero_grad()
            pred_xyz, pred_cls = model(images, pointclouds)

            loss_xyz = criterion_xyz(pred_xyz, targets_xyz)
            loss_cls = criterion_cls(pred_cls, targets_cls)
            total_loss = (alpha_xyz * loss_xyz) + (beta_cls * loss_cls)

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # === B. 验证阶段 (Validation) ===
        model.eval()  # 关闭 Dropout 和 BatchNorm 更新
        running_val_loss = 0.0
        running_val_xyz_mse = 0.0

        with torch.no_grad():  # 不计算梯度，节省显存提速
            for images, pointclouds, targets_xyz, targets_cls in val_loader:
                images, pointclouds = images.to(device), pointclouds.to(device)
                targets_xyz, targets_cls = targets_xyz.to(device), targets_cls.to(device)

                pred_xyz, pred_cls = model(images, pointclouds)

                loss_xyz = criterion_xyz(pred_xyz, targets_xyz)
                loss_cls = criterion_cls(pred_cls, targets_cls)
                total_loss = (alpha_xyz * loss_xyz) + (beta_cls * loss_cls)

                running_val_loss += total_loss.item()
                running_val_xyz_mse += loss_xyz.item()

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_xyz_mse = running_val_xyz_mse / len(val_loader)
        history_val_loss.append(avg_val_loss)

        # === C. 日志打印与模型保存 ===
        logger(f"Epoch [{epoch + 1:02d}/{cfg.EPOCHS}] "
               f"Train Loss: {avg_train_loss:.4f} | "
               f"Val Loss: {avg_val_loss:.4f} (XYZ MSE: {avg_val_xyz_mse:.4f})")

        # 实时保存曲线图
        save_curves(history_train_loss, history_val_loss, exp_dir)

        # 保存最新的模型 (Last)
        torch.save(model.state_dict(), os.path.join(exp_dir, "last.pth"))

        # 保存表现最好的模型 (Best)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best.pth"))
            logger(f"   🌟 发现更好的模型！(Val Loss 降至 {best_val_loss:.4f})")

    # 5. 训练结束总结
    end_time = time.time()
    logger(f"\n[SUCCESS] 训练完成！总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    logger(f"[INFO] 所有的模型权重、日志和曲线图已保存在: {os.path.abspath(exp_dir)}")


if __name__ == "__main__":
    train_model()