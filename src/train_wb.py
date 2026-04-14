# -*- coding: utf-8 -*-
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import random

# === 核心修复：全部使用绝对导入 (Absolute Import) ===
from datasets.uav_dataset import UAVFusionDataset
from models.fusion_net import UAVFusionNet  # 确保你的 fusion_net.py 里已经换上了 EnhancedPointNet
from config import cfg


def train_model():
    # ================= 1. 初始化实验与监控 =================
    print("\n" + "=" * 50)
    print("🚀 启动空反空多模态融合训练 (Exp_04: EnhancedPointNet)")
    print("=" * 50)

    # 初始化 WandB 实验
    run = wandb.init(
        project="UAV-Fusion-Anti-Drone",
        name="exp4_enhanced_pointnet_4096pts",  # 实验名称
        config={
            "learning_rate": cfg.LEARNING_RATE,
            "architecture": "ResNet18 + EnhancedPointNet",
            "num_points": cfg.NUM_POINTS,
            "batch_size": cfg.BATCH_SIZE,
            "drop_img_prob": cfg.DROP_IMG_PROB,
            "drop_pc_prob": cfg.DROP_PC_PROB,
            "loss_weight_xyz": cfg.LOSS_WEIGHT_XYZ,
            "loss_weight_cls": cfg.LOSS_WEIGHT_CLS
        }
    )

    # 创建本地保存目录 (与 WandB 的运行名称保持一致，方便对应)
    save_dir = f"./runs/{run.name}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] 实验数据与本地权重将保存在: {os.path.abspath(save_dir)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 正在使用设备: {device.type.upper()}")

    # ================= 2. 准备数据集 (80% 训练 / 20% 验证) =================
    print("[INFO] 正在准备和切分数据集...")
    train_dataset = UAVFusionDataset(csv_file=cfg.TRAIN_CSV, num_points=cfg.NUM_POINTS, is_train=True)
    val_dataset = UAVFusionDataset(csv_file=cfg.TRAIN_CSV, num_points=cfg.NUM_POINTS, is_train=False)

    # 生成固定的随机索引进行切分，保证每次实验考的都是同一套题
    indices = list(range(len(train_dataset)))
    random.seed(42)
    random.shuffle(indices)
    split_idx = int(0.2 * len(indices))
    val_idx, train_idx = indices[:split_idx], indices[split_idx:]

    # 替换内部记录
    train_dataset.data_records = [train_dataset.data_records[i] for i in train_idx]
    val_dataset.data_records = [val_dataset.data_records[i] for i in val_idx]

    print(f"[INFO] 数据集切分完毕 -> 训练集: {len(train_dataset)} 帧 | 验证集: {len(val_dataset)} 帧")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # ================= 3. 初始化模型与优化器 =================
    print("[INFO] 正在初始化网络模型...")
    model = UAVFusionNet(num_classes=2).to(device)

    # 监听模型梯度和参数分布 (可在 WandB 网页端查看模型健康度)
    wandb.watch(model, log="all", log_freq=10)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    criterion_xyz = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    best_val_loss = float('inf')  # 用于记录最佳成绩

    # ================= 4. 开始主训练循环 =================
    print("\n[INFO] 💥 开始训练循环...")
    start_time = time.time()

    for epoch in range(cfg.EPOCHS):
        # --- A. 训练阶段 ---
        model.train()
        train_loss = 0.0

        for images, pointclouds, targets_xyz, targets_cls in train_loader:
            images, pointclouds = images.to(device), pointclouds.to(device)
            targets_xyz, targets_cls = targets_xyz.to(device), targets_cls.to(device)

            optimizer.zero_grad()
            pred_xyz, pred_cls = model(images, pointclouds)

            loss_xyz = criterion_xyz(pred_xyz, targets_xyz)
            loss_cls = criterion_cls(pred_cls, targets_cls)

            # 总损失 = 坐标误差 * 权重 + 分类误差 * 权重
            total_loss = (cfg.LOSS_WEIGHT_XYZ * loss_xyz) + (cfg.LOSS_WEIGHT_CLS * loss_cls)

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- B. 验证阶段 ---
        model.eval()
        val_loss, val_mse = 0.0, 0.0

        with torch.no_grad():
            for images, pointclouds, targets_xyz, targets_cls in val_loader:
                images, pointclouds = images.to(device), pointclouds.to(device)
                targets_xyz, targets_cls = targets_xyz.to(device), targets_cls.to(device)

                pred_xyz, _ = model(images, pointclouds)
                loss_xyz = criterion_xyz(pred_xyz, targets_xyz)

                val_mse += loss_xyz.item()
                val_loss += (cfg.LOSS_WEIGHT_XYZ * loss_xyz.item())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)

        # --- C. 云端同步与本地保存 ---
        # 1. 整理核心指标发送给 WandB
        metrics = {
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/xyz_mse_meters": avg_val_mse,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)

        # 2. 每次 Epoch 都保存一次最新状态
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))

        # 3. 检查是否破了记录，如果是，保存为 best.pth
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            # 控制台高亮打印
            print(f"Epoch [{epoch + 1:02d}/{cfg.EPOCHS}] "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} "
                  f"👉 🌟 发现更好的模型! (MSE: {avg_val_mse:.4f})")
        else:
            # 控制台普通打印
            print(f"Epoch [{epoch + 1:02d}/{cfg.EPOCHS}] "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    # ================= 5. 训练结束 =================
    end_time = time.time()
    print(f"\n[SUCCESS] 训练完成！总耗时: {(end_time - start_time) / 60:.2f} 分钟")
    print(f"[INFO] 最终最佳权重保存在: {os.path.join(save_dir, 'best.pth')}")

    # 结束 WandB 进程
    run.finish()


if __name__ == "__main__":
    train_model()