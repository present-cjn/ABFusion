import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# 导入我们自己写的模块
from datasets.uav_dataset import UAVFusionDataset
from models.fusion_net import UAVFusionNet


def train_model():
    # ================= 1. 基础配置 =================
    CSV_PATH = "/media/hzbz/dataset/data/mmaud/train/dataset_index_20m.csv"
    BATCH_SIZE = 8  # 你的 Y9000P 显存应该足够跑 batch_size=8 或 16
    EPOCHS = 50  # 先跑 50 轮看看收敛情况
    LEARNING_RATE = 1e-4  # 学习率，不能太大，容易把预训练的 ResNet 权重破坏

    # 检测是否有 Y9000P 强大的 NVIDIA 显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 正在使用设备: {device.type.upper()}")

    # ================= 2. 加载数据与模型 =================
    print("[INFO] 正在准备数据集...")
    train_dataset = UAVFusionDataset(csv_file=CSV_PATH, num_points=1024)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("[INFO] 正在初始化模型...")
    model = UAVFusionNet(num_classes=2).to(device)

    # ================= 3. 定义损失函数与优化器 =================
    # 回归损失 (MSE)：用来约束 XYZ 坐标的误差
    criterion_xyz = nn.MSELoss()
    # 分类损失 (CrossEntropy)：用来约束分类结果
    criterion_cls = nn.CrossEntropyLoss()

    # 优化器 (AdamW 是目前最好用的优化器之一)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 损失权重 (Loss Weights)
    # 因为 MSE 算出来的值可能很大（比如几十），而 CrossEntropy 通常小于 1
    # 我们暂时给坐标回归 1.0 的权重，分类 10.0 的权重，让模型兼顾
    alpha_xyz = 1.0
    beta_cls = 10.0

    # ================= 4. 开始训练循环 =================
    print("[INFO] 🚀 开始训练...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()  # 设置为训练模式 (启用 Dropout 和 BatchNorm)
        running_loss = 0.0
        running_loss_xyz = 0.0
        running_loss_cls = 0.0

        for i, (images, pointclouds, targets_xyz, targets_cls) in enumerate(train_loader):
            # 将数据搬运到 GPU 上
            images = images.to(device)
            pointclouds = pointclouds.to(device)
            targets_xyz = targets_xyz.to(device)
            targets_cls = targets_cls.to(device)

            # 1. 梯度清零
            optimizer.zero_grad()

            # 2. 前向传播 (Forward pass)
            pred_xyz, pred_cls = model(images, pointclouds)

            # 3. 计算损失 (Calculate Loss)
            loss_xyz = criterion_xyz(pred_xyz, targets_xyz)
            loss_cls = criterion_cls(pred_cls, targets_cls)

            # 总损失
            total_loss = (alpha_xyz * loss_xyz) + (beta_cls * loss_cls)

            # 4. 反向传播与参数更新 (Backward pass & Optimize)
            total_loss.backward()
            optimizer.step()

            # 记录日志
            running_loss += total_loss.item()
            running_loss_xyz += loss_xyz.item()
            running_loss_cls += loss_cls.item()

        # 计算每个 Epoch 的平均 Loss
        avg_loss = running_loss / len(train_loader)
        avg_loss_xyz = running_loss_xyz / len(train_loader)
        avg_loss_cls = running_loss_cls / len(train_loader)

        # 打印当前 Epoch 进度
        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Total Loss: {avg_loss:.4f} | "
              f"XYZ MSE: {avg_loss_xyz:.4f} | "
              f"CLS Loss: {avg_loss_cls:.4f}")

    # ================= 5. 保存模型权重 =================
    end_time = time.time()
    print(f"\n[SUCCESS] 训练完成！总耗时: {(end_time - start_time) / 60:.2f} 分钟")

    save_path = "../weights/uav_fusion_baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] 模型权重已保存至: {save_path}")


if __name__ == "__main__":
    train_model()