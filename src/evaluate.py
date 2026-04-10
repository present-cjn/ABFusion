import torch
from torch.utils.data import DataLoader
import numpy as np

# 导入你的数据集和模型类
from dataset import UAVFusionDataset
from model import UAVFusionNet


def evaluate_model():
    # 1. 基础配置
    CSV_PATH = "/media/hzbz/dataset/data/mmaud/train/dataset_index_20m.csv"
    WEIGHTS_PATH = "uav_fusion_baseline.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 正在使用设备: {device}")

    # 2. 加载数据集 (shuffle=True 为了每次运行看不同的样本)
    dataset = UAVFusionDataset(csv_file=CSV_PATH, num_points=1024)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # 3. 初始化模型并加载权重
    model = UAVFusionNet(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print(f"[INFO] 成功加载模型权重: {WEIGHTS_PATH}")
    except Exception as e:
        print(f"[ERROR] 找不到权重文件，请确保文件名正确！\n{e}")
        return

    # 4. 开启推理模式 (至关重要：关闭 Dropout 和 BatchNorm 的动态更新)
    model.eval()

    print("\n" + "=" * 50)
    print("🚀 开始模型推理对比测试 🚀")
    print("=" * 50)

    # 5. 我们只抽 1 个 Batch (4 个样本) 来看一下直观结果
    with torch.no_grad():  # 告诉 PyTorch 不用算梯度了，省显存提速
        for images, pointclouds, targets_xyz, targets_cls in dataloader:

            # 搬运数据到 GPU
            images = images.to(device)
            pointclouds = pointclouds.to(device)

            # 模型预测
            pred_xyz, pred_cls = model(images, pointclouds)

            # 转回 CPU 的 NumPy 数组，方便打印阅读
            pred_xyz_np = pred_xyz.cpu().numpy()
            targets_xyz_np = targets_xyz.numpy()

            # 取分类概率最大的索引作为预测类别
            pred_cls_labels = torch.argmax(pred_cls, dim=1).cpu().numpy()
            targets_cls_np = targets_cls.numpy()

            # 6. 格式化打印对比
            for i in range(4):  # 遍历 Batch 里的 4 个样本
                print(f"\n--- 样本 {i + 1} ---")

                # 打印坐标
                p_x, p_y, p_z = pred_xyz_np[i]
                t_x, t_y, t_z = targets_xyz_np[i]

                # 计算这个样本的 3D 欧氏距离误差
                error_dist = np.sqrt((p_x - t_x) ** 2 + (p_y - t_y) ** 2 + (p_z - t_z) ** 2)

                print(f"📍 真实坐标 (GT) : X={t_x:>6.2f}, Y={t_y:>6.2f}, Z={t_z:>6.2f}")
                print(f"🎯 预测坐标 (Pred): X={p_x:>6.2f}, Y={p_y:>6.2f}, Z={p_z:>6.2f}")
                print(f"📐 坐标空间绝对误差: {error_dist:.2f} 米")
                print(f"🏷️ 真实类别: {targets_cls_np[i]} | 预测类别: {pred_cls_labels[i]}")

            break  # 测试完一个 Batch 就可以退出了


if __name__ == "__main__":
    evaluate_model()