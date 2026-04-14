import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.uav_dataset import UAVFusionDataset
from models.fusion_net import UAVFusionNet
from config import cfg


def evaluate_robustness(weights_path, test_mode="normal"):
    """
    test_mode 可以是:
    "normal" (双模态正常)
    "no_radar" (雷达失效，全靠视觉)
    "no_image" (视觉失效，全靠雷达)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 强制 is_train=False，保证 Dataset 内部不发生随机抹除
    val_dataset = UAVFusionDataset(csv_file=cfg.TRAIN_CSV, num_points=cfg.NUM_POINTS, is_train=False)

    # 为了公平对比，取前面固定的验证集切分（如果你之前固定了 random seed，这里取后 20%）
    import random
    indices = list(range(len(val_dataset)))
    random.seed(42)
    random.shuffle(indices)
    val_idx = indices[:int(0.2 * len(indices))]
    val_dataset.data_records = [val_dataset.data_records[i] for i in val_idx]

    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = UAVFusionNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    criterion_xyz = nn.MSELoss()
    total_mse = 0.0

    with torch.no_grad():
        for images, pointclouds, targets_xyz, _ in val_loader:
            images = images.to(device)
            pointclouds = pointclouds.to(device)
            targets_xyz = targets_xyz.to(device)

            # ==== 核心考题：人工制造传感器失效 ====
            if test_mode == "no_radar":
                pointclouds = torch.zeros_like(pointclouds)
            elif test_mode == "no_image":
                images = torch.zeros_like(images)

            pred_xyz, _ = model(images, pointclouds)
            loss_xyz = criterion_xyz(pred_xyz, targets_xyz)
            total_mse += loss_xyz.item()

    avg_mse = total_mse / len(val_loader)
    print(f"权重: {weights_path.split('/')[-1]} | 测试模式: {test_mode:10s} | 最终 XYZ MSE: {avg_mse:.4f} 米")


if __name__ == "__main__":
    exp_weights = "./runs/exp4_enhanced_pointnet_4096pts/best.pth"

    print("========== 1. 正常环境 (双模态) ==========")
    evaluate_robustness(exp_weights, "normal")

    print("\n========== 2. 极端环境 (雷达完全失效，只剩图像) ==========")
    evaluate_robustness(exp_weights, "no_radar")

    print("\n========== 3. 极端环境 (相机完全失效，只剩雷达) ==========")
    evaluate_robustness(exp_weights, "no_image")