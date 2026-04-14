import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random


class UAVFusionDataset(Dataset):
    def __init__(self, csv_file, num_points=1024, image_size=(224, 224), is_train=True):
        """
        Args:
            csv_file (str): 清洗后的索引文件路径 (例如 dataset_index_20m.csv)
            num_points (int): 每帧点云固定采样多少个点，保证 Batch 组装时不报错
            image_size (tuple): 图像缩放的目标大小
        """
        self.data_records = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data_records.append(row)

        self.num_points = num_points

        # 定义图像的预处理流水线
        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # 转为 [C, H, W] 并归一化到 [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准归一化
        ])

        self.is_train = is_train
        # 定义 Dropout 概率：15% 没图像，15% 没雷达，剩下 70% 都有
        self.drop_img_prob = 0.15
        self.drop_pc_prob = 0.15

    def __len__(self):
        return len(self.data_records)

    def _process_point_cloud(self, pcd_path):
        """
        处理点云：随机采样或补齐到固定的 self.num_points
        """
        # 加载点云，只取前 3 列 (X, Y, Z)
        points = np.load(pcd_path)[:, :3]
        num_current_points = points.shape[0]

        if num_current_points == 0:
            # 极端情况：如果遇到空点云，返回全 0 的 Tensor
            return np.zeros((self.num_points, 3), dtype=np.float32)

        if num_current_points >= self.num_points:
            # 随机下采样 (无放回)
            choice_idx = np.random.choice(num_current_points, self.num_points, replace=False)
        else:
            # 随机补齐 (有放回，允许重复采同一个点)
            choice_idx = np.random.choice(num_current_points, self.num_points, replace=True)

        sampled_points = points[choice_idx, :]
        return sampled_points.astype(np.float32)

    def __getitem__(self, idx):
        record = self.data_records[idx]

        # 1. 加载并处理图像
        img_path = record["image_path"]
        # 使用 PIL 读取图像，转换为 RGB (防某些黑白图报错)
        image = Image.open(img_path).convert("RGB")

        # === 动态裁剪左半边图 ===
        w, h = image.size
        # Image.crop 的参数是 (left, upper, right, lower)
        left_image = image.crop((0, 0, w // 2, h))

        image_tensor = self.img_transform(left_image)

        # 2. 加载并处理点云
        pcd_path = record["lidar_path"]
        points_np = self._process_point_cloud(pcd_path)
        points_tensor = torch.from_numpy(points_np)  # 形状: [num_points, 3]

        # 3. 加载真值目标 (Ground Truth XYZ)
        gt_x = float(record["ground_truth_x"])
        gt_y = float(record["ground_truth_y"])
        gt_z = float(record["ground_truth_z"])
        target_xyz = torch.tensor([gt_x, gt_y, gt_z], dtype=torch.float32)

        # 4. 加载分类标签 (暂时不重要，但先留着位置)
        class_id = int(record["class_id"]) if record["class_id"] else 0
        target_cls = torch.tensor(class_id, dtype=torch.long)
        # === 核心：模态 Dropout 逻辑 (仅在训练时开启) ===
        if self.is_train:
            rand_val = random.random()  # 生成 0 到 1 之间的随机数

            if rand_val < self.drop_img_prob:
                # 概率 [0.0, 0.15): 模拟视觉致盲 (把图像张量全变 0)
                image_tensor = torch.zeros_like(image_tensor)

            elif rand_val < (self.drop_img_prob + self.drop_pc_prob):
                # 概率 [0.15, 0.30): 模拟雷达失效 (把点云张量全变 0)
                points_tensor = torch.zeros_like(points_tensor)

            else:
                # 概率 [0.30, 1.0]: 正常双模态，啥也不做
                pass

        return image_tensor, points_tensor, target_xyz, target_cls


# ================= 测试你的 DataLoader =================
if __name__ == "__main__":
    # 替换为你刚才生成的 20m 数据索引的绝对路径
    CSV_PATH = "/media/hzbz/dataset/data/mmaud/train/dataset_index_20m.csv"

    # 实例化 Dataset
    dataset = UAVFusionDataset(csv_file=CSV_PATH, num_points=1024, image_size=(224, 224))
    print(f"[INFO] 成功加载数据集，共包含 {len(dataset)} 个有效帧")

    # 包装成 DataLoader (设置 batch_size 为 4，打乱数据)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # 抽取第一个 Batch 出来检查
    for batch_idx, (images, pcds, targets_xyz, targets_cls) in enumerate(dataloader):
        print("\n--- 第 1 个 Batch 的数据维度检查 ---")
        print(f"图像张量 (Images):      {images.shape}")  # 期望: [4, 3, 224, 224]
        print(f"点云张量 (PointClouds): {pcds.shape}")  # 期望: [4, 1024, 3]
        print(f"位置真值 (GT XYZ):      {targets_xyz.shape}")  # 期望: [4, 3]
        print(f"类别真值 (GT Class):    {targets_cls.shape}")  # 期望: [4]

        # 打印一下第一个 batch 的第一个 GT 坐标看看
        print(f"\n第一个样本的目标 XYZ 坐标: {targets_xyz[0].numpy()}")
        break  # 测试只看一个 batch 就够了