import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ================= 1. 点云分支：极简版 PointNet =================
class SimplePointNet(nn.Module):
    def __init__(self, out_channels=512):
        super(SimplePointNet, self).__init__()
        # 使用 1D 卷积对点云升维 (输入通道为 3，即 x, y, z)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_channels, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # PyTorch 的 Conv1d 期望的输入形状是 [Batch, Channels, Points]
        # DataLoader 给的是 [Batch, Points, 3]，所以需要转置一下
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # 形状: [Batch, 512, Points]

        # 全局最大池化 (Global Max Pooling)：提取每个维度的最强特征
        x = torch.max(x, 2, keepdim=True)[0]  # 形状: [Batch, 512, 1]
        x = x.view(-1, 512)  # 展平为 [Batch, 512]
        return x


# ================= 2. 双流融合主网络 =================
class UAVFusionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UAVFusionNet, self).__init__()

        # --- 视觉分支 ---
        # 加载预训练的 ResNet18
        # resnet = models.resnet18(pretrained=True)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 砍掉最后一层全连接层 (fc)，保留前面的特征提取层
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feat_dim = 512  # ResNet18 的输出维度是 512

        # --- 点云分支 ---
        self.lidar_encoder = SimplePointNet(out_channels=512)
        self.lidar_feat_dim = 512

        # --- 融合特征层 ---
        # 拼接后的总维度：512 (图像) + 512 (点云) = 1024
        combined_dim = self.image_feat_dim + self.lidar_feat_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout 防止过拟合
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # --- 任务输出头 (Dual Heads) ---
        # 1. 3D 位置回归头 (输出 X, Y, Z 三个值)
        self.xyz_head = nn.Linear(256, 3)

        # 2. 分类头 (输出无人机类别的 logits)
        self.cls_head = nn.Linear(256, num_classes)

    def forward(self, images, pointclouds):
        # 1. 提取图像特征
        # 输入: [B, 3, 224, 224] -> 输出: [B, 512, 1, 1]
        img_feats = self.image_encoder(images)
        img_feats = img_feats.view(img_feats.size(0), -1)  # 展平为 [B, 512]

        # 2. 提取点云特征
        # 输入: [B, 1024, 3] -> 输出: [B, 512]
        pc_feats = self.lidar_encoder(pointclouds)

        # 3. 特征级拼接 (Feature Concatenation)
        # 在列维度 (dim=1) 把它们拼起来 -> 形状: [B, 1024]
        fused_feats = torch.cat((img_feats, pc_feats), dim=1)

        # 4. 融合计算
        shared_feats = self.fusion_mlp(fused_feats)  # 形状: [B, 256]

        # 5. 输出结果
        pred_xyz = self.xyz_head(shared_feats)  # 形状: [B, 3]
        pred_cls = self.cls_head(shared_feats)  # 形状: [B, 2]

        return pred_xyz, pred_cls


# ================= 3. 模拟测试 =================
if __name__ == "__main__":
    # 创建模型实例
    model = UAVFusionNet(num_classes=2)

    # 模拟我们在 dataloader 里拿到的数据维度 (Batch Size = 4)
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_pointclouds = torch.randn(4, 1024, 3)

    # 将模拟数据喂给模型
    print("[INFO] 开始进行模型前向传播测试...")
    out_xyz, out_cls = model(dummy_images, dummy_pointclouds)

    print("\n--- 模型输出维度检查 ---")
    print(f"预测位置 (Predicted XYZ):   {out_xyz.shape} -> 期望是 [4, 3]")
    print(f"预测类别 (Predicted Class): {out_cls.shape} -> 期望是 [4, 2]")

    # 打印一下第一个样本的模型初始瞎猜的结果
    print(f"\n第一个样本的模型初始瞎猜 XYZ: {out_xyz[0].detach().numpy()}")
    print("[SUCCESS] 模型架构运行完美，没有任何维度不匹配的问题！")