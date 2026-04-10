import torch
import torch.nn as nn
import torch.nn.functional as F

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
