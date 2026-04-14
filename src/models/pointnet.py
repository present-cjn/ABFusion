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


class EnhancedPointNet(nn.Module):
    def __init__(self, out_channels=512):
        super(EnhancedPointNet, self).__init__()
        # 1. 浅层点特征提取
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # 2. 深层特征提取 (注意这里的输入维度是 64 + 64 = 128)
        # 这是为了接收拼接后的局部+全局特征
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, out_channels, 1)

        # 批归一化 (防止梯度爆炸，加速收敛)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x 预期形状: (B, N, 3)
        # PyTorch 的 Conv1d 需要特征在第二个维度: (B, 3, N)
        x = x.transpose(1, 2)
        B, _, N = x.size()

        # --- 阶段 1：提取各个点的浅层特征 ---
        x_local = F.relu(self.bn1(self.conv1(x)))
        x_local = F.relu(self.bn2(self.conv2(x_local)))  # 输出形状: (B, 64, N)

        # --- 阶段 2：提取局部全局特征并回传给每个点 (核心魔法 ✨) ---
        # 找出现有 64 维特征里的最强音
        global_feat_partial = torch.max(x_local, 2, keepdim=True)[0]  # (B, 64, 1)
        # 复制 N 份，准备发给每个点
        global_feat_expand = global_feat_partial.repeat(1, 1, N)  # (B, 64, N)

        # 拼接！每个点现在的信息是：[自身特征(64维), 所在环境特征(64维)]
        x_concat = torch.cat([x_local, global_feat_expand], dim=1)  # (B, 128, N)

        # --- 阶段 3：深层特征提取 (带有上下文感知能力) ---
        x_deep = F.relu(self.bn3(self.conv3(x_concat)))
        x_deep = F.relu(self.bn4(self.conv4(x_deep)))
        x_deep = F.relu(self.bn5(self.conv5(x_deep)))  # (B, 512, N)

        # --- 阶段 4：最终的全局最大池化 ---
        out_feat = torch.max(x_deep, 2, keepdim=False)[0]  # (B, 512)

        return out_feat