# -*- coding: utf-8 -*-
"""
File: config.py
Describe: 全局配置中心，存放所有路径、超参数和传感器外参经验值
"""
import os


class Config:
    # ================= 1. 路径配置 =================
    # 数据根目录
    DATA_ROOT = "/media/hzbz/dataset/data/mmaud"

    # 训练/验证相关路径
    TRAIN_CSV = os.path.join(DATA_ROOT, "train", "dataset_index_20m.csv")

    # 模型权重保存路径
    WEIGHTS_DIR = "../weights"  # 相对 src 的上一级
    BASELINE_WEIGHTS = os.path.join(WEIGHTS_DIR, "uav_fusion_baseline.pth")

    # 标定文件路径
    LEFT_CALIB_YAML = os.path.join(DATA_ROOT, "fisheye_calibration", "left", "we_want_rgb-camchain.yaml")
    RIGHT_CALIB_YAML = os.path.join(DATA_ROOT, "fisheye_calibration", "right", "we_want_rgb-camchain.yaml")

    # ================= 2. 训练超参数配置 =================
    NUM_POINTS = 1024  # 点云采样数
    IMAGE_SIZE = (224, 224)  # 图像 Resize 大小
    BATCH_SIZE = 16  # Y9000P 显存够大，可以开到 16 或 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4

    # ================= 3. 传感器外参 (经验值/微调值) =================
    # 假设 LiDAR 原点就是左相机原点
    LEFT_EXTRINSICS = {
        "tx": 0.0, "ty": 0.0, "tz": 0.0,
        "rx": 0.0, "ry": 0.0, "rz": 0.0
    }

    # 右相机相对于左相机的偏移 (例如向右平移 15cm，绕 Y 轴可能有一点偏航角)
    RIGHT_EXTRINSICS = {
        "tx": -15.0, "ty": 0.0, "tz": 0.0,  # 单位: 厘米
        "rx": 0.0, "ry": 0.0, "rz": 0.0  # 单位: 度
    }


# 实例化一个全局配置对象供其他文件导入
cfg = Config()