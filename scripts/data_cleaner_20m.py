# -*- coding: utf-8 -*-
"""
Time:     2026/4/9 17:30
Author:   present
Version:  1.0.0
File:     data_cleaner_20m.py.py
Describe: 
"""
import os
import csv
import json
import numpy as np


def find_closest_file(target_timestamp, file_list, threshold_ms=50):
    """
    在文件列表中找到与目标时间戳最接近的文件。
    """
    closest_file = None
    min_diff = float("inf")
    for file in file_list:
        file_timestamp = float(os.path.splitext(os.path.basename(file))[0])
        diff = abs(file_timestamp - target_timestamp)
        if diff < min_diff and diff <= threshold_ms / 1000.0:
            min_diff = diff
            closest_file = file
    return closest_file


def interpolate_ground_truth(target_timestamp, gt_data):
    """
    对真值数据进行线性插值，计算目标时间戳的真值。
    """
    timestamps = sorted(gt_data.keys())
    for i in range(len(timestamps) - 1):
        t1, t2 = timestamps[i], timestamps[i + 1]
        if t1 <= target_timestamp <= t2:
            ratio = (target_timestamp - t1) / (t2 - t1)
            interpolated = {
                key: gt_data[t1][key] + ratio * (gt_data[t2][key] - gt_data[t1][key])
                for key in gt_data[t1]
            }
            return interpolated
    return None


def clean_and_index_dataset(data_root, output_csv, distance_threshold=20.0):
    """
    遍历数据集，过滤掉距离大于 threshold 的数据。
    """
    records = []

    # 1. 预收集所有序列的 LiDAR 和 GT 数据，建立全局查找字典
    # 注意：为了防止内存占用过大，我们按序列处理会更稳妥
    seq_dirs = sorted([d for d in os.listdir(data_root) if d.startswith("seq")])

    total_processed = 0
    total_filtered = 0

    for seq_dir in seq_dirs:
        seq_path = os.path.join(data_root, seq_dir)
        print(f"\n[INFO] 正在处理序列: {seq_dir}")

        # 收集当前序列的 GT
        gt_data = {}
        gt_dir = os.path.join(seq_path, "ground_truth")
        if os.path.exists(gt_dir):
            for file in os.listdir(gt_dir):
                if file.endswith(".npy"):
                    ts = float(os.path.splitext(file)[0])
                    arr = np.load(os.path.join(gt_dir, file))
                    gt_data[ts] = {"x": arr[0], "y": arr[1], "z": arr[2]}

        # 收集当前序列的 LiDAR
        lidar_dir = os.path.join(seq_path, "lidar_360")
        lidar_files = []
        if os.path.exists(lidar_dir):
            lidar_files = [os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if f.endswith(".npy")]

        # 遍历图像并匹配
        image_dir = os.path.join(seq_path, "Image")
        if not os.path.exists(image_dir):
            continue

        for file in os.listdir(image_dir):
            if file.endswith(".png"):
                img_ts = float(os.path.splitext(file)[0])

                # 匹配 LiDAR 和 GT 插值
                lidar_file = find_closest_file(img_ts, lidar_files)
                gt = interpolate_ground_truth(img_ts, gt_data)

                if lidar_file and gt:
                    # === 核心逻辑：计算 3D 距离 ===
                    dist = np.sqrt(gt['x'] ** 2 + gt['y'] ** 2 + gt['z'] ** 2)
                    total_processed += 1

                    # === 核心过滤：仅保留距离在阈值内的数据 ===
                    if dist <= distance_threshold:
                        # 加载分类
                        class_file = os.path.join(seq_path, "class", f"{img_ts}.npy")
                        class_id = int(np.load(class_file)) if os.path.exists(class_file) else None

                        lidar_ts = float(os.path.splitext(os.path.basename(lidar_file))[0])

                        records.append({
                            "timestamp": img_ts,
                            "image_path": os.path.join(image_dir, file),
                            "lidar_path": lidar_file,
                            "ground_truth_x": gt["x"],
                            "ground_truth_y": gt["y"],
                            "ground_truth_z": gt["z"],
                            "distance": round(dist, 3),  # 新增距离字段方便查看
                            "class_id": class_id,
                            "lidar360_delay": lidar_ts - img_ts
                        })
                        total_filtered += 1

    # 写入新的 CSV
    if records:
        keys = records[0].keys()
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

        print("-" * 30)
        print(f"[完成] 总计扫描帧数: {total_processed}")
        print(f"[完成] 符合条件(≤{distance_threshold}m)帧数: {total_filtered}")
        print(f"[完成] 过滤掉的帧数: {total_processed - total_filtered}")
        print(f"[完成] 清洗后的索引已存至: {output_csv}")
    else:
        print("[警告] 未找到符合距离条件的数据记录。")


if __name__ == "__main__":
    # 配置路径
    # data_root = r"E:\data\mmaud\train"
    data_root = "/media/hzbz/dataset/data/mmaud/train"
    # 新的 CSV 文件名，标明是 20m 清洗版
    output_csv_20m = os.path.join(data_root, "dataset_index_20m.csv")

    # 执行清洗
    clean_and_index_dataset(data_root, output_csv_20m, distance_threshold=20.0)