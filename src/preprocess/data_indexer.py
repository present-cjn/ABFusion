# -*- coding: utf-8 -*-
"""
Time:     2026/4/9 10:50
Author:   present
Version:  1.0.0
File:     data_indexer.py
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
    return None  # 如果目标时间戳不在范围内

def index_dataset(data_root, output_csv, output_json):
    """
    遍历数据集目录，提取有效帧信息并生成索引文件。
    """
    records = []
    lidar_files = []
    ground_truth_data = {}

    # 直接处理 train/SeqXXXX 目录
    for seq_dir in sorted(os.listdir(data_root)):
        seq_path = os.path.join(data_root, seq_dir)
        if not os.path.isdir(seq_path) or not seq_dir.startswith("Seq"):
            print(f"[DEBUG] Skipping non-sequence directory: {seq_dir}")
            continue

        # 收集 ground_truth 数据
        ground_truth_dir = os.path.join(seq_path, "ground_truth")
        for file in os.listdir(ground_truth_dir):
            if file.endswith(".npy"):
                timestamp = float(os.path.splitext(file)[0])
                ground_truth_path = os.path.join(ground_truth_dir, file)
                ground_truth_array = np.load(ground_truth_path)
                ground_truth_data[timestamp] = {"x": ground_truth_array[0], "y": ground_truth_array[1], "z": ground_truth_array[2]}
                print(f"[DEBUG] Added ground truth: {timestamp} -> {ground_truth_data[timestamp]}")

        # 收集 LiDAR 数据
        lidar_dir = os.path.join(seq_path, "lidar_360")
        for file in os.listdir(lidar_dir):
            if file.endswith(".npy"):
                lidar_files.append(os.path.join(lidar_dir, file))
                print(f"[DEBUG] Added LiDAR file: {os.path.join(lidar_dir, file)}")

    # 遍历每个 SeqXXXX 目录
    for seq_dir in sorted(os.listdir(data_root)):
        seq_path = os.path.join(data_root, seq_dir)
        if not os.path.isdir(seq_path) or not seq_dir.startswith("Seq"):
            print(f"[DEBUG] Skipping non-sequence directory: {seq_dir}")
            continue

        # 处理 Image 文件夹
        image_dir = os.path.join(seq_path, "Image")
        for file in os.listdir(image_dir):
            if file.endswith(".png"):
                image_path = os.path.join(image_dir, file)
                timestamp = float(os.path.splitext(file)[0])

                # 找到最近的 LiDAR 文件
                lidar_file = find_closest_file(timestamp, lidar_files)
                if not lidar_file:
                    print(f"[DEBUG] No LiDAR file found for timestamp {timestamp}")
                    continue

                # 插值计算真值
                ground_truth = interpolate_ground_truth(timestamp, ground_truth_data)
                if not ground_truth:
                    print(f"[DEBUG] No ground truth found for timestamp {timestamp}")
                    continue

                # 加载分类信息
                class_file = os.path.join(seq_path, "class", f"{timestamp}.npy")
                if os.path.exists(class_file):
                    print(f"[DEBUG] Found class file for timestamp {timestamp}: {class_file}")
                    class_id = int(np.load(class_file))
                else:
                    class_id = None

                # 计算 LiDAR 时间偏移
                lidar_delay = float(os.path.splitext(os.path.basename(lidar_file))[0]) - timestamp

                record = {
                    "timestamp": timestamp,
                    "image_path": image_path,
                    "lidar_path": lidar_file,
                    "ground_truth_x": ground_truth["x"],
                    "ground_truth_y": ground_truth["y"],
                    "ground_truth_z": ground_truth["z"],
                    "class_id": class_id,
                    "lidar360_delay": lidar_delay
                }
                print(f"[DEBUG] Added record: {record}")
                records.append(record)

    # 写入 CSV 文件
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["timestamp", "image_path", "lidar_path", "class_id", "lidar360_delay", "ground_truth_x", "ground_truth_y", "ground_truth_z"])
        writer.writeheader()
        writer.writerows(records)

    # 写入 JSON 文件
    with open(output_json, mode="w", encoding="utf-8") as json_file:
        json.dump(records, json_file, indent=4, ensure_ascii=False)

    print(f"[INFO] 索引已生成：{output_csv} 和 {output_json}")

if __name__ == "__main__":
    # 数据集根目录
    data_root = os.path.join(r"E:\\data\\mmaud", "train")

    # 输出文件路径
    output_csv = os.path.join(data_root, "dataset_index.csv")
    output_json = os.path.join(data_root, "dataset_index.json")

    # 生成索引
    index_dataset(data_root, output_csv, output_json)