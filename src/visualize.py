# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from PIL import Image

# 1. 导入我们刚刚解耦的几何工具箱
from utils.calibration import load_camera_parameters, compose_extrinsic_matrix, project_lidar_to_image
# 2. 导入你的网络模型
from model import UAVFusionNet
# 3. 导入 Dataset 中写好的预处理方法 (保证喂给模型的数据和训练时完全一致)
from dataset import UAVFusionDataset


def main():
    # ================= 1. 配置路径 =================
    data_root = "/media/hzbz/dataset/data/mmaud"  # 你的 Ubuntu 路径
    csv_file = os.path.join(data_root, "train", "dataset_index_20m.csv")
    model_weights = "../weights/uav_fusion_baseline.pth"

    # 假设我们只可视化左相机的视野
    left_calib_file = os.path.join(data_root, "fisheye_calibration", "left", "we_want_rgb-camchain.yaml")

    # ================= 2. 加载基础参数与模型 =================
    # 加载相机内参和外参
    left_xi, left_intrinsics, left_distortion = load_camera_parameters(left_calib_file)
    left_extrinsics = compose_extrinsic_matrix(0, 0, 0, 0, 0, 0)  # 假设左图无偏移

    # 初始化 Dataset (只为了白嫖它的预处理逻辑)
    dataset = UAVFusionDataset(csv_file=csv_file, num_points=1024)

    # 初始化网络并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVFusionNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()  # 开启推理模式

    print("[INFO] 模型和参数加载完毕，开始可视化...")

    # ================= 3. 随机抽取样本进行推理与渲染 =================
    # 我们随便抽几个数据，比如第 10, 50, 100 行
    test_indices = [10, 50, 100, 200]

    for idx in test_indices:
        record = dataset.data_records[idx]
        img_path = record["image_path"]
        pcd_path = record["lidar_path"]

        # --- A. 为【模型推理】准备 Tensor 数据 ---
        # 这一步调用 dataset_item()，它会返回 Resize 过、归一化过的 Tensor
        img_tensor, pcd_tensor, target_xyz, target_cls = dataset[idx]

        # 增加 Batch 维度并送到 GPU
        img_tensor = img_tensor.unsqueeze(0).to(device)
        pcd_tensor = pcd_tensor.unsqueeze(0).to(device)

        # 让 AI 进行预测
        with torch.no_grad():
            pred_xyz, _ = model(img_tensor, pcd_tensor)
        pred_xyz_np = pred_xyz[0].cpu().numpy()  # 取出预测的 [X, Y, Z]
        gt_xyz_np = target_xyz.numpy()  # 取出真实的 [X, Y, Z]

        # --- B. 为【OpenCV 画图】准备原始高清数据 ---
        display_image = cv2.imread(img_path)
        # 注意：画雷达点云，要用原始包含全部点的点云，不能用采样后的 1024 个点
        raw_lidar_points = np.load(pcd_path)[:, :3]

        h, w = display_image.shape[:2]
        half_w = w // 2  # 因为原图是左右双目拼在一起的，左图只占左半边

        # 1. 画雷达点云投影 (绿色小点)
        left_pixels, valid_mask = project_lidar_to_image(
            raw_lidar_points, left_xi, left_intrinsics, left_distortion, left_extrinsics)

        depths = raw_lidar_points[valid_mask][:, 2]  # 取出 Z 值用于计算深度颜色
        for i, (px, py) in enumerate(left_pixels):
            u, v = int(px), int(py)
            if 0 <= u < half_w and 0 <= v < h:
                # 简单弄个颜色渐变，距离越近越红，远越绿
                depth = depths[i]
                color = (0, int(255 * (1 - min(depth, 30) / 30)), int(255 * min(depth, 30) / 30))
                cv2.circle(display_image, (u, v), 1, color, -1)

        # 2. 画真实 GT 中心点 (黄色实心圆)
        gt_points = np.array([gt_xyz_np])
        gt_pixels, _ = project_lidar_to_image(
            gt_points, left_xi, left_intrinsics, left_distortion, left_extrinsics)
        if len(gt_pixels) > 0:
            gu, gv = int(gt_pixels[0][0]), int(gt_pixels[0][1])
            if 0 <= gu < half_w and 0 <= gv < h:
                cv2.circle(display_image, (gu, gv), 6, (0, 255, 255), -1)

        # 3. 画 AI 预测框 (红色空心方框 + 预测距离)
        pred_points = np.array([pred_xyz_np])
        pred_pixels, _ = project_lidar_to_image(
            pred_points, left_xi, left_intrinsics, left_distortion, left_extrinsics)

        if len(pred_pixels) > 0:
            pu, pv = int(pred_pixels[0][0]), int(pred_pixels[0][1])
            if 0 <= pu < half_w and 0 <= pv < h:
                pred_z = pred_xyz_np[2]
                # 近大远小：假设 10米外框大小为 50 像素
                box_size = int(50 * (10.0 / pred_z))
                box_size = max(20, min(box_size, 200))  # 限制最大最小框

                tl = (pu - box_size // 2, pv - box_size // 2)
                br = (pu + box_size // 2, pv + box_size // 2)

                # 画红色预测框
                cv2.rectangle(display_image, tl, br, (0, 0, 255), 2)

                # 画准星
                cv2.line(display_image, (pu - 10, pv), (pu + 10, pv), (0, 0, 255), 1)
                cv2.line(display_image, (pu, pv - 10), (pu, pv + 10), (0, 0, 255), 1)

                # 写上预测距离和真实距离做对比
                error_dist = np.linalg.norm(pred_xyz_np - gt_xyz_np)
                text = f"Pred Z: {pred_z:.1f}m | Error: {error_dist:.2f}m"
                cv2.putText(display_image, text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- C. 展示图像 ---
        # 裁剪出左半边 (如果你想看全图可以把这行注释掉)
        display_left = display_image[:, :half_w]

        cv2.imshow("AI Prediction vs Ground Truth", display_left)
        print(f"当前展示 Idx {idx} | 按下任意键继续下一张，按 'q' 退出")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()