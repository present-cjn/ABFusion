# -*- coding: utf-8 -*-
"""
Time:     2026/4/9 10:55
Author:   present
Version:  1.0.0
File:     align_lidar_image.py
Describe:
"""
import os
import cv2
import numpy as np
import yaml
import csv


def load_camera_parameters(calibration_file):
    """
    从yaml文件加载相机内参和畸变参数。
    加入了兼容性处理，自动寻找 'cam0' 或 'cam1' 键值。
    """
    with open(calibration_file, 'r') as file:
        calibration_data = yaml.safe_load(file)

    # 自动寻找相机数据的键值（防止右侧标定文件里写的是 cam1）
    cam_key = None
    for key in calibration_data.keys():
        if key.startswith('cam'):
            cam_key = key
            break

    if not cam_key:
        raise ValueError(f"未在文件 {calibration_file} 中找到 cam 字段")

    intrinsics = calibration_data[cam_key]['intrinsics']
    xi, fx, fy, cx, cy = intrinsics

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    distortion = np.array(calibration_data[cam_key]['distortion_coeffs'])
    return xi, intrinsic, distortion


def compose_extrinsic_matrix(tx, ty, tz, rx, ry, rz):
    """
    tx, ty, tz 单位为 cm
    rx, ry, rz 单位为度
    """
    t_vec = np.array([[tx / 100.0], [ty / 100.0], [tz / 100.0]])
    rx_rad, ry_rad, rz_rad = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)

    Rx = np.array([[1, 0, 0], [0, np.cos(rx_rad), -np.sin(rx_rad)], [0, np.sin(rx_rad), np.cos(rx_rad)]])
    Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)], [0, 1, 0], [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
    Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0], [np.sin(rz_rad), np.cos(rz_rad), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = t_vec
    return T


def project_lidar_to_image(points_3d, xi, intrinsics, distortion, extrinsics):
    points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_cam = (extrinsics @ points_homo.T).T
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # 滤除相机背后的点
    valid_mask = Z > 0.05
    X, Y, Z = X[valid_mask], Y[valid_mask], Z[valid_mask]

    if len(X) == 0:
        return np.array([]), np.array([])

    rho = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    X_s, Y_s, Z_s = X / rho, Y / rho, Z / rho

    denom = Z_s + xi
    u_n, v_n = X_s / denom, Y_s / denom

    k1, k2, p1, p2 = distortion
    r2 = u_n ** 2 + v_n ** 2
    r4 = r2 ** 2
    u_d = u_n * (1 + k1 * r2 + k2 * r4) + 2 * p1 * u_n * v_n + p2 * (r2 + 2 * u_n ** 2)
    v_d = v_n * (1 + k1 * r2 + k2 * r4) + 2 * p2 * u_n * v_n + p1 * (r2 + 2 * v_n ** 2)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    pixels = np.stack((fx * u_d + cx, fy * v_d + cy), axis=1)
    return pixels, valid_mask


def main():
    # ================= 1. 配置区域 =================
    index_file = r"E:\data\mmaud\train\dataset_index_20m.csv"

    # 左右相机标定文件路径
    left_calib_file = os.path.normpath(r"E:\data\mmaud\fisheye_calibration\left\we_want_rgb-camchain.yaml")
    # 假设右相机的文件名也是一样的，如果不同请在此修改
    right_calib_file = os.path.normpath(r"E:\data\mmaud\fisheye_calibration\right\we_want_rgb-camchain.yaml")

    if not os.path.exists(index_file):
        print(f"[错误] 找不到索引文件: {index_file}")
        return
    if not os.path.exists(left_calib_file) or not os.path.exists(right_calib_file):
        print("[错误] 找不到左侧或右侧的标定文件，请检查路径。")
        return

    # 解析 CSV 文件
    data_records = []
    with open(index_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        f.seek(0)

        reader = csv.DictReader(f, delimiter=delimiter)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            data_records.append(row)

    if not data_records:
        print("[错误] 索引文件为空或解析失败！")
        return

    print(f"[INFO] 成功加载 {len(data_records)} 条数据记录。")

    # 分别加载左右相机的内参
    left_xi, left_intrinsics, left_distortion = load_camera_parameters(left_calib_file)
    right_xi, right_intrinsics, right_distortion = load_camera_parameters(right_calib_file)

    # 状态控制器
    state = {
        "current_idx": 0,
        "need_load_data": True
    }

    # === 【关键物理参数：外参配置】 ===
    # 以左相机为原点基准
    tx, ty, tz = 0, 0, 0
    rx, ry, rz = 0, 0, 0
    left_extrinsics = compose_extrinsic_matrix(tx, ty, tz, rx, ry, rz)

    # 【基线设置】：如果已知左右相机的距离（例如 12 cm），请设置 right_baseline_x = 12
    # 这意味着相对于左相机，右相机在 X 轴向右平移了 12cm
    right_baseline_x = 120
    right_extrinsics = compose_extrinsic_matrix(tx - right_baseline_x, ty, tz, rx, ry, rz)

    # ================= 2. 创建UI =================
    win_name = "Stereo Lidar-Camera Data Viewer"
    cv2.namedWindow(win_name, 0)
    cv2.resizeWindow(win_name, 1280, 720)

    print("[INFO] 交互说明:")
    print("  - 按 'N' 键切换到下一组数据")
    print("  - 按 'P' 键切换到上一组数据")
    print("  - 按 'S' 键保存当前画面的截图")
    print("  - 按 'Q' 键退出程序")

    # 在循环外初始化一个空白图像
    display_image = np.zeros((720, 1280, 3), dtype=np.uint8)

    # ================= 3. 主渲染循环 =================
    while True:
        record = data_records[state["current_idx"]]

        # 仅在需要切换数据时，执行高强度的计算和绘图 (解决 PyCharm 卡顿问题)
        if state["need_load_data"]:
            img_file = record.get("image_path", "").strip()
            pcd_file = record.get("lidar_path", "").strip()

            try:
                # 1. 获取 GT (如果有的话)
                gt_x = float(record.get("ground_truth_x", 0))
                gt_y = float(record.get("ground_truth_y", 0))
                gt_z = float(record.get("ground_truth_z", 0))
                current_gt = np.array([[gt_x, gt_y, gt_z]])

                # 2. 读取当前图片和点云
                current_image = cv2.imread(img_file)
                if current_image is None:
                    raise FileNotFoundError(f"无法读取图像: {img_file}")

                current_lidar_points = np.load(pcd_file)[:, :3]

                # 3. 准备绘制用的画布
                display_image = current_image.copy()
                h, w = display_image.shape[:2]
                half_w = w // 2

                # 4. 计算并绘制点云 (兵分两路)
                if len(current_lidar_points) > 0:
                    # [左图投影] - 使用左相机内外参
                    left_pixels, _ = project_lidar_to_image(current_lidar_points, left_xi, left_intrinsics,
                                                            left_distortion, left_extrinsics)
                    for px, py in left_pixels:
                        u, v = int(px), int(py)
                        if 0 <= u < half_w and 0 <= v < h:
                            cv2.circle(display_image, (u, v), 2, (0, 255, 0), -1)

                            # [右图投影] - 使用右相机内外参
                    right_pixels, _ = project_lidar_to_image(current_lidar_points, right_xi, right_intrinsics,
                                                             right_distortion, right_extrinsics)
                    for px, py in right_pixels:
                        u, v = int(px), int(py)
                        if 0 <= u < half_w and 0 <= v < h:
                            # 必须加上 half_w 偏移量，因为右图画在画布的右半边
                            cv2.circle(display_image, (u + half_w, v), 2, (255, 0, 0), -1)

                            # 5. 计算并绘制 Ground Truth
                if len(current_gt) > 0:
                    # GT 投左边
                    gt_left_pix, _ = project_lidar_to_image(current_gt, left_xi, left_intrinsics, left_distortion,
                                                            left_extrinsics)
                    if len(gt_left_pix) > 0:
                        gu, gv = int(gt_left_pix[0][0]), int(gt_left_pix[0][1])
                        if 0 <= gu < half_w and 0 <= gv < h:
                            cv2.circle(display_image, (gu, gv), 8, (0, 255, 255), 3)

                            # GT 投右边
                    gt_right_pix, _ = project_lidar_to_image(current_gt, right_xi, right_intrinsics, right_distortion,
                                                             right_extrinsics)
                    if len(gt_right_pix) > 0:
                        gu, gv = int(gt_right_pix[0][0]), int(gt_right_pix[0][1])
                        if 0 <= gu < half_w and 0 <= gv < h:
                            cv2.circle(display_image, (gu + half_w, gv), 8, (0, 255, 255), 3)

                            # 6. 绘制文本信息
                ts = record.get('timestamp', 'N/A')
                delay = record.get('lidar360_delay', 'N/A')

                if len(current_gt) > 0:
                    # 提取 xyz
                    gx, gy, gz = current_gt[0][0], current_gt[0][1], current_gt[0][2]

                    # 计算 3D 距离 (欧氏距离)
                    distance = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

                    # 格式化文本：保留两位小数，末尾加上单位 m
                    gt_text = f"GT: X={gx:.2f}, Y={gy:.2f}, Z={gz:.2f} | Dist: {distance:.2f}m"
                else:
                    gt_text = "GT: N/A"

                info_text = f"Idx: {state['current_idx']} / {len(data_records) - 1} | TS: {ts} | Delay: {delay}"

                cv2.putText(display_image, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_image, gt_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            except Exception as e:
                print(f"[加载或渲染失败] 行号 {state['current_idx']}: {e}")
                display_image = np.zeros((720, 1280, 3), dtype=np.uint8)

            # 计算和绘图全部完成，重置标志位，释放 CPU 占用
            state["need_load_data"] = False

        # === 极低负载显示循环 ===
        cv2.imshow(win_name, display_image)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            state["current_idx"] = min(state["current_idx"] + 20, len(data_records) - 1)
            state["need_load_data"] = True
        elif key == ord('p'):
            state["current_idx"] = max(state["current_idx"] - 1, 0)
            state["need_load_data"] = True
        elif key == ord('s'):
            save_path = f"viewer_result_idx_{state['current_idx']}.jpg"
            cv2.imwrite(save_path, display_image)
            print(f"[INFO] 图像已保存至: {os.path.abspath(save_path)}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()