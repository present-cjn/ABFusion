# -*- coding: utf-8 -*-
"""
File: calibration.py
Describe: 存放相机内外参解析、坐标系转换与投影的核心数学函数
"""
import numpy as np
import yaml

def load_camera_parameters(calibration_file):
    with open(calibration_file, 'r') as file:
        calibration_data = yaml.safe_load(file)

    cam_key = None
    for key in calibration_data.keys():
        if key.startswith('cam'):
            cam_key = key
            break

    if not cam_key:
        raise ValueError(f"未在文件 {calibration_file} 中找到 cam 字段")

    intrinsics = calibration_data[cam_key]['intrinsics']
    xi, fx, fy, cx, cy = intrinsics

    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion = np.array(calibration_data[cam_key]['distortion_coeffs'])
    return xi, intrinsic, distortion

def compose_extrinsic_matrix(tx, ty, tz, rx, ry, rz):
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