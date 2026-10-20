"""Reconstruccion de la nube local a partir de disparidad."""

import cv2
import numpy as np
import open3d as o3d


def compute_local_cloud(disparity, left_rect, Q):
    """Devuelve la nube coloreada en el marco de la camara izquierda rectificada."""
    points = cv2.reprojectImageTo3D(disparity, Q)
    valid = (disparity > 0) & np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
    colors = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[valid].astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors[valid] / 255.0)
    return cloud


def transform_cloud_to_world(local_cloud, camera_pose, R1):
    """Lleva una copia de la nube rectificada al sistema del tablero."""
    rectified_to_camera = np.eye(4)
    rectified_to_camera[:3, :3] = R1.T
    world_cloud = o3d.geometry.PointCloud(local_cloud)
    world_cloud.transform(camera_pose @ rectified_to_camera)
    return world_cloud
