"""Procesamiento de todas las capturas, una nube a la vez."""

from pathlib import Path

import cv2
import numpy as np

from .calibration import load_pair, rectify_pair
from .disparity import compute_disparity
from .pose import estimate_camera_pose
from .reconstruction import compute_local_cloud, transform_cloud_to_world


def reconstruct_dataset(data_dir, calibration, maps, board, max_pose_rms=5.0):
    """Genera una nube global por captura o informa el motivo del rechazo."""
    directory = Path(data_dir) / "captures"
    files = sorted(directory.glob("left_*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"No hay capturas en {directory}")
    for path in files:
        number = int(path.stem.split("_")[-1])
        result = {"capture": number}
        try:
            left, right = load_pair(directory, number)
            _, _, pose, rms = estimate_camera_pose(
                left, board, calibration["left_K"], calibration["left_dist"]
            )
            result["pose_rms"] = rms
            if rms > max_pose_rms:
                raise ValueError(f"RMS de pose {rms:.2f} px supera {max_pose_rms:g} px")
            left_rect, right_rect = rectify_pair(left, right, maps)
            disparity = compute_disparity(left_rect, right_rect)
            cloud = compute_local_cloud(disparity, left_rect, maps["Q"])
            if len(cloud.points) == 0:
                raise ValueError("No se obtuvieron puntos 3D validos")
            result["cloud"] = transform_cloud_to_world(cloud, pose, maps["R1"])
            result["camera_pose"] = pose
            result["points"] = len(cloud.points)
            result["coverage"] = float(np.mean(disparity > 0))
        except (ValueError, FileNotFoundError, cv2.error) as error:
            result["error"] = str(error)
        yield result
