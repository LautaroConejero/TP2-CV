"""Pose de la camara respecto del tablero ChArUco."""

import cv2
import numpy as np


def estimate_camera_pose(image, board, K, dist, min_corners=8, min_area_ratio=0.02):
    """Devuelve rvec, tvec, pose camara a tablero y RMS de reproyeccion.

    Los minimos de esquinas y area visible son filtros practicos de calidad.
    """
    params = cv2.aruco.CharucoParameters()
    params.cameraMatrix = K
    params.distCoeffs = dist
    # La calidad se evalua con las esquinas, su cobertura y el RMS.
    params.checkMarkers = False
    params.minMarkers = 2
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board, params).detectBoard(image)
    count = 0 if ids is None else len(ids)
    if count < min_corners:
        raise ValueError(f"Esquinas ChArUco detectadas: {count}; minimo requerido: {min_corners}")
    if board.checkCharucoCornersCollinear(ids):
        raise ValueError("Las esquinas ChArUco son colineales")
    area_ratio = cv2.contourArea(cv2.convexHull(corners)) / (image.shape[0] * image.shape[1])
    if area_ratio < min_area_ratio:
        raise ValueError(
            f"Las esquinas cubren {area_ratio:.1%} de la imagen; se requiere {min_area_ratio:.1%}"
        )

    object_points = board.getChessboardCorners()[ids.flatten()]
    ok, rvec, tvec = cv2.solvePnP(object_points, corners.reshape(-1, 2), K, dist)
    if not ok:
        raise ValueError("No se pudo estimar la pose de la camara")

    R, _ = cv2.Rodrigues(rvec)
    if np.any((R @ object_points.T + tvec)[2] <= 0):
        raise ValueError("La pose ubica el tablero detras de la camara")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    residuals = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    if not np.isfinite(rms):
        raise ValueError("La pose produce un error de reproyeccion no finito")
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R.T
    camera_pose[:3, 3] = (-R.T @ tvec).ravel()
    return rvec, tvec, camera_pose, rms
