"""Calibracion y rectificacion estereo con OpenCV."""

import pickle
from pathlib import Path

import cv2
import numpy as np


def load_calibration(data_dir):
    data_dir = Path(data_dir)
    with (data_dir / "stereo_calibration.pkl").open("rb") as f:
        calibration = pickle.load(f)
    with (data_dir / "stereo_maps.pkl").open("rb") as f:
        maps = pickle.load(f)
    return calibration, maps


def load_pair(directory, number):
    images = []
    for side in ("left", "right"):
        path = Path(directory) / f"{side}_{number}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    return tuple(images)


def rectify_pair(left, right, maps):
    for side, image in zip(("left", "right"), (left, right)):
        if image.shape[:2] != maps[f"{side}_map_x"].shape:
            raise ValueError("La imagen y su mapa tienen distinta resolucion")
    return (
        cv2.remap(left, maps["left_map_x"], maps["left_map_y"], cv2.INTER_LINEAR),
        cv2.remap(right, maps["right_map_x"], maps["right_map_y"], cv2.INTER_LINEAR),
    )


def _corners(image, pattern):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 0.7 if gray.shape[1] >= 1080 else 1.0
    small = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, points = cv2.findChessboardCorners(small, pattern, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, points / scale, (7, 7), (-1, -1), criteria).reshape(-1, 2)


def calibrate_stereo(directory, pattern, square_size, excluded_pairs=(), max_pair_error=2.0):
    """Calibra cada camara (Zhang) y luego el par; no escribe archivos PKL.

    Los pares cuyo error de reproyeccion estereo supera max_pair_error px se
    descartan y se recalibra con el resto.
    """
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size
    left_points, right_points, used = [], [], []
    image_size = None
    files = sorted(Path(directory).glob("left_*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))
    for path in files:
        number = int(path.stem.split("_")[-1])
        if number in excluded_pairs:
            continue
        images = load_pair(directory, number)
        if image_size is None:
            image_size = images[0].shape[1::-1]
        if any(im.shape[1::-1] != image_size for im in images):
            raise ValueError("Las imagenes de calibracion tienen distintas resoluciones")
        left, right = [_corners(im, pattern) for im in images]
        if left is None or right is None:
            continue
        left_points.append(left)
        right_points.append(right)
        used.append(number)
    if len(used) < 10:
        raise ValueError("Se requieren al menos 10 pares de calibracion validos")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    for iteration in range(5):
        # Intrinsecos y distorsion de cada camara por separado (calibrateCamera).
        left_rms, left_K, left_dist, _, _ = cv2.calibrateCamera(
            [obj] * len(used), left_points, image_size, None, None)
        right_rms, right_K, right_dist, _, _ = cv2.calibrateCamera(
            [obj] * len(used), right_points, image_size, None, None)
        # R y T entre camaras; los intrinsecos se usan como punto de partida.
        result = cv2.stereoCalibrateExtended(
            [obj] * len(used), left_points, right_points,
            left_K, left_dist, right_K, right_dist, image_size, None, None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=criteria
        )
        errors = np.sqrt(np.mean(result[-1] ** 2, axis=1))
        keep = errors <= max_pair_error
        if keep.all() or iteration == 4:
            break
        if keep.sum() < 10:
            raise ValueError("Quedan menos de 10 pares tras filtrar por error de reproyeccion")
        left_points = [p for p, valid in zip(left_points, keep) if valid]
        right_points = [p for p, valid in zip(right_points, keep) if valid]
        used = [number for number, valid in zip(used, keep) if valid]

    rms, left_K, left_dist, right_K, right_dist, R, T, E, F = result[:9]
    calibration = {
        "left_K": left_K, "left_dist": left_dist,
        "right_K": right_K, "right_dist": right_dist,
        "R": R, "T": T, "E": E, "F": F, "image_size": image_size,
        "rms": float(rms), "left_rms": float(left_rms), "right_rms": float(right_rms),
        "used_pairs": used,
        "excluded_pairs": sorted(int(p.stem.split("_")[-1]) for p in files
                                 if int(p.stem.split("_")[-1]) not in used),
        "pattern": tuple(pattern), "square_size_mm": square_size,
    }
    return calibration, _rectification_maps(calibration)


def refine_stereo(directory, calibration, board, reference_pairs):
    """Refina R y T con vistas ChArUco reservadas; conserva K y distorsion."""
    params = cv2.aruco.CharucoParameters()
    params.checkMarkers = False
    params.minMarkers = 2
    detector = cv2.aruco.CharucoDetector(board, params)
    objects, left_points, right_points, used = [], [], [], []
    for number in reference_pairs:
        left, right = load_pair(directory, number)
        if any(im.shape[1::-1] != tuple(calibration["image_size"]) for im in (left, right)):
            raise ValueError("Resolucion distinta de la calibracion")
        lc, li, _, _ = detector.detectBoard(left)
        rc, ri, _, _ = detector.detectBoard(right)
        if li is None or ri is None:
            continue
        ids, left_indices, right_indices = np.intersect1d(li.ravel(), ri.ravel(), return_indices=True)
        if len(ids) < 12 or board.checkCharucoCornersCollinear(ids):
            continue
        objects.append(board.getChessboardCorners()[ids])
        left_points.append(lc.reshape(-1, 2)[left_indices])
        right_points.append(rc.reshape(-1, 2)[right_indices])
        used.append(number)
    if len(used) < 10:
        raise ValueError("Se requieren al menos 10 pares ChArUco validos para refinar R y T")

    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objects, left_points, right_points,
        calibration["left_K"].copy(), calibration["left_dist"].copy(),
        calibration["right_K"].copy(), calibration["right_dist"].copy(),
        calibration["image_size"], flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    )
    refined = dict(calibration)
    refined.update(R=R, T=T, E=E, F=F, rms=float(rms), reference_pairs=used,
                   checkerboard_rms=calibration["rms"])
    return refined, _rectification_maps(refined)


def _rectification_maps(calibration):
    size = calibration["image_size"]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        calibration["left_K"], calibration["left_dist"],
        calibration["right_K"], calibration["right_dist"],
        size, calibration["R"], calibration["T"], alpha=0
    )
    maps = {
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "validRoi1": roi1, "validRoi2": roi2,
    }
    for side, R, P in [("left", R1, P1), ("right", R2, P2)]:
        maps[f"{side}_map_x"], maps[f"{side}_map_y"] = cv2.initUndistortRectifyMap(
            calibration[f"{side}_K"], calibration[f"{side}_dist"], R, P, size, cv2.CV_32FC1
        )
    return maps


def validate_stereo(directory, calibration, maps, board, pairs):
    """Error epipolar y de escala con esquinas ChArUco en pares no usados.

    Para cada par se rectifican las esquinas detectadas en ambas imagenes:
    |dy| mide la alineacion de lineas epipolares y la escala compara las
    distancias entre esquinas trianguladas con las del tablero real.
    """
    params = cv2.aruco.CharucoParameters()
    params.checkMarkers = False
    params.minMarkers = 2
    detector = cv2.aruco.CharucoDetector(board, params)
    rows = []
    for number in pairs:
        left, right = load_pair(directory, number)
        lc, li, _, _ = detector.detectBoard(left)
        rc, ri, _, _ = detector.detectBoard(right)
        if li is None or ri is None:
            continue
        ids, a, b = np.intersect1d(li.ravel(), ri.ravel(), return_indices=True)
        if len(ids) < 8:
            continue
        pl = cv2.undistortPoints(lc[a], calibration["left_K"], calibration["left_dist"],
                                 R=maps["R1"], P=maps["P1"]).reshape(-1, 2)
        pr = cv2.undistortPoints(rc[b], calibration["right_K"], calibration["right_dist"],
                                 R=maps["R2"], P=maps["P2"]).reshape(-1, 2)
        points = np.c_[pl, pl[:, 0] - pr[:, 0]].reshape(-1, 1, 3).astype(np.float64)
        X = cv2.perspectiveTransform(points, maps["Q"]).reshape(-1, 3)
        obj = board.getChessboardCorners()[ids]
        i, j = np.triu_indices(len(ids), 1)
        scale = np.linalg.norm(X[i] - X[j], axis=1) / np.linalg.norm(obj[i] - obj[j], axis=1)
        rows.append((number, len(ids), np.mean(np.abs(pl[:, 1] - pr[:, 1])), np.median(scale)))
    return np.array(rows)
