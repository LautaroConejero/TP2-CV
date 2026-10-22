"""Fusion de las nubes en el marco del tablero y optimizacion con Open3D."""

import cv2
import numpy as np
import open3d as o3d

from .pipeline import reconstruct_dataset


def crop_to_box(cloud, box):
    indices = box.get_point_indices_within_bounding_box(cloud.points)
    return cloud.select_by_index(indices)


def correct_board_plane(cloud, box, threshold=2.0, band=20.0):
    """Lleva el plano del tablero reconstruido a z = 0.

    Se ajusta un plano con RANSAC (segment_plane) a los puntos a menos de
    `band` mm de z = 0 que quedan fuera de la huella de la caja del objeto, y
    se devuelve la transformacion rigida minima (rotacion que alinea su normal
    con el eje z y traslacion en z) junto con el angulo y el desplazamiento.
    """
    points = np.asarray(cloud.points)
    low, high = box.get_min_bound(), box.get_max_bound()
    outside = ~((points[:, 0] > low[0]) & (points[:, 0] < high[0]) &
                (points[:, 1] > low[1]) & (points[:, 1] < high[1]))
    ring = np.flatnonzero((np.abs(points[:, 2]) < band) & outside)
    if len(ring) < 1000:
        raise ValueError("No hay suficientes puntos del tablero para ajustar el plano")
    plane, _ = cloud.select_by_index(ring).segment_plane(threshold, 3, 1000)
    normal, offset = np.array(plane[:3]), plane[3]
    if normal[2] < 0:
        normal, offset = -normal, -offset
    # Rotacion de Rodrigues que lleva la normal del plano al eje z.
    axis = np.cross(normal, [0.0, 0.0, 1.0])
    sin, cos = np.linalg.norm(axis), normal[2]
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - cos) / sin ** 2) if sin > 1e-9 else np.eye(3)
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[2, 3] = offset
    return transform, float(np.degrees(np.arcsin(sin))), float(offset)


def fuse_dataset(data_dir, calibration, maps, board, box, max_pose_rms=5.0,
                 max_correction_mm=20.0, max_correction_deg=5.0,
                 scene_voxel=8.0, scene_limit=1000.0):
    """Lleva cada captura al marco del tablero, corrige el plano y recorta la caja.

    Se procesa una captura a la vez para no acumular las nubes completas en
    memoria. Devuelve un diccionario captura -> nube recortada, una version
    muy submuestreada de la escena completa (solo para la figura del
    "antes"), las poses y un resumen por captura.
    """
    o3d.utility.random.seed(0)  # Semilla inicial de RANSAC
    clouds, scene = {}, o3d.geometry.PointCloud()
    camera_poses, report = {}, {}
    for result in reconstruct_dataset(data_dir, calibration, maps, board, max_pose_rms):
        number = result["capture"]
        if "error" in result:
            report[number] = {"error": result["error"]}
            continue
        cloud = result["cloud"]
        try:
            transform, degrees, offset = correct_board_plane(cloud, box)
        except (ValueError, RuntimeError) as error:
            report[number] = {"error": str(error)}
            continue
        if abs(offset) > max_correction_mm or degrees > max_correction_deg:
            report[number] = {"error": f"Correccion de plano no plausible: {degrees:.1f} grados, {offset:.1f} mm"}
            continue
        cloud.transform(transform)
        camera_poses[number] = transform @ result["camera_pose"]
        clouds[number] = crop_to_box(cloud, box)
        near = np.flatnonzero(np.linalg.norm(np.asarray(cloud.points), axis=1) < scene_limit)
        scene += cloud.select_by_index(near).voxel_down_sample(scene_voxel)
        report[number] = {
            "pose_rms": result["pose_rms"], "points": result["points"],
            "cropped": len(clouds[number].points), "coverage": result["coverage"],
            "correction_deg": degrees, "correction_mm": offset,
        }
        if len(report) % 10 == 0:
            print(f"Procesadas {len(report)} capturas")
    return clouds, scene, camera_poses, report


def merge(clouds):
    fused = o3d.geometry.PointCloud()
    for cloud in clouds.values():
        fused += cloud
    return fused


def refine_clouds_icp(clouds, camera_poses, reference=None, voxel_size=2.0,
                      max_distance=3.0, max_motion_mm=10.0, max_angle_deg=3.0):
    """Refina nubes y poses contra una referencia fija mediante ICP punto a plano.

    La inicializacion es la pose global ya estimada. Se aceptan correcciones
    pequenas que reducen el RMSE sin perder mas del 5% de correspondencias.
    Devuelve copias de las nubes, poses actualizadas y las metricas por vista.
    """
    if reference is None:
        reference = merge(clouds)
    target = reference.voxel_down_sample(voxel_size)
    target, _ = target.remove_statistical_outlier(30, 2.0)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=4 * voxel_size, max_nn=30))
    registration = o3d.pipelines.registration
    estimator = registration.TransformationEstimationPointToPlane(
        registration.HuberLoss(1.0))
    aligned, poses, report = {}, {}, {}
    for number, cloud in clouds.items():
        source = cloud.voxel_down_sample(voxel_size)
        before = registration.evaluate_registration(source, target, max_distance)
        result = registration.registration_icp(
            source, target, max_distance, np.eye(4), estimator,
            registration.ICPConvergenceCriteria(max_iteration=50))
        transform = result.transformation
        angle = float(np.degrees(np.arccos(np.clip(
            (np.trace(transform[:3, :3]) - 1) / 2, -1, 1))))
        center = source.get_center()
        motion = float(np.linalg.norm(
            transform[:3, :3] @ center + transform[:3, 3] - center))
        applied = (np.isfinite(transform).all() and before.fitness > 0 and
                   result.inlier_rmse < before.inlier_rmse and
                   result.fitness >= 0.95 * before.fitness and
                   motion <= max_motion_mm and angle <= max_angle_deg)
        aligned[number] = o3d.geometry.PointCloud(cloud)
        poses[number] = camera_poses[number].copy()
        if applied:
            aligned[number].transform(transform)
            poses[number] = transform @ poses[number]
        report[number] = {
            "applied": bool(applied), "rmse_before": before.inlier_rmse,
            "rmse_after": result.inlier_rmse if applied else before.inlier_rmse,
            "fitness_before": before.fitness,
            "fitness_after": result.fitness if applied else before.fitness,
            "motion_mm": motion if applied else 0.0,
            "angle_deg": angle if applied else 0.0,
        }
    return aligned, poses, report


def validate_icp(clouds, camera_poses, voxel_size=2.0, distance_limit=5.0):
    """Compara ICP sobre muestras iguales y vistas independientes del ajuste.

    Capturas modulo 3: 0 para referencia, 1 para evaluacion y 2 para alinear.
    Devuelve captura, distancia media antes/despues (limitada a 5 mm por
    defecto) y si se acepto la correccion.
    """
    reference = merge({n: c for n, c in clouds.items() if n % 3 == 0})
    evaluation = merge({n: c for n, c in clouds.items() if n % 3 == 1})
    evaluation = evaluation.voxel_down_sample(voxel_size)
    tested = {n: c for n, c in clouds.items() if n % 3 == 2}
    if not len(reference.points) or not len(evaluation.points) or not tested:
        raise ValueError("Se requieren nubes en los tres grupos de validacion ICP")
    _, poses, registrations = refine_clouds_icp(
        tested, camera_poses, reference=reference, voxel_size=voxel_size)
    rows = []
    for number, cloud in tested.items():
        sample = cloud.voxel_down_sample(voxel_size)
        corrected = o3d.geometry.PointCloud(sample)
        corrected.transform(poses[number] @ np.linalg.inv(camera_poses[number]))
        before = np.minimum(sample.compute_point_cloud_distance(evaluation), distance_limit).mean()
        after = np.minimum(corrected.compute_point_cloud_distance(evaluation), distance_limit).mean()
        rows.append((number, before, after, registrations[number]["applied"]))
    return np.array(rows)


def optimize_cloud(cloud, nb_neighbors, std_ratio, voxel_size, nb_points, radius):
    """Outliers estadisticos (nube densa) -> voxel -> outliers por radio.

    El filtro estadistico se aplica antes del submuestreo: la superficie real
    se acumula desde muchas vistas y queda densa, mientras que los puntos
    espurios de cada vista quedan dispersos. Despues del voxel esa diferencia
    de densidad desaparece. Devuelve cada etapa.
    """
    stages = {"caja": cloud}
    stages["estadistico"], _ = cloud.remove_statistical_outlier(nb_neighbors, std_ratio)
    stages["voxel"] = stages["estadistico"].voxel_down_sample(voxel_size)
    stages["radio"], _ = stages["voxel"].remove_radius_outlier(nb_points, radius)
    return stages


def orient_normals_outward(cloud):
    """Invierte las normales que apuntan hacia el centro de la nube."""
    normals = np.asarray(cloud.normals)
    inward = np.sum(normals * (np.asarray(cloud.points) - cloud.get_center()), axis=1) < 0
    normals[inward] *= -1
    cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def measure_object(cloud, n_directions=360):
    """Alto, largo y ancho en mm a partir de bounding boxes y puntos extremos.

    Alto: altura del punto mas alto sobre el plano del tablero, z = 0 (lado
    vertical de la bounding box alineada a los ejes del tablero; el eje z del
    ChArUco apunta hacia abajo, por eso se usa -z). Largo: distancia entre los
    dos puntos extremos de la proyeccion sobre el tablero. Ancho: extension de
    la proyeccion en la direccion perpendicular al largo.
    """
    aabb = cloud.get_axis_aligned_bounding_box()
    height = float(-aabb.get_min_bound()[2])
    # Los puntos extremos son los de mayor separacion al proyectar la nube
    # sobre direcciones horizontales cada 0.5 grados.
    points = np.asarray(cloud.points)[:, :2]
    angles = np.linspace(0, np.pi, n_directions, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    # Los extremos de cualquier proyeccion estan en la envolvente convexa.
    indices = cv2.convexHull(points.astype(np.float32), returnPoints=False).ravel()
    hull = points[indices]
    projections = hull @ directions.T
    best = np.argmax(projections.max(axis=0) - projections.min(axis=0))
    i, j = projections[:, best].argmin(), projections[:, best].argmax()
    axis = (hull[j] - hull[i]) / np.linalg.norm(hull[j] - hull[i])
    normal = np.array([-axis[1], axis[0]])
    along, across = (hull - hull[i]) @ axis, (hull - hull[i]) @ normal
    obb = cloud.get_oriented_bounding_box()
    return {
        "alto_mm": height,
        "largo_mm": float(along.max() - along.min()),
        "ancho_mm": float(across.max() - across.min()),
        "extremos": (hull[i], hull[j]), "eje_largo": axis,
        "aabb_extent_mm": np.asarray(aabb.get_extent()),
        "obb_extent_mm": np.sort(np.asarray(obb.extent))[::-1],
    }
