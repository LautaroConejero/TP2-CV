"""Fusion de las nubes en el marco del tablero y optimizacion con Open3D."""

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
    o3d.utility.random.seed(0)  # RANSAC reproducible
    clouds, scene = {}, o3d.geometry.PointCloud()
    camera_poses, report = {}, {}
    for result in reconstruct_dataset(data_dir, calibration, maps, board, max_pose_rms):
        number = result["capture"]
        if "error" in result:
            report[number] = {"error": result["error"]}
            continue
        cloud = result["cloud"]
        transform, degrees, offset = correct_board_plane(cloud, box)
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
    projections = points @ directions.T
    best = np.argmax(projections.max(axis=0) - projections.min(axis=0))
    i, j = projections[:, best].argmin(), projections[:, best].argmax()
    axis = (points[j] - points[i]) / np.linalg.norm(points[j] - points[i])
    normal = np.array([-axis[1], axis[0]])
    along, across = (points - points[i]) @ axis, (points - points[i]) @ normal
    obb = cloud.get_oriented_bounding_box()
    return {
        "alto_mm": height,
        "largo_mm": float(along.max() - along.min()),
        "ancho_mm": float(across.max() - across.min()),
        "extremos": (points[i], points[j]), "eje_largo": axis,
        "aabb_extent_mm": np.asarray(aabb.get_extent()),
        "obb_extent_mm": np.sort(np.asarray(obb.extent))[::-1],
    }
