"""Visualizaciones del pipeline."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def show_rectification(left, right, left_rect, right_rect):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    titles = ("Izquierda original", "Derecha original",
              "Izquierda rectificada", "Derecha rectificada")
    for ax, image, title in zip(axes.flat, (left, right, left_rect, right_rect), titles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[1]:
        for y in range(0, left_rect.shape[0], 100):
            ax.axhline(y, color="lime", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def show_disparity(disparity):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")
    image = ax.imshow(np.ma.masked_less_equal(disparity, 0), cmap=cmap)
    fig.colorbar(image, ax=ax, label="Disparidad (px)")
    ax.set_title("Disparidad - SGBM")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _camera_frustum(pose, K, image_size, scale):
    camera = o3d.geometry.LineSet.create_camera_visualization(
        image_size[0], image_size[1], K, np.linalg.inv(pose), scale
    )
    camera.paint_uniform_color([1, 0.5, 0])
    return camera


def show_cloud(cloud, camera_pose=None, K=None, image_size=None, axis_size=100):
    """Muestra la nube con ejes y, si se indica, la camara en el mismo marco."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    geometries = [cloud, frame]
    front, up = np.array([0., 0., -1.]), np.array([0., -1., 0.])
    if camera_pose is not None:
        camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size / 2)
        camera_axes.transform(camera_pose)
        camera = _camera_frustum(camera_pose, K, image_size, axis_size / 2)
        geometries.extend([camera_axes, camera])
        front, up = camera_pose[:3, :3] @ front, camera_pose[:3, :3] @ up
    # Centrar la vista sin que los puntos mas lejanos dominen el encuadre.
    center = np.median(np.asarray(cloud.points), axis=0)
    o3d.visualization.draw_geometries(
        geometries, window_name="Nube de puntos", lookat=center,
        front=front, up=up, zoom=0.02
    )


def show_camera_poses(camera_poses, K, image_size, axis_size=100):
    """Muestra las camaras aceptadas en el sistema del tablero."""
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)]
    for pose in camera_poses.values():
        camera = _camera_frustum(pose, K, image_size, axis_size / 2)
        geometries.append(camera)
    o3d.visualization.draw_geometries(geometries, window_name="Poses de las camaras")
