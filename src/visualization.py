"""Visualizaciones del pipeline."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def show_rectification(left, right, left_rect, right_rect, path=None):
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
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def show_disparity(disparity, path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")
    image = ax.imshow(np.ma.masked_less_equal(disparity, 0), cmap=cmap)
    fig.colorbar(image, ax=ax, label="Disparidad (px)")
    ax.set_title("Disparidad - SGBM")
    ax.axis("off")
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
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
    o3d.visualization.draw_geometries(camera_geometries(camera_poses, K, image_size, axis_size),
                                      window_name="Poses de las camaras")


def render_geometries(geometries, path, front, up, lookat=None, zoom=0.7,
                      width=1280, height=960, point_size=2.0):
    """Guarda una captura PNG sin abrir ventana (para el informe y Jupyter)."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for geometry in geometries:
        vis.add_geometry(geometry)
    options = vis.get_render_option()
    options.point_size = point_size
    options.background_color = np.ones(3)
    control = vis.get_view_control()
    control.set_front(front)
    control.set_up(up)
    if lookat is not None:
        control.set_lookat(lookat)
    control.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(path), do_render=True)
    vis.destroy_window()


def camera_geometries(camera_poses, K, image_size, axis_size=100):
    """Ejes del mundo + frustum de cada camara (en el marco del tablero)."""
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)]
    for pose in camera_poses.values():
        geometries.append(_camera_frustum(pose, K, image_size, axis_size / 2))
    return geometries


def plot_camera_trajectory(camera_poses, board_size, object_center=None, path=None):
    """Centros y ejes opticos de las camaras sobre el tablero (vista superior y lateral)."""
    numbers = sorted(camera_poses)
    centers = np.array([camera_poses[n][:3, 3] for n in numbers])
    axes_z = np.array([camera_poses[n][:3, 2] for n in numbers])
    fig, (top, side) = plt.subplots(1, 2, figsize=(14, 6))
    top.add_patch(plt.Rectangle((0, 0), *board_size, fill=False, color="black", label="ChArUco"))
    scatter = top.scatter(centers[:, 0], centers[:, 1], c=numbers, cmap="viridis", s=25)
    top.quiver(centers[:, 0], centers[:, 1], axes_z[:, 0], axes_z[:, 1],
               color="gray", width=0.003, scale=15)
    if object_center is not None:
        top.plot(*object_center[:2], "r*", markersize=15, label="Objeto")
    top.set(xlabel="x (mm)", ylabel="y (mm)", title="Vista superior", aspect="equal")
    top.invert_yaxis()
    top.legend(loc="upper right")
    radius = np.linalg.norm(centers[:, :2] - np.array(board_size) / 2, axis=1)
    side.scatter(radius, -centers[:, 2], c=numbers, cmap="viridis", s=25)
    side.set(xlabel="Distancia horizontal al centro del tablero (mm)",
             ylabel="Altura sobre el tablero (mm)", title="Vista lateral")
    fig.colorbar(scatter, ax=[top, side], label="Numero de captura")
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_measurement(cloud, measurement, path=None):
    """Proyeccion superior con los puntos extremos y perfil lateral con el alto."""
    points = np.asarray(cloud.points)[::3]
    colors = np.asarray(cloud.colors)[::3]
    p, q = measurement["extremos"]
    axis = measurement["eje_largo"]
    normal = np.array([-axis[1], axis[0]])
    along, across = (points[:, :2] - p) @ axis, (points[:, :2] - p) @ normal
    # Rectangulo alineado al eje largo que encierra la proyeccion.
    corners = np.array([[along.min(), across.min()], [along.max(), across.min()],
                        [along.max(), across.max()], [along.min(), across.max()],
                        [along.min(), across.min()]])
    corners = p + corners[:, :1] * axis + corners[:, 1:] * normal
    fig, (top, side) = plt.subplots(1, 2, figsize=(15, 6))
    top.scatter(points[:, 0], points[:, 1], c=colors, s=0.3)
    top.plot(*corners.T, "r-", linewidth=1.5)
    top.plot(*np.vstack([p, q]).T, "bo--", linewidth=1.5, label="Puntos extremos")
    top.set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal",
            title=f"Vista superior: {measurement['largo_mm']:.0f} x {measurement['ancho_mm']:.0f} mm")
    top.invert_yaxis()
    top.legend(loc="upper right")
    side.scatter(along, -points[:, 2], c=colors, s=0.3)
    side.axhline(0, color="black", linewidth=1)
    side.axhline(measurement["alto_mm"], color="red", linestyle="--", linewidth=1.5)
    side.set(xlabel="Posicion sobre el eje largo (mm)", ylabel="Altura sobre el tablero (mm)",
             aspect="equal", title=f"Vista lateral: alto {measurement['alto_mm']:.0f} mm")
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
