from typing import Any
from pathlib import Path

from PIL import Image
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from numpy import ndarray, dtype


def download_image_gray(image_path: str) -> ndarray[tuple[int, ...], dtype[Any]]:
    try:
        img = Image.open(image_path)
        return np.array(img)
    except FileNotFoundError:
        print(f"Ошибка: Файл {image_path} не найден.")
        exit()


def create_cube(size=1.0):
    cube_mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    cube_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    return cube_mesh




base_detail = create_cube(size=5)
img_array = download_image_gray('img/robot.jpeg')

grid_size_x = 200
grid_size_y = 100
cell_width = img_array.shape[1] // grid_size_x
cell_height = img_array.shape[0] // grid_size_y
points = []

for j in range(grid_size_y):
    for i in range(grid_size_x):
        center_x = i * cell_width + cell_width // 2
        center_y = j * cell_height + cell_height // 2
        cell_values = img_array[j * cell_height:(j + 1) * cell_height, i * cell_width:(i + 1) * cell_width]
        z = np.mean(cell_values) * 0.2
        print([center_x, center_y, z])
        points.append([center_x, center_y, z])

points = np.array([box for box in points if box[2] > 0.5])



print(f"Количество точек: {len(points)}")

model = o3d.geometry.TriangleMesh()
for point in points:
    detail = base_detail.translate(np.array(point), relative=False)
    model += detail

o3d.io.write_triangle_mesh(Path('output_model.ply'), model)


o3d.visualization.draw_geometries([model],
                                 zoom=0.2,
                                 front=[0.5, 0, 0.5],
                                 lookat=[0, 0, 0],
                                 up=[0, 1, 0])
