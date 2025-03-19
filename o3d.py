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
    cube_mesh.paint_uniform_color([0.0, 0.0, 0.0])
    return cube_mesh




base_detail = create_cube(size=7)
img_array = download_image_gray('img/robot.jpeg')

grid_size_x = 80
grid_size_y = 80
cell_width = img_array.shape[1] // grid_size_x
cell_height = img_array.shape[0] // grid_size_y
points = []

for j in range(grid_size_y):
    for i in range(grid_size_x):
        center_x = i * cell_width + cell_width // 2
        center_y = j * cell_height + cell_height // 2
        cell_values = img_array[j * cell_height:(j + 1) * cell_height, i * cell_width:(i + 1) * cell_width]
        z = np.mean(cell_values) * 0.5
        print([center_x, center_y, z])
        points.append([center_x, center_y, z])

points = np.array([box for box in points if box[2] > 0.5])

if len(points) > 1000:
    indices = np.random.choice(len(points), 1000, replace=False)
    points = points[indices]

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

'''
Объяснение кода и шаги решения:

    Импорт библиотек: Импортируем необходимые библиотеки (PIL, NumPy, Open3D, matplotlib).

    Загрузка и подготовка изображения:
        Загружаем изображение в градациях серого. Это упрощает анализ.
        Преобразуем изображение в NumPy массив.
        Важно: Этот код предполагает, что изображение имеет относительно простые формы и контрасты. Для более сложных изображений потребуется более продвинутая обработка (например, обнаружение краев, распознавание объектов, сегментация).

    Обработка изображения и определение координат деталей:
        Этот этап является ключевым и наиболее сложным. Он зависит от типа фотографии и от того, как вы хотите представить ее в 3D.
        Пример 1 (Простое преобразование): Каждый пиксель изображения используется как точка в 3D-пространстве. Значение яркости пикселя используется для Z-координаты. Это самый простой вариант, который может хорошо работать для градиентных изображений или изображений с плавными переходами.
        Пример 2 (Поиск паттернов): Изображение разбивается на участки. Центр каждого участка становится координатой детали, а средняя яркость участка используется для Z-координаты. Этот подход подходит для изображений с регулярной структурой (кирпичная стена, мозаика, и т.д.). Этот код представляет собой упрощенный пример.
        Важно: Вам нужно будет адаптировать этот код к конкретному типу изображения. Возможно, вам потребуется использовать алгоритмы компьютерного зрения для обнаружения объектов, сегментации, или поиска повторяющихся паттернов. Можно использовать библиотеки, такие как OpenCV, для более продвинутой обработки изображений.
        Ограничение количества точек (если их больше 1000): Чтобы не превысить лимит в 1000 деталей, код случайно выбирает подмножество точек.

'''
