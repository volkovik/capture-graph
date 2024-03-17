import numpy as np
from matplotlib import pyplot as plt

# Параметры
# Длины сторон прямоугольника наблюдаемой области
S_n = 30
S_m = 10
# Координаты центра прямоугольника наблюдаемой области
S_x = 150
S_z = 60
# Угол поворота прямоугольника наблюдаемой области
S_phi = 50
# Параметры линейной функции, задающей траекторию полета самолета
B_phi = 30  # Угловой коэффициент
B_h = -3  # Сдвиг по оси ординат


def trajectory_function(x, phi, b):
    return np.tan(np.deg2rad(phi)) * x + b


def calculate_vertices_of_rectangle(x, y, width, height, angle):
    # Матрица поворота
    R = np.array([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]
    ])

    # Матрица вершин прямоугольника
    vertices = np.array([
        [width / 2, -width / 2, -width / 2, width / 2],
        [height / 2, height / 2, -height / 2, -height / 2]
    ])

    # Поворот вершин
    vertices = np.dot(R, vertices)

    # Перенос вершин к точке пересечения
    vertices[0, :] += x
    vertices[1, :] += y

    return vertices


# Функция для вычисления проекции точки на прямую
def calculate_projection(phi, b, a_x, a_y):
    k = np.tan(np.deg2rad(phi))
    x_projection = (a_x + k * a_y - k * b) / (k**2 + 1)
    y_projection = k * x_projection + b
    return x_projection, y_projection


plt.figure(figsize=(10, 10))

# Показ траектории полета самолета
fig, ax = plt.subplots()
ax.axline(
    (100, trajectory_function(100, B_phi, B_h)),
    slope=np.tan(np.deg2rad(B_phi)),
    label='Траектория полета самолета'
)

# Показ прямоугольника наблюдаемой области
rectangle_vertices = calculate_vertices_of_rectangle(S_x, S_z, S_n, S_m, S_phi)
plt.plot(
    np.append(rectangle_vertices[0, :], rectangle_vertices[0, 0]),
    np.append(rectangle_vertices[1, :], rectangle_vertices[1, 0]),
    'y-', label='Наблюдаемая область'
)
plt.gca().add_patch(plt.Polygon(rectangle_vertices.T, facecolor='y', alpha=0.3))
point_and_projection = [
    (
        calculate_projection(B_phi, B_h, rectangle_vertices[0, i], rectangle_vertices[1, i]),
        (rectangle_vertices[0, i], rectangle_vertices[1, i])
    )
    for i in range(4)
]

min_point_and_projection = min(point_and_projection, key=lambda x: x[0][0] + x[0][1])

x_projection, y_projection = min_point_and_projection[0]
x_vertical, y_vertical = min_point_and_projection[1]
plt.plot([x_projection, x_vertical], [y_projection, y_vertical], 'g--', alpha=0.3)
plt.plot(x_projection, y_projection, 'go', label='Точка начала видеофиксации')

max_point_and_projection = max(point_and_projection, key=lambda x: x[0][0] + x[0][1])

x_projection, y_projection = max_point_and_projection[0]
x_vertical, y_vertical = max_point_and_projection[1]
plt.plot([x_projection, x_vertical], [y_projection, y_vertical], 'r--', alpha=0.3)
plt.plot(x_projection, y_projection, 'ro', label='Точка оконачания видеофиксации')


plt.title('Видеофиксация наблюдаемой области летательным аппаратом')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
