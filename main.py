import numpy as np
from matplotlib import pyplot as plt

# Параметры
# Длины сторон прямоугольника наблюдаемой области
S_x = 30
S_y = 10
# Координаты центра прямоугольника наблюдаемой области
S_lambda = 40
S_phi = 50
# Угол поворота прямоугольника наблюдаемой области
S_alpha = 50
# Точка начала вылета и угол полета
B_x = 0
B_y = 0
# Точка входа
D_x = 30
# Запуск оборудования
D_k = 10


def linear_equation(x, y, slope, x_0):
    k = np.tan(np.deg2rad(slope))
    b = y - k * x
    return k * x_0 + b


def calculate_distance_between_dots(x_1, y_1, x_2, y_2):
    return np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)


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
def calculate_projection(phi, x, y, a_x, a_y):
    k = np.tan(np.deg2rad(phi))
    b = y - k * x
    x_projection = (a_x + k * a_y - k * b) / (k**2 + 1)
    y_projection = k * x_projection + b
    return x_projection, y_projection


def calculate_intersection_of_lines(k_1, b_1, k_2, b_2):
    x = (b_2 - b_1) / (k_1 - k_2)
    y = k_1 * x + b_1
    return x, y


def calculate_third_dot_of_right_triangle(x, y, slope, hypotenuse):
    x_r = x + np.cos(np.deg2rad(slope)) * hypotenuse
    y_r = y + np.sin(np.deg2rad(slope)) * hypotenuse
    return x_r, y_r


plt.figure(figsize=(10, 10))

# Показ прямоугольника наблюдаемой области
fig, ax = plt.subplots()

rectangle_vertices = calculate_vertices_of_rectangle(S_lambda, S_phi, S_x, S_y, S_alpha)
plt.plot(
    np.append(rectangle_vertices[0, :], rectangle_vertices[0, 0]),
    np.append(rectangle_vertices[1, :], rectangle_vertices[1, 0]),
    'y-'
)
plt.gca().add_patch(plt.Polygon(rectangle_vertices.T, facecolor='y', alpha=0.1))

B_betas = [60, 75, 90, 105]
for B_beta in B_betas:
    # Показ траектории полета самолета

    point_and_projection = [
        (
            calculate_projection(B_beta, B_x, B_y, rectangle_vertices[0, i], rectangle_vertices[1, i]),
            (rectangle_vertices[0, i], rectangle_vertices[1, i])
        )
        for i in range(4)
    ]

    plt.plot(S_lambda, S_phi, 'yo', label='Центр наблюдаемой области')
    x_projection, y_projection = calculate_projection(B_beta, B_x, B_y, S_lambda, S_phi)

    if calculate_distance_between_dots(S_lambda, S_phi, x_projection, y_projection) <= D_x:
        ax.axline((B_x, B_y), slope=np.tan(np.deg2rad(B_beta)), color='b', label='Правильная траектория полета')
        plt.plot(x_projection, y_projection, 'bo')
        plt.plot([S_lambda, x_projection], [S_phi, y_projection], 'b--', alpha=0.3)

        min_point_and_projection = min(point_and_projection, key=lambda x: x[0][0] + x[0][1])

        x_projection, y_projection = min_point_and_projection[0]

        x_3, y_3 = calculate_third_dot_of_right_triangle(
            x_projection, y_projection,
            B_beta,
            -D_k
        )
        plt.plot(x_3, y_3, 'go', label='Точка запуска аппаратуры')
    else:
        ax.axline((B_x, B_y), slope=np.tan(np.deg2rad(B_beta)), color='r', label='Неправильная траектория полета')
        plt.plot(x_projection, y_projection, 'ro')
        plt.plot([S_lambda, x_projection], [S_phi, y_projection], 'r--', alpha=0.3)

plt.title('Видеофиксация наблюдаемой области самолетом')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
