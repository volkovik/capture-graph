import numpy as np
from matplotlib import pyplot as plt

# Параметры
# Длины сторон прямоугольника наблюдаемой области
x_MA = 30
y_MA = 15
# Координаты центра прямоугольника наблюдаемой области
lambda_MA = 0
phi_MA = 25
# Угол поворота прямоугольника наблюдаемой области
alpha_MA = 50
# Точка начала вылета и угол полета
lambda_St = 0
phi_St = 0
alpha_St = 30
# Перпендикулярное расстояния способности камеры снимать
x_CMax = 10
x_CMin = 3
# Запуск оборудования
x_WU = 10


def get_linear_equation_axline(x_0, y_0, slope, x_r):
    k = np.tan(np.deg2rad(slope))
    b = y_0 - k * x_0
    return k * x_r + b


def get_vertices_of_rectangle(x_0, y_0, width, height, angle):
    """Нахождение вершин прямоугольника"""
    # Матрица поворота
    rotate_matrix = np.array([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]
    ])

    # Матрица вершин прямоугольника
    vertices_matrix = np.array([
        [width / 2, -width / 2, -width / 2, width / 2],
        [height / 2, height / 2, -height / 2, -height / 2]
    ])

    # Поворот вершин
    vertices_matrix = np.dot(rotate_matrix, vertices_matrix)

    # Перенос вершин к точке пересечения
    vertices_matrix[0, :] += x_0
    vertices_matrix[1, :] += y_0

    return vertices_matrix


def get_projection_on_axline(slope, x_0, y_0, a_x, a_y):
    # Находим уравнение прямой
    k_0 = np.tan(np.deg2rad(slope))
    b_0 = y_0 - k_0 * x_0

    # Находим проекцию точки на прямую
    x_p = (a_x + k_0 * a_y - k_0 * b_0) / (k_0 ** 2 + 1)
    y_p = k_0 * x_p + b_0

    return x_p, y_p


def get_intersection_of_lines(k_1, b_1, k_2, b_2):
    x = (b_2 - b_1) / (k_1 - k_2)
    y = k_1 * x + b_1
    return x, y


def calculate_polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

fig, ax = plt.subplots()


# Наблюдаемая область
RECT_VERTICES = get_vertices_of_rectangle(lambda_MA, phi_MA, x_MA, y_MA, alpha_MA)
plt.plot(
    np.append(RECT_VERTICES[0, :], RECT_VERTICES[0, 0]),
    np.append(RECT_VERTICES[1, :], RECT_VERTICES[1, 0]),
    'y-'
)
plt.gca().add_patch(plt.Polygon(RECT_VERTICES.T, facecolor='y', alpha=0.1))
plt.plot(lambda_MA, phi_MA, 'yo')

# Траектория полета самолета
ax.axline((lambda_St, phi_St), slope=np.tan(np.deg2rad(alpha_St)), color='b')

# Находим положение наблюдаемой области относительно траектории полета самолета
projection_lengths = []
for vertical in RECT_VERTICES.T:
    x_projection, y_projection = get_projection_on_axline(alpha_St, lambda_St, phi_St, vertical[0], vertical[1])
    projection_lengths.append((
        np.sqrt((x_projection - vertical[0]) ** 2 + (y_projection - vertical[1]) ** 2),
        (vertical[0], vertical[1])
    ))

max_length_projection = max(projection_lengths, key=lambda p: p[0])
LEFT_POS_OF_RECT = (max_length_projection[1][1]
                    > get_linear_equation_axline(lambda_St, phi_St, alpha_St, max_length_projection[1][0]))


# Рисуем ограничительные линии возможности съемки камеры
hyp_min = np.sqrt(2 * x_CMin ** 2)
hyp_max = np.sqrt(2 * x_CMax ** 2)
if LEFT_POS_OF_RECT:
    lambda_CMin, phi_CMin = lambda_St - hyp_min, phi_St + hyp_min / 2
    lambda_CMax, phi_CMax = lambda_St - hyp_max, phi_St + hyp_max / 2

else:
    lambda_CMin, phi_CMin = lambda_St + hyp_min, phi_St - hyp_min / 2
    lambda_CMax, phi_CMax = lambda_St + hyp_max, phi_St - hyp_max / 2

ax.axline((lambda_CMin, phi_CMin), slope=np.tan(np.deg2rad(alpha_St)), color='b', ls=':', alpha=0.5)
ax.axline((lambda_CMax, phi_CMax), slope=np.tan(np.deg2rad(alpha_St)), color='b', ls=':', alpha=0.5)

# Находим точки пересечения линий ограничительных возможностей камеры с прямоугольником наблюдаемой области
recorded_area_points = []
vertical_pairs = [(RECT_VERTICES.T[i], RECT_VERTICES.T[i + 1 if i != 3 else 0]) for i in range(4)]
# Находим уравнение ограничений камеры
k_limit = np.tan(np.deg2rad(alpha_St))
b_min_limit = (phi_St + hyp_min / 2) - k_limit * (lambda_St - hyp_min)
b_max_limit = (phi_St + hyp_max / 2) - k_limit * (lambda_St - hyp_max)
for pair in vertical_pairs:
    # Находим уравнение прямой через две точки
    k_rectangle_line = (pair[0][1] - pair[1][1]) / (pair[0][0] - pair[1][0])
    b_rectangle_line = pair[0][1] - k_rectangle_line * pair[0][0]

    y_limit_min_linear = get_linear_equation_axline(lambda_CMin, phi_CMin, alpha_St, pair[0][0])
    y_limit_max_linear = get_linear_equation_axline(lambda_CMax, phi_CMax, alpha_St, pair[0][0])
    if y_limit_min_linear <= pair[0][1] <= y_limit_max_linear and LEFT_POS_OF_RECT:
        recorded_area_points.append((pair[0][0], pair[0][1]))
    elif y_limit_min_linear >= pair[0][1] >= y_limit_max_linear and not LEFT_POS_OF_RECT:
        recorded_area_points.append((pair[0][0], pair[0][1]))

    # Находим точку пересечения прямой и ограничения
    for b_limit in [b_min_limit, b_max_limit]:
        x, y = get_intersection_of_lines(k_limit, b_limit, k_rectangle_line, b_rectangle_line)
        if pair[0][0] <= x <= pair[1][0] or pair[1][0] <= x <= pair[0][0]:
            recorded_area_points.append((x, y))


# Выделяем область, которую можно снять
plt.gca().add_patch(plt.Polygon(recorded_area_points, edgecolor='b', facecolor='y', hatch='///', alpha=0.5))


# Находим точку старта съемки и рисуем перпендикуляр к траектории полета
recorded_area_projections = [
    get_projection_on_axline(alpha_St, lambda_St, phi_St, recorded_area_point[0], recorded_area_point[1])
    for recorded_area_point in recorded_area_points
]
recorded_area_projections.sort(key=lambda p: p[0])

start_point = recorded_area_projections[0]
plt.plot(start_point[0], start_point[1], 'go', label='Точка начала съемки')
start_point_on_limit_max = get_projection_on_axline(
    alpha_St, lambda_CMax, phi_CMax, start_point[0], start_point[1]
)
plt.plot(
    (start_point[0], start_point_on_limit_max[0]),
    (start_point[1], start_point_on_limit_max[1]),
    'g--', alpha=0.5
)

end_point = recorded_area_projections[-1]
plt.plot(end_point[0], end_point[1], 'ro', label='Точка окончания съемки')
end_point_on_limit_max = get_projection_on_axline(
    alpha_St, lambda_CMax, phi_CMax, end_point[0], end_point[1]
)
plt.plot(
    (end_point[0], end_point_on_limit_max[0]),
    (end_point[1], end_point_on_limit_max[1]),
    'r--', alpha=0.5
)

# Находим площадь наблюдаемой области, которую можно снять
recorded_area = calculate_polygon_area(recorded_area_points)
searched_area = calculate_polygon_area(RECT_VERTICES.T)
print(recorded_area, searched_area, recorded_area / searched_area)

info_text = f'Заснятой территории: {(recorded_area / searched_area) * 100:.2f}%'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.03, 0.03, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

plt.title('Съемка наблюдаемой области самолетом')
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
ax.margins(0.5)
plt.show()
