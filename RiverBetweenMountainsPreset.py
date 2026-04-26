import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.spatial import cKDTree
from PIL import Image


p0 = np.array([0, rd.randint(0, 1025)])
p1 = np.array([rd.randint(0, 1025), rd.randint(0, 1025)])
p2 = np.array([rd.randint(0, 1025), rd.randint(0, 1025)])
p3 = np.array([1025, rd.randint(0, 1025)])

STEP = 0.001
RIVER_WIDTH = 1
MNTN_SPACE = 40
IMAGE_SIZE = 1025
BORDER_HEIGHT = 1.0
MOUNTAIN_HEIGHT = 10.0
SIGMA_FACTOR = 0.1
FADE_DIST = 60.0

print(f"seed: {p0[0]}.{p0[1]}.{p1[0]}.{p1[1]}.{p2[0]}"
      f".{p2[1]}.{p3[0]}.{p3[1]}.{STEP}.{RIVER_WIDTH}"
      f".{MNTN_SPACE}.{IMAGE_SIZE}.{BORDER_HEIGHT}."
      f"{MOUNTAIN_HEIGHT}.{SIGMA_FACTOR}")

def p(t):
    return (p0 * (-(t ** 3) + 3 * t ** 2 - 3 * t + 1) +
            p1 * (3 * t ** 3 - 6 * t ** 2 + 3 * t) +
            p2 * (-3 * (t ** 3) + 3 * (t ** 2)) +
            p3 * t ** 3)


def dp(t):
    return (p0 * (-3 * t ** 2 + 6 * t - 3) +
            p1 * (9 * t ** 2 - 12 * t + 3) +
            p2 * (-9 * t ** 2 + 6 * t) +
            p3 * (3 * t ** 2))


def ddp(t):
    return (p0 * (-6 * t + 6) +
            p1 * (18 * t - 12) +
            p2 * (-18 * t + 6) +
            p3 * (6 * t))


x = np.arange(0, 1 + STEP, STEP)

curve_x, curve_y = [], []
der_x, der_y = [], []
der_x2, der_y2 = [], []
curvative_x, curvative_y = [], []
curve_dots_x = []
normals_list = []

for t in x:
    curve_x.append(p(t)[0])
    curve_y.append(p(t)[1])

    # Нормаль
    tangent = dp(t)
    tangent_norm = tangent / np.linalg.norm(tangent)
    normal = np.array([tangent_norm[1], -tangent_norm[0]])
    normals_list.append(normal)

    # Берега
    der_x.append((p(t) + normal * RIVER_WIDTH)[0])
    der_y.append((p(t) + normal * RIVER_WIDTH)[1])
    der_x2.append((p(t) - normal * RIVER_WIDTH)[0])
    der_y2.append((p(t) - normal * RIVER_WIDTH)[1])

    # Кривизна
    v, a = dp(t), ddp(t)
    numerator = np.linalg.norm(v) ** 3
    denominator = v[0] * a[1] - v[1] * a[0]
    curvative_x.append(t)
    curvative_y.append(denominator / numerator)

# Поиск вершин (максимумы/минимумы кривизны)
for i in range(1, len(curvative_y) - 1):
    if curvative_y[i - 1] < curvative_y[i] > curvative_y[i + 1]:
        curve_dots_x.append([curvative_x[i], -1])
    if curvative_y[i - 1] > curvative_y[i] < curvative_y[i + 1]:
        curve_dots_x.append([curvative_x[i], 1])

# Преобразуем списки в массивы
center_curve = np.column_stack((curve_x, curve_y))
upper_curve = np.column_stack((der_x, der_y))
lower_curve = np.column_stack((der_x2, der_y2))
all_banks = np.vstack([upper_curve, lower_curve])
normals = np.array(normals_list)


# ==========================================
# 2. КЛАССИФИКАЦИЯ ГОР
# ==========================================
mountains_left = []
mountains_right = []

print(f"Найдено вершин: {len(curve_dots_x)}")

for item in curve_dots_x:
    t, side_flag = item[0], item[1]
    curve_point = p(t)

    tangent = dp(t)
    tangent_norm = tangent / np.linalg.norm(tangent)
    normal = np.array([tangent_norm[1], -tangent_norm[0]])

    mountain_coord = curve_point + side_flag * normal * MNTN_SPACE * RIVER_WIDTH

    if side_flag == 1:
        mountains_right.append(mountain_coord)
    else:
        mountains_left.append(mountain_coord)

mountains_right = np.array(mountains_right) if len(mountains_right) > 0 else np.empty((0, 2))
mountains_left = np.array(mountains_left) if len(mountains_left) > 0 else np.empty((0, 2))
print(f"Гор справа: {mountains_right.shape[0]}, слева: {mountains_left.shape[0]}")

# ==========================================
# 3. КОНФИГУРАЦИЯ И KD-TREES
# ==========================================
tree_center = cKDTree(center_curve)
tree_banks = cKDTree(all_banks)
tree_mtn_right = cKDTree(mountains_right) if len(mountains_right) > 0 else None
tree_mtn_left = cKDTree(mountains_left) if len(mountains_left) > 0 else None


# ==========================================
# 4. ГЕНЕРАЦИЯ СЕТКИ И РАСЧЕТ РАССТОЯНИЙ
# ==========================================
y_grid, x_grid = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
pixels = np.column_stack([x_grid.ravel(), y_grid.ravel()])

# А. До центра реки и определение стороны
dist_to_center, idx_center = tree_center.query(pixels)
normals_at_pixels = normals[idx_center]
vec_to_pixel = pixels - center_curve[idx_center]

dot_products = np.einsum('ij,ij->i', vec_to_pixel, normals_at_pixels)
is_right_bank = dot_products > 0
is_left_bank = dot_products <= 0
is_river_mask = dist_to_center < RIVER_WIDTH

# Б. До берегов
dist_to_bank, _ = tree_banks.query(pixels)

# В. До границы
dist_to_border = np.minimum.reduce([
    x_grid.ravel(), (IMAGE_SIZE - x_grid.ravel()),
    y_grid.ravel(), (IMAGE_SIZE - y_grid.ravel())
])

# Г. До гор
dist_to_mountain = np.full(len(pixels), np.inf)
if tree_mtn_right:
    d_r, _ = tree_mtn_right.query(pixels[is_right_bank])
    dist_to_mountain[is_right_bank] = d_r
if tree_mtn_left:
    d_l, _ = tree_mtn_left.query(pixels[is_left_bank])
    dist_to_mountain[is_left_bank] = d_l

# ==========================================
# 5. РАСЧЕТ ВЫСОТЫ (С ВАРИАНТОМ SMOOTHSTEP)
# ==========================================

# 1. Базовый склон (от реки к границе)
denominator = dist_to_bank + dist_to_border + 1e-5
base_slope = BORDER_HEIGHT * (dist_to_bank / denominator)

# 2. Влияние гор (Гауссово распределение)
sigma = MOUNTAIN_HEIGHT / SIGMA_FACTOR
gaussian_mtn = MOUNTAIN_HEIGHT * np.exp(- (dist_to_mountain ** 2) / (2 * sigma ** 2))

# Формула Smoothstep: 3x^2 - 2x^3
# Мы нормализуем расстояние от берега в диапазон [0, 1]
t = np.clip(dist_to_bank / FADE_DIST, 0.0, 1.0)
falloff = t * t * (3.0 - 2.0 * t)

# Домножаем высоту горы на этот коэффициент (у берега = 0, вдали = 1)
gaussian_mtn_adjusted = gaussian_mtn * falloff

# 4. Итоговая высота (Максимум из склона и горы)
final_height = np.maximum(base_slope, gaussian_mtn_adjusted)

# 5. Жесткая маска реки (чтобы вода была идеально ровной)
final_height[is_river_mask] = 0.0

# Возвращаем в форму картинки
height_map = final_height.reshape(IMAGE_SIZE, IMAGE_SIZE)

# ==========================================
# 6. РАСЧЕТ ЦВЕТОВ (ЕСТЕСТВЕННАЯ ПАЛИТРА)
# ==========================================

from matplotlib.colors import LinearSegmentedColormap

# 1. Нормализация и приведение к 2D
norm_height = np.clip(final_height / MOUNTAIN_HEIGHT, 0, 1)
norm_height_2d = norm_height.reshape(IMAGE_SIZE, IMAGE_SIZE)

# 2. Создаём кастомную палитру
colors = [
    (0.0, (0.0, 0.1, 0.4)),   # Темно-синий (глубокая река)
    (0.05, (0.0, 0.3, 0.8)),  # Бирюзовый (мелководье/берег)
    (0.1, (0.1, 0.7, 0.1)),   # Тёмно-зелёный (низменности)
    (0.3, (0.9, 0.95, 0.1)),  # Светло-жёлтый (холмы)
    (0.7, (0.45, 0.3, 0.1)),  # Коричневый (горные склоны)
    (1.0, (1.0, 1.0, 1.0))    # Белый (снежные вершины)
]
custom_cmap = LinearSegmentedColormap.from_list('natural_map', colors, N=256)

# 3. Применяем палитру (вернёт массив HxWx4 в формате RGBA)
color_map = custom_cmap(norm_height_2d)

# 4. Жёстко задаём цвет реки в цветной карте (RGBA)
river_mask_2d = is_river_mask.reshape(IMAGE_SIZE, IMAGE_SIZE)
color_map[river_mask_2d] = [0.0, 0.1, 0.4, 1.0]

# 5. Карта высот в оттенках серого (Grayscale)
height_palette = np.stack([norm_height_2d] * 3, axis=-1)
height_palette[river_mask_2d] = [0, 0, 0]


# ==========================================
# 7. СОХРАНЕНИЕ И ВЫВОД (ГАРАНТИРОВАННАЯ БИТНОСТЬ)
# ==========================================


# 1. Карта высот -> 16-bit grayscale (1 канал, 16 бит на пиксель)
height_map_16bit = (norm_height_2d * 65535).astype(np.uint16)
Image.fromarray(height_map_16bit, mode='I;16').save('heightmap.png')

# 2. Карта цветов -> 24-bit RGB (3 канала, 8 бит на канал)
# color_map от colormapa имеет 4 канала (RGBA), отрезаем альфу
color_map_rgb = (color_map[:, :, :3] * 255).astype(np.uint8)
Image.fromarray(color_map_rgb, mode='RGB').save('texture.png')

print("✅ Карты сохранены с корректной битностью:")
print("   - heightmap.png: 16-bit grayscale")
print("   - texture.png:   24-bit RGB")

# --- Предпросмотр (опционально) ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(height_map_16bit, cmap='gray', vmin=0, vmax=65535)
plt.title("Height Map (16-bit)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(color_map_rgb)
plt.title("Texture Map (24-bit RGB)")
plt.axis('off')

plt.tight_layout()
plt.show()