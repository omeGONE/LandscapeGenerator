import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.spatial import cKDTree
from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# ==========================================
# СОЗДАНИЕ СИДА
# ==========================================
p0 = np.array([0, rd.randint(0, 1025)])
p1 = np.array([rd.randint(0, 1025), rd.randint(0, 1025)])
p2 = np.array([rd.randint(0, 1025), rd.randint(0, 1025)])
p3 = np.array([1025, rd.randint(0, 1025)])

# p0 = np.array([0, 760])
# p1 = np.array([1013, 662])
# p2 = np.array([500, 43])
# p3 = np.array([1025, 285])

STEP = 0.001
RIVER_WIDTH = 10
MNTN_SPACE = 40
IMAGE_SIZE = 1025
BORDER_HEIGHT = 0.5
MOUNTAIN_HEIGHT = 10.0
SIGMA_FACTOR = 0.05
FADE_DIST = 60.0
NOISE_SIGMA = 1
COLOR_NOISE_SIGMA = 4
HUE_NOISE_SIGMA = 6
SATURATION_NOISE_SIGMA = 3
VALUE_NOISE_SIGMA = 3
NOISE_STRENGTH = 0.01
HUE_STRENGTH = 0.005
SATURATION_STRENGTH = 0.1
VALUE_STRENGTH = 0.08

print(f"seed: {p0[0]}:{p0[1]}:{p1[0]}:{p1[1]}:{p2[0]}"
      f":{p2[1]}:{p3[0]}:{p3[1]}:{STEP}:{RIVER_WIDTH}"
      f":{MNTN_SPACE}:{IMAGE_SIZE}:{BORDER_HEIGHT}"
      f":{MOUNTAIN_HEIGHT}:{SIGMA_FACTOR}:{FADE_DIST}"
      f":{NOISE_SIGMA}:{COLOR_NOISE_SIGMA}:{HUE_NOISE_SIGMA}"
      f":{SATURATION_NOISE_SIGMA}:{VALUE_NOISE_SIGMA}"
      f":{NOISE_STRENGTH}:{HUE_STRENGTH}:{SATURATION_STRENGTH}:{VALUE_STRENGTH}")


# ==========================================
# ГЕНЕРАЦИЯ КРИВОЙ
# ==========================================


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

    tangent = dp(t)
    tangent_norm = tangent / np.linalg.norm(tangent)
    normal = np.array([tangent_norm[1], -tangent_norm[0]])
    normals_list.append(normal)

    der_x.append((p(t) + normal * RIVER_WIDTH)[0])
    der_y.append((p(t) + normal * RIVER_WIDTH)[1])
    der_x2.append((p(t) - normal * RIVER_WIDTH)[0])
    der_y2.append((p(t) - normal * RIVER_WIDTH)[1])

    v, a = dp(t), ddp(t)
    numerator = np.linalg.norm(v) ** 3
    denominator = v[0] * a[1] - v[1] * a[0]
    curvative_x.append(t)
    curvative_y.append(denominator / numerator)

for i in range(1, len(curvative_y) - 1):
    if curvative_y[i - 1] < curvative_y[i] > curvative_y[i + 1]:
        curve_dots_x.append([curvative_x[i], -1])
    if curvative_y[i - 1] > curvative_y[i] < curvative_y[i + 1]:
        curve_dots_x.append([curvative_x[i], 1])

center_curve = np.column_stack((curve_x, curve_y))
upper_curve = np.column_stack((der_x, der_y))
lower_curve = np.column_stack((der_x2, der_y2))
all_banks = np.vstack([upper_curve, lower_curve])
normals = np.array(normals_list)


# ==========================================
# КЛАССИФИКАЦИЯ ГОР
# ==========================================


mountains_left = []
mountains_right = []

for item in curve_dots_x:
    t, side_flag = item[0], item[1]
    curve_point = p(t)

    tangent = dp(t)
    tangent_norm = tangent / np.linalg.norm(tangent)
    normal = np.array([tangent_norm[1], -tangent_norm[0]])

    mountain_coord = curve_point + side_flag * normal * MNTN_SPACE

    if side_flag == 1:
        mountains_right.append(mountain_coord)
    else:
        mountains_left.append(mountain_coord)

mountains_right = np.array(mountains_right) if len(mountains_right) > 0 else np.empty((0, 2))
mountains_left = np.array(mountains_left) if len(mountains_left) > 0 else np.empty((0, 2))

# ==========================================
# КОНФИГУРАЦИЯ И KD-TREES
# ==========================================


tree_center = cKDTree(center_curve)
tree_banks = cKDTree(all_banks)
tree_mtn_right = cKDTree(mountains_right) if len(mountains_right) > 0 else None
tree_mtn_left = cKDTree(mountains_left) if len(mountains_left) > 0 else None


# ==========================================
# ГЕНЕРАЦИЯ СЕТКИ И РАСЧЕТ РАССТОЯНИЙ
# ==========================================


y_grid, x_grid = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
pixels = np.column_stack([x_grid.ravel(), y_grid.ravel()])

dist_to_center, idx_center = tree_center.query(pixels)
normals_at_pixels = normals[idx_center]
vec_to_pixel = pixels - center_curve[idx_center]

dot_products = np.einsum('ij,ij->i', vec_to_pixel, normals_at_pixels)
is_right_bank = dot_products > 0
is_left_bank = dot_products <= 0
is_river_mask = dist_to_center < RIVER_WIDTH

dist_to_bank, _ = tree_banks.query(pixels)

dist_to_border = np.minimum.reduce([
    x_grid.ravel(), (IMAGE_SIZE - x_grid.ravel()),
    y_grid.ravel(), (IMAGE_SIZE - y_grid.ravel())
])

dist_to_mountain = np.full(len(pixels), np.inf)
if tree_mtn_right:
    d_r, _ = tree_mtn_right.query(pixels[is_right_bank])
    dist_to_mountain[is_right_bank] = d_r
if tree_mtn_left:
    d_l, _ = tree_mtn_left.query(pixels[is_left_bank])
    dist_to_mountain[is_left_bank] = d_l


# ==========================================
# РАСЧЕТ ВЫСОТЫ
# ==========================================


denominator = dist_to_bank + dist_to_border + 1e-5
base_slope = BORDER_HEIGHT * (dist_to_bank / denominator)

sigma = MOUNTAIN_HEIGHT / SIGMA_FACTOR
gaussian_mtn = MOUNTAIN_HEIGHT * np.exp(- (dist_to_mountain ** 2) / (2 * sigma ** 2))

t = np.clip(dist_to_bank / FADE_DIST, 0.0, 1.0)
falloff = t * t * (3.0 - 2.0 * t)

gaussian_mtn_adjusted = gaussian_mtn * falloff

final_height = np.maximum(base_slope, gaussian_mtn_adjusted)

final_height[is_river_mask] = 0.0

height_map = final_height.reshape(IMAGE_SIZE, IMAGE_SIZE)


# ==========================================
# РАСЧЕТ ЦВЕТОВ
# ==========================================


norm_height = np.clip(final_height / MOUNTAIN_HEIGHT, 0, 1)
norm_height_2d = norm_height.reshape(IMAGE_SIZE, IMAGE_SIZE)

raw_noise_h = np.random.rand(IMAGE_SIZE, IMAGE_SIZE)
smooth_noise_h = gaussian_filter(raw_noise_h, sigma=NOISE_SIGMA) # Крупные формы
smooth_noise_h = (smooth_noise_h - 0.5) * 2 * NOISE_STRENGTH     # Сила 5%
norm_height_2d = np.clip(norm_height_2d + smooth_noise_h, 0, 1)
river_mask_2d = is_river_mask.reshape(IMAGE_SIZE, IMAGE_SIZE)
norm_height_2d[river_mask_2d] = 0.0

raw_noise_c = np.random.rand(IMAGE_SIZE, IMAGE_SIZE)
smooth_noise_c = gaussian_filter(raw_noise_c, sigma=COLOR_NOISE_SIGMA)
smooth_noise_c = (smooth_noise_c - 0.5) * 2  # Диапазон [-1, 1]

land_colors = [
    (0.0, (0.05, 0.25, 0.05)),
    (0.15, (0.1, 0.45, 0.1)),
    (0.35, (0.4375, 0.5, 0.4)),
    (0.6,  (0.28515625, 0.2734375, 0.2734375)),
    (0.8,  (0.16015625, 0.15625, 0.15625)),
    (1.0,  (1.0, 1.0, 1.0))
]
land_cmap = LinearSegmentedColormap.from_list('land_palette', land_colors, N=256)
color_map = land_cmap(norm_height_2d)  # Форма (H, W, 4)

rgb_part = color_map[:, :, :3]
hsv_part = rgb_to_hsv(rgb_part)

noise_h = gaussian_filter(np.random.rand(IMAGE_SIZE, IMAGE_SIZE), sigma=HUE_NOISE_SIGMA) * 2 - 1
noise_s = gaussian_filter(np.random.rand(IMAGE_SIZE, IMAGE_SIZE), sigma=SATURATION_NOISE_SIGMA) * 2 - 1
noise_v = gaussian_filter(np.random.rand(IMAGE_SIZE, IMAGE_SIZE), sigma=VALUE_NOISE_SIGMA) * 2 - 1

hsv_part[:, :, 0] = (hsv_part[:, :, 0] + noise_h * HUE_STRENGTH) % 0.5

hsv_part[:, :, 1] = np.clip(hsv_part[:, :, 1] + noise_s * SATURATION_STRENGTH, 0.0, 1.0)
hsv_part[:, :, 2] = np.clip(hsv_part[:, :, 2] + noise_v * VALUE_STRENGTH, 0.0, 1.0)

color_map[:, :, :3] = hsv_to_rgb(hsv_part)

river_color = [0.02, 0.18, 0.42, 1.0]
color_map[river_mask_2d] = river_color

height_palette = np.stack([norm_height_2d] * 3, axis=-1)
height_palette[river_mask_2d] = [0, 0, 0]


# ==========================================
# 7. СОХРАНЕНИЕ И ВЫВОД (ГАРАНТИРОВАННАЯ БИТНОСТЬ)
# ==========================================


height_map_16bit = (norm_height_2d * 65535).astype(np.uint16)
Image.fromarray(height_map_16bit, mode='I;16').save('heightmap.png')

color_map_rgb = (color_map[:, :, :3] * 255).astype(np.uint8)
Image.fromarray(color_map_rgb, mode='RGB').save('texture.png')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(height_map_16bit, cmap='gray', vmin=0, vmax=65535)
plt.title("Height Map")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(color_map_rgb)
plt.title("Texture Map")
plt.axis('off')

plt.tight_layout()
plt.show()