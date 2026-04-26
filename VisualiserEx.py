import numpy as np
from PIL import Image
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

class Point:
    def __init__(endl, a, b):
        endl.x = a
        endl.y = b

def load_terrain_maps(colormap_path="texture.png", heightmap_path="heightmap.png"):
    h_img = Image.open(heightmap_path)
    heightmap = np.array(h_img, dtype=np.float32) / 256.0

    c_img = Image.open(colormap_path).convert('RGB')
    colormap = np.array(c_img, dtype=np.uint8)

    heightmap = heightmap.T
    colormap = colormap.transpose(1, 0, 2)

    return heightmap, colormap

heightmap, colormap = load_terrain_maps()

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 800)
ax.set_ylim(600, 0)
ax.axis('off')

def DrawVerticalLine(x, y1, y2, color):
    color = to_hex(color / 255.0)
    ax.vlines(x=x, ymin=y2, ymax=y1, colors=color, linewidth=2)

# DrawVerticalLine(600, 600, 1025, "#ff0000")
# plt.show()

def Render(p, height, horizon, scale_height, distance, screen_width, screen_height):
    # Отрисовка от заднего плана к переднему (от больших значений Z к меньшим)
    for z in range(distance, 1, -1):
        # Найти линию на карте. Этот расчёт соответствует полю зрения 90°
        pleft  = Point(-z + p.x, -z + p.y)
        pright = Point( z + p.x, -z + p.y)
        # Разбить линию на сегменты
        dx = (pright.x - pleft.x) / screen_width
        # Растеризовать линию и нарисовать вертикальную линию для каждого сегмента
        for i in range(screen_width):
            map_x = int(round(pleft.x))
            map_y = int(round(pleft.y))
            h_val = heightmap[map_x, map_y]
            rgb = colormap[map_x, map_y]

            height_on_screen = (height - h_val) / z * scale_height + horizon

            DrawVerticalLine(i, height_on_screen, screen_height, rgb)
            print(z, i)
            pleft.x += dx

Render( Point(600, 0), 50, 120, 120, 300, 800, 600)

plt.show()