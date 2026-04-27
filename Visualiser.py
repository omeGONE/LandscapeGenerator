import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def load_terrain_maps(colormap_path="texture.png", heightmap_path="heightmap.png"):
    h_img = Image.open(heightmap_path)
    heightmap = np.array(h_img, dtype=np.float32) / 256.0
    c_img = Image.open(colormap_path).convert('RGB')
    colormap = np.array(c_img, dtype=np.uint8)
    heightmap = heightmap.T
    colormap = colormap.transpose(1, 0, 2)
    return heightmap, colormap


heightmap, colormap = load_terrain_maps()
H, W = heightmap.shape


def Render(p, height, horizon, scale_height, distance, screen_width, screen_height):
    framebuffer = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    framebuffer[:] = [135, 206, 235]

    for z in range(distance, 1, -1):
        pleft_x = -z + p.x
        pleft_y = -z + p.y
        pright_x = z + p.x
        dx = (pright_x - pleft_x) / screen_width

        for i in range(screen_width):
            map_x = int(round(pleft_x))
            map_y = int(round(pleft_y))

            if map_x < 0 or map_x >= W or map_y < 0 or map_y >= H:
                pleft_x += dx
                continue

            h_val = heightmap[map_x, map_y]
            y_screen = int((height - h_val) / z * scale_height + horizon)

            if y_screen < 0: y_screen = 0
            if y_screen >= screen_height:
                pleft_x += dx
                continue

            framebuffer[y_screen:, i, :] = colormap[map_x, map_y]
            pleft_x += dx

    return framebuffer

def visualise():
    frame = Render(Point(512, 1025), 100, 120, 550, 3000, 800, 600)
    plt.figure(figsize=(8, 6))
    plt.imshow(frame)
    plt.axis('off')
    plt.tight_layout()
    plt.show()