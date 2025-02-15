from PIL import Image
from pprint import pprint
import numpy as np
import time
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

# Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)


def classify_color(rgb):
    """Classifies the color based on RGB values."""
    return WHITE if rgb[0] >= 125 else BLUE if rgb[2] >= 125 else BLACK

def is_blue_point_center(position, image):
    """Checks if a pixel position is the center of a blue region."""
    width, height = image.size
    offsets = [-1, 0, 1]
    blue_pixel_count = sum(
        1 for dx in offsets for dy in offsets
        if 0 <= position[0] + dx < width and 0 <= position[1] + dy < height
        and image.getpixel((position[0] + dx, position[1] + dy)) == BLUE
    )
    return blue_pixel_count >= 7

def calculate_distance(x1, y1, x2, y2):
    """Calculates the squared Euclidean distance between two points."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def find_center(positions):
    """Finds the geometric center of a cluster of points."""
    x_sum, y_sum = map(sum, zip(*positions))
    return (x_sum // len(positions), y_sum // len(positions))

def are_pixels_connected(pixel1, pixel2, image):
    """Determines if two pixels are connected by checking the color pattern along the path."""
    if pixel1 == pixel2:
        return False
    x1, y1 = pixel1
    x2, y2 = pixel2
    vx, vy = x2 - x1, y2 - y1
    steps = round(max(abs(vx), abs(vy)) * 0.7)
    path_colors = [
        image.getpixel(tuple(map(round, (x1 + vx * t / steps, y1 + vy * t / steps))))
        for t in range(steps + 1)
    ]
    return path_colors.count(BLACK) >= round(len(path_colors) * 0.5) if is_collinear(path_colors) else False

def is_collinear(colors_list):
    """Checks if the color transition follows a collinear pattern."""
    for i, color in enumerate(colors_list[:-1]):
        if color == BLACK and colors_list[i + 1] == BLUE:
            return i >= round(len(colors_list) * 0.8)
    return False

def process_image(image_path):
    """Loads an image, processes it."""
    image = Image.open(image_path).convert("RGB")
    processed_image = Image.new("RGB", image.size, WHITE)
    for x in range(image.width):
        for y in range(image.height):
            processed_image.putpixel((x, y), classify_color(image.getpixel((x, y))))
    return processed_image

def find_blue_regions(image):
    """Finds clusters of blue pixels and groups them into regions."""
    width, height = image.size
    blue_regions = []
    for x in range(width):
        for y in range(height):
            if is_blue_point_center((x, y), image):
                for region in reversed(blue_regions):
                    if calculate_distance(x, y, region[0][0], region[0][1]) < 30:
                        region.append((x, y))
                        break
                else:
                    blue_regions.append([(x, y)])
    return blue_regions

def build_adj_matrix(center_points, image):
    """Builds an adjacency matrix representing pixel connections."""
    num_points = len(center_points)
    adj_matrix = np.zeros((num_points, num_points), dtype=int)
    for i in range(num_points):
        for j in range(i, num_points):
            if are_pixels_connected(center_points[i], center_points[j], image):
                adj_matrix[i, j] = adj_matrix[j, i] = 1
    return adj_matrix

def img2adj(image_path, index=0):
    """Main function that processes the image, finds regions, builds the adjacency matrix, and computes the Wiener number."""
    processed_image = process_image(image_path)
    blue_regions = find_blue_regions(processed_image)
    center_points = list(map(find_center, blue_regions))
    adj_matrix = build_adj_matrix(center_points, processed_image)
    return center_points, adj_matrix


if __name__ == "__main__":
    images_path = 'images'
    image_path = os.path.join(images_path, "image_0.png")
    adj_matrix = img2adj(image_path=image_path)
    pprint(adj_matrix)
