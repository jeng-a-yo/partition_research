from generate_data_set import generate_data
from img2adj import img2adj
from adj2part import adj2part
from tools import time_it

def img2part(image_path):
    adj_matrix = img2adj(image_path=image_path)
    partitioning, modularity = adj2part(adj_matrix)
    return partitioning, modularity

@time_it
def main():
    num_images = 10
    points_quantity = 25
    images_path="images"
    matrices_path="adjacency_matrices"
    generate_data(num_images=num_images, points_quantity=points_quantity, images_path=images_path, matrices_path=matrices_path)

    for i in range(num_images):
        image_path = os.path.join(images_path, f"image_{i}.png")
        partitioning, modularity = img2part(image_path)
        for k, group in partitioning.items():
            print(f"Virtual Microgrid {k + 1}: {group}")
        print(f"Modularity: {modularity}")
        print("----------------------------------------------------------------")

if __name__ == '__main__':
    main()