import os
import random
import time

import numpy as np
from PIL import Image


def check(path):
    for _, _, files in os.walk(path):
        for file in files:
            print(file)
            if file == ".DS_Store":
                os.remove(path + file)


def read_image(path):
    image = Image.open(path)
    matrix = np.reshape(list(image.getdata()), [image.size[1], image.size[0]])
    return matrix


def write_image(matrix, path):
    matrix = np.array(matrix)
    image = Image.fromarray(matrix.astype(np.int8), "L")
    now = lambda: int(round(time.time() * 10000))
    image.save(path + str(now()) + ".png")


def normalize(matrix):
    matrix = np.array(matrix)
    normalized_matrix = np.reshape(matrix / 127.5 - 1, [1, matrix.shape[0], matrix.shape[1], 1])
    return normalized_matrix


def denormalize(matrix):
    matrix = np.array(matrix)
    denormalized_matrix = np.reshape((matrix + 1) * 127.5, [matrix.shape[0], matrix.shape[1]])
    return denormalized_matrix


def concat(matrix):
    matrix = np.array(matrix)
    side_len = int(np.sqrt(matrix.shape[0]))
    first_row = matrix[0].reshape([matrix.shape[1], matrix.shape[2]])
    for i in range(1, side_len):
        img = matrix[i].reshape([matrix.shape[1], matrix.shape[2]])
        first_row = np.concatenate([first_row, img], axis=1)
    image_matrix = first_row
    for j in range(1, side_len):
        row = matrix[j * side_len].reshape([matrix.shape[1], matrix.shape[2]])
        for k in range(1, side_len):
            img = matrix[j * side_len + k].reshape([matrix.shape[1], matrix.shape[2]])
            row = np.concatenate([row, img], axis=1)
        image_matrix = np.concatenate([image_matrix, row], axis=0)
    image_matrix = image_matrix.reshape([1, image_matrix.shape[0], image_matrix.shape[1], 1])
    return image_matrix


def sampling(matrix):
    matrix = np.array(matrix)
    size = matrix.shape[0]
    matrix = matrix.tolist()
    return random.sample(matrix, int(size / 2))


if __name__ == "__main__":
    write_image(denormalize(normalize(read_image("./data/syn/0.png"))[0]), "./")
