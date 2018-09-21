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
    matrix = np.reshape(list(image.getdata()), [image.size[1], image.size[0], 3])
    return matrix


def write_image(matrix, path):
    matrix = np.array(matrix)
    image = Image.fromarray(matrix.astype(np.int8), "RGB")
    now = lambda: int(round(time.time() * 10000))
    image.save(path + str(now()) + ".png")


def normalize(matrix):
    matrix = np.array(matrix)
    normalized_matrix = np.reshape(matrix / 127.5 - 1, [1, matrix.shape[0], matrix.shape[1], 3])
    return normalized_matrix


def denormalize(matrix):
    matrix = np.array(matrix)
    denormalized_matrix = np.reshape((matrix + 1) * 127.5, [matrix.shape[0], matrix.shape[1], 3])
    return denormalized_matrix


def concat(matrix):
    matrix = np.array(matrix)
    side_len = int(np.sqrt(matrix.shape[0]))
    first_row = matrix[0].reshape([matrix.shape[1], matrix.shape[2], 3])
    for i in range(1, side_len):
        img = matrix[i].reshape([matrix.shape[1], matrix.shape[2], 3])
        first_row = np.concatenate([first_row, img], axis=1)
    image_matrix = first_row
    for j in range(1, side_len):
        row = matrix[j * side_len].reshape([matrix.shape[1], matrix.shape[2], 3])
        for k in range(1, side_len):
            img = matrix[j * side_len + k].reshape([matrix.shape[1], matrix.shape[2], 3])
            row = np.concatenate([row, img], axis=1)
        image_matrix = np.concatenate([image_matrix, row], axis=0)
    image_matrix = image_matrix.reshape([1, image_matrix.shape[0], image_matrix.shape[1], 3])
    return image_matrix


def sampling(matrix):
    matrix = np.array(matrix)
    size = matrix.shape[0]
    matrix = matrix.tolist()
    return random.sample(matrix, int(size / 2))


def PSNR(matrix1, matrix2):
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    height = matrix1.shape[1]
    width = matrix1.shape[2]
    channels = matrix1.shape[3]
    MSE = 0
    for i in range(height):
        for j in range(width):
            pixel = 0
            for k in range(channels):
                pixel = pixel + np.square(matrix1[0][i][j][k] - matrix2[0][i][j][k])
            MSE = MSE + pixel / 3.0
    MSE = MSE / (height * width)
    PSNR = 10 * np.log10(2 * 2 / MSE)
    return PSNR


if __name__ == "__main__":
    write_image(denormalize(normalize(read_image("./data/n/0.png"))[0]), "./samples/")
