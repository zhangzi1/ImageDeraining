from utils import *


class Buffer:

    # Matrix:[None,35,55,1]

    def __init__(self, path):
        if os.path.exists(path):
            self.path = path
            self.clear()
        else:
            print("[!] Buffer path does not exist! ")

    def push(self, matrix):
        matrix = np.array(matrix)
        for i in range(matrix.shape[0]):
            write_image(denormalize(matrix[i]), self.path)
        return self.path

    def sample(self, num):
        matrix = []
        for _, _, files in os.walk(self.path):
            file_list = random.sample(files, num)
            for file in file_list:
                matrix.append(normalize(read_image(self.path + file))[0])
        return matrix

    def random_replace(self, matrix):
        matrix = np.array(matrix)
        for _, _, files in os.walk(self.path):
            file_list = random.sample(files, matrix.shape[0])
            for file in file_list:
                os.remove(self.path + file)
        for i in range(matrix.shape[0]):
            write_image(denormalize(matrix[i]), self.path)

    def clear(self):
        for _, _, files in os.walk(self.path):
            for file in files:
                os.remove(self.path + file)
