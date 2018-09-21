from utils import *


class Sample:
    def __init__(self, path):
        if os.path.exists(path):
            self.path = path + time.asctime(time.localtime(time.time())) + "/"
            os.makedirs(self.path)
        else:
            print("[!] Sample path does not exist! ")

    def push(self, matrix):
        matrix = np.array(matrix)
        for i in range(matrix.shape[0]):
            write_image(denormalize(matrix[i]), self.path)
        return self.path
