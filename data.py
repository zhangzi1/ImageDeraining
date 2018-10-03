from utils import *


class Data:
    def __init__(self):
        n_path = "./data/n/"
        r_path = "./data/r/"
        if os.path.exists(n_path) and os.path.exists(r_path):
            self.n_path = n_path
            self.r_path = r_path
        else:
            print("[!] Data path does not exist!")

    def n_sample(self, num):
        matrix = []
        for _, _, files in os.walk(self.n_path):
            np.random.shuffle(files)
            file_list = random.sample(files, num)
            for file in file_list:
                matrix.append(normalize(read_image(self.n_path + file))[0])
        return matrix

    def r_sample(self, num):
        matrix = []
        for _, _, files in os.walk(self.r_path):
            np.random.shuffle(files)
            file_list = random.sample(files, num)
            for file in file_list:
                matrix.append(normalize(read_image(self.r_path + file))[0])
        return matrix
