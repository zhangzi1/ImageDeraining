from utils import *


class Data:
    def __init__(self, syn_path, real_path):
        if os.path.exists(syn_path) and os.path.exists(real_path):
            self.syn_path = syn_path
            self.real_path = real_path
        else:
            print("[!] Data path does not exist!")

    def syn_sample(self, num):
        matrix = []
        for _, _, files in os.walk(self.syn_path):
            file_list = random.sample(files, num)
            for file in file_list:
                matrix.append(normalize(read_image(self.syn_path + file))[0])
        return matrix

    def real_sample(self, num):
        matrix = []
        for _, _, files in os.walk(self.real_path):
            file_list = random.sample(files, num)
            for file in file_list:
                matrix.append(normalize(read_image(self.real_path + file))[0])
        return matrix
