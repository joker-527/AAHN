import pickle

import os
import pickle

def read_pickle(work_path):
    data_list = []
    with open(work_path, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)
                    data_list.append(data)
                except EOFError:
                    break
    return data_list

def main():
    pkl_path = '/home/zheng/Yexin/Project02/nnUNet/' \
               'nnUNet_trained_models/nnUNet/2d/Task043_BraTS2019' \
               '/nnUNetTrainer_missFormer__nnUNetPlansv2.1/all/model_best.model.pkl'
    data_list = read_pickle(pkl_path)
    print(data_list)
if __name__=="__main__":
    main()

