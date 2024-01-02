from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse

def main():

    val_id = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]
    img_folder = '/home/zheng/Yexin/Project02/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task043_BraTS2019/labelsTr/'
    img_folder1 = '/home/zheng/Yexin/Project02/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task043_BraTS2019/imagesTr/'
    train_patient_names = []
    train_patients = os.listdir(img_folder)
    all_file = os.listdir(img_folder1)
   # print(train_patients)

    for p in train_patients:
        train_patient_name = p.replace('.nii.gz','')
        print(train_patient_name)
        train_patient_names.append(train_patient_name)


    splits = []
    for i,name in enumerate(train_patient_names):
        if i not in val_id:
            for img in img_folder1:
                new_img = img[:-11]
                print(new_img)
                if name == new_img:
                    splits[-1]['train'] = img
        else:
            for img in img_folder1:
                new_img = img[:-11]
                print(new_img)
                if name == new_img:
                    splits[-1]['val'] = img
    print(splits)
    #save_pickle(splits, join(out_base, "splits_final.pkl"))


if __name__ == "__main__":
   main()
