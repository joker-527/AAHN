from collections import OrderedDict
import glob
import os
import re
import json
from batchgenerators.utilities.file_and_folder_operations import *


def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l



path_originalData = '/home/zheng/Yexin/Project02/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task999_1K/'

train_image = list_sort_nicely(glob.glob(path_originalData+"imagesTr/*"))
train_label = list_sort_nicely(glob.glob(path_originalData+"labelsTr/*"))
test_image = list_sort_nicely(glob.glob(path_originalData+"imagesTs/*"))
test_label = list_sort_nicely(glob.glob(path_originalData+"labelsTs/*"))

train_image = ["{}".format(item.split('/')[-1]) for item in train_image]
train_label = ["{}".format(item.split('/')[-1]) for item in train_label]
test_image = ["{}".format(item.split('/')[-1]) for item in test_image]
test_label = ["{}".format(item.split('/')[-1]) for item in test_label]
#输出一下目录的情况，看是否成功
print(train_image)
print(train_label)
print(test_image)
print(test_label)

# 自行修改
json_dict = OrderedDict()
json_dict['name'] = "Task999_1K"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see Task999_1K"
json_dict['licence'] = "see Task999_1K license"
json_dict['release'] = "0.0"
json_dict['modality'] = {
        "0": "CT",
    }
json_dict['labels'] = {
        "0": "background",
        "1": "Liver",
        "2": "Kidney",
        "3": "Spleen",
        "4": "Pancreas",
    }
json_dict['numTraining'] = len(train_image)
json_dict['numTest'] = len(test_image)
json_dict['training'] = [{'image': "./imagesTr/%s" % i , "label": "./labelsTr/%s" % i} for i in train_label]
json_dict['test'] = ["./imagesTs/%s" % i for i in test_image]
save_json(json_dict, join(path_originalData, "dataset.json"))
