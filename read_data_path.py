import numpy as np
import cv2
import os

class_dictionary = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                    10: 'cow', 11: 'dining_table', 12: 'dog', 13: 'horse', 14: 'motorbike',
                    15: 'person', 16: 'potted_plant', 17: 'sheep', 18: 'sofa', 19: 'train',
                    20: 'TV_monitor'}

# VOC2012_AUG数据集简介：

# 两个文件夹： img文件夹包含11355张rgb图片，cls文件夹包含11355个语义分割.mat文件，id序号完全对应
# 利用scipy.io.loadmat函数读取cls中的.mat文件，可以得到标注信息。
# 读取得到 (h，w) 单通道矩阵，像素值总共有21个类别，由21个数字代替：0、1、2、...、20。

# 0代表背景信息
# 1-20代表图片中目标物体种类


def read_path():

    data_x = []
    data_y = []

    filename = os.listdir('cls')
    filename.sort()
    for name in filename:

        serial_number = name.split('.')[0]
        img_path = 'img/' + serial_number + '.jpg'
        seg_path = 'cls/' + serial_number + '.mat'

        data_x.append(img_path)
        data_y.append(seg_path)

    return data_x, data_y


def make_data():

    data_x, data_y = read_path()
    print('all image quantity : ', len(data_y))    # 11355

    train_x = data_x[:10000]
    train_y = data_y[:10000]
    val_x = data_x[10000:]
    val_y = data_y[10000:]
    test_x = data_x[10000:]
    test_y = data_y[10000:]

    return train_x, train_y, val_x, val_y, test_x, test_y

