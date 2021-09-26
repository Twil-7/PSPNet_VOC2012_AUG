import numpy as np
import cv2
import os
from read_data_path import make_data
from psp_model import get_psp_model
from train import SequenceData
from train import train_network
from PIL import Image
import scipy.io


# 真实目标物体像素值的标记类别
class_dictionary = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                    10: 'cow', 11: 'dining_table', 12: 'dog', 13: 'horse', 14: 'motorbike',
                    15: 'person', 16: 'potted_plant', 17: 'sheep', 18: 'sofa', 19: 'train',
                    20: 'TV_monitor'}
inputs_size = (473, 473, 3)

# 语义分割结果的颜色表示空间
np.random.seed(1)
color_array = np.zeros((21, 3))
color_array[0, :] = np.array([255, 255, 0]) / 255    # 背景信息设置为天蓝色

# 20个目标物体的颜色表示随机设置
for row in range(1, 21):

    r = np.random.random_integers(0, 255)
    b = np.random.random_integers(0, 255)
    g = np.random.random_integers(0, 255)

    color_array[row, :] = np.array([r, b, g]) / 255


def detect_semantic(test_x, test_y):

    psp_model = get_psp_model()
    psp_model.load_weights('best_val_accuracy0.92490.h5')

    # img ： 原始rbg图像
    # pre_semantic ： 模型预测的图像语义分割结果
    # true_semantic2 ： 真实的语义分割标注信息

    for i in range(100):

        img = cv2.imread(test_x[i])
        size = img.shape

        img1 = cv2.resize(img, (inputs_size[1], inputs_size[0]), interpolation=cv2.INTER_AREA)
        img2 = img1 / 255
        img3 = img2[np.newaxis, :, :, :]

        result1 = psp_model.predict(img3)  # (1, 473, 473, 2)
        result2 = result1[0]
        result3 = cv2.resize(result2, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((size[0], size[1]))
        pre_semantic = np.zeros((size[0], size[1], 3))

        for j in range(size[0]):
            for k in range(size[1]):

                index = np.argmax(result3[j, k, :])
                mask[j, k] = index
                pre_semantic[j, k, :] = color_array[index, :]

        # 利用scipy.io.loadmat函数，label['GTcls'].Segmentation函数，得到语义分割标注信息
        # 此时得到一个类别矩阵，像素位置上的数值用0-20记录，分别代表不同目标物体

        true_semantic = scipy.io.loadmat(test_y[i], mat_dtype=True, squeeze_me=True, struct_as_record=False)
        true_semantic1 = true_semantic['GTcls'].Segmentation
        true_semantic2 = np.zeros((size[0], size[1], 3))

        for j in range(size[0]):
            for k in range(size[1]):

                index = int(true_semantic1[j, k])
                true_semantic2[j, k, :] = color_array[index, :]

        # cv2.namedWindow("img")
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #
        # cv2.namedWindow("true_semantic")
        # cv2.imshow("true_semantic", true_semantic2)
        # cv2.waitKey(0)
        #
        # cv2.namedWindow("pre_semantic")
        # cv2.imshow("pre_semantic", pre_semantic)
        # cv2.waitKey(0)

        cv2.imwrite("demo/" + str(i) + '_img' + '.jpg', img/1.0)
        cv2.imwrite("demo/" + str(i) + '_true_semantic' + '.jpg', true_semantic2*255)
        cv2.imwrite("demo/" + str(i) + '_pre_semantic' + '.jpg', pre_semantic*255)








