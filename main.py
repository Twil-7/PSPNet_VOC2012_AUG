import numpy as np
import cv2
import os

from read_data_path import make_data
from psp_model import get_psp_model
from train import SequenceData
from train import train_network
from train import load_network_then_train
from detect import detect_semantic

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class_dictionary = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                    10: 'cow', 11: 'dining_table', 12: 'dog', 13: 'horse', 14: 'motorbike',
                    15: 'person', 16: 'potted_plant', 17: 'sheep', 18: 'sofa', 19: 'train',
                    20: 'TV_monitor'}


if __name__ == "__main__":

    train_x, train_y, val_x, val_y, test_x, test_y = make_data()
    psp_model = get_psp_model()
    psp_model.summary()

    train_generator = SequenceData(train_x, train_y, 32)
    test_generator = SequenceData(test_x, test_y, 32)

    # train_network(train_generator, test_generator, epoch=10)
    # load_network_then_train(train_generator, test_generator, epoch=20, input_name='first_weights.hdf5',
    #                         output_name='second_weights.hdf5')

    # detect_semantic(test_x, test_y)







