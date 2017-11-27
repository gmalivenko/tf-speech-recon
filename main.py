import atexit
import datetime
import os

import random
import numpy as np
import tensorflow as tf

import h5py

from feature_extractor import *
from ann import *


DATA_FILE = "./data/train/features/train_features.h5"
LABELS = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}


def termination_funk():
    print("Terminating..")
    save_path = network.save_model(sess, path + 'model-final.cptk',)
    print("Model saved in file: %s" % save_path)

atexit.register(termination_funk)

global path
path = "./graph/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"

if not os.path.exists(path):
    os.makedirs(path)


def count_vars():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


class Data:
    def __init__(self):
        self.data = []

    def read_data(self, file_name):
        h5f = h5py.File(file_name, 'r')

        for lbl_id, lbl in enumerate(LABELS):
            for sample_id in h5f.get(lbl):
                self.data.append((np.array(h5f.get(lbl).get(sample_id)), lbl_id))

        self.data = np.array(self.data)
        h5f.close()

    def shuffle(self):
        random.shuffle(self.data)

    def get_samples(self):
        res = [x[0] for x in self.data]
        return res

    def get_labels(self):
        res = [x[1] for x in self.data]
        return res

# fe = FeatureExtractor("./data/train/audio/one/", ",")

# print(fe.sample(10))
# fe.visualize()

print('loading data')
train_data = Data()
train_data.read_data(DATA_FILE)
train_data.shuffle()
d = train_data.get_samples()
l = train_data.get_labels()
print('data loaded')

data_shape = np.shape(d)

# data = h5f.get('one').get('1')
# data = np.array(data)

# print(data)

# data = np.random.rand(100, 61, 40, 1)
# labels = np.random.randint(10, size=100)

batch_size = 100

LOAD_PATH = './graph/2017-11-27-22:42:22/'

with tf.Session() as sess:
    # init = tf.global_variables_initializer()

    global network
    network = ANN(sess, batch_size, save_path = path)
    network.build_lace(len(LABELS), input_size = [data_shape[1], data_shape[2], 1], channel_start = 32)
    # sess.run(init)
    network.restore_model(LOAD_PATH)




    network.train(d, l)

