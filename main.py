from feature_extractor import *
from ann import *

import random
import numpy as np
import tensorflow as tf

import h5py

DATA_FILE = "./data/train/features/train_features.h5"

LABELS = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}


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

with tf.Session() as sess:
    init = tf.global_variables_initializer()

    network = ANN(sess, batch_size)
    network.build_lace(len(LABELS), input_size=[data_shape[1], data_shape[2], 1])
    # sess.run(init)

    network.train(d, l)

