import atexit
import datetime
import os

import random
import numpy as np
import tensorflow as tf

import h5py

from feature_extractor import *
from ann import *

## for cluster
DATA_FOLDER = "/work/asr2/bozheniuk/train/"
#DATA_FOLDER = "./data/train/"

## for cluster
GRAPH_FOLDER = "/work/asr2/bozheniuk/graph/"
#GRAPH_FOLDER = "./graph/"

LOAD_PATH = GRAPH_FOLDER + 'graph1/'

FEATURE_FILE = "features/train_features.h5"
LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


def termination_funk():
    print("Terminating..")
    # save_path = network.save_model(sess, path + 'model-final.cptk',)
    # print("Model saved in file: %s" % save_path)

atexit.register(termination_funk)

global path
path = GRAPH_FOLDER + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"

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
        self.data_size = 0

    def read_data(self, file_name):
        h5f = h5py.File(file_name, 'r')

        for lbl_id, lbl in enumerate(LABELS):
            print(lbl)
            co = 0
            for sample_id in h5f.get(lbl):
                self.data.append((np.array(h5f.get(lbl).get(sample_id)), lbl_id))
                self.data_size += 1
                co += 1

            print(co)

        self.data = np.array(self.data)
        h5f.close()

    def get_by_label(self, filename, lbl):
        h5f = h5py.File(filename, 'r')
        samples = []
        labels = []

        for sample_id in h5f.get(lbl):
            samples.append(np.array(h5f.get(lbl).get(sample_id)))
            labels.append(LABELS.index(lbl))

        h5f.close()
        
        return samples, labels

    def shuffle(self):
        self.data = np.random.permutation(self.data)
        
    def get_train_samples(self):
        res = [x[0] for x in self.data]
        return res

    def get_train_labels(self):
        res = [x[1] for x in self.data]
        return res

    def get_validation_samples(self):
        res = [x[0] for x in self.data[int(0.8 * self.data_size):]]
        return res

    def get_validation_labels(self):
        res = [x[1] for x in self.data[int(0.8 * self.data_size):]]
        return res


if os.path.isfile(DATA_FOLDER + FEATURE_FILE):
    print("feture file exist")
else:
    print("featutures doesn't exist")
    print("extracting features")
    fe = FeatureExtractor()
    fe.extract_features(DATA_FOLDER + 'audio/', DATA_FOLDER + 'features/')




# print(fe.sample(10))
# fe.visualize()

print('loading features')
train_data = Data()
train_data.read_data(DATA_FOLDER + FEATURE_FILE)
train_data.shuffle()
d_train = train_data.get_train_samples()
l_train = train_data.get_train_labels()
d_valid = train_data.get_validation_samples()
l_valid = train_data.get_validation_labels()
print('data loaded')

data_shape = np.shape(d_train)

# data = h5f.get('one').get('1')
# data = np.array(data)

# print(data)

# data = np.random.rand(100, 61, 40, 1)
# labels = np.random.randint(10, size=100)

batch_size = 100


def train_net():
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()

        global network
        network = ANN(sess, batch_size, save_path = path)
        network.build_lace(len(LABELS), input_size = [data_shape[1], data_shape[2], 1], channel_start = 128)
        # sess.run(init)
        # network.restore_model(LOAD_PATH)
        # print('cl: ', cl)
        # print('pred: ', pred)
        # print('true: ', l[0:10])

        network.train(d_train, l_train, d_valid, l_valid)

def test_net():
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()

        global network
        network = ANN(sess, batch_size, save_path=path)
        network.build_lace(len(LABELS), input_size=[data_shape[1], data_shape[2], 1], channel_start=128)
        # sess.run(init)
        network.restore_model(LOAD_PATH)
    
        shape = np.shape(go_data)

        batch = 100
        i = 0

        while ((i+1) * batch) < shape[0]:
            d = go_data[i * batch : (i + 1) * batch]
            l = go_label[i * batch : (i + 1) * batch] 
            ac, pred = network.test(d, l)
            print('accuracy: ', ac)
            i += 1

        #    print('pred: ', pred)
        #    print('true: ', l)
        # print('cl: ', cl)
        # print('pred: ', pred)
        # print('true: ', l[0:10])


if __name__ == '__main__':
    train_net()
