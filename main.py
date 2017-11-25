
from feature_extractor import *
from ann import *


import numpy as np
import tensorflow as tf


# fe = FeatureExtractor("./data/train/audio/one/", ",")

# print(fe.sample(10))
# fe.visualize()
data = np.random.rand(100, 61, 40, 1)
labels = np.random.randint(10, size=100)

with tf.Session() as sess:
    init = tf.global_variables_initializer()

    network = ANN(sess)
    network.build_lace(10, input_size=[61, 40, 1])
    # sess.run(init)

    network.train(data, labels, 0.0001)