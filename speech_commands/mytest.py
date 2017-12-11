import tensorflow as tf
import numpy as np


from ops import *


sess = tf.InteractiveSession()

x = tf.constant(np.random.rand(2,4,4,2))

out, w = depthwise_separable_conv(x, True)

tf.global_variables_initializer().run()
