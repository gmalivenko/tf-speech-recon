import tensorflow as tf
import numpy as np


from ops import *


sess = tf.InteractiveSession()

x = tf.constant(np.random.rand(2,10,10,2), dtype=tf.float32)

out, w1, w2 = depthwise_separable_conv(x, output_size=4, is_training=True)

tf.global_variables_initializer().run()

print(np.shape(sess.run(out)))
