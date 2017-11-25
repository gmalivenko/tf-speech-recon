
import tensorflow as tf
from functools import reduce
from ops import *



class ANN:
    def __init__(self, sess):

        # self.config = config
        self.sess = sess

    def build_lace(self, output_size, input_size = [100, 13, 1], channel_start = 32):
        self.w = {}
        self.layer = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        jump_block_num = 4
        jump_net_num = 2

        with tf.variable_scope('lace'):

            self.phase = tf.placeholder(tf.bool, name = 'phase')

            self.input = tf.placeholder('float32', [None, input_size[0], input_size[1], input_size[2]], name = 'input')
            self.layer['output_0'] = self.input

            current_channel_num = channel_start

            for i in range(jump_block_num):
                self.layer['l_' + str(i + 1) + '_0_o'], \
                self.w['l_' + str(i + 1) + '_0_ow'], \
                self.w['l_' + str(i + 1) + '_0_o'] = conv2d(self.layer['output_' + str(i)],
                                                            current_channel_num,[3, 3], [2, 2],
                                                            initializer,
                                                            activation_fn = None,
                                                            padding='SAME',
                                                            name = 'l_' + str(i + 1) + '_c0')

                print(tf.shape(self.w['l_' + str(i + 1) + '_0_o']))

                for j in range(jump_net_num):
                    index = '_' + str(i + 1) + '_' + str(j + 1) + '_'
                    prev_output = 'l_' + str(i + 1) + '_' + str(j) + '_o'

                    self.layer['l' + index +'c1'], self.w['l' + index +'c1w'], self.w['l' + index +'c1b'] = conv2d(self.layer[prev_output],
                                                                                                                   current_channel_num,
                                                                                                                   [3, 3], [1, 1],
                                                                                                                   initializer,
                                                                                                                   activation_fn = None,
                                                                                                                   padding='SAME',
                                                                                                                   name='l' + index + 'c1')
                    self.layer['l' + index + 'bn1'] = tf.contrib.layers.batch_norm(self.layer['l'+index+'c1'],
                                                                                   center = True, scale = True,
                                                                                   is_training = self.phase)
                    self.layer['l' + index + 'y1'] = tf.nn.relu(self.layer['l' + index + 'bn1'])
                    self.layer['l' + index + 'c2'], self.w['l' + index + 'c2w'], self.w['l' + index + 'c2b'] = conv2d(self.layer['l' + index + 'y1'],
                                                                                                                      current_channel_num,
                                                                                                                      [3, 3], [1, 1],
                                                                                                                      initializer,
                                                                                                                      activation_fn = None,
                                                                                                                      padding='SAME',
                                                                                                                      name='l' + index + 'c2')

                    self.layer['l' + index + 'p'] = tf.add(self.layer['l' + index + 'c2'], self.layer[prev_output])
                    self.layer['l' + index + 'bn2'] = tf.contrib.layers.batch_norm(self.layer['l' + index + 'p'],
                                                                                   center = True, scale = True,
                                                                                   is_training = self.phase)

                    self.layer['l' + index + 'o'] = tf.nn.relu(self.layer['l' + index + 'bn2'])
                    last_layer = self.layer['l' + index + 'o']

                self.layer['output_' + str(i + 1)], self.w['output_' + str(i + 1)] = elementwise_mat_prod(last_layer, name = 'elementwise_' + str(i + 1))
                last_layer = self.layer['output_' + str(i + 1)]

                current_channel_num *= 2


            self.layer['output'], self.w['output'] = weighted_sum(last_layer)
            shape = self.layer['output'].get_shape().as_list()
            print('Shape', shape)
            self.layer['output_flat'] = tf.reshape(self.layer['output'], [-1, reduce(lambda x, y: x * y, shape[1:])])

            self.layer['size_mapping'], self.w['size_mapping_w'], self.w['size_mapping_b'] = linear(self.layer['output_flat'],
                                                                                                      output_size,
                                                                                                      name = 'linear_mapping')

            self.classification = tf.nn.softmax(self.layer['size_mapping'])

            shape = self.classification.get_shape().as_list()
            print('classification ', shape)

            self.ground_truth = tf.placeholder('int64', [None], name = 'true_label')
            one_hot = tf.one_hot(self.ground_truth, output_size, 1.0, 0.0, name='gt_one_hot')
            shape = one_hot.get_shape().as_list()
            print('one hot ', shape)

            self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(tf.multiply(one_hot, tf.log(self.classification))), name = 'cross_enropy')

            shape = self.cross_entropy.get_shape().as_list()
            print('cross entropy ', shape)

            # self.learning_rate = tf.placeholder('float64', 1, name='learning_rate')

            self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cross_entropy)

            tf.initialize_all_variables().run()

    def train(self, data, labels, lrate):
        for i in range(1000):
            _, ent = self.sess.run([self.optimizer, self.cross_entropy], {
                self.input: data,
                self.ground_truth: labels,
                self.phase: True,
                # self.learning_rate: lrate
            })
            print(ent)


