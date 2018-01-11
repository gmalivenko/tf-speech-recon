# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ops import *
from functools import reduce
import configparser
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Graph(object):
    def __init__(self, model_settings):
        self.model_settings = model_settings
        self.model_architecture = self.model_settings['arch']
        self.prepare_placeholders()
        output = self.create_model()
        self.add_optimizer(output)

    def prepare_placeholders(self):
        if self.model_settings['features'] == 'mfcc':
          self.input_frequency_size = int(self.model_settings['dct_coefficient_count'])
        else:
          self.input_frequency_size = int(self.model_settings['fft_window_size'] + 1)
        self.input_time_size = self.model_settings['spectrogram_length']

        self.fingerprint_size = self.model_settings['fingerprint_size']
        self.fingerprint_input = tf.placeholder(
            tf.float32, [None, self.fingerprint_size], name='fingerprint_input')

        self.fingerprint_4d = tf.reshape(self.fingerprint_input,
                                         [-1, self.input_time_size, self.input_frequency_size, 1])

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')


        self.w = {}
        self.layer = {}

        self.label_count = self.model_settings['label_count']

        if self.is_adversarial():
            self.noise_label_count = self.model_settings['noise_label_count']
            self.noise_labels = tf.placeholder(
                tf.float32, [None, self.noise_label_count], name='adversarial_groundtruth_input')

        self.ground_truth_input = tf.placeholder(
            tf.float32, [None, self.label_count], name='groundtruth_input')


    def create_model(self, runtime_settings=None):
      """Builds a model of the requested architecture compatible with the settings.

      There are many possible ways of deriving predictions from a spectrogram
      input, so this function provides an abstract interface for creating different
      kinds of models in a black-box way. You need to pass in a TensorFlow node as
      the 'fingerprint' input, and this should output a batch of 1D features that
      describe the audio. Typically this will be derived from a spectrogram that's
      been run through an MFCC, but in theory it can be any feature vector of the
      size specified in model_settings['fingerprint_size'].

      The function will build the graph it needs in the current TensorFlow graph,
      and return the tensorflow output that will contain the 'logits' input to the
      softmax prediction process. If training flag is on, it will also return a
      placeholder node that can be used to control the dropout amount.

      See the implementations below for the possible model architectures that can be
      requested.

      Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        model_architecture: String specifying which kind of model to create.
        is_training: Whether the model is going to be used for training.
        runtime_settings: Dictionary of information about the runtime.

      Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.

      Raises:
        Exception: If the architecture type isn't recognized.
      """

      if self.model_architecture == 'single_fc':
        return self.create_single_fc_model()
      elif self.model_architecture == 'conv':
        return self.create_conv_model()
      elif self.model_architecture == 'lace':
        return self.create_lace_model()
      elif self.model_architecture == 'adversarial_lace':
        return self.create_adversarial_lace_model()
      elif self.model_architecture == 'lace_no_batch_norm':
        return self.create_lace_no_batch_norm_model()
      elif self.model_architecture == 'mobile_cnn':
        return self.create_mobile_cnn()
      elif self.model_architecture == 'wave_net':
        return self.create_wave_net()
      elif self.model_architecture == 'adv_wave_net':
        return self.create_adversarial_wave_net()
      elif self.model_architecture == 'mfcc_wave_net':
        return self.create_mfcc_wave_net()
      elif self.model_architecture == 'gated_mfcc_wave_net':
        return self.create_gated_mfcc_wave_net()
      elif self.model_architecture == 'low_latency_conv':
        return self.create_low_latency_conv_model()
      elif self.model_architecture == 'low_latency_svdf':
        return self.create_low_latency_svdf_model()
      elif self.model_architecture == 'crnn':
        return self.create_crnn_model()
      elif self.model_architecture == 'conv1d':
        return self.create_conv1d_model()
      elif self.model_architecture == 'ds_cnn':
        return self.create_ds_cnn_model()
      else:
        raise Exception('model_architecture argument "' + self.model_architecture +
                        '" not recognized, should be one of "single_fc", "conv",' +
                        ' "low_latency_conv, or "low_latency_svdf"')


    def add_optimizer(self, net_output):
        control_dependencies = []
        # checks = tf.add_check_numerics_ops()
        # control_dependencies = [checks]

        # Create the back propagation and training evaluation machinery in the graph.
        if self.is_adversarial():
            with tf.name_scope('target_cross_entropy'):
                self.cross_entropy_mean = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.ground_truth_input, logits=net_output[0]))
            tf.summary.scalar('target_cross_entropy', self.cross_entropy_mean)
            with tf.name_scope('target_train'), tf.control_dependencies(control_dependencies):
                self.learning_rate_input = tf.placeholder(
                    tf.float32, [], name='learning_rate_input')
                self.optimizer = tf.train.AdamOptimizer(
                    self.learning_rate_input)

                self.grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy_mean)
                self.train_step = self.optimizer.apply_gradients(self.grads_and_vars)
                # self.train_step = self.optimizer.minimize(self.cross_entropy_mean)

                self.predicted_indices = tf.argmax(net_output[0], 1)
                self.expected_indices = tf.argmax(self.ground_truth_input, 1)
                self.correct_prediction = tf.equal(self.predicted_indices, self.expected_indices)
                self.confusion_matrix = tf.confusion_matrix(self.expected_indices, self.predicted_indices,
                                                            num_classes=self.label_count)
                self.evaluation_step = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            with tf.name_scope('adv_cross_entropy'):
                self.adv_cross_entropy_mean = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.noise_labels, logits=net_output[1]))

            with tf.name_scope('adv_train'):
                self.adv_learning_rate_input = tf.placeholder(
                    tf.float32, [], name='adv_learning_rate_input')
                self.adv_optimizer = tf.train.AdamOptimizer(
                    self.adv_learning_rate_input)
                self.adv_train_step = self.adv_optimizer.minimize(self.adv_cross_entropy_mean)

            self.adv_predicted_indices = tf.argmax(net_output[1], 1)
            self.adv_expected_indices = tf.argmax(self.noise_labels, 1)
            self.adv_correct_prediction = tf.equal(self.adv_predicted_indices, self.adv_expected_indices)
            self.adv_confusion_matrix = tf.confusion_matrix(self.adv_expected_indices, self.adv_predicted_indices,
                                                        num_classes=self.noise_label_count)
            self.adv_evaluation_step = tf.reduce_mean(tf.cast(self.adv_correct_prediction, tf.float32))

        else:
            with tf.name_scope('cross_entropy'):
                self.cross_entropy_mean = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.ground_truth_input, logits=net_output))
            tf.summary.scalar('cross_entropy', self.cross_entropy_mean)
            with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
                self.learning_rate_input = tf.placeholder(
                    tf.float32, [], name='learning_rate_input')
                self.optimizer = tf.train.AdamOptimizer(
                    self.learning_rate_input)

                self.grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy_mean)

                self.train_step = self.optimizer.apply_gradients(self.grads_and_vars)
                # self.train_step = self.optimizer.minimize(self.cross_entropy_mean)
            self.predicted_indices = tf.argmax(net_output, 1)
            self.expected_indices = tf.argmax(self.ground_truth_input, 1)
            self.correct_prediction = tf.equal(self.predicted_indices, self.expected_indices)
            self.confusion_matrix = tf.confusion_matrix(self.expected_indices, self.predicted_indices,
                                                        num_classes=self.label_count)
            self.evaluation_step = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_arch_name(self):
      return self.model_architecture

    def load_variables_from_checkpoint(self, sess, start_checkpoint):
      """Utility function to centralize checkpoint restoration.

      Args:
        sess: TensorFlow session.
        start_checkpoint: Path to saved checkpoint on disk.
      """
      saver = tf.train.Saver()
      saver.restore(sess, start_checkpoint)

    def is_adversarial(self):
      if 'is_adversarial' in self.model_settings.keys():
        return int(self.model_settings['is_adversarial'])
      else:
        return 0

    def create_single_fc_model(self):
      """Builds a model with a single hidden fully-connected layer.

      This is a very simple model with just one matmul and bias layer. As you'd
      expect, it doesn't produce very accurate results, but it is very fast and
      simple, so it's useful for sanity testing.

      Here's the layout of the graph:

      (fingerprint_input)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v

      Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

      Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
      """
      fingerprint_size = self.model_settings['fingerprint_size']
      # self.fingerprint_input = tf.placeholder(
      #     tf.float32, [None, fingerprint_size], name='fingerprint_input')

      fingerprint_size = self.model_settings['fingerprint_size']

      weights = tf.Variable(
          tf.truncated_normal([fingerprint_size, self.label_count], stddev=0.001))
      bias = tf.Variable(tf.zeros([self.label_count]))
      self.logits = tf.matmul(self.fingerprint_input, weights) + bias

      return self.logits

    def create_lace_model(self):

        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        channel_start = int(self.model_settings['channel_size'])
        jump_block_num = int(self.model_settings['jump_block_num'])
        jump_net_num = int(self.model_settings['jump_net_num'])

        self.layer['output_0'] = self.fingerprint_4d

        current_channel_num = channel_start

        for i in range(jump_block_num):
            self.layer['l_' + str(i + 1) + '_0_o'], \
            self.w['l_' + str(i + 1) + '_0_ow'] = conv2d(
                self.layer['output_' + str(i)],
                current_channel_num, [3, 3], [2, 2],
                self.initializer,
                activation_fn=None,
                padding='SAME',
                name='l_' + str(i + 1) + '_c0')
            # print(tf.shape(w['l_' + str(i + 1) + '_0_o']))

            for j in range(jump_net_num):
                index = '_' + str(i + 1) + '_' + str(j + 1) + '_'
                prev_output = 'l_' + str(i + 1) + '_' + str(j) + '_o'

                self.layer['l' + index + 'c1'], self.w['l' + index + 'c1w'] = conv2d(
                    self.layer[prev_output],
                    current_channel_num,
                    [3, 3], [1, 1],
                    self.initializer,
                    activation_fn=None,
                    padding='SAME',
                    name='l' + index + 'c1')
                # self.layer['l' + index + 'bn1'] = tf.contrib.layers.batch_norm(
                #     self.layer['l' + index + 'c1'],
                #     center=True, scale=True,
                #     is_training=self.is_training)
                self.layer['l' + index + 'bn1'] = slim.batch_norm(self.layer['l' + index + 'c1'],
                                                                 is_training=self.is_training,
                                                                 decay=0.96,
                                                                 updates_collections=None,
                                                                 activation_fn=tf.nn.relu,
                                                                 scope='l' + index + 'bn1')
                # self.layer['l' + index + 'y1'] = tf.nn.relu(self.layer['l' + index + 'bn1'])
                self.layer['l' + index + 'c2'], self.w['l' + index + 'c2w'] = conv2d(
                    self.layer['l' + index + 'bn1'],
                    current_channel_num,
                    [3, 3], [1, 1],
                    self.initializer,
                    activation_fn=None,
                    padding='SAME',
                    name='l' + index + 'c2')
                self.layer['l' + index + 'p'] = tf.add(
                    self.layer['l' + index + 'c2'],
                    self.layer[prev_output])
                self.layer['l' + index + 'bn2'] = slim.batch_norm(self.layer['l' + index + 'p'],
                                                                 is_training=self.is_training,
                                                                 decay=0.96,
                                                                 updates_collections=None,
                                                                 activation_fn=tf.nn.relu,
                                                                 scope='l' + index + 'bn2')
                # self.layer['l' + index + 'bn2'] = tf.contrib.layers.batch_norm(
                #     self.layer['l' + index + 'p'],
                #     center=True, scale=True,
                #     is_training=self.is_training)
                # self.layer['l' + index + 'o'] = tf.nn.relu(self.layer['l' + index + 'bn2'])
                self.layer['l' + index + 'o'] = self.layer['l' + index + 'bn2']
                last_layer = self.layer['l' + index + 'o']

            self.layer['output_' + str(i + 1)], self.w['output_' + str(i + 1)] = elementwise_mat_prod(
                last_layer,
                name='elementwise_' + str(i + 1))
            last_layer = self.layer['output_' + str(i + 1)]

            current_channel_num *= 2

        self.layer['output'], self.w['output'] = weighted_sum(last_layer)
        shape = self.layer['output'].get_shape().as_list()
        # print('Shape', shape)
        # layer['output_flat'] = tf.reshape(layer['output'], [-1, reduce(lambda x, y: x * y, shape[1:])])
        self.layer['output_flat'] = tf.squeeze(self.layer['output'], [1, 2, 4])

        self.layer['size_mapping'], self.w['size_mapping_w'], self.w['size_mapping_b'] = linear(
            self.layer['output_flat'],
            self.label_count,
            name='linear_mapping')

        self.final_fc = self.layer['size_mapping']
        return self.final_fc

    def create_adversarial_lace_model(self):
        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        channel_start = int(self.model_settings['channel_size'])
        jump_block_num = int(self.model_settings['jump_block_num'])
        jump_net_num = int(self.model_settings['jump_net_num'])

        self.layer['output_0'] = self.fingerprint_4d

        current_channel_num = channel_start

        with tf.variable_scope('main_graph'):
            for i in range(jump_block_num):
                self.layer['l_' + str(i + 1) + '_0_o'], \
                self.w['l_' + str(i + 1) + '_0_ow'] = conv2d(
                    self.layer['output_' + str(i)],
                    current_channel_num, [3, 3], [2, 2],
                    self.initializer,
                    activation_fn=None,
                    padding='SAME',
                    name='l_' + str(i + 1) + '_c0')
                # print(tf.shape(w['l_' + str(i + 1) + '_0_o']))

                for j in range(jump_net_num):
                    index = '_' + str(i + 1) + '_' + str(j + 1) + '_'
                    prev_output = 'l_' + str(i + 1) + '_' + str(j) + '_o'

                    self.layer['l' + index + 'c1'], self.w['l' + index + 'c1w'] = conv2d(
                        self.layer[prev_output],
                        current_channel_num,
                        [3, 3], [1, 1],
                        self.initializer,
                        activation_fn=None,
                        padding='SAME',
                        name='l' + index + 'c1')
                    self.layer['l' + index + 'bn1'] = tf.contrib.layers.batch_norm(
                        self.layer['l' + index + 'c1'],
                        center=True, scale=True,
                        is_training=self.is_training)
                    self.layer['l' + index + 'y1'] = tf.nn.relu(self.layer['l' + index + 'bn1'])
                    self.layer['l' + index + 'c2'], self.w['l' + index + 'c2w'] = conv2d(
                        self.layer['l' + index + 'y1'],
                        current_channel_num,
                        [3, 3], [1, 1],
                        self.initializer,
                        activation_fn=None,
                        padding='SAME',
                        name='l' + index + 'c2')
                    self.layer['l' + index + 'p'] = tf.add(
                        self.layer['l' + index + 'c2'],
                        self.layer[prev_output])
                    self.layer['l' + index + 'bn2'] = tf.contrib.layers.batch_norm(
                        self.layer['l' + index + 'p'],
                        center=True, scale=True,
                        is_training=self.is_training)
                    self.layer['l' + index + 'o'] = tf.nn.relu(self.layer['l' + index + 'bn2'])
                    last_layer = self.layer['l' + index + 'o']

                self.layer['output_' + str(i + 1)], self.w['output_' + str(i + 1)] = elementwise_mat_prod(
                    last_layer,
                    name='elementwise_' + str(i + 1))
                last_layer = self.layer['output_' + str(i + 1)]

                current_channel_num *= 2

            self.layer['output'], self.w['output'] = weighted_sum(last_layer)
            self.layer['output_flat'] = tf.squeeze(self.layer['output'], [1, 2, 4])


        with tf.variable_scope('target_subnet'):
            self.layer['size_mapping'], self.w['size_mapping_w'], self.w['size_mapping_b'] = linear(
                self.layer['output_flat'],
                self.label_count,
                name='linear_mapping')

            self.final_fc = self.layer['size_mapping']

        with tf.variable_scope('adversarial'):
            #gradient reversal layer
            # https://stackoverflow.com/questions/36456436/how-can-i-define-only-the-gradient-for-a-tensorflow-subgraph
            # Suppose you want group of ops that behave as f(x) in forward mode, but as g(x) in the backward mode.
            # t = g(x)
            # y = t + tf.stop_gradient(f(x) - t)
            self.adv_lamda = 0.1
            grl_back = -self.adv_lamda * self.layer['output_flat']
            grl_forward = grl_back + tf.stop_gradient(self.layer['output_flat'] - grl_back)


            self.layer['adv_output'], self.w['adv_w'], self.w['adv_b'] = linear(
                grl_forward,
                self.noise_label_count,
                name='adversarial_linear_mapping')

        return self.final_fc, self.layer['adv_output']

    def create_mobile_cnn(self):

        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # channel_start = int(parser['arch-parameters']['channel_size'])
        channel_start = 32

        current_channel_num = channel_start
        dw_layers_num = 13

        kernel_shape = [3, 3]

        self.layer['init_conv'], \
        self.w['init_kernel'] = conv2d(self.fingerprint_4d,
                                       output_dim=current_channel_num,
                                       kernel_size=kernel_shape,
                                       stride=(2, 2),
                                       initializer=None,
                                       padding='SAME',
                                       name='init')

        self.layer['init_bn'] = tf.layers.batch_normalization(self.layer['init_conv'],
                                                              training=self.is_training)
        self.layer['init_relu'] = tf.nn.relu(self.layer['init_bn'])
        last_output = self.layer['init_relu']

        self.layer['conv1'], \
        self.w['conv1_dw'], \
        self.w['conv1_pw'] = depthwise_separable_conv(self.layer['init_relu'],
                                                      2 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw1')
        self.layer['conv2'], \
        self.w['conv2_dw'], \
        self.w['conv2_pw'] = depthwise_separable_conv(self.layer['conv1'],
                                                      4 * channel_start,
                                                      self.is_training,
                                                      stride=(2, 2),
                                                      padding='SAME',
                                                      name='dw2')
        self.layer['conv3'], \
        self.w['conv3_dw'], \
        self.w['conv3_pw'] = depthwise_separable_conv(self.layer['conv2'],
                                                      4 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw3')
        self.layer['conv4'], \
        self.w['conv4_dw'], \
        self.w['conv4_pw'] = depthwise_separable_conv(self.layer['conv3'],
                                                      8 * channel_start,
                                                      self.is_training,
                                                      stride=(2, 2),
                                                      padding='SAME',
                                                      name='dw4')
        self.layer['conv5'], \
        self.w['conv5_dw'], \
        self.w['conv5_pw'] = depthwise_separable_conv(self.layer['conv4'],
                                                      8 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw5')
        self.layer['conv6'], \
        self.w['conv6_dw'], \
        self.w['conv6_pw'] = depthwise_separable_conv(self.layer['conv5'],
                                                      16 * channel_start,
                                                      self.is_training,
                                                      stride=(2, 2),
                                                      padding='SAME',
                                                      name='dw6')
        self.layer['conv7'], \
        self.w['conv7_dw'], \
        self.w['conv7_pw'] = depthwise_separable_conv(self.layer['conv6'],
                                                      16 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw7')
        self.layer['conv8'], \
        self.w['conv8_dw'], \
        self.w['conv8_pw'] = depthwise_separable_conv(self.layer['conv7'],
                                                      16 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw8')
        self.layer['conv9'], \
        self.w['conv9_dw'], \
        self.w['conv9_pw'] = depthwise_separable_conv(self.layer['conv8'],
                                                      16 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw9')
        self.layer['conv10'], \
        self.w['conv10_dw'], \
        self.w['conv10_pw'] = depthwise_separable_conv(self.layer['conv9'],
                                                      16 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw10')
        self.layer['conv11'], \
        self.w['conv11_dw'], \
        self.w['conv11_pw'] = depthwise_separable_conv(self.layer['conv10'],
                                                      16 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name = 'dw11')
        self.layer['conv12'], \
        self.w['conv12_dw'], \
        self.w['conv12_pw'] = depthwise_separable_conv(self.layer['conv11'],
                                                      32 * channel_start,
                                                      self.is_training,
                                                      stride=(2, 2),
                                                      padding='SAME',
                                                      name='dw12')
        self.layer['conv13'], \
        self.w['conv13_dw'], \
        self.w['conv13_pw'] = depthwise_separable_conv(self.layer['conv12'],
                                                      32 * channel_start,
                                                      self.is_training,
                                                      stride=(1, 1),
                                                      padding='SAME',
                                                      name='dw13')

        shape = self.layer['conv13'].get_shape().as_list()
        self.layer['pool'] = tf.nn.pool(self.layer['conv13'], window_shape=(shape[1], shape[2]), pooling_type='AVG', padding='VALID')
        self.layer['squeeze'] = tf.squeeze(self.layer['pool'], axis=[1, 2])

        self.layer['fc'], self.w['fc_w'], self.w['fc_b']  = linear(self.layer['squeeze'], self.label_count, name='final_fc')

        return self.layer['fc']

    def create_lace_no_batch_norm_model(self, fingerprint_input, model_settings, is_training):
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])

        if is_training:
            dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        model_name = 'lace'
        w = {}
        layer = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        channel_start = 32
        jump_block_num = 4
        jump_net_num = 2

        #phase = tf.placeholder(tf.bool, name='phase')

        layer['output_0'] = fingerprint_4d

        current_channel_num = channel_start

        for i in range(jump_block_num):
            layer['l_' + str(i + 1) + '_0_o'], \
            w['l_' + str(i + 1) + '_0_ow'] = conv2d(
                layer['output_' + str(i)],
                current_channel_num, [3, 3], [2, 2],
                initializer,
                activation_fn=None,
                padding='SAME',
                name='l_' + str(i + 1) + '_c0')

            # print(tf.shape(w['l_' + str(i + 1) + '_0_o']))

            for j in range(jump_net_num):
                index = '_' + str(i + 1) + '_' + str(j + 1) + '_'
                prev_output = 'l_' + str(i + 1) + '_' + str(j) + '_o'

                layer['l' + index + 'c1'], w['l' + index + 'c1w'] = conv2d(
                    layer[prev_output],
                    current_channel_num,
                    [3, 3], [1, 1],
                    initializer,
                    activation_fn=None,
                    padding='SAME',
                    name='l' + index + 'c1')
                # layer['l' + index + 'bn1'] = tf.contrib.layers.batch_norm(
                #     layer['l' + index + 'c1'],
                #     center=True, scale=True,
                #     is_training=phase)
                layer['l' + index + 'y1'] = tf.nn.relu(layer['l' + index + 'c1'])

                if is_training:
                    layer['l' + index + 'drop1'] = tf.nn.dropout(layer['l' + index + 'y1'], dropout_prob)
                else:
                    layer['l' + index + 'drop1'] = layer['l' + index + 'y1']

                layer['l' + index + 'c2'], w['l' + index + 'c2w'] = conv2d(
                    layer['l' + index + 'drop1'],
                    current_channel_num,
                    [3, 3], [1, 1],
                    initializer,
                    activation_fn=None,
                    padding='SAME',
                    name='l' + index + 'c2')
                layer['l' + index + 'p'] = tf.add(
                    layer['l' + index + 'c2'],
                    layer[prev_output])
                # layer['l' + index + 'bn2'] = tf.contrib.layers.batch_norm(
                #     layer['l' + index + 'p'],
                #     center=True, scale=True,
                #     is_training=phase)
                layer['l' + index + 'y2'] = tf.nn.relu(layer['l' + index + 'p'])

                if is_training:
                    layer['l' + index + 'o'] = tf.nn.dropout(layer['l' + index + 'y2'], dropout_prob)
                else:
                    layer['l' + index + 'o'] = layer['l' + index + 'y2']

                last_layer = layer['l' + index + 'o']

            layer['output_' + str(i + 1)], w['output_' + str(i + 1)] = elementwise_mat_prod(
                last_layer,
                name='elementwise_' + str(i + 1))
            last_layer = layer['output_' + str(i + 1)]

            current_channel_num *= 2

        print('creating')
        layer['output'], w['output'] = weighted_sum(last_layer)
        shape = layer['output'].get_shape().as_list()
        # print('Shape', shape)
        # layer['output_flat'] = tf.reshape(layer['output'], [-1, reduce(lambda x, y: x * y, shape[1:])])
        layer['output_flat'] = tf.squeeze(layer['output'], [1, 2, 4])

        label_count = model_settings['label_count']
        layer['size_mapping'], w['size_mapping_w'], w['size_mapping_b'] = linear(
            layer['output_flat'],
            label_count,
            name='linear_mapping')

        final_fc = layer['size_mapping']

        if is_training:
            return final_fc, dropout_prob, layer
        else:
            return final_fc

    def create_conv_model(self, model_settings):
      """Builds a standard convolutional model.

      This is roughly the network labeled as 'cnn-trad-fpool3' in the
      'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
      http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

      Here's the layout of the graph:

      (fingerprint_input)
              v
          [Conv2D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MaxPool]
              v
          [Conv2D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MaxPool]
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v

      This produces fairly good quality results, but can involve a large number of
      weight parameters and computations. For a cheaper alternative from the same
      paper with slightly less accuracy, see 'low_latency_conv' below.

      During training, dropout nodes are introduced after each relu, controlled by a
      placeholder.

      Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

      Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
      """

      fingerprint_size = model_settings['fingerprint_size']
      self.fingerprint_input = tf.placeholder(
          tf.float32, [None, fingerprint_size], name='fingerprint_input')

      self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
      input_frequency_size = model_settings['dct_coefficient_count']
      input_time_size = model_settings['spectrogram_length']
      fingerprint_4d = tf.reshape(self.fingerprint_input,
                                  [-1, input_time_size, input_frequency_size, 1])
      first_filter_width = 8
      first_filter_height = 20
      first_filter_count = 64
      first_weights = tf.Variable(
          tf.truncated_normal(
              [first_filter_height, first_filter_width, 1, first_filter_count],
              stddev=0.01))
      first_bias = tf.Variable(tf.zeros([first_filter_count]))
      first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                                'SAME') + first_bias
      first_relu = tf.nn.relu(first_conv)

      first_dropout = tf.nn.dropout(first_relu, self.dropout_prob)

      max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
      second_filter_width = 4
      second_filter_height = 10
      second_filter_count = 64
      second_weights = tf.Variable(
          tf.truncated_normal(
              [
                  second_filter_height, second_filter_width, first_filter_count,
                  second_filter_count
              ],
              stddev=0.01))
      second_bias = tf.Variable(tf.zeros([second_filter_count]))
      second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                                 'SAME') + second_bias
      second_relu = tf.nn.relu(second_conv)

      second_dropout = tf.nn.dropout(second_relu, self.dropout_prob)

      second_conv_shape = second_dropout.get_shape()
      second_conv_output_width = second_conv_shape[2]
      second_conv_output_height = second_conv_shape[1]
      second_conv_element_count = int(
          second_conv_output_width * second_conv_output_height *
          second_filter_count)
      flattened_second_conv = tf.reshape(second_dropout,
                                         [-1, second_conv_element_count])
      label_count = model_settings['label_count']
      final_fc_weights = tf.Variable(
          tf.truncated_normal(
              [second_conv_element_count, label_count], stddev=0.01))
      final_fc_bias = tf.Variable(tf.zeros([label_count]))
      self.final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
      # if is_training:
      #   return final_fc, dropout_prob
      # else:
      #   return final_fc
      return

    def create_low_latency_conv_model(self, fingerprint_input, model_settings,
                                      is_training):
      """Builds a convolutional model with low compute requirements.

      This is roughly the network labeled as 'cnn-one-fstride4' in the
      'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
      http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

      Here's the layout of the graph:

      (fingerprint_input)
              v
          [Conv2D]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v

      This produces slightly lower quality results than the 'conv' model, but needs
      fewer weight parameters and computations.

      During training, dropout nodes are introduced after the relu, controlled by a
      placeholder.

      Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.

      Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.
      """
      if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
      input_frequency_size = model_settings['dct_coefficient_count']
      input_time_size = model_settings['spectrogram_length']
      fingerprint_4d = tf.reshape(fingerprint_input,
                                  [-1, input_time_size, input_frequency_size, 1])
      first_filter_width = 8
      first_filter_height = input_time_size
      first_filter_count = 186
      first_filter_stride_x = 1
      first_filter_stride_y = 1
      first_weights = tf.Variable(
          tf.truncated_normal(
              [first_filter_height, first_filter_width, 1, first_filter_count],
              stddev=0.01))
      first_bias = tf.Variable(tf.zeros([first_filter_count]))
      first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
          1, first_filter_stride_y, first_filter_stride_x, 1
      ], 'VALID') + first_bias
      first_relu = tf.nn.relu(first_conv)
      if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
      else:
        first_dropout = first_relu
      first_conv_output_width = math.floor(
          (input_frequency_size - first_filter_width + first_filter_stride_x) /
          first_filter_stride_x)
      first_conv_output_height = math.floor(
          (input_time_size - first_filter_height + first_filter_stride_y) /
          first_filter_stride_y)
      first_conv_element_count = int(
          first_conv_output_width * first_conv_output_height * first_filter_count)
      flattened_first_conv = tf.reshape(first_dropout,
                                        [-1, first_conv_element_count])
      first_fc_output_channels = 128
      first_fc_weights = tf.Variable(
          tf.truncated_normal(
              [first_conv_element_count, first_fc_output_channels], stddev=0.01))
      first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
      first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
      if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
      else:
        second_fc_input = first_fc
      second_fc_output_channels = 128
      second_fc_weights = tf.Variable(
          tf.truncated_normal(
              [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
      second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
      second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
      if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
      else:
        final_fc_input = second_fc
      label_count = model_settings['label_count']
      final_fc_weights = tf.Variable(
          tf.truncated_normal(
              [second_fc_output_channels, label_count], stddev=0.01))
      final_fc_bias = tf.Variable(tf.zeros([label_count]))
      final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
      if is_training:
        return final_fc, dropout_prob
      else:
        return final_fc

    def create_low_latency_svdf_model(self, fingerprint_input, model_settings,
                                      is_training, runtime_settings):
      """Builds an SVDF model with low compute requirements.

      This is based in the topology presented in the 'Compressing Deep Neural
      Networks using a Rank-Constrained Topology' paper:
      https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

      Here's the layout of the graph:

      (fingerprint_input)
              v
            [SVDF]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
            [Relu]
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v
          [MatMul]<-(weights)
              v
          [BiasAdd]<-(bias)
              v

      This model produces lower recognition accuracy than the 'conv' model above,
      but requires fewer weight parameters and, significantly fewer computations.

      During training, dropout nodes are introduced after the relu, controlled by a
      placeholder.

      Args:
        fingerprint_input: TensorFlow node that will output audio feature vectors.
        The node is expected to produce a 2D Tensor of shape:
          [batch, model_settings['dct_coefficient_count'] *
                  model_settings['spectrogram_length']]
        with the features corresponding to the same time slot arranged contiguously,
        and the oldest slot at index [:, 0], and newest at [:, -1].
        model_settings: Dictionary of information about the model.
        is_training: Whether the model is going to be used for training.
        runtime_settings: Dictionary of information about the runtime.

      Returns:
        TensorFlow node outputting logits results, and optionally a dropout
        placeholder.

      Raises:
          ValueError: If the inputs tensor is incorrectly shaped.
      """
      if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

      input_frequency_size = model_settings['dct_coefficient_count']
      input_time_size = model_settings['spectrogram_length']

      # Validation.
      input_shape = fingerprint_input.get_shape()
      if len(input_shape) != 2:
        raise ValueError('Inputs to `SVDF` should have rank == 2.')
      if input_shape[-1].value is None:
        raise ValueError('The last dimension of the inputs to `SVDF` '
                         'should be defined. Found `None`.')
      if input_shape[-1].value % input_frequency_size != 0:
        raise ValueError('Inputs feature dimension %d must be a multiple of '
                         'frame size %d', fingerprint_input.shape[-1].value,
                         input_frequency_size)

      # Set number of units (i.e. nodes) and rank.
      rank = 2
      num_units = 1280
      # Number of filters: pairs of feature and time filters.
      num_filters = rank * num_units
      # Create the runtime memory: [num_filters, batch, input_time_size]
      batch = 1
      memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                           trainable=False, name='runtime-memory')
      # Determine the number of new frames in the input, such that we only operate
      # on those. For training we do not use the memory, and thus use all frames
      # provided in the input.
      # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
      if is_training:
        num_new_frames = input_time_size
      else:
        window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                               model_settings['sample_rate'])
        num_new_frames = tf.cond(
            tf.equal(tf.count_nonzero(memory), 0),
            lambda: input_time_size,
            lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
      new_fingerprint_input = fingerprint_input[
          :, -num_new_frames*input_frequency_size:]
      # Expand to add input channels dimension.
      new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

      # Create the frequency filters.
      weights_frequency = tf.Variable(
          tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
      # Expand to add input channels dimensions.
      # weights_frequency: [input_frequency_size, 1, num_filters]
      weights_frequency = tf.expand_dims(weights_frequency, 1)
      # Convolve the 1D feature filters sliding over the time dimension.
      # activations_time: [batch, num_new_frames, num_filters]
      activations_time = tf.nn.conv1d(
          new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
      # Rearrange such that we can perform the batched matmul.
      # activations_time: [num_filters, batch, num_new_frames]
      activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

      # Runtime memory optimization.
      if not is_training:
        # We need to drop the activations corresponding to the oldest frames, and
        # then add those corresponding to the new frames.
        new_memory = memory[:, :, num_new_frames:]
        new_memory = tf.concat([new_memory, activations_time], 2)
        tf.assign(memory, new_memory)
        activations_time = new_memory

      # Create the time filters.
      weights_time = tf.Variable(
          tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
      # Apply the time filter on the outputs of the feature filters.
      # weights_time: [num_filters, input_time_size, 1]
      # outputs: [num_filters, batch, 1]
      weights_time = tf.expand_dims(weights_time, 2)
      outputs = tf.matmul(activations_time, weights_time)
      # Split num_units and rank into separate dimensions (the remaining
      # dimension is the input_shape[0] -i.e. batch size). This also squeezes
      # the last dimension, since it's not used.
      # [num_filters, batch, 1] => [num_units, rank, batch]
      outputs = tf.reshape(outputs, [num_units, rank, -1])
      # Sum the rank outputs per unit => [num_units, batch].
      units_output = tf.reduce_sum(outputs, axis=1)
      # Transpose to shape [batch, num_units]
      units_output = tf.transpose(units_output)

      # Appy bias.
      bias = tf.Variable(tf.zeros([num_units]))
      first_bias = tf.nn.bias_add(units_output, bias)

      # Relu.
      first_relu = tf.nn.relu(first_bias)

      if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
      else:
        first_dropout = first_relu

      first_fc_output_channels = 256
      first_fc_weights = tf.Variable(
          tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
      first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
      first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
      if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
      else:
        second_fc_input = first_fc
      second_fc_output_channels = 256
      second_fc_weights = tf.Variable(
          tf.truncated_normal(
              [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
      second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
      second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
      if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
      else:
        final_fc_input = second_fc
      label_count = model_settings['label_count']
      final_fc_weights = tf.Variable(
          tf.truncated_normal(
              [second_fc_output_channels, label_count], stddev=0.01))
      final_fc_bias = tf.Variable(tf.zeros([label_count]))
      final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
      if is_training:
        return final_fc, dropout_prob
      else:
        return final_fc

    def create_crnn_model(self):
      if self.is_training is not None:
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

      filter_width = int(self.model_settings['filter_width'])
      filter_height = int(self.model_settings['filter_height'])
      filter_count = int(self.model_settings['filter_count'])
      weights = tf.Variable(tf.truncated_normal([filter_height, filter_width, 1, filter_count], stddev=0.01))
      bias = tf.Variable(tf.zeros([filter_count]))
      conv_out = tf.nn.conv2d(self.fingerprint_4d, weights, [1, 2, 1, 1], 'VALID') + bias

      batch_mean, batch_var = tf.nn.moments(conv_out, [0])
      scale = tf.Variable(tf.ones([filter_count]))
      beta = tf.Variable(tf.zeros([filter_count]))
      conv = tf.nn.batch_normalization(conv_out, batch_mean, batch_var, beta, scale, 1e-3)
      conv_relu = tf.nn.relu(conv)


      conv_shape = conv_relu.get_shape()
      conv_output_width = conv_shape[2]
      conv_output_height = conv_shape[1]
      flattened_conv = tf.reshape(conv_relu, [-1, conv_output_height, conv_output_width * filter_count])

      if self.is_training is not None:
        rnn_input = tf.nn.dropout(flattened_conv, self.dropout_prob)
      else:
        rnn_input = flattened_conv


      gru_cell = tf.contrib.rnn.GRUCell(num_units=60)
      cells = []
      num_layers = int(self.model_settings['num_gru_layers'])
      num_units = int(self.model_settings['num_gru_units'])
      for _ in range(num_layers):
        cell = tf.contrib.rnn.GRUCell(num_units=num_units)
        cells.append(cell)
      multi_layer_gru_cell = tf.contrib.rnn.MultiRNNCell(cells)
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(multi_layer_gru_cell, multi_layer_gru_cell,
                                                               inputs=rnn_input, dtype=tf.float32, time_major=False)
      rnn_output = tf.concat([outputs[0], outputs[1]], 2)

      rnn_relu = tf.nn.relu(rnn_output)

      rnn_result_height = rnn_relu.get_shape()[1]
      rnn_result_width = rnn_relu.get_shape()[2]
      rnn_result_flattened = tf.reshape(rnn_relu, [-1, rnn_result_width * rnn_result_height])

      if self.is_training is not None:
        fc_input = tf.nn.dropout(rnn_result_flattened, self.dropout_prob)
      else:
        fc_input = rnn_result_flattened


      num_fc_units = int(self.model_settings['num_units_in_fc_layer'])
      fc_weights = tf.Variable(
        tf.truncated_normal([tf.cast(rnn_result_width * rnn_result_height, tf.int32), num_fc_units], stddev=0.01))
      fc_bias = tf.Variable(tf.zeros([num_fc_units]))
      fc_layer_out = tf.matmul(fc_input, fc_weights) + fc_bias

      fc_batch_mean, fc_batch_var = tf.nn.moments(fc_layer_out, [0])
      fc_scale = tf.Variable(tf.ones([1]))
      fc_beta = tf.Variable(tf.zeros([1]))
      fc_layer = tf.nn.batch_normalization(fc_layer_out, fc_batch_mean, fc_batch_var, fc_beta, fc_scale, 1e-3)

      relu_fc = tf.nn.relu(fc_layer)

      if self.is_training is not None:
        final_fc_input = tf.nn.dropout(relu_fc, self.dropout_prob)
      else:
        final_fc_input = relu_fc

      fc_shape = final_fc_input.get_shape()
      label_count = self.model_settings['label_count']
      final_fc_weights = tf.Variable(tf.truncated_normal([tf.cast(fc_shape[1], tf.int32), label_count], stddev=0.01))
      final_fc_bias = tf.Variable(tf.zeros([label_count]))
      final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias

      return final_fc

    def create_adversarial_wave_net(self):

      num_of_filters = int(self.model_settings['num_of_filters'])
      default_init = tf.contrib.layers.xavier_initializer()
      num_blocks = int(self.model_settings['number_of_wave_net_blocks'])
      filter_size = int(self.model_settings['filter_size'])
      dilation_rates = list(map(int, self.model_settings['dilation_rates'].split(',')))
      fingerprint_3d = tf.expand_dims(self.fingerprint_input, -1)

      def res_block(input, filter_length, num_of_filters, rate, block):
        with tf.variable_scope(name_or_scope='block_%d_%d' % (block, rate)):
          kernel_shape = [filter_length, num_of_filters, num_of_filters]
          filter_weights = tf.get_variable('w_filter', kernel_shape, tf.float32, initializer=default_init)
          filter = tf.nn.convolution(input, filter_weights, 'SAME', dilation_rate=[rate])
          filter_out = tf.nn.tanh(filter)
          filter_bn = slim.batch_norm(filter_out, is_training=self.is_training, decay=0.96, updates_collections=None)
          out = filter_bn

          outWeights = tf.get_variable('w_out', [1, num_of_filters, num_of_filters], tf.float32,
                                       initializer=default_init)
          out = tf.nn.convolution(out, outWeights, 'SAME')
          out = tf.tanh(out)
          out_bn = slim.batch_norm(out, is_training=self.is_training, decay=0.96, updates_collections=None)
          res = out_bn + input

        return res, out

      with tf.variable_scope('input_conv'):
        input_weights = tf.get_variable('w_inp', [filter_size, 1, num_of_filters], tf.float32, initializer=default_init)
        res = tf.tanh(tf.nn.convolution(fingerprint_3d, input_weights, 'SAME', dilation_rate=[1]))
        res = slim.batch_norm(res, is_training=self.is_training, decay=0.96, updates_collections=None)
      skip = 0
      for i in range(num_blocks):
        for r in dilation_rates:
          res, s = res_block(res, filter_size, num_of_filters, r, i)
          skip += s
      with tf.variable_scope('pre_pooling_conv'):
        skip_sum_weights = tf.get_variable('w_pre_pooling', [1, num_of_filters, num_of_filters], tf.float32,
                                           initializer=default_init)
        pre_pooling_conv = tf.tanh(tf.nn.convolution(skip, skip_sum_weights, 'SAME'))
        # pre_pooling_conv_bn = tf.layers.batch_normalization(pre_pooling_conv, training=self.is_training)
        pre_pooling_conv_bn = slim.batch_norm(pre_pooling_conv, is_training=self.is_training, decay=0.96,
                                              updates_collections=None)
      global_pool = tf.reduce_mean(pre_pooling_conv_bn, axis=1)

      label_count = self.model_settings['label_count']
      with tf.variable_scope('final_layer'):
        final_fc_weights = tf.get_variable('w_softmax', [num_of_filters, label_count], tf.float32,
                                           initializer=default_init)
        final_fc_bias = tf.get_variable('b_softmax', [label_count], tf.float32, initializer=tf.constant_initializer(0))
        final_fc = tf.matmul(global_pool, final_fc_weights) + final_fc_bias



      with tf.variable_scope('adversarial'):
        # gradient reversal layer
        # https://stackoverflow.com/questions/36456436/how-can-i-define-only-the-gradient-for-a-tensorflow-subgraph
        # Suppose you want group of ops that behave as f(x) in forward mode, but as g(x) in the backward mode.
        # t = g(x)
        # y = t + tf.stop_gradient(f(x) - t)
        adv_lamda = 0.1
        grl_back = -adv_lamda * global_pool
        grl_forward = grl_back + tf.stop_gradient(global_pool - grl_back)

        adv_weights = tf.get_variable('w_adv', [num_of_filters, self.noise_label_count], tf.float32, initializer=default_init)
        adv_bias = tf.get_variable('b_adv', [self.noise_label_count], tf.float32, initializer=tf.constant_initializer(0))
        adv_output = tf.matmul(global_pool, adv_weights) + adv_bias


      return final_fc, adv_output



    def create_wave_net(self):

      num_of_filters = int(self.model_settings['num_of_filters'])
      default_init = tf.contrib.layers.xavier_initializer()
      num_blocks = int(self.model_settings['number_of_wave_net_blocks'])
      filter_size = int(self.model_settings['filter_size'])
      dilation_rates = list(map(int, self.model_settings['dilation_rates'].split(',')))
      fingerprint_3d = tf.expand_dims(self.fingerprint_input, -1)

      def res_block(input, filter_length, num_of_filters, rate, block):
        with tf.variable_scope(name_or_scope='block_%d_%d' % (block, rate)):
          kernel_shape = [filter_length, num_of_filters, num_of_filters]
          filter_weights = tf.get_variable('w_filter', kernel_shape, tf.float32, initializer=default_init)
          filter = tf.nn.convolution(input, filter_weights, 'SAME', dilation_rate=[rate])
          filter_out = tf.nn.tanh(filter)
          filter_bn = slim.batch_norm(filter_out,is_training=self.is_training,decay=0.96,updates_collections=None)
          out = filter_bn

          outWeights = tf.get_variable('w_out', [1, num_of_filters, num_of_filters], tf.float32, initializer=default_init)
          out = tf.nn.convolution(out, outWeights, 'SAME')
          out = tf.tanh(out)
          out_bn = slim.batch_norm(out,is_training=self.is_training,decay=0.96,updates_collections=None)
          res = out_bn + input

        return res, out

      with tf.variable_scope('input_conv'):
        input_weights = tf.get_variable('w_inp', [filter_size, 1, num_of_filters], tf.float32, initializer=default_init)
        res = tf.tanh(tf.nn.convolution(fingerprint_3d, input_weights, 'SAME', dilation_rate=[1]))
        res = slim.batch_norm(res,is_training=self.is_training,decay=0.96,updates_collections=None)
      skip = 0
      for i in range(num_blocks):
        for r in dilation_rates:
          res, s = res_block(res, filter_size, num_of_filters, r, i)
          skip += s
      with tf.variable_scope('pre_pooling_conv'):
        skip_sum_weights = tf.get_variable('w_pre_pooling', [1, num_of_filters, num_of_filters], tf.float32, initializer=default_init)
        pre_pooling_conv = tf.tanh(tf.nn.convolution(skip, skip_sum_weights, 'SAME'))
        # pre_pooling_conv_bn = tf.layers.batch_normalization(pre_pooling_conv, training=self.is_training)
        pre_pooling_conv_bn = slim.batch_norm(pre_pooling_conv,is_training=self.is_training,decay=0.96,updates_collections=None)
      global_pool = tf.reduce_mean(pre_pooling_conv_bn, axis=1)

      label_count = self.model_settings['label_count']
      with tf.variable_scope('final_layer'):
        final_fc_weights = tf.get_variable('w_softmax', [num_of_filters, label_count], tf.float32, initializer=default_init)
        final_fc_bias = tf.get_variable('b_softmax', [label_count], tf.float32, initializer=tf.constant_initializer(0))
        final_fc = tf.matmul(global_pool, final_fc_weights) + final_fc_bias

      return final_fc



    def create_mfcc_wave_net(self):

      num_of_filters = int(self.model_settings['num_of_filters'])
      default_init = tf.contrib.layers.xavier_initializer()
      num_blocks = int(self.model_settings['number_of_wave_net_blocks'])
      filter_size = int(self.model_settings['filter_size'])
      dilation_rates = list(map(int, self.model_settings['dilation_rates'].split(',')))
      fingerprint3d = tf.squeeze(self.fingerprint_4d, 3)

      def res_block(input, filter_length, num_of_filters, rate, block):
        with tf.variable_scope(name_or_scope='block_%d_%d' % (block, rate)):
          kernel_shape = [filter_length, num_of_filters, num_of_filters]
          filter_weights = tf.get_variable('w_filter', kernel_shape, tf.float32, initializer=default_init)
          filter = tf.nn.convolution(input, filter_weights, 'SAME', dilation_rate=[rate])
          filter_out = tf.nn.tanh(filter)
          filter_bn = slim.batch_norm(filter_out,is_training=self.is_training,decay=0.96,updates_collections=None)
          out = filter_bn

          outWeights = tf.get_variable('w_out', [1, num_of_filters, num_of_filters], tf.float32, initializer=default_init)
          out = tf.nn.convolution(out, outWeights, 'SAME')
          out = tf.tanh(out)
          out_bn = slim.batch_norm(out,is_training=self.is_training,decay=0.96,updates_collections=None)
          res = out_bn + input

        return res, out

      with tf.variable_scope('input_conv'):
        input_weights = tf.get_variable('w_inp', [1, self.input_frequency_size, num_of_filters], tf.float32, initializer=default_init)
        res = tf.tanh(tf.nn.convolution(fingerprint3d, input_weights, 'SAME'))
        res = slim.batch_norm(res,is_training=self.is_training,decay=0.96,updates_collections=None)
      skip = 0
      for i in range(num_blocks):
        for r in dilation_rates:
          res, s = res_block(res, filter_size, num_of_filters, r, i)
          skip += s
      with tf.variable_scope('pre_pooling_conv'):
        skip_sum_weights = tf.get_variable('w_pre_pooling', [1, num_of_filters, num_of_filters], tf.float32, initializer=default_init)
        pre_pooling_conv = tf.tanh(tf.nn.convolution(skip, skip_sum_weights, 'SAME'))
        # pre_pooling_conv_bn = tf.layers.batch_normalization(pre_pooling_conv, training=self.is_training)
        pre_pooling_conv_bn = slim.batch_norm(pre_pooling_conv,is_training=self.is_training,decay=0.96,updates_collections=None)
      global_pool = tf.reduce_mean(pre_pooling_conv_bn, axis=1)

      label_count = self.model_settings['label_count']
      with tf.variable_scope('final_layer'):
        final_fc_weights = tf.get_variable('w_softmax', [num_of_filters, label_count], tf.float32, initializer=default_init)
        final_fc_bias = tf.get_variable('b_softmax', [label_count], tf.float32, initializer=tf.constant_initializer(0))
        final_fc = tf.matmul(global_pool, final_fc_weights) + final_fc_bias

      return final_fc

    def create_gated_mfcc_wave_net(self):

      num_of_filters = int(self.model_settings['num_of_filters'])
      default_init = tf.contrib.layers.xavier_initializer()
      num_blocks = int(self.model_settings['number_of_wave_net_blocks'])
      filter_size = int(self.model_settings['filter_size'])
      dilation_rates = list(map(int, self.model_settings['dilation_rates'].split(',')))
      fingerprint3d = tf.squeeze(self.fingerprint_4d, 3)

      def res_block(input, filter_length, num_of_filters, rate, block):
        with tf.variable_scope(name_or_scope='block_%d_%d' % (block, rate)):
          kernel_shape = [filter_length, num_of_filters, num_of_filters]
          filter_weights = tf.get_variable('w_filter', kernel_shape, tf.float32, initializer=default_init)
          gate_weights = tf.get_variable('w_gate', kernel_shape, tf.float32, initializer=default_init)
          filter = tf.nn.convolution(input, filter_weights, 'SAME', dilation_rate=[rate])
          gate = tf.nn.convolution(input, gate_weights, 'SAME', dilation_rate=[rate])
          filter_out = tf.nn.tanh(filter)
          # filter_bn = tf.layers.batch_normalization(filter_out, training=self.is_training)
          filter_bn = slim.batch_norm(filter_out, is_training=self.is_training, decay=0.96, updates_collections=None)
          gate_out = tf.nn.relu(gate)
          # gate_bn = tf.layers.batch_normalization(gate_out, training=self.is_training)
          gate_bn = slim.batch_norm(gate_out, is_training=self.is_training, decay=0.96, updates_collections=None)
          out = filter_bn * gate_bn

          outWeights = tf.get_variable('w_out', [1, num_of_filters, num_of_filters], tf.float32, initializer=default_init)
          out = tf.nn.convolution(out, outWeights, 'SAME')
          out = tf.tanh(out)
          out_bn = slim.batch_norm(out,is_training=self.is_training,decay=0.96,updates_collections=None)
          res = out_bn + input

        return res, out

      with tf.variable_scope('input_conv'):
        input_weights = tf.get_variable('w_inp', [1, self.input_frequency_size, num_of_filters], tf.float32, initializer=default_init)
        res = tf.tanh(tf.nn.convolution(fingerprint3d, input_weights, 'SAME'))
        res = slim.batch_norm(res,is_training=self.is_training,decay=0.96,updates_collections=None)
      skip = 0
      for i in range(num_blocks):
        for r in dilation_rates:
          res, s = res_block(res, filter_size, num_of_filters, r, i)
          skip += s
      with tf.variable_scope('pre_pooling_conv'):
        skip_sum_weights = tf.get_variable('w_pre_pooling', [1, num_of_filters, num_of_filters], tf.float32, initializer=default_init)
        pre_pooling_conv = tf.tanh(tf.nn.convolution(skip, skip_sum_weights, 'SAME'))
        # pre_pooling_conv_bn = tf.layers.batch_normalization(pre_pooling_conv, training=self.is_training)
        pre_pooling_conv_bn = slim.batch_norm(pre_pooling_conv,is_training=self.is_training,decay=0.96,updates_collections=None)
      global_pool = tf.reduce_mean(pre_pooling_conv_bn, axis=1)

      label_count = self.model_settings['label_count']
      with tf.variable_scope('final_layer'):
        final_fc_weights = tf.get_variable('w_softmax', [num_of_filters, label_count], tf.float32, initializer=default_init)
        final_fc_bias = tf.get_variable('b_softmax', [label_count], tf.float32, initializer=tf.constant_initializer(0))
        final_fc = tf.matmul(global_pool, final_fc_weights) + final_fc_bias

      return final_fc

    def create_conv1d_model(self):
      fingerprint_3d = tf.reshape(self.fingerprint_input, [-1, self.fingerprint_size, 1])  # [batch, in_width, in_channels]

      # conv blocks
      block_1 = conv_pooling_block(fingerprint_3d, 80, 4, 64, 1, self.is_training, self.dropout_prob, 'block_1', 'max', 4)
      block_2 = conv_pooling_block(block_1, 3, 1, 64, 1, self.is_training, self.dropout_prob, 'block_2', 'max', 4)
      block_3 = conv_pooling_block(block_2, 3, 1, 128, 1, self.is_training, self.dropout_prob, 'block_3', 'max', 4)
      block_4 = conv_pooling_block(block_3, 3, 1, 256, 1, self.is_training, self.dropout_prob, 'block_4', 'max', 4)
      block_5 = conv_pooling_block(block_4, 3, 1, 512, 1, self.is_training, self.dropout_prob, 'block_5', 'max', 4)
      block_6 = conv_pooling_block(block_5, 3, 1, 1024, 1, self.is_training, self.dropout_prob, 'block_6', 'avg', 4)

      with tf.variable_scope('fc_layer'):
        fc_input = block_6

        label_count = self.model_settings['label_count']
        final_fc_weights = tf.Variable(
          tf.truncated_normal([tf.cast(fc_input.get_shape()[1], tf.int32), label_count], stddev=0.01))
        final_fc_bias = tf.Variable(tf.zeros([label_count]))
        final_fc = tf.matmul(fc_input, final_fc_weights) + final_fc_bias
        print(final_fc)

        return final_fc

    def create_ds_cnn_model(self):
      """Builds a model with depthwise separable convolutional neural network
      Model definition is based on https://arxiv.org/abs/1704.04861 and
      Tensorflow implementation: https://github.com/Zehaos/MobileNet

      model_size_info: defines number of layers, followed by the DS-Conv layer
        parameters in the order {number of conv features, conv filter height,
        width and stride in y,x dir.} for each of the layers.
      Note that first layer is always regular convolution, but the remaining
        layers are all depthwise separable convolutions.
      """

      def ds_cnn_arg_scope(weight_decay=0):
        """Defines the default ds_cnn argument scope.
        Args:
          weight_decay: The weight decay to use for regularizing the model.
        Returns:
          An `arg_scope` to use for the DS-CNN model.
        """
        with slim.arg_scope(
                [slim.convolution2d, slim.separable_convolution2d],
                weights_initializer=slim.initializers.xavier_initializer(),
                biases_initializer=slim.init_ops.zeros_initializer(),
                weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
          return sc

      def _depthwise_separable_conv(inputs,
                                    num_pwc_filters,
                                    sc,
                                    kernel_size,
                                    stride):
        """ Helper function to build the depth-wise separable convolution layer.
        """

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1,
                                                      kernel_size=kernel_size,
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(bn,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

      if self.is_training is not None:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

      t_dim = self.input_time_size
      f_dim = self.input_frequency_size
      self.model_size_info = list(map(int, self.model_settings['model_size_info'].split(',')))
      # Extract model dimensions from model_size_info
      num_layers = self.model_size_info[0]
      conv_feat = [None] * num_layers
      conv_kt = [None] * num_layers
      conv_kf = [None] * num_layers
      conv_st = [None] * num_layers
      conv_sf = [None] * num_layers
      i = 1
      for layer_no in range(0, num_layers):
        conv_feat[layer_no] = self.model_size_info[i]
        i += 1
        conv_kt[layer_no] = self.model_size_info[i]
        i += 1
        conv_kf[layer_no] = self.model_size_info[i]
        i += 1
        conv_st[layer_no] = self.model_size_info[i]
        i += 1
        conv_sf[layer_no] = self.model_size_info[i]
        i += 1

      scope = 'DS-CNN'
      with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            weights_initializer=slim.initializers.xavier_initializer(),
                            biases_initializer=slim.init_ops.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
          with slim.arg_scope([slim.batch_norm],
                              is_training=self.is_training,
                              decay=0.96,
                              updates_collections=None,
                              activation_fn=tf.nn.relu):
            for layer_no in range(0, num_layers):
              if layer_no == 0:
                net = slim.convolution2d(self.fingerprint_4d, conv_feat[layer_no],[conv_kt[layer_no], conv_kf[layer_no]],
                                         stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME', scope='conv_1')
                net = slim.batch_norm(net, scope='conv_1/batch_norm')
              else:
                net = _depthwise_separable_conv(net, conv_feat[layer_no],kernel_size=[conv_kt[layer_no], conv_kf[layer_no]],
                                                stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                sc='conv_ds_' + str(layer_no))
              t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
              f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

            net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')

        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(net, self.label_count, activation_fn=None, scope='fc1')

      return logits
