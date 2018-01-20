import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.slim as slim

def clipped_error(x):
  # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        # b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # out = tf.nn.bias_add(conv, b, data_format)
        out = conv

    if activation_fn != None:
        out = activation_fn(out)

    return out, w


def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    shape=input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev = stddev))
        b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
        return activation_fn(out), w, b
    else:
        return out, w, b


def elementwise_mat_prod(input_, data_format='NHWC', name='elemnt_wise'):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], shape[2], 1],
                            tf.float32,
                            tf.constant_initializer(1.0))

        multiplier = tf.tile(w, [1, 1, shape[-1]])

        out = tf.multiply(input_, multiplier)

    return out, w


def weighted_sum(input_, data_format='NDHWC', padding='VALID', name='weighted_sum'):
    shape = input_.get_shape().as_list()
    #tesing: ignore kernel_size and derive it from the input shape

    with tf.variable_scope(name):
        # kernel_shape = [kernel_size[0], kernel_size[1], input_.get_shape()[-1], 1]
        kernel_shape = [shape[1], shape[2], 1, 1, 1]
        stride = [1, 1, 1, 1, 1]
        initializer = tf.constant_initializer(1.0 / (shape[1] * shape[2]))

        input_expanded = tf.expand_dims(input_, axis=-1) # [batch, height,  width, channels, 1]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        out = tf.nn.conv3d(input_expanded, w, stride, padding=padding, data_format=data_format)


    return out, w


def depthwise_separable_conv(input_, output_size, is_training, kernel=(3, 3), stride=(1, 1), data_format='NDHWC', padding='VALID', name='dw_conv'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        # kernel_shape = [kernel_size[0], kernel_size[1], input_.get_shape()[-1], 1]
        kernel_shape = [kernel[0], kernel[1], shape[-1], 1]
        stride_shape = [1, stride[0], stride[1], 1]

        initializer = tf.contrib.layers.xavier_initializer()
        filter_dw = tf.get_variable('filter_dw', kernel_shape, tf.float32, initializer=initializer)

        depthwise_conv = tf.nn.depthwise_conv2d(input_, filter_dw, stride_shape, padding=padding)

        batch_norm1 = slim.batch_norm(depthwise_conv,
                                     is_training=is_training,
                                     decay=0.96,
                                     updates_collections=None,
                                     activation_fn=tf.nn.relu,
                                     scope='dw_batch_norm')
        # batch_norm_1 = tf.layers.batch_normalization(depthwise_conv, momentum=0.96, training=is_training)
        # relu_1 = tf.nn.relu(batch_norm_1)

        pointwise_kernel_shape = [1, 1, shape[-1], output_size]
        pointwise_stride_shape = [1, 1, 1, 1]

        filter_pw = tf.get_variable('filter_pw', pointwise_kernel_shape, tf.float32, initializer=initializer)

        pointwise_conv = tf.nn.conv2d(batch_norm1, filter_pw, pointwise_stride_shape, padding=padding)

        batch_norm2 = slim.batch_norm(pointwise_conv,
                                      is_training=is_training,
                                      decay=0.96,
                                      updates_collections=None,
                                      activation_fn=tf.nn.relu,
                                      scope='pw_batch_norm')


        # batch_norm_2 = tf.layers.batch_normalization(pointwise_conv, training=is_training)
        # relu_2 = tf.nn.relu(batch_norm_2)
        out = batch_norm2

    return out, filter_dw, filter_pw

def stacked_conv_pooling(input, filter_width, stride, num_filters, num_stacked_conv, is_training, dropout_prob, scope,
                         pooling_type='max', pooling_size=4):
  with tf.variable_scope(scope):
    receptive_field_size_ = filter_width
    stride_ = stride
    num_filters_ = num_filters
    input_ = input

    # stacked convolutions
    for conv_id in range(num_stacked_conv):
      input_ = conv_relu(input_, receptive_field_size_, stride_, num_filters_, is_training, dropout_prob,
                             'conv_relu-' + str(conv_id))
    conv_out = input_

    # pooling
    if pooling_type == 'avg':
      pooling_ = tf.reduce_mean(conv_out, 1)
    else:
      pooling_ = tf.nn.pool(conv_out, [pooling_size], strides=[4], pooling_type='MAX', padding='VALID')

    return pooling_

def conv_relu(input_, filter_width, stride_, num_filters, is_training, dropout_prob, scope, padding='SAME'):
  with tf.variable_scope(scope):
    in_channels = input_.get_shape()[2]
    weights_ = tf.get_variable('weights', [filter_width, in_channels, num_filters], tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
    bias_ = tf.get_variable('bias', [num_filters], tf.float32, initializer=tf.constant_initializer(0))
    conv_ = tf.nn.conv1d(input_, weights_, stride=stride_, padding=padding) + bias_

    # batch normalization
    bn = slim.batch_norm(conv_, is_training=is_training, decay=0.9, updates_collections=None)

    # relu
    relu_ = tf.nn.relu(bn)

    return relu_

def residual_block_1d(input, filter_width, stride_, num_filters, is_training, dropout_prob, scope, pooling_type='', pooling_size=4):
  with tf.variable_scope(scope):
    conv_1 = conv_relu(input, filter_width, stride_, num_filters, is_training, dropout_prob, scope + '/conv_1')
    conv_2 = conv_relu(conv_1, filter_width, stride_, num_filters, is_training, dropout_prob, scope + '/conv_2')
    conv = tf.concat([conv_1, conv_2], axis=2)
    if num_filters != input.get_shape()[2]:
      conv_out = input + conv
    else:
      conv_out = conv

    # pooling
    if pooling_type == 'avg':
      result = tf.reduce_mean(conv_out, 1)
    elif pooling_type == 'max':
      result = tf.nn.pool(conv_out, [pooling_size], strides=[4], pooling_type='MAX', padding='VALID')
    else:
      result = conv_out

  return result



