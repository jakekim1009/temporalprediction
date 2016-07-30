"""
conv, deconv, and linear ops and a few others are adapted from https://github.com/carpedm20/DCGAN-tensorflow
constrained conv and special deconv are adapted from https://github.com/openai/improved-gan/tree/master/imagenet
minibatch discrimination layer is adapted from https://github.com/openai/improved-gan/tree/master/imagenet, although
     they do not seem to package minibatch discrim as a layer
"""

import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        if needs_reshape:
           normed = tf.reshape(normed, orig_shape)

        return normed

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.
    For brevity, let `x = `, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def max_pool(input_, k_h=2, k_w=2, d_h=2, d_w=2, name='max_pool'):
    with tf.variable_scope(name):
         return tf.nn.max_pool(input_, [1, k_h, k_w, 1], [1, d_h, d_w, 1], padding='SAME')

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def conv2d_2(input_1, input_2, output_dim,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d_2'):
    with tf.variable_scope(name):
        w_1 = tf.get_variable('w_1', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv_1 = tf.nn.conv2d(input_1, w_1, strides=[1, d_h, d_w, 1], padding='SAME')

        w_2 = tf.get_variable('w_2', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv_2 = tf.nn.conv2d(input_2, w_2, strides=[1, d_h, d_w, 1], padding='SAME')

        conv = conv_1 + conv_2

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def constrained_conv2d(input_, output_dim,
           k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    assert k_h % d_h == 0
    assert k_w % d_w == 0
    # constrained to have stride be a factor of kernel width
    # this is intended to reduce convolution artifacts
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        # This is meant to reduce boundary artifacts
        padded = tf.pad(input_, [[0, 0],
            [k_h-1, 0],
            [k_w-1, 0],
            [0, 0]])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv

def special_deconv2d(input_, output_shape,
             k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False,
             init_bias=0.):
    # designed to reduce padding and stride artifacts in the generator

    # If the following fail, it is hard to avoid grid pattern artifacts
    assert k_h % d_h == 0
    assert k_w % d_w == 0

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        def check_shape(h_size, im_size, stride):
            if h_size != (im_size + stride - 1) // stride:
                print "Need h_size == (im_size + stride - 1) // stride"
                print "h_size: ", h_size
                print "im_size: ", im_size
                print "stride: ", stride
                print "(im_size + stride - 1) / float(stride): ", (im_size + stride - 1) / float(stride)
                raise ValueError()

        check_shape(int(input_.get_shape()[1]), output_shape[1] + k_h, d_h)
        check_shape(int(input_.get_shape()[2]), output_shape[2] + k_w, d_w)

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[output_shape[0],
            output_shape[1] + k_h, output_shape[2] + k_w, output_shape[3]],
                                strides=[1, d_h, d_w, 1])
        deconv = tf.slice(deconv, [0, k_h // 2, k_w // 2, 0], output_shape)


        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear_2(input_1, input_2, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape_1 = input_1.get_shape().as_list()
    shape_2 = input_2.get_shape().as_list()

    with tf.variable_scope(scope or "Linear_2"):
         matrix_1 = tf.get_variable("Matrix_1", [shape_1[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
         matrix_2 = tf.get_variable("Matrix_2", [shape_2[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
         bias = tf.get_variable("bias", [output_size],
                initializer=tf.constant_initializer(bias_start))

         out = tf.matmul(input_1, matrix_1) + tf.matmul(input_2, matrix_2) + bias

         if with_w:
            return out, matrix_1, matrix_2, bias
         return out

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def horizontal_gradient(images):
    images = tf.image.rgb_to_grayscale(images)
    dim = tf.shape(images)[1]
    return tf.abs(tf.slice(images, [0, 1, 0, 0], [-1, -1, -1, -1]) - tf.slice(images, [0,0,0,0], [-1, dim-1, -1, -1]))

def vertical_gradient(images):
    images = tf.image.rgb_to_grayscale(images)
    dim = tf.shape(images)[2]
    return tf.abs(tf.slice(images, [0,0,1,0], [-1, -1, -1, -1]) - tf.slice(images, [0,0,0,0], [-1, -1, dim-1, -1]))


def minibatch_discrimination(input_, n_kernels=60, dim_per_kernel=10, name='minibatch_discrimination'):
    # implementation of minibatch_discrimintation from https://arxiv.org/abs/1606.03498
    # n_kernels is B, dim_per_kernel is C in the notation in the paper
    shape = input_.get_shape().as_list()
    batch_size = shape[0]

    # The comments below use the notation from the original paper
    with tf.variable_scope(name):
         input_reshaped = tf.reshape(input_, [batch_size, -1])
         # T is x length by n_kernels by dim_per_kernel
         x = linear(input_reshaped, n_kernels * dim_per_kernel, scope="minibatch_T")
         M = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

         # Compute pairwise distances
         big = np.zeros((batch_size, batch_size), dtype='float32')
         big += np.eye(batch_size)
         big = tf.expand_dims(big, 1)

         abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(M, 3) - tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0)), 2)
         # abs_dif[a,b,c] = || M_a,row_b - M_c,row_b ||_L1
         mask = 1. - big
         masked = tf.exp(-abs_dif) * mask # masked[a, b, a] == 0, masked[a, b, c (!=a)] = c_b(x_a, x_c)
         # tf.reduce_sum(masked, 2)[a, b] = sum_c c_b(x_a, x_c) = o(x_a)_b
         minibatch_features = tf.reduce_sum(masked, 2) / (batch_size * batch_size - batch_size) # tf.reduce_sum(mask)

         return tf.concat(1, [input_, minibatch_features])